"""Zerodha Kite Connect live feed: binary full-mode packets -> raw trade ``Tick``s and 5-level depth.

Kite streams quote snapshots, not individual prints. A tick is emitted only when cumulative volume
increases, with ``size`` = volume delta (all trades since the previous snapshot), priced at the last
traded price and stamped with the exchange last-trade time. Every full-mode packet also yields an
``OrderBookSnapshot`` (emitted before the tick) used by tick hygiene to confirm large moves.
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import io
import json
import logging
import struct
from collections.abc import AsyncIterator, Iterable, Mapping
from datetime import UTC, datetime
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx
from pydantic import SecretStr
from websockets.asyncio.client import connect
from websockets.exceptions import InvalidStatus, WebSocketException

from quantterm.connectors.base import SourceError
from quantterm.connectors.http import HttpFetcher
from quantterm.models import BookLevel, Exchange, Market, OrderBookSnapshot, Tick

log = logging.getLogger(__name__)

FULL_MODE_LENGTH = 184
_SEGMENT_DIVISOR = {3: 10_000_000.0, 6: 10_000.0}  # CDS, BCD; everything else is paise -> /100
_FULL_HEAD = struct.Struct(">IIIII")  # token, ltp, ltq, avg price, cumulative volume
_U32 = struct.Struct(">I")
_DEPTH_ENTRY = struct.Struct(">IIH2x")  # quantity, price, orders, padding
_DEPTH_OFFSET = 64
_DEPTH_LEVELS = 5
_IST_OFFSET_SECONDS = 19_800

KiteEvent = Tick | OrderBookSnapshot


def split_packets(message: bytes) -> list[bytes]:
    if len(message) < 2:
        return []  # 1-byte heartbeat
    (count,) = struct.unpack_from(">H", message, 0)
    packets: list[bytes] = []
    offset = 2
    for _ in range(count):
        if offset + 2 > len(message):
            raise SourceError("truncated Kite frame header")
        (length,) = struct.unpack_from(">H", message, offset)
        offset += 2
        if offset + length > len(message):
            raise SourceError("truncated Kite packet")
        packets.append(message[offset : offset + length])
        offset += length
    return packets


class KiteTickDecoder:
    def __init__(self, instruments: Mapping[int, tuple[Exchange, str]]) -> None:
        self._instruments = dict(instruments)
        self._last: dict[int, tuple[int, int]] = {}  # token -> (cumulative volume, IST day number)

    @property
    def tokens(self) -> list[int]:
        return sorted(self._instruments)

    def _depth(self, packet: bytes, divisor: float) -> tuple[tuple[BookLevel, ...], tuple[BookLevel, ...]]:
        sides: list[list[BookLevel]] = [[], []]
        for i in range(2 * _DEPTH_LEVELS):
            qty, price, orders = _DEPTH_ENTRY.unpack_from(packet, _DEPTH_OFFSET + i * _DEPTH_ENTRY.size)
            if price > 0:  # empty levels are zero-filled
                sides[i // _DEPTH_LEVELS].append(BookLevel(price=price / divisor, quantity=qty, orders=orders))
        return tuple(sides[0]), tuple(sides[1])

    def decode(self, message: bytes) -> list[KiteEvent]:
        events: list[KiteEvent] = []
        for packet in split_packets(message):
            if len(packet) != FULL_MODE_LENGTH:
                continue  # LTP/quote/index packets carry no trade time or depth
            token, ltp, ltq, _avg, volume = _FULL_HEAD.unpack_from(packet, 0)
            (ltt,) = _U32.unpack_from(packet, 44)
            (exchange_ts,) = _U32.unpack_from(packet, 60)
            instrument = self._instruments.get(token)
            if instrument is None:
                continue
            exchange, symbol = instrument
            divisor = _SEGMENT_DIVISOR.get(token & 0xFF, 100.0)
            book_ts = exchange_ts or ltt
            if book_ts:
                bids, asks = self._depth(packet, divisor)
                try:
                    if bids or asks:
                        events.append(OrderBookSnapshot(
                            symbol=symbol, exchange=exchange, timestamp=datetime.fromtimestamp(book_ts, tz=UTC),
                            bids=bids, asks=asks,
                        ))
                except ValueError:
                    log.warning("kite depth for %s:%s not price-ordered; skipped", exchange.value, symbol)
            if ltt == 0 or ltp == 0:
                continue
            day = (ltt + _IST_OFFSET_SECONDS) // 86_400
            prev = self._last.get(token)
            if prev is not None and prev[1] == day:
                if volume <= prev[0]:
                    continue
                size = volume - prev[0]
            else:
                size = ltq  # first snapshot of the session: only the last print is attributable
            self._last[token] = (volume, day)
            events.append(
                Tick(
                    symbol=symbol,
                    exchange=exchange,
                    timestamp=datetime.fromtimestamp(ltt, tz=UTC),
                    price=ltp / divisor,
                    size=size,
                )
            )
        return events


def parse_kite_instruments(
    csv_text: str, exchange: Exchange, symbols: Iterable[str]
) -> dict[int, tuple[Exchange, str]]:
    wanted = set(symbols)
    found: dict[int, tuple[Exchange, str]] = {}
    for row in csv.DictReader(io.StringIO(csv_text)):
        if row["exchange"] == exchange.value and row["instrument_type"] == "EQ" and row["tradingsymbol"] in wanted:
            found[int(row["instrument_token"])] = (exchange, row["tradingsymbol"])
    missing = wanted - {s for _, s in found.values()}
    if missing:
        raise SourceError(f"Kite instrument master has no {exchange.value} EQ for {sorted(missing)}")
    return found


def kite_login_url(api_key: SecretStr) -> str:
    return "https://kite.zerodha.com/connect/login?" + urlencode({"v": "3", "api_key": api_key.get_secret_value()})


def parse_request_token(value: str) -> str:
    """Accepts the bare ``request_token`` or the full redirect URL Kite sends the browser to after login."""
    value = value.strip()
    if "://" in value or "?" in value:
        query = parse_qs(urlsplit(value).query)
        if query.get("status", ["success"])[0] != "success" or not query.get("request_token"):
            raise ValueError("redirect URL carries no successful request_token")
        value = query["request_token"][0]
    if not value.isalnum():
        raise ValueError("request_token must be alphanumeric")
    return value


def kite_session_checksum(api_key: SecretStr, request_token: str, api_secret: SecretStr) -> str:
    raw = api_key.get_secret_value() + request_token + api_secret.get_secret_value()
    return hashlib.sha256(raw.encode()).hexdigest()


async def create_kite_session(
    client: httpx.AsyncClient,
    api_key: SecretStr,
    api_secret: SecretStr,
    request_token: str,
    base_url: str = "https://api.kite.trade",
) -> SecretStr:
    """Exchanges a one-time login ``request_token`` for the day's access token (valid until ~06:00 IST)."""
    response = await client.post(
        f"{base_url}/session/token",
        headers={"X-Kite-Version": "3"},
        data={
            "api_key": api_key.get_secret_value(),
            "request_token": request_token,
            "checksum": kite_session_checksum(api_key, request_token, api_secret),
        },
    )
    try:
        payload = response.json()
    except ValueError:
        payload = {}
    if response.status_code != 200 or payload.get("status") != "success":
        reason = payload.get("message") or f"HTTP {response.status_code}"
        raise SourceError(f"Kite session exchange failed: {reason}")
    return SecretStr(payload["data"]["access_token"])


def _auth_headers(api_key: SecretStr, access_token: SecretStr) -> dict[str, str]:
    return {
        "X-Kite-Version": "3",
        "Authorization": f"token {api_key.get_secret_value()}:{access_token.get_secret_value()}",
    }


async def load_kite_instruments(
    http: HttpFetcher,
    api_key: SecretStr,
    access_token: SecretStr,
    exchange: Exchange,
    symbols: Iterable[str],
    base_url: str = "https://api.kite.trade",
) -> dict[int, tuple[Exchange, str]]:
    if exchange.market is not Market.IN:
        raise ValueError("Kite instruments are NSE/BSE only")
    response = await http.get(f"{base_url}/instruments/{exchange.value}", headers=_auth_headers(api_key, access_token))
    return parse_kite_instruments(response.text, exchange, symbols)


class KiteTickerClient:
    def __init__(
        self,
        api_key: SecretStr,
        access_token: SecretStr,
        decoder: KiteTickDecoder,
        *,
        url: str = "wss://ws.kite.trade",
        max_backoff: float = 60.0,
    ) -> None:
        self._api_key = api_key
        self._token = access_token
        self._decoder = decoder
        self._url = url
        self._max_backoff = max_backoff

    async def events(self) -> AsyncIterator[KiteEvent]:
        """Reconnects with exponential backoff; authentication failures are raised, not retried."""
        tokens = self._decoder.tokens
        uri = f"{self._url}?api_key={self._api_key.get_secret_value()}&access_token={self._token.get_secret_value()}"
        backoff = 1.0
        while True:
            try:
                async with connect(uri, max_size=2**22) as ws:
                    await ws.send(json.dumps({"a": "subscribe", "v": tokens}))
                    await ws.send(json.dumps({"a": "mode", "v": ["full", tokens]}))
                    backoff = 1.0
                    async for message in ws:
                        if isinstance(message, bytes):
                            for event in self._decoder.decode(message):
                                yield event
                        else:
                            payload = json.loads(message)
                            if payload.get("type") == "error":
                                log.warning("kite feed error: %s", payload.get("data"))
            except InvalidStatus as exc:
                if exc.response.status_code in (401, 403):
                    raise
                log.warning("kite handshake rejected (HTTP %s); retrying", exc.response.status_code)
            except (OSError, WebSocketException, SourceError) as exc:
                log.warning("kite feed disconnected: %s", type(exc).__name__)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, self._max_backoff)
