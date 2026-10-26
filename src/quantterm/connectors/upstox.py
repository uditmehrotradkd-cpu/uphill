"""Upstox Market Data Feed V3: protobuf ``full`` mode -> raw trade ``Tick``s and 5-level depth.

Authenticates with the read-only Analytics Token (1-year validity, no daily login). Like Kite, Upstox
streams snapshots: a tick is emitted when cumulative volume (``vtt``) increases, sized by the delta.
The protobuf schema (MarketDataFeed.proto v3) is decoded by a small wire-format reader, so no
generated code or protobuf runtime is required.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import logging
import struct
import uuid
from collections.abc import AsyncIterator, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from pydantic import SecretStr
from websockets.asyncio.client import connect
from websockets.exceptions import InvalidStatus, WebSocketException

from quantterm.connectors.base import SourceError
from quantterm.connectors.http import HttpFetcher, SourceHTTPError
from quantterm.models import BookLevel, Exchange, Market, OrderBookSnapshot, Tick

log = logging.getLogger(__name__)

UPSTOX_INSTRUMENTS_URL = "https://assets.upstox.com/market-quote/instruments/exchange/{exchange}.json.gz"
_IST_OFFSET_SECONDS = 19_800
_DOUBLE = struct.Struct("<d")
# FeedResponse.type / MarketInfo.segmentStatus enums from MarketDataFeed.proto v3
FEED_TYPES = ("initial_feed", "live_feed", "market_info")
MARKET_STATUS = ("PRE_OPEN_START", "PRE_OPEN_END", "NORMAL_OPEN", "NORMAL_CLOSE", "CLOSING_START", "CLOSING_END")

UpstoxEvent = Tick | OrderBookSnapshot


# ------------------------------------------------------------------------------ protobuf wire format
def _varint(buf: bytes, pos: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        if pos >= len(buf):
            raise SourceError("truncated protobuf varint")
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, pos
        shift += 7
        if shift > 63:
            raise SourceError("protobuf varint too long")


def _fields(buf: bytes) -> Iterator[tuple[int, int, int | bytes]]:
    """Yields (field number, wire type, value): ints for varint, raw bytes for fixed/length-delimited."""
    pos = 0
    while pos < len(buf):
        key, pos = _varint(buf, pos)
        number, wire = key >> 3, key & 7
        value: int | bytes
        if wire == 0:
            value, pos = _varint(buf, pos)
        elif wire == 1:
            value, pos = buf[pos : pos + 8], pos + 8
        elif wire == 2:
            length, pos = _varint(buf, pos)
            value, pos = buf[pos : pos + length], pos + length
        elif wire == 5:
            value, pos = buf[pos : pos + 4], pos + 4
        else:
            raise SourceError(f"unsupported protobuf wire type {wire}")
        if pos > len(buf):
            raise SourceError("truncated protobuf field")
        yield number, wire, value


def _int64(value: int | bytes) -> int:
    assert isinstance(value, int)
    return value - (1 << 64) if value >= 1 << 63 else value


def _double(value: int | bytes) -> float:
    assert isinstance(value, bytes)
    return float(_DOUBLE.unpack(value)[0])


def _bytes(value: int | bytes) -> bytes:
    assert isinstance(value, bytes)
    return value


@dataclass
class _Ltpc:
    ltp: float = 0.0
    ltt: int = 0  # epoch ms
    ltq: int = 0
    cp: float = 0.0


@dataclass
class _Snapshot:
    ltpc: _Ltpc | None = None
    quotes: list[tuple[int, float, int, float]] = field(default_factory=list)  # bidQ, bidP, askQ, askP
    vtt: int = 0


def _ltpc(buf: bytes) -> _Ltpc:
    out = _Ltpc()
    for n, w, v in _fields(buf):
        if n == 1 and w == 1:
            out.ltp = _double(v)
        elif n == 2 and w == 0:
            out.ltt = _int64(v)
        elif n == 3 and w == 0:
            out.ltq = _int64(v)
        elif n == 4 and w == 1:
            out.cp = _double(v)
    return out


def _quote(buf: bytes) -> tuple[int, float, int, float]:
    bid_q = ask_q = 0
    bid_p = ask_p = 0.0
    for n, w, v in _fields(buf):
        if n == 1 and w == 0:
            bid_q = _int64(v)
        elif n == 2 and w == 1:
            bid_p = _double(v)
        elif n == 3 and w == 0:
            ask_q = _int64(v)
        elif n == 4 and w == 1:
            ask_p = _double(v)
    return bid_q, bid_p, ask_q, ask_p


def _market_full(buf: bytes, snap: _Snapshot) -> None:
    for n, w, v in _fields(buf):
        if n == 1 and w == 2:
            snap.ltpc = _ltpc(_bytes(v))
        elif n == 2 and w == 2:
            snap.quotes = [_quote(_bytes(q)) for qn, qw, q in _fields(_bytes(v)) if qn == 1 and qw == 2]
        elif n == 6 and w == 0:
            snap.vtt = _int64(v)


def _feed(buf: bytes) -> _Snapshot:
    snap = _Snapshot()
    for n, w, v in _fields(buf):
        if w != 2:
            continue
        if n == 1:  # ltpc mode
            snap.ltpc = _ltpc(_bytes(v))
        elif n == 2:  # FullFeed { marketFF = 1 | indexFF = 2 }
            for fn, fw, fv in _fields(_bytes(v)):
                if fn == 1 and fw == 2:
                    _market_full(_bytes(fv), snap)
                elif fn == 2 and fw == 2:
                    for inner_n, inner_w, inner_v in _fields(_bytes(fv)):
                        if inner_n == 1 and inner_w == 2:
                            snap.ltpc = _ltpc(_bytes(inner_v))
    return snap


def _map_entry(buf: bytes) -> tuple[str, int | bytes | None]:
    key, value = "", None
    for n, _w, v in _fields(buf):
        if n == 1:
            key = _bytes(v).decode()
        elif n == 2:
            value = v
    return key, value


@dataclass
class FeedResponse:
    type: str
    current_ts: int
    feeds: dict[str, _Snapshot]
    segment_status: dict[str, str]


def parse_feed_response(message: bytes) -> FeedResponse:
    kind, ts = 0, 0
    feeds: dict[str, _Snapshot] = {}
    status: dict[str, str] = {}
    for n, w, v in _fields(message):
        if n == 1 and w == 0:
            kind = _int64(v)
        elif n == 2 and w == 2:
            key, value = _map_entry(_bytes(v))
            feeds[key] = _feed(value) if isinstance(value, bytes) else _Snapshot()
        elif n == 3 and w == 0:
            ts = _int64(v)
        elif n == 4 and w == 2:
            for mn, mw, mv in _fields(_bytes(v)):
                if mn == 1 and mw == 2:
                    segment, code = _map_entry(_bytes(mv))
                    idx = code if isinstance(code, int) else 0
                    status[segment] = MARKET_STATUS[idx] if idx < len(MARKET_STATUS) else str(idx)
    return FeedResponse(FEED_TYPES[kind] if 0 <= kind < len(FEED_TYPES) else str(kind), ts, feeds, status)


# ------------------------------------------------------------------------------------------- decoder
class UpstoxFeedDecoder:
    def __init__(self, instruments: Mapping[str, tuple[Exchange, str]]) -> None:
        self._instruments = dict(instruments)
        self._last: dict[str, tuple[int, int, int]] = {}  # key -> (cumulative volume, IST day, ltt)
        self.segment_status: dict[str, str] = {}

    @property
    def instrument_keys(self) -> list[str]:
        return sorted(self._instruments)

    def decode(self, message: bytes) -> list[UpstoxEvent]:
        response = parse_feed_response(message)
        if response.segment_status:
            self.segment_status.update(response.segment_status)
        events: list[UpstoxEvent] = []
        for key, snap in response.feeds.items():
            instrument = self._instruments.get(key)
            if instrument is None or snap.ltpc is None:
                continue
            exchange, symbol = instrument
            ltpc = snap.ltpc
            book_ms = response.current_ts or ltpc.ltt
            if snap.quotes and book_ms:
                bids = tuple(BookLevel(price=p, quantity=q) for q, p, _, _ in snap.quotes if p > 0)
                asks = tuple(BookLevel(price=p, quantity=q) for _, _, q, p in snap.quotes if p > 0)
                try:
                    if bids or asks:
                        events.append(OrderBookSnapshot(
                            symbol=symbol, exchange=exchange, timestamp=datetime.fromtimestamp(book_ms / 1000, tz=UTC),
                            bids=bids, asks=asks,
                        ))
                except ValueError:
                    log.warning("upstox depth for %s:%s not price-ordered; skipped", exchange.value, symbol)
            if ltpc.ltt <= 0 or ltpc.ltp <= 0:
                continue
            day = (ltpc.ltt // 1000 + _IST_OFFSET_SECONDS) // 86_400
            prev = self._last.get(key)
            if prev is not None and prev[1] == day:
                if snap.vtt:
                    if snap.vtt <= prev[0]:
                        continue
                    size = snap.vtt - prev[0]
                elif ltpc.ltt <= prev[2]:  # ltpc mode carries no cumulative volume
                    continue
                else:
                    size = ltpc.ltq
            else:
                size = ltpc.ltq  # first snapshot of the session: only the last print is attributable
            self._last[key] = (snap.vtt, day, ltpc.ltt)
            events.append(Tick(
                symbol=symbol, exchange=exchange, timestamp=datetime.fromtimestamp(ltpc.ltt / 1000, tz=UTC),
                price=ltpc.ltp, size=max(size, 0),
            ))
        return events


# ------------------------------------------------------------------------------------- instruments
def parse_upstox_instruments(
    rows: Iterable[Mapping[str, Any]], exchange: Exchange, symbols: Iterable[str], *, strict: bool = True
) -> dict[str, tuple[Exchange, str]]:
    wanted = set(symbols)
    segment = f"{exchange.value}_EQ"
    found: dict[str, tuple[Exchange, str]] = {}
    for row in rows:
        if row.get("segment") == segment and row.get("instrument_type") == "EQ" and row.get("trading_symbol") in wanted:
            found[str(row["instrument_key"])] = (exchange, str(row["trading_symbol"]))
    missing = wanted - {s for _, s in found.values()}
    if missing and strict:
        raise SourceError(f"Upstox instrument master has no {exchange.value} EQ for {sorted(missing)}")
    if missing:
        log.warning("Upstox instrument master has no %s EQ for %s; skipped", exchange.value, sorted(missing))
    return found


async def load_upstox_instruments(
    http: HttpFetcher, exchange: Exchange, symbols: Iterable[str], url_template: str = UPSTOX_INSTRUMENTS_URL,
    *, strict: bool = True,
) -> dict[str, tuple[Exchange, str]]:
    if exchange.market is not Market.IN:
        raise ValueError("Upstox instruments are NSE/BSE only")
    body = (await http.get(url_template.format(exchange=exchange.value))).content
    if body[:2] == b"\x1f\x8b":
        body = gzip.decompress(body)
    return parse_upstox_instruments(json.loads(body), exchange, symbols, strict=strict)


# ------------------------------------------------------------------------------------------ client
class UpstoxFeedClient:
    def __init__(
        self,
        token: SecretStr,
        decoder: UpstoxFeedDecoder,
        http: HttpFetcher,
        *,
        base_url: str = "https://api.upstox.com",
        mode: str = "full",
        max_backoff: float = 60.0,
    ) -> None:
        self._token = token
        self._decoder = decoder
        self._http = http
        self._base_url = base_url
        self._mode = mode
        self._max_backoff = max_backoff

    async def _authorized_uri(self) -> str:
        payload = await self._http.get_json(
            f"{self._base_url}/v3/feed/market-data-feed/authorize",
            headers={"Accept": "application/json", "Authorization": f"Bearer {self._token.get_secret_value()}"},
        )
        try:
            return str(payload["data"]["authorized_redirect_uri"])
        except (KeyError, TypeError) as exc:
            raise SourceError("Upstox authorize response has no authorized_redirect_uri") from exc

    async def events(self) -> AsyncIterator[UpstoxEvent]:
        """Reconnects with exponential backoff; authentication failures are raised, not retried."""
        request = json.dumps({
            "guid": uuid.uuid4().hex,
            "method": "sub",
            "data": {"mode": self._mode, "instrumentKeys": self._decoder.instrument_keys},
        }).encode()
        backoff = 1.0
        while True:
            try:
                uri = await self._authorized_uri()  # one-time code: never logged
                async with connect(uri, max_size=2**22) as ws:
                    await ws.send(request)  # Upstox requires the subscription as a binary frame
                    backoff = 1.0
                    async for message in ws:
                        if not isinstance(message, bytes):
                            continue
                        before = dict(self._decoder.segment_status)
                        for event in self._decoder.decode(message):
                            yield event
                        if self._decoder.segment_status != before:
                            log.info("upstox market status: %s", self._decoder.segment_status)
            except SourceHTTPError as exc:
                if exc.status in (401, 403):
                    raise SourceError("Upstox rejected the analytics token (expired or revoked?)") from exc
                log.warning("upstox authorize failed: HTTP %s", exc.status)
            except InvalidStatus as exc:
                log.warning("upstox handshake rejected (HTTP %s); retrying", exc.response.status_code)
            except (OSError, WebSocketException, SourceError) as exc:
                log.warning("upstox feed disconnected: %s", type(exc).__name__)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, self._max_backoff)
