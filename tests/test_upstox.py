from __future__ import annotations

import asyncio
import gzip
import json
import struct
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import httpx
import pytest

from quantterm.connectors.base import SourceError
from quantterm.connectors.http import HttpFetcher
from quantterm.connectors.upstox import (
    UpstoxFeedDecoder,
    load_upstox_instruments,
    parse_feed_response,
    parse_upstox_instruments,
)
from quantterm.live import start_live_feeds
from quantterm.models import Exchange, OrderBookSnapshot, Tick

IST = ZoneInfo("Asia/Kolkata")
INFY_KEY = "NSE_EQ|INE009A01021"


# ---------------------------------------------------------------- minimal protobuf encoder (test-only)
def _varint(v: int) -> bytes:
    v &= (1 << 64) - 1
    out = bytearray()
    while True:
        b = v & 0x7F
        v >>= 7
        if v:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _i(n: int, v: int) -> bytes:
    return _varint(n << 3) + _varint(v)


def _d(n: int, v: float) -> bytes:
    return _varint(n << 3 | 1) + struct.pack("<d", v)


def _m(n: int, payload: bytes) -> bytes:
    return _varint(n << 3 | 2) + _varint(len(payload)) + payload


def _s(n: int, text: str) -> bytes:
    return _m(n, text.encode())


def _ltpc(ltp: float, ltt_ms: int, ltq: int, cp: float = 0.0) -> bytes:
    return _d(1, ltp) + _i(2, ltt_ms) + _i(3, ltq) + _d(4, cp)


def _full_feed(ltp: float, ltt_ms: int, ltq: int, vtt: int,
               quotes: tuple[tuple[int, float, int, float], ...] = ()) -> bytes:
    level = b"".join(_m(1, _i(1, bq) + _d(2, bp) + _i(3, aq) + _d(4, ap)) for bq, bp, aq, ap in quotes)
    market_ff = _m(1, _ltpc(ltp, ltt_ms, ltq)) + _m(2, level) + _i(6, vtt)
    return _m(2, _m(1, market_ff)) + _i(4, 1)


def _response(kind: int, feeds: dict[str, bytes], ts_ms: int, status: dict[str, int] | None = None) -> bytes:
    body = _i(1, kind) + b"".join(_m(2, _s(1, k) + _m(2, v)) for k, v in feeds.items()) + _i(3, ts_ms)
    if status:
        body += _m(4, b"".join(_m(1, _s(1, seg) + _i(2, code)) for seg, code in status.items()))
    return body


def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


# --------------------------------------------------------------------------------------------- tests
def test_full_feed_yields_depth_then_trade() -> None:
    decoder = UpstoxFeedDecoder({INFY_KEY: (Exchange.NSE, "INFY")})
    ltt = _ms(datetime(2026, 10, 26, 9, 30, tzinfo=IST))
    quotes = ((1200, 1928.4, 500, 1928.5), (800, 1928.35, 900, 1928.55))
    events = decoder.decode(_response(1, {INFY_KEY: _full_feed(1928.45, ltt, 25, 1_000_000, quotes)}, ltt + 50))
    book, tick = events
    assert isinstance(book, OrderBookSnapshot) and isinstance(tick, Tick)
    assert [(lvl.price, lvl.quantity) for lvl in book.bids] == [(1928.4, 1200), (1928.35, 800)]
    assert [(lvl.price, lvl.quantity) for lvl in book.asks] == [(1928.5, 500), (1928.55, 900)]
    assert (tick.symbol, tick.price, tick.size) == ("INFY", 1928.45, 25)
    assert tick.timestamp == datetime(2026, 10, 26, 4, 0, tzinfo=UTC)


def test_trade_size_is_cumulative_volume_delta() -> None:
    decoder = UpstoxFeedDecoder({INFY_KEY: (Exchange.NSE, "INFY")})
    ltt = _ms(datetime(2026, 10, 26, 9, 30, tzinfo=IST))

    def ticks(ltp: float, t: int, vtt: int) -> list[tuple[float, int]]:
        evs = decoder.decode(_response(1, {INFY_KEY: _full_feed(ltp, t, 5, vtt)}, t))
        return [(e.price, e.size) for e in evs if isinstance(e, Tick)]

    assert ticks(1928.45, ltt, 1_000_000) == [(1928.45, 5)]
    assert ticks(1928.45, ltt, 1_000_000) == []  # quote-only update
    assert ticks(1928.5, ltt + 1000, 1_000_340) == [(1928.5, 340)]


def test_ltpc_mode_emits_on_new_trade_time_and_ignores_unknown_keys() -> None:
    decoder = UpstoxFeedDecoder({INFY_KEY: (Exchange.NSE, "INFY")})
    ltt = _ms(datetime(2026, 10, 26, 9, 30, tzinfo=IST))
    msg = _response(1, {INFY_KEY: _m(1, _ltpc(1928.0, ltt, 7)), "NSE_EQ|OTHER": _m(1, _ltpc(1.0, ltt, 1))}, ltt)
    assert [(e.symbol, e.size) for e in decoder.decode(msg)] == [("INFY", 7)]
    assert decoder.decode(msg) == []
    later = _response(1, {INFY_KEY: _m(1, _ltpc(1929.0, ltt + 500, 3))}, ltt + 500)
    assert [(e.price, e.size) for e in decoder.decode(later)] == [(1929.0, 3)]


def test_market_info_status_is_parsed() -> None:
    response = parse_feed_response(_response(2, {}, 1, {"NSE_EQ": 3, "NSE_INDEX": 2}))
    assert response.type == "market_info"
    assert response.segment_status == {"NSE_EQ": "NORMAL_CLOSE", "NSE_INDEX": "NORMAL_OPEN"}


def test_truncated_message_raises() -> None:
    with pytest.raises(SourceError):
        parse_feed_response(_m(2, b"\x0a\x05ab")[:-1] + b"\x80")


ROWS = [
    {"segment": "NSE_EQ", "instrument_type": "EQ", "instrument_key": INFY_KEY, "trading_symbol": "INFY"},
    {"segment": "NSE_FO", "instrument_type": "FUT", "instrument_key": "NSE_FO|1", "trading_symbol": "INFY"},
    {"segment": "NSE_EQ", "instrument_type": "BE", "instrument_key": "NSE_EQ|X", "trading_symbol": "TCS"},
]


def test_instrument_master_requires_every_symbol() -> None:
    assert parse_upstox_instruments(ROWS, Exchange.NSE, ["INFY"]) == {INFY_KEY: (Exchange.NSE, "INFY")}
    with pytest.raises(SourceError, match="TCS"):
        parse_upstox_instruments(ROWS, Exchange.NSE, ["INFY", "TCS"])


def test_instrument_master_is_gunzipped() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/NSE.json.gz")
        return httpx.Response(200, content=gzip.compress(json.dumps(ROWS).encode()))

    async def scenario() -> dict[str, tuple[Exchange, str]]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await load_upstox_instruments(HttpFetcher(client, rate_per_second=1000), Exchange.NSE, ["INFY"])

    assert asyncio.run(scenario()) == {INFY_KEY: (Exchange.NSE, "INFY")}


def test_no_feed_starts_without_credentials() -> None:
    class _S:
        upstox_symbols = ["NSE:INFY"]
        upstox_analytics_token = None
        kite_symbols: list[str] = []

    async def scenario() -> int:
        return len(start_live_feeds(_S(), object()))  # type: ignore[arg-type]

    assert asyncio.run(scenario()) == 0
