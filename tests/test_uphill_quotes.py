from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import Any

from fakes import InMemoryBroker
from pydantic import SecretStr

import uphill.market.quotes as quotes_mod
from quantterm.models import Exchange
from quantterm.streaming import tick_channel
from uphill.market.quotes import LiveQuotePoller, nse_hours

IST = Exchange.NSE.timezone


def test_nse_hours() -> None:
    assert nse_hours(datetime(2026, 10, 27, 10, 0, tzinfo=IST))
    assert not nse_hours(datetime(2026, 10, 27, 16, 0, tzinfo=IST))
    assert not nse_hours(datetime(2026, 10, 25, 10, 0, tzinfo=IST))  # Sunday


class FakeHttp:
    def __init__(self, prices: dict[str, float]) -> None:
        self.prices, self.calls = prices, []

    async def get_json(self, url: str, *, params: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        keys = params["instrument_key"].split(",")
        self.calls.append(len(keys))
        return {"status": "success", "data": {
            f"NSE_EQ:{k}": {"instrument_token": k, "last_price": self.prices[k], "cp": 100.0, "volume": 5}
            for k in keys if k in self.prices}}


def test_poller_batches_publishes_changes_only(monkeypatch: Any) -> None:
    symbols = [f"S{i}" for i in range(250)]

    async def instruments(http: Any, exchange: Exchange, wanted: list[str], strict: bool) -> dict[str, Any]:
        return {f"KEY{s}": (Exchange.NSE, s) for s in wanted}

    monkeypatch.setattr(quotes_mod, "load_upstox_instruments", instruments)
    http = FakeHttp({f"KEYS{i}": 100.0 + i for i in range(250)})
    broker = InMemoryBroker()
    now = datetime(2026, 10, 27, 5, 0, tzinfo=UTC)
    poller = LiveQuotePoller(http, SecretStr("t"), broker, symbols, clock=lambda: now)  # type: ignore[arg-type]

    async def scenario() -> tuple[int, int, list[str]]:
        async with broker.subscribe([tick_channel(Exchange.NSE, "S7")]) as messages:
            first = await poller.poll()
            http.prices["KEYS7"] = 999.5
            second = await poller.poll()
            got = [json.loads((await messages.__anext__())[1])["tick"]["price"] for _ in range(2)]
        return first, second, got

    first, second, got = asyncio.run(scenario())
    assert http.calls == [200, 50, 200, 50]
    assert (first, second) == (250, 1) and got == [107.0, 999.5]
    assert poller.quotes["NSE:S7"]["price"] == 999.5 and len(poller.listings) == 250


class CandleHttp:
    def __init__(self) -> None:
        self.urls: list[str] = []

    async def get_json(self, url: str, *, headers: dict[str, str]) -> dict[str, Any]:
        self.urls.append(url)
        return {"status": "success", "data": {"candles": [  # Upstox returns newest first
            ["2026-10-27T09:45:00+05:30", 101.0, 104.0, 100.5, 103.0, 700, 0],
            ["2026-10-27T09:15:00+05:30", 100.0, 102.0, 99.0, 101.0, 500, 0],
        ]}}


def test_intraday_candles_sorted_cached_and_limited_to_universe(monkeypatch: Any) -> None:
    async def instruments(http: Any, exchange: Exchange, wanted: list[str], strict: bool) -> dict[str, Any]:
        return {"NSE_EQ|INE0": (Exchange.NSE, "INFY")}

    monkeypatch.setattr(quotes_mod, "load_upstox_instruments", instruments)
    http = CandleHttp()
    poller = LiveQuotePoller(http, SecretStr("t"), InMemoryBroker(), ["INFY"])  # type: ignore[arg-type]
    candles = asyncio.run(poller.intraday("INFY", 30))
    assert candles is not None and [c[0] for c in candles] == [1793072700, 1793074500]
    assert candles[0] == (1793072700, 100.0, 102.0, 99.0, 101.0, 500.0)
    assert asyncio.run(poller.intraday("INFY", 30)) == candles and len(http.urls) == 1  # TTL cache
    assert http.urls[0].endswith("/v3/historical-candle/intraday/NSE_EQ%7CINE0/minutes/30")
    assert asyncio.run(poller.intraday("EVIL", 30)) is None and len(http.urls) == 1
