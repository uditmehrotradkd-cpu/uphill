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
