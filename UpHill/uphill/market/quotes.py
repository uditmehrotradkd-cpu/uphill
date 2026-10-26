"""Live last-traded prices for the whole universe via Upstox's batched LTP quote API.

One request covers up to ~200 stocks, so ~500 listings cost 3 requests per poll: cheap enough for a 0.1-CPU
host, unlike streaming every tick. Prices go out on the same broker channel as streamed ticks, so the website's
live board needs no changes. Quotes are display-only and never written to ``raw_market_bars``.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, time
from typing import Any

from pydantic import SecretStr

from quantterm.connectors.http import HttpFetcher
from quantterm.connectors.upstox import load_upstox_instruments
from quantterm.models import Exchange
from quantterm.pubsub import Broker
from quantterm.streaming import tick_channel

log = logging.getLogger(__name__)

IST = Exchange.NSE.timezone
BATCH = 200
OPEN, CLOSE = time(9, 0), time(15, 35)  # pre-open through the closing print


def nse_hours(now: datetime) -> bool:
    local = now.astimezone(IST)
    return local.weekday() < 5 and OPEN <= local.time() <= CLOSE


class LiveQuotePoller:
    def __init__(self, http: HttpFetcher, token: SecretStr, broker: Broker, symbols: Sequence[str], *,
                 base_url: str = "https://api.upstox.com",
                 clock: Callable[[], datetime] = lambda: datetime.now(UTC)) -> None:
        self._http = http
        self._token = token
        self._broker = broker
        self._symbols = list(dict.fromkeys(symbols))
        self._base = base_url.rstrip("/")
        self._clock = clock
        self._keys: dict[str, str] = {}  # Upstox instrument key -> NSE symbol
        self.quotes: dict[str, dict[str, Any]] = {}  # "NSE:SYM" -> {"price", "prev_close", "volume", "timestamp"}
        self.updated: datetime | None = None

    @property
    def listings(self) -> list[str]:
        return [f"NSE:{s}" for s in self._symbols]

    async def poll(self) -> int:
        """Fetch every quote once; returns how many prices changed (and were published)."""
        if not self._keys:
            found = await load_upstox_instruments(self._http, Exchange.NSE, self._symbols, strict=False)
            self._keys = {key: sym for key, (_, sym) in found.items()}
        now = self._clock()
        changed = 0
        keys = sorted(self._keys)
        for i in range(0, len(keys), BATCH):
            payload = await self._http.get_json(
                f"{self._base}/v3/market-quote/ltp", params={"instrument_key": ",".join(keys[i:i + BATCH])},
                headers={"Accept": "application/json", "Authorization": f"Bearer {self._token.get_secret_value()}"},
            )
            for row in (payload.get("data") or {}).values():
                symbol = self._keys.get(str(row.get("instrument_token")))
                price = row.get("last_price")
                if symbol is None or not isinstance(price, int | float) or price <= 0:
                    continue
                listing = f"NSE:{symbol}"
                prev = self.quotes.get(listing)
                self.quotes[listing] = {"price": float(price), "prev_close": row.get("cp"),
                                        "volume": row.get("volume"), "timestamp": now.isoformat()}
                if prev is None or prev["price"] != price:
                    changed += 1
                    await self._broker.publish(tick_channel(Exchange.NSE, symbol), json.dumps(
                        {"tick": {"price": float(price), "size": 0, "timestamp": now.isoformat(),
                                  "prev_close": row.get("cp")}}))
        self.updated = now
        return changed

    async def run(self, interval_seconds: float) -> None:
        while True:
            if nse_hours(self._clock()) or self.updated is None:  # one poll off-hours fills the last price
                try:
                    changed = await self.poll()
                    log.info("live quotes: %d/%d prices changed", changed, len(self.quotes))
                except Exception as exc:
                    log.warning("live quote poll failed: %s", exc)
            await asyncio.sleep(interval_seconds)
