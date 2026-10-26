"""All-stocks screener: per-listing EOD statistics for the whole universe, refreshed in the background."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any

from quantterm.models import Exchange
from quantterm.terminal.command_router import WorkspaceLike
from uphill.market.universe import UniverseStock

log = logging.getLogger(__name__)

LOOKBACK_DAYS = 366
# sessions back for each return horizon
HORIZONS = {"chg_1w": 5, "chg_1m": 21, "chg_3m": 63, "chg_6m": 126}


def _pct(now: float, then: float | None) -> float | None:
    return None if not then else (now / then - 1) * 100


def stats(closes: Sequence[float], volumes: Sequence[float]) -> dict[str, float | None]:
    if not closes:
        return {}
    last = closes[-1]
    out: dict[str, float | None] = {
        "last_close": last,
        "prev_close": closes[-2] if len(closes) > 1 else None,
        "chg_1d": _pct(last, closes[-2]) if len(closes) > 1 else None,
        "chg_1y": _pct(last, closes[0]) if len(closes) >= 240 else None,
        "high_52w": max(closes),
        "low_52w": min(closes),
        "avg_volume_20d": sum(volumes[-20:]) / len(volumes[-20:]) if volumes else None,
    }
    for key, n in HORIZONS.items():
        out[key] = _pct(last, closes[-n - 1]) if len(closes) > n else None
    out["from_high"] = _pct(last, out["high_52w"])
    return out


class Screener:
    def __init__(self, workspace: WorkspaceLike, universe: Sequence[UniverseStock],
                 clock: Callable[[], datetime] = lambda: datetime.now(UTC), concurrency: int = 8) -> None:
        self._ws = workspace
        self._universe = list(universe)
        self._clock = clock
        self._sem = asyncio.Semaphore(concurrency)
        self.rows: dict[str, dict[str, Any]] = {
            s.listing: self._base(s) | {"error": "loading"} for s in self._universe}
        self.updated: datetime | None = None

    @staticmethod
    def _base(s: UniverseStock) -> dict[str, Any]:
        return {"listing": s.listing, "symbol": s.symbol, "name": s.name, "industry": s.industry, "cap": s.cap}

    async def _one(self, s: UniverseStock, as_of: datetime) -> dict[str, Any]:
        async with self._sem:
            try:
                sec = await self._ws.resolve(s.symbol, (Exchange.NSE,))
                view = await self._ws.gp(sec, as_of, lookback_days=LOOKBACK_DAYS, adjust_splits=True,
                                         adjust_dividends=False)
            except Exception as exc:  # unregistered / no history: shown as such, never fatal
                return self._base(s) | {"error": f"{type(exc).__name__}: {exc}"[:160]}
        rsi = next((d.value for d in view.summary if d.label == "RSI(14)"), None)
        return self._base(s) | stats(view.close, view.volume) | {
            "rsi": rsi, "last_date": view.dates[-1].isoformat() if view.dates else None, "error": None}

    async def refresh(self) -> int:
        as_of = self._clock()
        results = await asyncio.gather(*(self._one(s, as_of) for s in self._universe))
        self.rows = {r["listing"]: r for r in results}
        self.updated = self._clock()
        return sum(r["error"] is None for r in results)

    async def run(self, interval_seconds: float) -> None:
        while True:
            try:
                ok = await self.refresh()
                log.info("screener refreshed: %d/%d listings with data", ok, len(self._universe))
            except Exception:
                log.exception("screener refresh failed")
            await asyncio.sleep(interval_seconds)
