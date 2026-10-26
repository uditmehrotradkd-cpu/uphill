"""Daily NSE end-of-day bars: load every session missed since the newest stored bar from the official bhavcopy.

Catching up (instead of a fixed daily cron) also covers days the free Render instance spent asleep.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, date, datetime, time, timedelta
from typing import Any

from quantterm.connectors.prices import NseBhavcopySource
from quantterm.db.repositories import DAILY, PriceRepository
from quantterm.models import Exchange

log = logging.getLogger(__name__)

IST = Exchange.NSE.timezone
PUBLISH_AFTER = time(18, 30)  # NSE posts the UDiFF bhavcopy around 18:00-19:00 IST
MAX_CATCH_UP_DAYS = 45

LATEST_DAILY_BAR = """
SELECT max("timestamp") AS latest FROM raw_market_bars WHERE exchange = 'NSE' AND bar_interval = '1 day'
"""
ACTIVE_NSE_LISTINGS = "SELECT ticker FROM securities WHERE exchange = 'NSE' AND delisted_on IS NULL"


def pending_sessions(last: date | None, now: datetime) -> list[date]:
    """Weekdays after ``last`` whose bhavcopy should exist by ``now`` (holidays are skipped on fetch)."""
    local = now.astimezone(IST)
    newest = local.date() if local.time() >= PUBLISH_AFTER else local.date() - timedelta(days=1)
    floor = local.date() - timedelta(days=MAX_CATCH_UP_DAYS)
    day = max(last + timedelta(days=1), floor) if last else floor
    out = []
    while day <= newest:
        if day.weekday() < 5:
            out.append(day)
        day += timedelta(days=1)
    return out


class EodUpdater:
    def __init__(self, db: Any, prices: PriceRepository, source: NseBhavcopySource, *,
                 on_update: Callable[[], Awaitable[object]] | None = None,
                 clock: Callable[[], datetime] = lambda: datetime.now(UTC)) -> None:
        self._db = db
        self._prices = prices
        self._source = source
        self._on_update = on_update
        self._clock = clock
        self.last_session: date | None = None
        self.checked: datetime | None = None

    async def catch_up(self) -> int:
        """Returns the number of bars inserted."""
        now = self._clock()
        rows = await self._db.fetch(LATEST_DAILY_BAR)
        latest = rows[0]["latest"] if rows else None
        last = latest.astimezone(IST).date() if latest else None
        listed = {r["ticker"] for r in await self._db.fetch(ACTIVE_NSE_LISTINGS)}
        today = now.astimezone(IST).date()
        inserted = 0
        for session in pending_sessions(last, now):
            bars = await self._source.fetch_session(session)
            if not bars:
                if session == today:  # not published yet; the next check retries
                    break
                continue  # exchange holiday
            n = 0
            for symbol in sorted(listed & bars.keys()):
                n += await self._prices.append_bars(symbol, Exchange.NSE, DAILY, [bars[symbol]],
                                                    NseBhavcopySource.source)
            log.info("EOD %s: %d bars inserted for %d listings", session, n, len(listed))
            inserted += n
            last = session
        self.last_session, self.checked = last, now
        if inserted and self._on_update is not None:
            await self._on_update()
        return inserted

    async def run(self, interval_seconds: float) -> None:
        while True:
            try:
                await self.catch_up()
            except Exception:
                log.exception("EOD update failed")
            await asyncio.sleep(interval_seconds)
