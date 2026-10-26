from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime, timedelta
from typing import Any

from quantterm.models import Exchange, OhlcvBar, session_bar_ts
from uphill.market.eod import ACTIVE_NSE_LISTINGS, LATEST_DAILY_BAR, EodUpdater, pending_sessions

IST = Exchange.NSE.timezone


def _ist(y: int, m: int, d: int, hh: int, mm: int = 0) -> datetime:
    return datetime(y, m, d, hh, mm, tzinfo=IST).astimezone(UTC)


def test_pending_sessions_skip_weekends_and_wait_for_publication() -> None:
    # Fri 23 Oct 2026 stored; Tue 27 Oct before 18:30 IST -> Mon 26 only
    assert pending_sessions(date(2026, 10, 23), _ist(2026, 10, 27, 12)) == [date(2026, 10, 26)]
    assert pending_sessions(date(2026, 10, 23), _ist(2026, 10, 27, 19)) == [date(2026, 10, 26), date(2026, 10, 27)]
    assert pending_sessions(date(2026, 10, 27), _ist(2026, 10, 27, 19)) == []
    assert pending_sessions(date(2026, 10, 23), _ist(2026, 10, 25, 20)) == []  # weekend
    assert len(pending_sessions(None, _ist(2026, 10, 27, 19))) <= 46  # bounded first catch-up


class FakeDb:
    def __init__(self, latest: datetime | None, tickers: list[str]) -> None:
        self.latest, self.tickers = latest, tickers

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        if query is LATEST_DAILY_BAR:
            return [{"latest": self.latest}]
        assert query is ACTIVE_NSE_LISTINGS
        return [{"ticker": t} for t in self.tickers]


class FakePrices:
    def __init__(self) -> None:
        self.rows: list[tuple[str, date]] = []

    async def append_bars(self, symbol: str, exchange: Exchange, interval: timedelta, bars: list[OhlcvBar],
                          source: str) -> int:
        self.rows += [(symbol, b.ts.astimezone(IST).date()) for b in bars]
        return len(bars)


class FakeBhavcopy:
    def __init__(self, sessions: dict[date, list[str]]) -> None:
        self.sessions, self.requested = sessions, []

    async def fetch_session(self, session: date) -> dict[str, OhlcvBar]:
        self.requested.append(session)
        return {s: OhlcvBar(ts=session_bar_ts(Exchange.NSE, session), open=10, high=11, low=9, close=10.5,
                            volume=100) for s in self.sessions.get(session, [])}


def test_eod_catch_up_inserts_listed_symbols_skips_holidays_and_refreshes() -> None:
    db = FakeDb(session_bar_ts(Exchange.NSE, date(2026, 10, 21)), ["INFY", "TCS"])
    prices = FakePrices()
    # Thu 22 = holiday (no file), Fri 23 traded, Mon 26 traded, Tue 27 (today) not yet published
    source = FakeBhavcopy({date(2026, 10, 23): ["INFY", "TCS", "OTHER"], date(2026, 10, 26): ["INFY"]})
    refreshed: list[bool] = []

    async def on_update() -> None:
        refreshed.append(True)

    updater = EodUpdater(db, prices, source, on_update=on_update,  # type: ignore[arg-type]
                         clock=lambda: _ist(2026, 10, 27, 19))
    assert asyncio.run(updater.catch_up()) == 3
    assert prices.rows == [("INFY", date(2026, 10, 23)), ("TCS", date(2026, 10, 23)), ("INFY", date(2026, 10, 26))]
    assert source.requested == [date(2026, 10, 22), date(2026, 10, 23), date(2026, 10, 26), date(2026, 10, 27)]
    assert updater.last_session == date(2026, 10, 26) and refreshed == [True]
