"""Exercises the real SQL against PostgreSQL + TimescaleDB. Skipped unless QT_TEST_DATABASE_URL is set.

Use a disposable database: the test drops and recreates every table.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from quantterm.corporate_actions import CorporateAction, CorporateActionType, bonus_factor
from quantterm.db.pool import apply_schema
from quantterm.db.repositories import (
    DAILY,
    CorporateActionRepository,
    FundamentalsRepository,
    MacroRepository,
    OutlierRow,
    OutlierTickRepository,
    PriceRepository,
    ReconciliationRepository,
    SecurityRepository,
    TickRepository,
    TickRow,
)
from quantterm.engine.adjustments import get_adjusted_series
from quantterm.engine.pit_query import fetch_latest_snapshot, fetch_pit_snapshot, fetch_revision_history
from quantterm.models import Exchange, FundamentalFact, MacroObservation, OhlcvBar, PeriodType, session_bar_ts
from quantterm.pit import PointInTimeFundamentalsRepository, facts_to_frame
from quantterm.reconciliation import reconcile

DSN = os.environ.get("QT_TEST_DATABASE_URL")
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(DSN is None, reason="QT_TEST_DATABASE_URL not set"),
]
IST = ZoneInfo("Asia/Kolkata")
_TABLES = ("reconciliation_breaks", "ticks_raw", "outlier_ticks", "raw_market_bars", "corporate_actions",
           "macro_observations", "financial_metrics_pit", "option_quotes", "securities", "ohlcv_raw",
           "fundamental_facts")


def _with_pool(body: Callable[[Any], Awaitable[None]]) -> None:
    import asyncpg

    async def main() -> None:
        pool = await asyncpg.create_pool(DSN, min_size=1, max_size=4, server_settings={"timezone": "UTC"})
        try:
            await pool.execute(f"DROP TABLE IF EXISTS {', '.join(_TABLES)} CASCADE")
            await apply_schema(pool)
            await apply_schema(pool)  # idempotent re-apply
            await body(pool)
        finally:
            await pool.close()

    asyncio.run(main())


def _fact(value: str, filed: datetime, source: str, confidence: float = 1.0) -> FundamentalFact:
    return FundamentalFact(symbol="INFY", metric="revenue", period_type=PeriodType.QUARTER,
                           period_end_date=date(2024, 12, 31), filing_timestamp=filed, value=Decimal(value),
                           currency="INR", source=source, source_document_id="NSE:1189811",
                           confidence_score=confidence)


def test_end_to_end_sql() -> None:
    import asyncpg

    async def body(pool: Any) -> None:
        securities = SecurityRepository(pool)
        await securities.upsert("INFY", Exchange.NSE, isin="INE009A01021")
        facts = FundamentalsRepository(pool, securities)
        filed = datetime(2025, 1, 16, 19, 40, 31, tzinfo=IST)
        restated = datetime(2025, 4, 17, 19, 45, tzinfo=IST)
        assert await facts.append(Exchange.NSE, [_fact("417640000000", filed, "NSE_XBRL")]) == 1
        assert await facts.append(Exchange.NSE, [_fact("417640000000", filed, "NSE_XBRL")]) == 0
        assert await facts.append(Exchange.NSE, [_fact("417000000000", restated, "NSE_XBRL")]) == 1
        with pytest.raises(asyncpg.CheckViolationError, match="out-of-order"):
            await facts.append(Exchange.NSE, [_fact("1", datetime(2025, 2, 1, tzinfo=IST), "NSE_XBRL")])
        vendor = _fact("41764", datetime(2025, 2, 1, tzinfo=UTC), "FMP", confidence=0.8)
        assert await facts.append(Exchange.NSE, [vendor], only_if_changed=True) == 1
        assert await facts.append(
            Exchange.NSE, [vendor.model_copy(update={"filing_timestamp": datetime(2025, 2, 2, tzinfo=UTC)})],
            only_if_changed=True,
        ) == 0

        with pytest.raises(asyncpg.RaiseError, match="append-only"):
            await pool.execute("UPDATE financial_metrics_pit SET metric_value = 0")

        as_of = datetime(2025, 2, 3, tzinfo=UTC)
        snap = await fetch_pit_snapshot(pool, ["INFY"], Exchange.NSE, ["revenue"], PeriodType.QUARTER, as_of,
                                        "NSE_XBRL")
        assert [(s.revision_number, s.metric_value) for s in snap] == [(1, Decimal("417640000000"))]
        later = await fetch_pit_snapshot(pool, ["INFY"], Exchange.NSE, ["revenue"], PeriodType.QUARTER,
                                         datetime(2025, 5, 1, tzinfo=UTC), "NSE_XBRL")
        assert later[0].revision_number == 2
        best = await fetch_pit_snapshot(pool, ["INFY"], Exchange.NSE, ["revenue"], PeriodType.QUARTER, as_of)
        assert best[0].source_authority == "NSE_XBRL"
        latest = await fetch_latest_snapshot(pool, ["INFY"], Exchange.NSE, ["revenue"], PeriodType.QUARTER, as_of)
        assert len(latest) == 1
        history = await fetch_revision_history(pool, "INFY", Exchange.NSE, "revenue", PeriodType.QUARTER,
                                               date(2024, 12, 31), "NSE_XBRL", datetime(2025, 5, 1, tzinfo=UTC))
        assert [h.revision_number for h in history] == [1, 2]

        pit = PointInTimeFundamentalsRepository(pool)
        primary = await pit.fetch_as_of(["INFY"], Exchange.NSE, ["revenue"], PeriodType.QUARTER, "NSE_XBRL", as_of)
        secondary = await pit.fetch_as_of(["INFY"], Exchange.NSE, ["revenue"], PeriodType.QUARTER, "FMP", as_of)
        result = reconcile(facts_to_frame(primary), facts_to_frame(secondary))
        assert await ReconciliationRepository(pool).record_breaks(
            "INFY", Exchange.NSE, as_of, "NSE_XBRL", "FMP", result) == 1

        prices = PriceRepository(pool, securities)
        sessions = [date(2024, 6, 3), date(2024, 6, 4), date(2024, 6, 5)]
        bars = [OhlcvBar(ts=session_bar_ts(Exchange.NSE, d), open=c, high=c, low=c, close=c, volume=1000, vwap=c)
                for d, c in zip(sessions, [3000.0, 3010.0, 1508.0])]
        assert await prices.append_bars("INFY", Exchange.NSE, DAILY, bars, "NSE_BHAVCOPY") == 3
        assert await prices.append_bars("INFY", Exchange.NSE, DAILY, bars[:1], "OTHER") == 0
        await CorporateActionRepository(pool, securities).append([CorporateAction(
            symbol="INFY", exchange=Exchange.NSE, action_type=CorporateActionType.BONUS, ex_date=date(2024, 6, 5),
            announced_at=datetime(2024, 5, 20, tzinfo=IST), adjustment_factor=bonus_factor(Decimal(1), Decimal(1)),
            source="NSE_CA")])
        adjusted = await get_adjusted_series("INFY", date(2024, 6, 3), date(2024, 6, 5),
                                             datetime(2024, 6, 6, tzinfo=UTC), exchange=Exchange.NSE, db=pool)
        assert adjusted["close"].to_list() == [1500.0, 1505.0, 1508.0]
        assert adjusted["vwap"].to_list() == [1500.0, 1505.0, 1508.0]

        await TickRepository(pool).append([TickRow("INFY", "NSE", datetime(2025, 1, 16, 4, 0, tzinfo=UTC),
                                                   1928.45, 25, True, None, "KITE")])
        assert await pool.fetchval("SELECT count(*) FROM ticks_raw") == 1
        await OutlierTickRepository(pool).append([OutlierRow(
            "INFY", "NSE", datetime(2025, 1, 16, 4, 0, 1, tzinfo=UTC), 19.28, 25, "UNCONFIRMED_MAD_OUTLIER",
            1928.1, 0.05, 37889.0, 1928.0, 1928.2, 120.0, "KITE")])
        assert await pool.fetchval("SELECT reason FROM outlier_ticks") == "UNCONFIRMED_MAD_OUTLIER"

        macro = MacroRepository(pool)
        await macro.append([MacroObservation(series_id="RBI:REPO", observation_date=date(2025, 2, 7),
                                             release_timestamp=datetime(2025, 2, 7, 10, 0, tzinfo=IST),
                                             value=Decimal("6.25"), source="RBI")])
        frame = await macro.fetch_as_of("RBI:REPO", date(2025, 1, 1), date(2025, 12, 31), as_of)
        assert frame.is_empty()  # released after as_of
        frame = await macro.fetch_as_of("RBI:REPO", date(2025, 1, 1), date(2025, 12, 31),
                                        datetime(2025, 2, 7, 10, 0, tzinfo=IST))
        assert frame["value"].to_list() == [6.25]

    _with_pool(body)


def test_terminal_functions_on_real_sql() -> None:
    from market_seed import AS_OF, seed, settings

    from quantterm.terminal.command_router import CommandRouter
    from quantterm.workspace import FAView, GPView, OVMEView, RVView, Workspace

    async def body(pool: Any) -> None:
        await seed(pool)
        await OutlierTickRepository(pool).append([OutlierRow(
            "RELIANCE", "NSE", datetime(2024, 12, 31, 9, 0, tzinfo=UTC), 12.6, 5, "MAD_OUTLIER", 1261.0, 1.1, 1135.0,
            None, None, None, "KITE")])
        ws = Workspace(pool, settings(), clock=lambda: AS_OF)
        router = CommandRouter(ws, clock=lambda: AS_OF)
        views = [await router.execute(c) for c in ("RELIANCE IN Equity GP 6M", "FA", "RV", "OVME",
                                                    "HDFCBANK IN Equity FA", "AAPL US Equity GP")]
        assert [type(v.view) for v in views] == [GPView, FAView, RVView, OVMEView, FAView, GPView], \
            [v.message for v in views]
        rv = views[2].view
        assert isinstance(rv, RVView) and [r.security.symbol for r in rv.rows] == ["RELIANCE", "ONGC"]
        sec = await ws.resolve("RELIANCE", [Exchange.NSE])
        hygiene = await ws.tick_hygiene(sec, AS_OF)
        assert hygiene[0].value == 1 and hygiene[1].value == 1135.0
        revisions = await ws.revisions(sec, "revenue", PeriodType.FISCAL_YEAR, date(2024, 3, 31), AS_OF)
        assert [r.label for r in revisions] == ["revenue 2024-03-31 rev 1", "revenue 2024-03-31 rev 2"]

    _with_pool(body)


def test_demo_seed_on_real_sql() -> None:
    import asyncpg
    from market_seed import settings

    from quantterm.demo import DemoSeedRefused, assert_demo_only, demo_settings, seed_demo
    from quantterm.terminal.command_router import CommandRouter
    from quantterm.workspace import FAView, GPView, OVMEView, Workspace

    as_of = datetime(2026, 10, 23, 21, 0, tzinfo=UTC)

    async def body(pool: Any) -> None:
        await assert_demo_only(pool)
        demo = demo_settings(settings(), DSN or "")
        first = await seed_demo(pool, demo, as_of)
        again = await seed_demo(pool, demo, as_of)
        assert first.option_quotes > 200 and (again.bars, again.fundamentals, again.option_quotes) == (0, 0, 0)
        with pytest.raises(asyncpg.RaiseError, match="append-only"):
            await pool.execute("UPDATE option_quotes SET bid = 0")
        router = CommandRouter(Workspace(pool, demo, clock=lambda: as_of), clock=lambda: as_of)
        views = [await router.execute(c) for c in ("RELIANCE IN Equity GP", "TCS IN Equity FA", "AAPL US Equity OVME")]
        assert [type(v.view) for v in views] == [GPView, FAView, OVMEView], [v.message for v in views]
        ovme = views[2].view
        assert isinstance(ovme, OVMEView) and ovme.inputs[1].source == "DEMO_OPTIONS"
        assert all(r.call_quote is not None and r.call_quote.implied_vol is not None for r in ovme.chain)
        await pool.execute("INSERT INTO macro_observations VALUES ('FRED:DGS10', '2026-10-22', '2026-10-23', 4.1, 'FRED')")
        with pytest.raises(DemoSeedRefused, match="FRED"):
            await assert_demo_only(pool)

    _with_pool(body)
