from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any, TypeVar
from zoneinfo import ZoneInfo

import polars as pl
import pytest
from fakes import FakeDatabase

from quantterm.corporate_actions import CorporateAction, CorporateActionType, bonus_factor
from quantterm.db.repositories import (
    DAILY,
    CorporateActionRepository,
    FundamentalsRepository,
    MacroRepository,
    PriceRepository,
    ReconciliationRepository,
    SecurityRepository,
    UnknownSecurityError,
)
from quantterm.models import (
    Exchange,
    FundamentalFact,
    MacroObservation,
    OhlcvBar,
    PeriodType,
    SectorClass,
    session_bar_ts,
)
from quantterm.pit import PointInTimeFundamentalsRepository, facts_to_frame
from quantterm.reconciliation import reconcile

T = TypeVar("T")
IST = ZoneInfo("Asia/Kolkata")


def run(coro: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coro)


@pytest.fixture
def db() -> FakeDatabase:
    return FakeDatabase()


@pytest.fixture
def securities(db: FakeDatabase) -> SecurityRepository:
    repo = SecurityRepository(db)
    run(repo.upsert("INFY", Exchange.NSE, isin="INE009A01021"))
    run(repo.upsert("HDFCBANK", Exchange.NSE, sector_class=SectorClass.BANK))
    return repo


def _fact(value: str, filed: datetime, source: str = "NSE_XBRL", end: date = date(2024, 12, 31)) -> FundamentalFact:
    return FundamentalFact(
        symbol="INFY", metric="revenue", period_type=PeriodType.QUARTER, period_end_date=end,
        filing_timestamp=filed, value=Decimal(value), currency="INR", source=source,
    )


def test_security_upsert_defaults_currency_and_unknown_raises(securities: SecurityRepository) -> None:
    rec = run(securities.get("INFY", Exchange.NSE))
    assert rec.currency == "INR" and rec.isin == "INE009A01021"
    assert run(securities.get("HDFCBANK", Exchange.NSE)).sector_class is SectorClass.BANK
    with pytest.raises(UnknownSecurityError):
        run(securities.get("INFY", Exchange.BSE))


def test_regulatory_facts_are_append_only_and_idempotent(db: FakeDatabase, securities: SecurityRepository) -> None:
    repo = FundamentalsRepository(db, securities)
    first = _fact("417640000000", datetime(2025, 1, 16, 19, 42, 44, tzinfo=IST))
    restated = _fact("417000000000", datetime(2025, 4, 17, 19, 0, tzinfo=IST))
    assert run(repo.append(Exchange.NSE, [first])) == 1
    assert run(repo.append(Exchange.NSE, [first])) == 0  # re-ingest of the same filing
    assert run(repo.append(Exchange.NSE, [restated])) == 1
    assert [(m["revision_number"], m["metric_value"]) for m in db.metrics] == [
        (1, Decimal("417640000000")), (2, Decimal("417000000000"))
    ]  # original vintage kept, restatement added as revision 2


def test_vendor_snapshot_only_versions_on_change(db: FakeDatabase, securities: SecurityRepository) -> None:
    repo = FundamentalsRepository(db, securities)
    day1 = _fact("417640000000", datetime(2025, 2, 1, tzinfo=UTC), source="FMP")
    day2_same = _fact("417640000000", datetime(2025, 2, 2, tzinfo=UTC), source="FMP")
    day3_changed = _fact("417700000000", datetime(2025, 2, 3, tzinfo=UTC), source="FMP")
    assert run(repo.append(Exchange.NSE, [day1], only_if_changed=True)) == 1
    assert run(repo.append(Exchange.NSE, [day2_same], only_if_changed=True)) == 0
    assert run(repo.append(Exchange.NSE, [day3_changed], only_if_changed=True)) == 1

    pit = PointInTimeFundamentalsRepository(db)
    seen = run(pit.fetch_as_of(["INFY"], Exchange.NSE, ["revenue"], PeriodType.QUARTER, "FMP",
                               datetime(2025, 2, 2, 12, tzinfo=UTC)))
    assert [f.value for f in seen] == [Decimal("417640000000")]


def test_append_for_unregistered_symbol_fails(db: FakeDatabase, securities: SecurityRepository) -> None:
    repo = FundamentalsRepository(db, securities)
    bad = _fact("1", datetime(2025, 1, 16, tzinfo=UTC)).model_copy(update={"symbol": "NOPE"})
    with pytest.raises(UnknownSecurityError):
        run(repo.append(Exchange.NSE, [bad]))


@pytest.fixture
def prices(db: FakeDatabase, securities: SecurityRepository) -> PriceRepository:
    return PriceRepository(db, securities)


def _daily(d: date, close: float, volume: int = 1_000_000) -> OhlcvBar:
    return OhlcvBar(ts=session_bar_ts(Exchange.NSE, d), open=close, high=close * 1.01, low=close * 0.99,
                    close=close, volume=volume, vwap=close)


def test_raw_bar_is_never_overwritten(prices: PriceRepository, db: FakeDatabase) -> None:
    d = date(2024, 6, 3)
    assert run(prices.append_bars("INFY", Exchange.NSE, DAILY, [_daily(d, 1500.0)], "NSE_BHAVCOPY")) == 1
    assert run(prices.append_bars("INFY", Exchange.NSE, DAILY, [_daily(d, 9999.0)], "VENDOR")) == 0
    raw = run(prices.fetch_raw("INFY", Exchange.NSE, DAILY, session_bar_ts(Exchange.NSE, d),
                               session_bar_ts(Exchange.NSE, d + DAILY)))
    assert raw["close"].to_list() == [1500.0] and raw["vwap"].to_list() == [1500.0]
    assert [b["source"] for b in db.bars.values()] == ["NSE_BHAVCOPY"]


def test_bars_for_unregistered_listing_are_refused(prices: PriceRepository) -> None:
    with pytest.raises(UnknownSecurityError):
        run(prices.append_bars("INFY", Exchange.BSE, DAILY, [_daily(date(2024, 6, 3), 1500.0)], "BSE"))


def test_corporate_actions_are_filtered_by_announcement_time(
    db: FakeDatabase, securities: SecurityRepository
) -> None:
    repo = CorporateActionRepository(db, securities)
    bonus = CorporateAction(symbol="INFY", exchange=Exchange.NSE, action_type=CorporateActionType.BONUS,
                            ex_date=date(2024, 6, 5), announced_at=datetime(2024, 5, 20, 16, 0, tzinfo=IST),
                            adjustment_factor=bonus_factor(Decimal(1), Decimal(1)), source="NSE_CA")
    assert run(repo.append([bonus])) == 1
    assert run(repo.append([bonus.model_copy(update={"adjustment_factor": Decimal("0.25")})])) == 0
    assert run(repo.fetch_known("INFY", Exchange.NSE, datetime(2024, 5, 20, 10, 0, tzinfo=IST))) == []
    (known,) = run(repo.fetch_known("INFY", Exchange.NSE, datetime(2024, 6, 1, tzinfo=IST)))
    assert known == bonus


def test_reconciliation_breaks_persisted(db: FakeDatabase, securities: SecurityRepository) -> None:
    filed = datetime(2025, 1, 16, 19, 42, tzinfo=IST)
    primary = facts_to_frame([_fact("417640000000", filed)])
    secondary = facts_to_frame([_fact("41764", filed, source="FMP")])  # vendor reported in crores
    result = reconcile(primary, secondary)
    n = run(ReconciliationRepository(db).record_breaks("INFY", Exchange.NSE, datetime(2025, 2, 1, tzinfo=UTC),
                                                       "NSE_XBRL", "FMP", result))
    assert n == 1
    assert db.breaks[0]["status"] == "SCALE_MISMATCH"
    assert db.breaks[0]["primary_value"] == Decimal("417640000000.0")


def test_macro_vintage_as_of(db: FakeDatabase) -> None:
    repo = MacroRepository(db)
    obs = [
        MacroObservation(series_id="FRED:PAYEMS", observation_date=date(2024, 8, 1),
                         release_timestamp=datetime(2024, 9, 6, 12, 30, tzinfo=UTC), value=Decimal("159095"),
                         source="FRED"),
        MacroObservation(series_id="FRED:PAYEMS", observation_date=date(2024, 8, 1),
                         release_timestamp=datetime(2024, 10, 4, 12, 30, tzinfo=UTC), value=Decimal("159173"),
                         source="FRED"),
    ]
    assert run(repo.append(obs)) == 2
    first_print = run(repo.fetch_as_of("FRED:PAYEMS", date(2024, 1, 1), date(2024, 12, 31),
                                       datetime(2024, 9, 30, tzinfo=UTC)))
    revised = run(repo.fetch_as_of("FRED:PAYEMS", date(2024, 1, 1), date(2024, 12, 31),
                                   datetime(2024, 10, 5, tzinfo=UTC)))
    assert first_print["value"].to_list() == [159095.0]
    assert revised["value"].to_list() == [159173.0]
    assert isinstance(revised, pl.DataFrame)


def test_bar_model_mirrors_db_constraints() -> None:
    with pytest.raises(ValueError):
        OhlcvBar(ts=datetime(2024, 1, 2, tzinfo=UTC), open=10, high=9, low=8, close=9.5, volume=1)
    with pytest.raises(ValueError):
        OhlcvBar(ts=datetime(2024, 1, 2), open=10, high=11, low=9, close=10, volume=1)
    with pytest.raises(ValueError, match="vwap"):
        OhlcvBar(ts=datetime(2024, 1, 2, tzinfo=UTC), open=10, high=11, low=9, close=10, volume=1, vwap=11.5)
    # Turnover/volume rounding a hair above the high is tolerated, as in the SQL CHECK.
    assert OhlcvBar(ts=datetime(2024, 1, 2, tzinfo=UTC), open=10, high=11, low=9, close=10, volume=1,
                    vwap=11.000001).vwap
    assert timedelta(days=1) == DAILY
