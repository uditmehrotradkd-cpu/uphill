from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any
from zoneinfo import ZoneInfo

import pytest
from fakes import FakeCheckViolation, FakeDatabase

from quantterm.db.repositories import FundamentalsRepository, SecurityRepository
from quantterm.engine.pit_query import (
    fetch_latest_snapshot,
    fetch_pit_snapshot,
    fetch_revision_history,
    snapshots_to_frame,
)
from quantterm.models import Exchange, FundamentalFact, PeriodType

IST = ZoneInfo("Asia/Kolkata")
Q3_FY25 = date(2024, 12, 31)
Q2_FY25 = date(2024, 9, 30)
ORIGINAL = datetime(2025, 1, 16, 19, 40, 31, tzinfo=IST)
RESTATED = datetime(2025, 4, 17, 19, 45, tzinfo=IST)


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def _fact(value: str, filed: datetime, *, source: str = "NSE_XBRL", end: date = Q3_FY25,
          confidence: float = 1.0, metric: str = "revenue") -> FundamentalFact:
    return FundamentalFact(
        symbol="INFY", metric=metric, period_type=PeriodType.QUARTER, period_end_date=end,
        filing_timestamp=filed, value=Decimal(value), currency="INR", source=source,
        source_document_id=f"{source}:{filed:%Y%m%d}", confidence_score=confidence,
    )


@pytest.fixture
def db() -> FakeDatabase:
    db = FakeDatabase()
    securities = SecurityRepository(db)
    _run(securities.upsert("INFY", Exchange.NSE))
    repo = FundamentalsRepository(db, securities)
    _run(repo.append(Exchange.NSE, [
        _fact("409860000000", datetime(2024, 10, 17, 19, 46, tzinfo=IST), end=Q2_FY25),
        # Delivered out of order within the batch; the writer sorts by knowledge time.
        _fact("417000000000", RESTATED),
        _fact("417640000000", ORIGINAL),
    ]))
    _run(repo.append(Exchange.NSE, [_fact("41764", datetime(2025, 2, 1, tzinfo=UTC), source="FMP",
                                          confidence=0.8)], only_if_changed=True))
    return db


def _snap(db: FakeDatabase, as_of: datetime, source: str | None = "NSE_XBRL", **kw: Any) -> list[Any]:
    return _run(fetch_pit_snapshot(db, ["INFY"], Exchange.NSE, ["revenue"], PeriodType.QUARTER, as_of, source, **kw))


def test_revisions_numbered_in_knowledge_time_order(db: FakeDatabase) -> None:
    q3 = [m for m in db.metrics if m["period_end_date"] == Q3_FY25 and m["source_authority"] == "NSE_XBRL"]
    assert sorted((m["revision_number"], m["metric_value"]) for m in q3) == [
        (1, Decimal("417640000000")), (2, Decimal("417000000000"))
    ]


def test_snapshot_picks_highest_revision_visible_at_as_of(db: FakeDatabase) -> None:
    before_any = _snap(db, datetime(2025, 1, 16, 19, 40, 30, tzinfo=IST))
    at_original = _snap(db, ORIGINAL)  # filing exactly at as_of is visible
    after_restatement = _snap(db, datetime(2025, 6, 1, tzinfo=UTC))
    assert [s.period_end_date for s in before_any] == [Q2_FY25]
    assert at_original[-1].revision_number == 1 and at_original[-1].metric_value == Decimal("417640000000")
    assert after_restatement[-1].revision_number == 2 and after_restatement[-1].metric_value == Decimal("417000000000")


def test_reingesting_a_known_vintage_is_a_no_op(db: FakeDatabase) -> None:
    repo = FundamentalsRepository(db, SecurityRepository(db))
    assert _run(repo.append(Exchange.NSE, [_fact("417640000000", ORIGINAL)])) == 0
    assert len(db.metrics) == 4


def test_out_of_order_vintage_is_rejected(db: FakeDatabase) -> None:
    repo = FundamentalsRepository(db, SecurityRepository(db))
    with pytest.raises(FakeCheckViolation, match="out-of-order"):
        _run(repo.append(Exchange.NSE, [_fact("417100000000", datetime(2025, 2, 1, tzinfo=IST))]))


def test_best_authority_prefers_confidence_then_falls_back(db: FakeDatabase) -> None:
    best = {s.period_end_date: s for s in _snap(db, datetime(2025, 3, 1, tzinfo=UTC), source=None)}
    assert best[Q3_FY25].source_authority == "NSE_XBRL"  # 1.0 beats vendor 0.8
    vendor_only = _snap(db, datetime(2025, 3, 1, tzinfo=UTC), source="FMP")
    assert vendor_only[0].metric_value == Decimal("41764") and vendor_only[0].confidence_score == Decimal("0.800")

    securities = SecurityRepository(db)
    _run(FundamentalsRepository(db, securities).append(Exchange.NSE, [
        _fact("68060000000", datetime(2025, 2, 1, tzinfo=UTC), source="FMP", confidence=0.8, metric="net_income"),
    ], only_if_changed=True))
    net = _run(fetch_pit_snapshot(db, ["INFY"], Exchange.NSE, ["net_income"], PeriodType.QUARTER,
                                  datetime(2025, 3, 1, tzinfo=UTC)))
    assert net[0].source_authority == "FMP"  # only source with the period


def test_latest_snapshot_returns_newest_period_only(db: FakeDatabase) -> None:
    latest = _run(fetch_latest_snapshot(db, ["INFY"], Exchange.NSE, ["revenue"], PeriodType.QUARTER,
                                        datetime(2024, 12, 1, tzinfo=UTC), "NSE_XBRL"))
    assert [s.period_end_date for s in latest] == [Q2_FY25]
    latest_now = _run(fetch_latest_snapshot(db, ["INFY"], Exchange.NSE, ["revenue"], PeriodType.QUARTER,
                                            datetime(2025, 6, 1, tzinfo=UTC), "NSE_XBRL"))
    assert [(s.period_end_date, s.revision_number) for s in latest_now] == [(Q3_FY25, 2)]


def test_revision_history_is_bounded_by_as_of(db: FakeDatabase) -> None:
    def history(as_of: datetime) -> list[int]:
        rows = _run(fetch_revision_history(db, "INFY", Exchange.NSE, "revenue", PeriodType.QUARTER, Q3_FY25,
                                           "NSE_XBRL", as_of))
        return [r.revision_number for r in rows]

    assert history(datetime(2025, 2, 1, tzinfo=UTC)) == [1]
    assert history(datetime(2025, 6, 1, tzinfo=UTC)) == [1, 2]


def test_periods_from_trims_history_and_frame_conversion(db: FakeDatabase) -> None:
    snaps = _snap(db, datetime(2025, 6, 1, tzinfo=UTC), periods_from=date(2024, 10, 1))
    frame = snapshots_to_frame(snaps)
    assert frame["period_end_date"].to_list() == [Q3_FY25]
    assert frame["revision_number"].to_list() == [2]


@pytest.mark.parametrize("bad", [datetime(2025, 1, 1), date(2025, 1, 1)])
def test_as_of_must_be_timezone_aware_datetime(db: FakeDatabase, bad: Any) -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        _snap(db, bad)
