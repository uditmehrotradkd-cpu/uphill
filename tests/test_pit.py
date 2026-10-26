from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

import polars as pl
import pytest
from pydantic import ValidationError

from quantterm.engine.pit_query import SNAPSHOT_SQL
from quantterm.models import Exchange, FundamentalFact, PeriodType
from quantterm.pit import (
    PointInTimeFundamentalsRepository,
    asof_join_fundamentals,
    facts_to_frame,
    latest_period_as_of,
    resolve_as_of,
)


def _fact(period_end: date, filed: datetime, value: str, symbol: str = "INFY") -> FundamentalFact:
    return FundamentalFact(
        symbol=symbol,
        metric="revenue",
        period_type=PeriodType.FISCAL_YEAR,
        period_end_date=period_end,
        filing_timestamp=filed,
        value=Decimal(value),
        currency="INR",
        source="NSE_XBRL",
    )


@pytest.fixture
def facts() -> pl.DataFrame:
    return facts_to_frame(
        [
            _fact(date(2023, 3, 31), datetime(2023, 4, 13, 11, 0, tzinfo=UTC), "146767"),
            _fact(date(2024, 3, 31), datetime(2024, 4, 18, 11, 0, tzinfo=UTC), "153670"),
            # Restatement of the newest period.
            _fact(date(2024, 3, 31), datetime(2024, 6, 20, 11, 0, tzinfo=UTC), "153000"),
            # Late restatement of an OLD period after the newer period exists.
            _fact(date(2023, 3, 31), datetime(2024, 7, 1, 11, 0, tzinfo=UTC), "146000"),
        ]
    )


def test_resolve_as_of_ignores_future_filings(facts: pl.DataFrame) -> None:
    out = resolve_as_of(facts, datetime(2024, 4, 18, 10, 59, tzinfo=UTC))
    assert out["period_end_date"].to_list() == [date(2023, 3, 31)]
    assert out["value"].to_list() == [146767.0]


def test_resolve_as_of_picks_restatement_once_public(facts: pl.DataFrame) -> None:
    before = latest_period_as_of(facts, datetime(2024, 6, 19, tzinfo=UTC))
    after = latest_period_as_of(facts, datetime(2024, 6, 21, tzinfo=UTC))
    assert before["value"].item() == 153670.0
    assert after["value"].item() == 153000.0


def test_filing_exactly_at_as_of_is_visible(facts: pl.DataFrame) -> None:
    out = latest_period_as_of(facts, datetime(2024, 4, 18, 11, 0, tzinfo=UTC))
    assert out["value"].item() == 153670.0


def test_naive_as_of_rejected(facts: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        resolve_as_of(facts, datetime(2024, 1, 1))


def test_fact_cannot_be_filed_before_period_end() -> None:
    with pytest.raises(ValidationError):
        _fact(date(2024, 3, 31), datetime(2024, 3, 30, tzinfo=UTC), "1")


def test_asof_join_old_period_restatement_does_not_clobber_newer_period(facts: pl.DataFrame) -> None:
    panel = pl.DataFrame(
        {
            "symbol": ["INFY"] * 5,
            "timestamp": [
                datetime(2023, 4, 1, tzinfo=UTC),  # nothing public yet
                datetime(2023, 5, 1, tzinfo=UTC),
                datetime(2024, 5, 1, tzinfo=UTC),
                datetime(2024, 6, 25, tzinfo=UTC),
                datetime(2024, 7, 5, tzinfo=UTC),  # after FY23 restatement
            ],
        },
        schema={"symbol": pl.Utf8, "timestamp": pl.Datetime("us", "UTC")},
    )
    out = asof_join_fundamentals(panel, facts, "revenue", PeriodType.FISCAL_YEAR, "NSE_XBRL")
    assert out["revenue"].to_list() == [None, 146767.0, 153670.0, 153000.0, 153000.0]
    assert out["revenue_period_end"][-1] == date(2024, 3, 31)


class _FakeDB:
    def __init__(self, rows: list[Mapping[str, Any]]) -> None:
        self.rows = rows
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    async def fetch(self, query: str, *args: Any) -> Sequence[Mapping[str, Any]]:
        self.calls.append((query, args))
        return self.rows


def test_repository_passes_as_of_bound_parameter() -> None:
    as_of = datetime(2024, 5, 1, tzinfo=UTC)
    db = _FakeDB(
        [
            {
                "ticker": "INFY",
                "exchange": "NSE",
                "metric_name": "revenue",
                "period_type": "FY",
                "period_end_date": date(2024, 3, 31),
                "filing_timestamp": datetime(2024, 4, 18, 11, 0, tzinfo=UTC),
                "revision_number": 1,
                "metric_value": Decimal("153670"),
                "currency": "INR",
                "source_authority": "NSE_XBRL",
                "source_document_id": "NSE:1169123",
                "confidence_score": Decimal("1.000"),
            }
        ]
    )
    repo = PointInTimeFundamentalsRepository(db)
    out = asyncio.run(
        repo.fetch_as_of(["INFY"], Exchange.NSE, ["revenue"], PeriodType.FISCAL_YEAR, "NSE_XBRL", as_of)
    )
    query, args = db.calls[0]
    assert query == SNAPSHOT_SQL
    assert "filing_timestamp <= $5" in query and "revision_number DESC" in query
    assert args[4] == as_of and args[-1] == "NSE_XBRL"
    assert out[0].value == Decimal("153670") and out[0].source_document_id == "NSE:1169123"


def test_repository_rejects_naive_as_of() -> None:
    repo = PointInTimeFundamentalsRepository(_FakeDB([]))
    with pytest.raises(ValueError):
        asyncio.run(
            repo.fetch_as_of(["X"], Exchange.NSE, ["revenue"], PeriodType.QUARTER, "NSE_XBRL", datetime(2024, 1, 1))
        )
