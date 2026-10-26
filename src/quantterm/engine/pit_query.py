"""Point-in-time fundamentals over ``financial_metrics_pit`` (asyncpg).

Every query filters ``filing_timestamp <= as_of_date`` and, per (ticker, metric, period), keeps the
highest ``revision_number`` visible at that instant. Revisions are numbered in knowledge-time order by
the ``assign_metric_revision`` trigger, so the highest visible revision is the latest public vintage.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Protocol

import polars as pl
from pydantic import BaseModel, ConfigDict

from quantterm.models import Exchange, PeriodType

_COLUMNS = (
    "ticker, exchange, metric_name, period_type, period_end_date, filing_timestamp, revision_number, "
    "metric_value, currency, source_authority, source_document_id, confidence_score"
)
_VISIBLE = """
    ticker = ANY($1::text[]) AND exchange = $2 AND metric_name = ANY($3::text[]) AND period_type = $4
    AND filing_timestamp <= $5 AND period_end_date >= $6
"""

# Single authority: highest revision visible at as_of per (ticker, metric, period).
SNAPSHOT_SQL = f"""
SELECT DISTINCT ON (ticker, metric_name, period_end_date) {_COLUMNS}
FROM financial_metrics_pit
WHERE {_VISIBLE} AND source_authority = $7
ORDER BY ticker, metric_name, period_end_date, revision_number DESC
"""

# All authorities: highest revision per authority, then the most trusted authority per period.
BEST_SNAPSHOT_SQL = f"""
WITH per_authority AS (
    SELECT DISTINCT ON (ticker, metric_name, period_end_date, source_authority) {_COLUMNS}
    FROM financial_metrics_pit
    WHERE {_VISIBLE}
    ORDER BY ticker, metric_name, period_end_date, source_authority, revision_number DESC
)
SELECT DISTINCT ON (ticker, metric_name, period_end_date) {_COLUMNS}
FROM per_authority
ORDER BY ticker, metric_name, period_end_date, confidence_score DESC, filing_timestamp DESC, source_authority
"""


def _latest_period(snapshot_sql: str) -> str:
    return f"""
SELECT DISTINCT ON (ticker, metric_name) {_COLUMNS}
FROM ({snapshot_sql}) s
ORDER BY ticker, metric_name, period_end_date DESC
"""


LATEST_SQL = _latest_period(SNAPSHOT_SQL)
BEST_LATEST_SQL = _latest_period(BEST_SNAPSHOT_SQL)

REVISION_HISTORY_SQL = f"""
SELECT {_COLUMNS}
FROM financial_metrics_pit
WHERE ticker = $1 AND exchange = $2 AND metric_name = $3 AND period_type = $4 AND period_end_date = $5
  AND source_authority = $6 AND filing_timestamp <= $7
ORDER BY revision_number
"""

_EARLIEST_PERIOD = date(1900, 1, 1)


class Fetcher(Protocol):
    async def fetch(self, query: str, *args: Any) -> Sequence[Mapping[str, Any]]: ...


class MetricSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    exchange: Exchange
    metric_name: str
    period_type: PeriodType
    period_end_date: date
    filing_timestamp: datetime
    revision_number: int
    metric_value: Decimal
    currency: str
    source_authority: str
    source_document_id: str | None
    confidence_score: Decimal


def _require_aware(as_of_date: datetime) -> None:
    if not isinstance(as_of_date, datetime) or as_of_date.tzinfo is None or as_of_date.utcoffset() is None:
        raise ValueError("as_of_date must be a timezone-aware datetime (filings are timestamped to the second)")


def _snapshots(rows: Sequence[Mapping[str, Any]]) -> list[MetricSnapshot]:
    return [
        MetricSnapshot(
            ticker=r["ticker"],
            exchange=Exchange(r["exchange"]),
            metric_name=r["metric_name"],
            period_type=PeriodType(r["period_type"]),
            period_end_date=r["period_end_date"],
            filing_timestamp=r["filing_timestamp"],
            revision_number=r["revision_number"],
            metric_value=Decimal(r["metric_value"]),
            currency=r["currency"],
            source_authority=r["source_authority"],
            source_document_id=r["source_document_id"],
            confidence_score=Decimal(r["confidence_score"]),
        )
        for r in rows
    ]


async def _run(
    db: Fetcher,
    single_sql: str,
    best_sql: str,
    tickers: Sequence[str],
    exchange: Exchange,
    metric_names: Sequence[str],
    period_type: PeriodType,
    as_of_date: datetime,
    source_authority: str | None,
    periods_from: date | None,
) -> list[MetricSnapshot]:
    _require_aware(as_of_date)
    args: list[Any] = [
        list(tickers), exchange.value, list(metric_names), period_type.value, as_of_date,
        periods_from or _EARLIEST_PERIOD,
    ]
    if source_authority is None:
        return _snapshots(await db.fetch(best_sql, *args))
    return _snapshots(await db.fetch(single_sql, *args, source_authority))


async def fetch_pit_snapshot(
    db: Fetcher,
    tickers: Sequence[str],
    exchange: Exchange,
    metric_names: Sequence[str],
    period_type: PeriodType,
    as_of_date: datetime,
    source_authority: str | None = None,
    *,
    periods_from: date | None = None,
) -> list[MetricSnapshot]:
    """Every period's value as known at ``as_of_date``.

    With ``source_authority`` the latest visible revision of that source is returned; without it, the
    latest revision of each source is taken and the highest ``confidence_score`` wins per period.
    """
    return await _run(db, SNAPSHOT_SQL, BEST_SNAPSHOT_SQL, tickers, exchange, metric_names, period_type,
                      as_of_date, source_authority, periods_from)


async def fetch_latest_snapshot(
    db: Fetcher,
    tickers: Sequence[str],
    exchange: Exchange,
    metric_names: Sequence[str],
    period_type: PeriodType,
    as_of_date: datetime,
    source_authority: str | None = None,
) -> list[MetricSnapshot]:
    """Only the most recent period per (ticker, metric) that was public at ``as_of_date``."""
    return await _run(db, LATEST_SQL, BEST_LATEST_SQL, tickers, exchange, metric_names, period_type,
                      as_of_date, source_authority, None)


async def fetch_revision_history(
    db: Fetcher,
    ticker: str,
    exchange: Exchange,
    metric_name: str,
    period_type: PeriodType,
    period_end_date: date,
    source_authority: str,
    as_of_date: datetime,
) -> list[MetricSnapshot]:
    """Audit trail: every vintage of one period visible at ``as_of_date``, oldest revision first."""
    _require_aware(as_of_date)
    rows = await db.fetch(
        REVISION_HISTORY_SQL, ticker, exchange.value, metric_name, period_type.value, period_end_date,
        source_authority, as_of_date,
    )
    return _snapshots(rows)


def snapshots_to_frame(snapshots: Sequence[MetricSnapshot]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ticker": [s.ticker for s in snapshots],
            "exchange": [s.exchange.value for s in snapshots],
            "metric_name": [s.metric_name for s in snapshots],
            "period_type": [s.period_type.value for s in snapshots],
            "period_end_date": [s.period_end_date for s in snapshots],
            "filing_timestamp": [s.filing_timestamp for s in snapshots],
            "revision_number": [s.revision_number for s in snapshots],
            "metric_value": [float(s.metric_value) for s in snapshots],
            "currency": [s.currency for s in snapshots],
            "source_authority": [s.source_authority for s in snapshots],
            "confidence_score": [float(s.confidence_score) for s in snapshots],
        },
        schema={
            "ticker": pl.Utf8,
            "exchange": pl.Utf8,
            "metric_name": pl.Utf8,
            "period_type": pl.Utf8,
            "period_end_date": pl.Date,
            "filing_timestamp": pl.Datetime("us", "UTC"),
            "revision_number": pl.Int32,
            "metric_value": pl.Float64,
            "currency": pl.Utf8,
            "source_authority": pl.Utf8,
            "confidence_score": pl.Float64,
        },
    )
