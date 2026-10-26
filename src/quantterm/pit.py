"""Point-in-time (bitemporal) access to fundamentals. Every read path requires an ``as_of``."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import polars as pl

from quantterm.engine.pit_query import Fetcher, fetch_pit_snapshot
from quantterm.models import Exchange, FundamentalFact, PeriodType

FACT_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "symbol": pl.Utf8,
    "metric": pl.Utf8,
    "period_type": pl.Utf8,
    "period_end_date": pl.Date,
    "filing_timestamp": pl.Datetime("us", "UTC"),
    "value": pl.Float64,
    "currency": pl.Utf8,
    "source": pl.Utf8,
}


def _require_aware(as_of: datetime) -> None:
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("as_of must be timezone-aware to compare against filing timestamps")


class PointInTimeFundamentalsRepository:
    """``FundamentalFact`` view over ``engine.pit_query`` for one source authority."""

    def __init__(self, db: Fetcher) -> None:
        self._db = db

    async def fetch_as_of(
        self,
        symbols: Sequence[str],
        exchange: Exchange,
        metrics: Sequence[str],
        period_type: PeriodType,
        source: str,
        as_of: datetime,
    ) -> list[FundamentalFact]:
        """Highest revision per (symbol, metric, period) among filings with ``filing_timestamp <= as_of``."""
        _require_aware(as_of)
        snapshots = await fetch_pit_snapshot(self._db, symbols, exchange, metrics, period_type, as_of, source)
        return [
            FundamentalFact(
                symbol=s.ticker,
                metric=s.metric_name,
                period_type=s.period_type,
                period_end_date=s.period_end_date,
                filing_timestamp=s.filing_timestamp,
                value=s.metric_value,
                currency=s.currency,
                source=s.source_authority,
                source_document_id=s.source_document_id,
                confidence_score=float(s.confidence_score),
            )
            for s in snapshots
        ]


def facts_to_frame(facts: Sequence[FundamentalFact]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": [f.symbol for f in facts],
            "metric": [f.metric for f in facts],
            "period_type": [f.period_type.value for f in facts],
            "period_end_date": [f.period_end_date for f in facts],
            "filing_timestamp": [f.filing_timestamp for f in facts],
            "value": [float(f.value) for f in facts],
            "currency": [f.currency for f in facts],
            "source": [f.source for f in facts],
        },
        schema=FACT_SCHEMA,
    )


def resolve_as_of(facts: pl.DataFrame, as_of: datetime) -> pl.DataFrame:
    """In-memory equivalent of ``pit_query.SNAPSHOT_SQL``: newest filing per period visible at ``as_of``."""
    _require_aware(as_of)
    keys = ["symbol", "metric", "period_type", "period_end_date", "source"]
    return (
        facts.filter(pl.col("filing_timestamp") <= as_of)
        .sort("filing_timestamp")
        .group_by(keys, maintain_order=True)
        .last()
        .sort(keys)
    )


def latest_period_as_of(facts: pl.DataFrame, as_of: datetime) -> pl.DataFrame:
    """Most recent period known at ``as_of`` per (symbol, metric, period_type, source), restatements applied."""
    resolved = resolve_as_of(facts, as_of)
    return (
        resolved.sort("period_end_date")
        .group_by(["symbol", "metric", "period_type", "source"], maintain_order=True)
        .last()
        .sort(["symbol", "metric"])
    )


def asof_join_fundamentals(
    panel: pl.DataFrame,
    facts: pl.DataFrame,
    metric: str,
    period_type: PeriodType,
    source: str,
    date_col: str = "timestamp",
) -> pl.DataFrame:
    """Attach the latest-period value known at each panel timestamp (backtest-safe).

    ``panel`` needs ``symbol`` and a tz-aware UTC ``date_col``. A restatement of an older
    period never replaces a newer period; a restatement of the newest period does.
    """
    events = (
        facts.filter(
            (pl.col("metric") == metric)
            & (pl.col("period_type") == period_type.value)
            & (pl.col("source") == source)
        )
        .sort(["symbol", "filing_timestamp"])
        .with_columns(
            pl.col("period_end_date")
            .cast(pl.Int32)
            .cum_max()
            .over("symbol")
            .alias("_newest_known_period")
        )
        .filter(pl.col("period_end_date").cast(pl.Int32) == pl.col("_newest_known_period"))
        .select(
            "symbol",
            pl.col("filing_timestamp").alias("_known_since"),
            pl.col("period_end_date").alias(f"{metric}_period_end"),
            pl.col("value").alias(metric),
        )
    )
    return (
        panel.sort(date_col)
        .join_asof(
            events.sort("_known_since"),
            left_on=date_col,
            right_on="_known_since",
            by="symbol",
            strategy="backward",
            check_sortedness=False,  # both sides sorted above
        )
        .drop("_known_since")
    )
