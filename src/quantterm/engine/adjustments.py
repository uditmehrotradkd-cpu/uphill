"""Dynamic corporate-action adjustment over immutable raw bars (Bloomberg ``GP`` adjustment toggles).

``adjust_splits``    ~ GP "Capital changes": splits, bonus issues, rights issues (prices x factor,
                       volume / factor).
``adjust_dividends`` ~ GP "Normal + abnormal cash": prices x (P - D) / P, where P is the raw close of
                       the session before the ex-date, restated onto the post-capital-change share basis
                       when a capital change shares the ex-date. Volume is unaffected.

Every bar whose session date is strictly before an ex-date takes that ex-date's factor; factors compound
backwards. Only actions announced at or before ``as_of_date`` with ``ex_date <= as_of_date`` are applied,
and only bars complete at ``as_of_date`` are returned. Raw rows are never written or mutated.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, time, timedelta
from typing import Any, Protocol

import polars as pl

from quantterm.corporate_actions import CAPITAL_CHANGE_TYPES, CorporateAction, CorporateActionType
from quantterm.models import Exchange, session_bar_ts

DAILY = timedelta(days=1)
PRICE_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "vwap")
_CAPITAL_TYPES = [t.value for t in CAPITAL_CHANGE_TYPES]

RAW_BAR_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "timestamp": pl.Datetime("us", "UTC"),
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Int64,
    "vwap": pl.Float64,
}
ACTION_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "ex_date": pl.Date,
    "action_type": pl.Utf8,
    "adjustment_factor": pl.Float64,
    "raw_dividend_amount": pl.Float64,
}
ADJUSTED_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "timestamp": pl.Datetime("us", "UTC"),
    "session_date": pl.Date,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
    "vwap": pl.Float64,
    "price_factor": pl.Float64,
    "volume_factor": pl.Float64,
}
_FACTOR_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "ex_date": pl.Date,
    "price_factor": pl.Float64,
    "volume_factor": pl.Float64,
}

RAW_BARS_SQL = """
SELECT "timestamp", open, high, low, close, volume, vwap
FROM raw_market_bars
WHERE ticker = $1 AND exchange = $2 AND bar_interval = $3 AND "timestamp" >= $4 AND "timestamp" < $5
ORDER BY "timestamp"
"""

KNOWN_ACTIONS_SQL = """
SELECT ex_date, action_type, adjustment_factor, raw_dividend_amount
FROM corporate_actions
WHERE ticker = $1 AND exchange = $2 AND announced_at <= $3 AND ex_date <= $4
ORDER BY ex_date, action_type
"""


class Fetcher(Protocol):
    """``asyncpg.Pool`` / ``asyncpg.Connection`` subset."""

    async def fetch(self, query: str, *args: Any) -> Sequence[Mapping[str, Any]]: ...


def knowledge_cutoff(as_of_date: date | datetime, exchange: Exchange) -> datetime:
    """A bare date means "end of that calendar day on the exchange"; datetimes must be tz-aware."""
    if isinstance(as_of_date, datetime):
        if as_of_date.tzinfo is None or as_of_date.utcoffset() is None:
            raise ValueError("as_of_date must be timezone-aware when a datetime is given")
        return as_of_date
    return datetime.combine(as_of_date, time.max, tzinfo=exchange.timezone)


def last_complete_session(cutoff: datetime, exchange: Exchange) -> date:
    local = cutoff.astimezone(exchange.timezone)
    return local.date() if local.time() >= exchange.session_close else local.date() - DAILY


def actions_frame(actions: Sequence[CorporateAction]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ex_date": [a.ex_date for a in actions],
            "action_type": [a.action_type.value for a in actions],
            "adjustment_factor": [None if a.adjustment_factor is None else float(a.adjustment_factor) for a in actions],
            "raw_dividend_amount": [
                None if a.raw_dividend_amount is None else float(a.raw_dividend_amount) for a in actions
            ],
        },
        schema=ACTION_SCHEMA,
    )


def _rows_frame(rows: Sequence[Mapping[str, Any]], schema: Mapping[str, Any]) -> pl.DataFrame:
    return pl.DataFrame({c: [r[c] for r in rows] for c in schema}, schema=dict(schema))


def cumulative_factors(
    closes: pl.DataFrame,
    actions: pl.DataFrame,
    *,
    adjust_splits: bool,
    adjust_dividends: bool,
) -> pl.DataFrame:
    """One row per ex-date with the product of that and every later ex-date's factor.

    ``closes``: raw session closes (``session_date``, ``close``) used for dividend factors.
    """
    capital_by_date: dict[date, float] = {}
    for ex_date, factor in actions.filter(pl.col("action_type").is_in(_CAPITAL_TYPES)).select(
        "ex_date", "adjustment_factor"
    ).iter_rows():
        capital_by_date[ex_date] = capital_by_date.get(ex_date, 1.0) * factor

    per_date: dict[date, list[float]] = {}
    if adjust_splits:
        per_date = {d: [f, 1.0 / f] for d, f in capital_by_date.items()}

    if adjust_dividends:
        dividends = actions.filter(pl.col("action_type") == CorporateActionType.CASH_DIVIDEND.value)
        ordered = closes.sort("session_date")
        for ex_date, amount in dividends.select("ex_date", "raw_dividend_amount").iter_rows():
            prior = ordered.filter(pl.col("session_date") < ex_date).select(pl.col("close").last()).item()
            if prior is None:
                raise ValueError(f"no raw close before dividend ex_date {ex_date}")
            # Dividends are declared per post-ex share; restate the prior close onto that basis.
            prior_post = float(prior) * capital_by_date.get(ex_date, 1.0)
            if amount >= prior_post:
                raise ValueError(f"dividend {amount} on {ex_date} >= prior close {prior_post}; check data")
            per_date.setdefault(ex_date, [1.0, 1.0])[0] *= (prior_post - amount) / prior_post

    if not per_date:
        return pl.DataFrame(schema=_FACTOR_SCHEMA)
    return (
        pl.DataFrame(
            {
                "ex_date": list(per_date),
                "price_factor": [v[0] for v in per_date.values()],
                "volume_factor": [v[1] for v in per_date.values()],
            },
            schema=_FACTOR_SCHEMA,
        )
        .sort("ex_date")
        .with_columns(
            pl.col("price_factor").cum_prod(reverse=True),
            pl.col("volume_factor").cum_prod(reverse=True),
        )
    )


def apply_adjustments(
    bars: pl.DataFrame,
    actions: pl.DataFrame,
    *,
    adjust_splits: bool = True,
    adjust_dividends: bool = True,
    closes: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Pure transform: returns a new adjusted frame; ``bars`` is not modified.

    ``bars`` needs ``session_date``, ``open``, ``high``, ``low``, ``close``, ``volume`` (``vwap`` and
    ``timestamp`` optional). ``closes`` defaults to the last bar close of each session in ``bars``.
    """
    missing = {"session_date", "open", "high", "low", "close", "volume"} - set(bars.columns)
    if missing:
        raise ValueError(f"bars missing columns: {sorted(missing)}")
    order = "timestamp" if "timestamp" in bars.columns else "session_date"
    ordered = bars.sort(order)
    if ordered.is_empty():
        return ordered.with_columns(
            pl.col("volume").cast(pl.Float64), pl.lit(1.0).alias("price_factor"), pl.lit(1.0).alias("volume_factor")
        )
    if closes is None:
        closes = ordered.group_by("session_date", maintain_order=True).agg(pl.col("close").last())

    first_session = ordered["session_date"].min()
    # An ex-date on/before the first bar adjusts nothing here and may have no prior close available.
    relevant = actions.filter(pl.col("ex_date") > first_session)
    factors = cumulative_factors(closes, relevant, adjust_splits=adjust_splits, adjust_dividends=adjust_dividends)

    joined = (
        ordered.with_columns((pl.col("session_date") + DAILY).alias("_probe"))
        .join_asof(factors, left_on="_probe", right_on="ex_date", strategy="forward", check_sortedness=False)
        .with_columns(pl.col("price_factor").fill_null(1.0), pl.col("volume_factor").fill_null(1.0))
    )
    prices = [c for c in PRICE_COLUMNS if c in joined.columns]
    return joined.with_columns(
        *[(pl.col(c) * pl.col("price_factor")).alias(c) for c in prices],
        (pl.col("volume").cast(pl.Float64) * pl.col("volume_factor")).alias("volume"),
    ).drop("_probe", "ex_date")


def _with_session_date(raw: pl.DataFrame, exchange: Exchange) -> pl.DataFrame:
    return raw.with_columns(
        pl.col("timestamp").dt.convert_time_zone(exchange.timezone.key).dt.date().alias("session_date")
    )


async def get_adjusted_series(
    ticker: str,
    start_date: date,
    end_date: date,
    as_of_date: date | datetime,
    adjust_splits: bool = True,
    adjust_dividends: bool = True,
    *,
    exchange: Exchange,
    db: Fetcher,
    interval: timedelta = DAILY,
) -> pl.DataFrame:
    """Adjusted OHLCV/VWAP for sessions in [start_date, end_date] exactly as computable at ``as_of_date``."""
    if start_date > end_date:
        raise ValueError("start_date must be <= end_date")
    if interval <= timedelta(0):
        raise ValueError("interval must be positive")
    cutoff = knowledge_cutoff(as_of_date, exchange)
    intraday = interval < DAILY
    # EOD bars are knowable only after the session close; intraday bars once their interval has elapsed.
    last_session = cutoff.astimezone(exchange.timezone).date() if intraday else last_complete_session(cutoff, exchange)
    empty = pl.DataFrame(schema=ADJUSTED_SCHEMA)
    if last_session < start_date:
        return empty

    # Fetch through the as-of session: later ex-dates still adjust bars inside [start_date, end_date].
    window = (ticker, exchange.value)
    lo, hi = session_bar_ts(exchange, start_date), session_bar_ts(exchange, last_session + DAILY)
    bars = _with_session_date(_rows_frame(await db.fetch(RAW_BARS_SQL, *window, interval, lo, hi), RAW_BAR_SCHEMA),
                              exchange)
    if intraday:
        bars = bars.filter(pl.col("timestamp") + interval <= cutoff.astimezone(UTC))
    if bars.is_empty():
        return empty

    actions = _rows_frame(
        await db.fetch(KNOWN_ACTIONS_SQL, *window, cutoff, cutoff.astimezone(exchange.timezone).date()),
        ACTION_SCHEMA,
    )

    closes: pl.DataFrame | None = None
    if intraday and adjust_dividends:
        # Prefer official daily closes for dividend factors; fall back to the last intraday print.
        daily = _with_session_date(_rows_frame(await db.fetch(RAW_BARS_SQL, *window, DAILY, lo, hi), RAW_BAR_SCHEMA),
                                   exchange)
        last_prints = bars.sort("timestamp").group_by("session_date", maintain_order=True).agg(pl.col("close").last())
        closes = pl.concat([daily.select("session_date", "close"), last_prints]).unique("session_date", keep="first")

    adjusted = apply_adjustments(
        bars, actions, adjust_splits=adjust_splits, adjust_dividends=adjust_dividends, closes=closes
    )
    return adjusted.filter(pl.col("session_date").is_between(start_date, end_date)).select(list(ADJUSTED_SCHEMA))
