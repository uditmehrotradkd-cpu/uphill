"""Cross-source reconciliation of fundamentals: regulatory filing (primary) vs vendor API (secondary)."""

from __future__ import annotations

from enum import StrEnum

import polars as pl

DEFAULT_TOLERANCE = 0.005  # 0.5 %
# Common unit errors: thousands, lakhs, millions, crores, billions.
SCALE_CANDIDATES: tuple[float, ...] = (1e3, 1e5, 1e6, 1e7, 1e9)
_KEYS = ["symbol", "metric", "period_type", "period_end_date"]


class ReconStatus(StrEnum):
    MATCH = "MATCH"
    VARIANCE_BREACH = "VARIANCE_BREACH"
    SCALE_MISMATCH = "SCALE_MISMATCH"
    SIGN_MISMATCH = "SIGN_MISMATCH"
    CURRENCY_MISMATCH = "CURRENCY_MISMATCH"
    MISSING_PRIMARY = "MISSING_PRIMARY"
    MISSING_SECONDARY = "MISSING_SECONDARY"


def _side(df: pl.DataFrame, suffix: str) -> pl.DataFrame:
    missing = {*_KEYS, "value", "currency", "source"} - set(df.columns)
    if missing:
        raise ValueError(f"{suffix} frame missing columns: {sorted(missing)}")
    dupes = df.group_by(_KEYS).len().filter(pl.col("len") > 1)
    if not dupes.is_empty():
        raise ValueError(f"{suffix} frame has duplicate keys; resolve point-in-time first")
    return df.select(
        *_KEYS,
        pl.col("value").cast(pl.Float64).alias(f"value_{suffix}"),
        pl.col("currency").alias(f"currency_{suffix}"),
        pl.col("source").alias(f"source_{suffix}"),
    )


def reconcile(
    primary: pl.DataFrame,
    secondary: pl.DataFrame,
    tolerance: float = DEFAULT_TOLERANCE,
    zero_abs_tolerance: float = 0.5,
) -> pl.DataFrame:
    """Full outer comparison keyed on (symbol, metric, period_type, period_end_date).

    Inputs must already be point-in-time resolved to the same ``as_of``. Relative variance is
    ``|secondary - primary| / |primary|``; when primary is 0 an absolute tolerance applies.
    Any status other than MATCH is a break (``is_break``).
    """
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    p, s = pl.col("value_primary"), pl.col("value_secondary")
    ratio = (s / p).abs()
    scale_hit = pl.lit(False)
    for k in SCALE_CANDIDATES:
        scale_hit = scale_hit | ((ratio / k - 1).abs() <= tolerance) | ((ratio * k - 1).abs() <= tolerance)

    rel_var = pl.when(p != 0).then((s - p).abs() / p.abs()).otherwise(None)
    within = pl.when(p != 0).then(rel_var <= tolerance).otherwise((s - p).abs() <= zero_abs_tolerance)

    status = (
        pl.when(p.is_null())
        .then(pl.lit(ReconStatus.MISSING_PRIMARY.value))
        .when(s.is_null())
        .then(pl.lit(ReconStatus.MISSING_SECONDARY.value))
        .when(pl.col("currency_primary") != pl.col("currency_secondary"))
        .then(pl.lit(ReconStatus.CURRENCY_MISMATCH.value))
        .when(within)
        .then(pl.lit(ReconStatus.MATCH.value))
        .when((p * s) < 0)
        .then(pl.lit(ReconStatus.SIGN_MISMATCH.value))
        .when((p != 0) & (s != 0) & scale_hit)
        .then(pl.lit(ReconStatus.SCALE_MISMATCH.value))
        .otherwise(pl.lit(ReconStatus.VARIANCE_BREACH.value))
    )

    return (
        _side(primary, "primary")
        .join(_side(secondary, "secondary"), on=_KEYS, how="full", coalesce=True)
        .with_columns(rel_var.alias("relative_variance"), status.alias("status"))
        .with_columns((pl.col("status") != ReconStatus.MATCH.value).alias("is_break"))
        .sort(_KEYS)
    )
