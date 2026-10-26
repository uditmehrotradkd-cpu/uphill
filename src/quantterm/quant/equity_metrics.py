"""Bloomberg-convention equity metrics: ``BETA <GO>`` regression and ``FA``-style Enterprise Value.

Beta: OLS of security on benchmark periodic returns (default 2 years of weekly returns against the
local index), Blume-adjusted toward the market: ``0.67 * raw + 0.33``.

Enterprise Value: market cap + total debt + lease liabilities + preferred + minority - cash & ST
investments. Banks and NBFCs (and insurers) fund themselves with debt/deposits as raw material, so
debt, leases and cash are not added back; their valuation basis switches to price / tangible book.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date
from decimal import Decimal
from enum import StrEnum

import numpy as np
import numpy.typing as npt
import polars as pl
from pydantic import BaseModel, ConfigDict

from quantterm.models import SectorClass

BLUME_RAW_WEIGHT = 0.67
BLUME_PRIOR_WEIGHT = 0.33
_ZERO = Decimal(0)


class ReturnFrequency(StrEnum):
    DAILY = "DAILY"
    WEEKLY = "WEEKLY"
    MONTHLY = "MONTHLY"


_TRUNCATE = {ReturnFrequency.WEEKLY: "1w", ReturnFrequency.MONTHLY: "1mo"}
_MIN_OBS = {ReturnFrequency.DAILY: 60, ReturnFrequency.WEEKLY: 26, ReturnFrequency.MONTHLY: 24}

ReturnsLike = Sequence[float] | npt.NDArray[np.float64] | pl.Series


class BetaEstimate(BaseModel):
    model_config = ConfigDict(frozen=True)

    raw_beta: float
    adjusted_beta: float
    alpha: float
    beta_std_error: float
    alpha_std_error: float
    r_squared: float
    t_stat: float
    n_observations: int


class BetaResult(BetaEstimate):
    start: date
    end: date
    frequency: ReturnFrequency


def blume_adjust(raw_beta: float) -> float:
    return BLUME_RAW_WEIGHT * raw_beta + BLUME_PRIOR_WEIGHT


def _as_array(values: ReturnsLike) -> npt.NDArray[np.float64]:
    if isinstance(values, pl.Series):
        return values.cast(pl.Float64).to_numpy()
    return np.asarray(values, dtype=np.float64)


def calculate_bloomberg_beta(stock_returns: ReturnsLike, benchmark_returns: ReturnsLike) -> BetaEstimate:
    """OLS ``r_stock = alpha + beta * r_benchmark + e`` on date-aligned periodic returns.

    Pairs where either return is missing/non-finite are dropped together. Standard errors are the
    classical (homoskedastic) OLS errors that ``BETA <GO>`` reports.
    """
    y, x = _as_array(stock_returns), _as_array(benchmark_returns)
    if y.shape != x.shape or y.ndim != 1:
        raise ValueError(f"returns must be aligned 1-D series, got shapes {y.shape} and {x.shape}")
    keep = np.isfinite(y) & np.isfinite(x)
    y, x = y[keep], x[keep]
    n = int(y.size)
    if n < 3:
        raise ValueError(f"need at least 3 paired observations, got {n}")

    x_mean, y_mean = float(x.mean()), float(y.mean())
    dx = x - x_mean
    sxx = float(dx @ dx)
    if sxx == 0.0:
        raise ValueError("benchmark returns have zero variance")
    beta = float(dx @ (y - y_mean)) / sxx
    alpha = y_mean - beta * x_mean
    resid = y - (alpha + beta * x)
    ssr = float(resid @ resid)
    sst = float((y - y_mean) @ (y - y_mean))
    sigma2 = ssr / (n - 2)
    beta_se = math.sqrt(sigma2 / sxx)
    alpha_se = math.sqrt(sigma2 * (1.0 / n + x_mean * x_mean / sxx))
    return BetaEstimate(
        raw_beta=beta,
        adjusted_beta=blume_adjust(beta),
        alpha=alpha,
        beta_std_error=beta_se,
        alpha_std_error=alpha_se,
        r_squared=1.0 - ssr / sst if sst > 0 else 0.0,
        t_stat=beta / beta_se if beta_se > 0 else math.inf,
        n_observations=n,
    )


def periodic_returns(
    security: pl.DataFrame, benchmark: pl.DataFrame, frequency: ReturnFrequency
) -> pl.DataFrame:
    """Align on common trading dates, sample the last close per period, then compute simple returns.

    Inputs: columns ``date`` and ``close`` (use corporate-action adjusted closes).
    """
    joined = (
        security.select("date", pl.col("close").alias("sec"))
        .join(benchmark.select("date", pl.col("close").alias("bmk")), on="date", how="inner")
        .drop_nulls()
        .sort("date")
    )
    if frequency is not ReturnFrequency.DAILY:
        joined = (
            joined.with_columns(pl.col("date").dt.truncate(_TRUNCATE[frequency]).alias("_period"))
            .group_by("_period", maintain_order=True)
            .agg(pl.col("date").last(), pl.col("sec").last(), pl.col("bmk").last())
            .drop("_period")
        )
    return joined.with_columns(
        pl.col("sec").pct_change().alias("r_sec"),
        pl.col("bmk").pct_change().alias("r_bmk"),
    ).drop_nulls()


def compute_beta(
    security: pl.DataFrame,
    benchmark: pl.DataFrame,
    frequency: ReturnFrequency = ReturnFrequency.WEEKLY,
    start: date | None = None,
    end: date | None = None,
) -> BetaResult:
    """``BETA <GO>`` from closes: default 2 years of weekly (last close of each week) returns."""
    if start is not None:
        security = security.filter(pl.col("date") >= start)
        benchmark = benchmark.filter(pl.col("date") >= start)
    if end is not None:
        security = security.filter(pl.col("date") <= end)
        benchmark = benchmark.filter(pl.col("date") <= end)

    rets = periodic_returns(security, benchmark, frequency)
    if rets.height < _MIN_OBS[frequency]:
        raise ValueError(f"need >= {_MIN_OBS[frequency]} {frequency} observations, got {rets.height}")
    est = calculate_bloomberg_beta(rets["r_sec"], rets["r_bmk"])
    first, last = rets["date"].min(), rets["date"].max()
    assert isinstance(first, date) and isinstance(last, date)
    return BetaResult(**est.model_dump(), start=first, end=last, frequency=frequency)


# ------------------------------------------------------------------------------ Enterprise Value
class ValuationBasis(StrEnum):
    ENTERPRISE_VALUE = "ENTERPRISE_VALUE"
    TANGIBLE_BOOK = "TANGIBLE_BOOK"


class InstitutionalEV(BaseModel):
    model_config = ConfigDict(frozen=True)

    sector: SectorClass
    valuation_basis: ValuationBasis
    market_cap: Decimal
    debt_included: Decimal
    lease_liabilities_included: Decimal
    preferred_stock: Decimal
    minority_interest: Decimal
    cash_netted: Decimal
    enterprise_value: Decimal
    tangible_book_value: Decimal | None = None
    price_to_tangible_book: Decimal | None = None

    def ev_to_ebitda(self, ebitda: Decimal) -> Decimal | None:
        """None where not meaningful: financials (no EBITDA concept) or non-positive EBITDA."""
        if self.valuation_basis is not ValuationBasis.ENTERPRISE_VALUE or ebitda <= 0:
            return None
        return self.enterprise_value / ebitda

    def ev_to_sales(self, revenue: Decimal) -> Decimal | None:
        if self.valuation_basis is not ValuationBasis.ENTERPRISE_VALUE or revenue <= 0:
            return None
        return self.enterprise_value / revenue


def tangible_book_value(
    common_equity: Decimal, goodwill: Decimal = _ZERO, other_intangibles: Decimal = _ZERO
) -> Decimal:
    """Common shareholders' equity less goodwill and other intangibles (preferred already excluded)."""
    if goodwill < 0 or other_intangibles < 0:
        raise ValueError("goodwill and intangibles cannot be negative")
    return common_equity - goodwill - other_intangibles


def calculate_institutional_ev(
    market_cap: Decimal,
    total_debt: Decimal,
    lease_liabilities: Decimal,
    preferred_stock: Decimal,
    minority_interest: Decimal,
    cash_and_st_investments: Decimal,
    sector: SectorClass | str,
    *,
    tangible_book_value: Decimal | None = None,
) -> InstitutionalEV:
    """EV = market cap + debt + leases + preferred + minority - cash & short-term investments.

    For BANK/NBFC/INSURANCE the debt, lease and cash terms are excluded (EV = market cap + preferred
    + minority) and ``tangible_book_value`` is required to produce the P/TBV multiple instead.
    """
    sector = SectorClass(sector)
    if market_cap <= 0:
        raise ValueError("market_cap must be positive")
    for name, v in (("total_debt", total_debt), ("lease_liabilities", lease_liabilities),
                    ("preferred_stock", preferred_stock), ("cash_and_st_investments", cash_and_st_investments)):
        if v < 0:
            raise ValueError(f"{name} cannot be negative")

    if sector.is_financial:
        if tangible_book_value is None:
            raise ValueError(f"{sector.value} valuation needs tangible_book_value for the P/TBV multiple")
        return InstitutionalEV(
            sector=sector,
            valuation_basis=ValuationBasis.TANGIBLE_BOOK,
            market_cap=market_cap,
            debt_included=_ZERO,
            lease_liabilities_included=_ZERO,
            preferred_stock=preferred_stock,
            minority_interest=minority_interest,
            cash_netted=_ZERO,
            enterprise_value=market_cap + preferred_stock + minority_interest,
            tangible_book_value=tangible_book_value,
            # Negative tangible equity makes the multiple meaningless rather than negative.
            price_to_tangible_book=market_cap / tangible_book_value if tangible_book_value > 0 else None,
        )
    return InstitutionalEV(
        sector=sector,
        valuation_basis=ValuationBasis.ENTERPRISE_VALUE,
        market_cap=market_cap,
        debt_included=total_debt,
        lease_liabilities_included=lease_liabilities,
        preferred_stock=preferred_stock,
        minority_interest=minority_interest,
        cash_netted=cash_and_st_investments,
        enterprise_value=(
            market_cap + total_debt + lease_liabilities + preferred_stock + minority_interest
            - cash_and_st_investments
        ),
        tangible_book_value=tangible_book_value,
        price_to_tangible_book=(
            market_cap / tangible_book_value if tangible_book_value is not None and tangible_book_value > 0 else None
        ),
    )
