from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

import numpy as np
import polars as pl
import pytest
import statsmodels.api as sm

from quantterm.models import SectorClass
from quantterm.quant.equity_metrics import (
    ReturnFrequency,
    ValuationBasis,
    blume_adjust,
    calculate_bloomberg_beta,
    calculate_institutional_ev,
    compute_beta,
    periodic_returns,
    tangible_book_value,
)


def _business_days(start: date, n: int) -> list[date]:
    out: list[date] = []
    d = start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _series(dates: list[date], returns: np.ndarray, start_price: float) -> pl.DataFrame:
    prices = start_price * np.cumprod(np.concatenate([[1.0], 1.0 + returns]))
    return pl.DataFrame({"date": dates, "close": prices[: len(dates)]})


def test_blume_adjustment() -> None:
    assert blume_adjust(1.5) == pytest.approx(1.335)
    assert blume_adjust(1.0) == pytest.approx(1.0)
    assert blume_adjust(-0.2) == pytest.approx(0.196)


def test_daily_beta_exact_linear_relationship() -> None:
    dates = _business_days(date(2023, 1, 2), 300)
    rng = np.random.default_rng(7)
    r_bmk = rng.normal(0.0004, 0.01, len(dates) - 1)
    r_sec = 0.0001 + 1.3 * r_bmk
    res = compute_beta(
        _series(dates, r_sec, 1600.0), _series(dates, r_bmk, 22000.0), ReturnFrequency.DAILY
    )
    assert res.raw_beta == pytest.approx(1.3, abs=1e-9)
    assert res.adjusted_beta == pytest.approx(0.67 * 1.3 + 0.33)
    assert res.r_squared == pytest.approx(1.0)
    assert res.alpha == pytest.approx(0.0001, abs=1e-9)
    assert res.n_observations == 299


def test_weekly_beta_with_noise_and_mismatched_holidays() -> None:
    dates = _business_days(date(2022, 11, 1), 520)
    rng = np.random.default_rng(42)
    r_bmk = rng.normal(0.0003, 0.011, len(dates) - 1)
    r_sec = 0.8 * r_bmk + rng.normal(0.0, 0.006, len(dates) - 1)
    sec = _series(dates, r_sec, 500.0)
    # Benchmark missing a few sessions (e.g. data gaps); alignment must be on common dates.
    bmk = _series(dates, r_bmk, 70000.0).filter(~pl.col("date").is_in(dates[100:103]))
    res = compute_beta(sec, bmk, ReturnFrequency.WEEKLY, start=date(2022, 11, 1))
    assert res.frequency is ReturnFrequency.WEEKLY
    assert 100 <= res.n_observations <= 104
    assert res.raw_beta == pytest.approx(0.8, abs=0.15)
    assert res.beta_std_error > 0
    assert res.adjusted_beta == pytest.approx(blume_adjust(res.raw_beta))


def test_weekly_sampling_uses_last_close_of_week() -> None:
    dates = _business_days(date(2024, 1, 1), 10)  # Mon 1 Jan .. Fri 12 Jan
    df = pl.DataFrame({"date": dates, "close": [float(i + 1) for i in range(10)]})
    rets = periodic_returns(df, df, ReturnFrequency.WEEKLY)
    assert rets["date"].to_list() == [date(2024, 1, 12)]
    assert rets["r_sec"].item() == pytest.approx(10.0 / 5.0 - 1.0)


def test_insufficient_observations() -> None:
    dates = _business_days(date(2024, 1, 1), 30)
    df = _series(dates, np.full(29, 0.001), 100.0)
    with pytest.raises(ValueError, match="observations"):
        compute_beta(df, df, ReturnFrequency.WEEKLY)


def test_ev_corporate_includes_debt_leases_and_nets_cash() -> None:
    # Reliance-style corporate: debt, leases, minority and cash all count.
    mcap = Decimal("1300") * Decimal("13532472634")
    ev = calculate_institutional_ev(
        market_cap=mcap,
        total_debt=Decimal("3.5e12"),
        lease_liabilities=Decimal("2.1e11"),
        preferred_stock=Decimal("0"),
        minority_interest=Decimal("1.2e12"),
        cash_and_st_investments=Decimal("2.2e12"),
        sector=SectorClass.GENERAL,
    )
    assert ev.valuation_basis is ValuationBasis.ENTERPRISE_VALUE
    assert ev.enterprise_value == mcap + Decimal("3.5e12") + Decimal("2.1e11") + Decimal("1.2e12") - Decimal("2.2e12")
    assert ev.ev_to_ebitda(Decimal("1.6e12")) == ev.enterprise_value / Decimal("1.6e12")
    assert ev.ev_to_ebitda(Decimal("-5")) is None
    assert ev.price_to_tangible_book is None


@pytest.mark.parametrize("sector", [SectorClass.BANK, SectorClass.NBFC, "INSURANCE"])
def test_financials_switch_to_tangible_book(sector: SectorClass | str) -> None:
    mcap = Decimal("1700") * Decimal("7640000000")
    tbv = tangible_book_value(Decimal("4.6e12"), goodwill=Decimal("1.0e11"), other_intangibles=Decimal("5e10"))
    ev = calculate_institutional_ev(
        market_cap=mcap,
        total_debt=Decimal("6.6e12"),  # borrowings would dwarf market cap
        lease_liabilities=Decimal("1e11"),
        preferred_stock=Decimal("0"),
        minority_interest=Decimal("4.5e11"),
        cash_and_st_investments=Decimal("2.0e12"),
        sector=sector,
        tangible_book_value=tbv,
    )
    assert ev.valuation_basis is ValuationBasis.TANGIBLE_BOOK
    assert ev.debt_included == ev.lease_liabilities_included == ev.cash_netted == 0
    assert ev.enterprise_value == mcap + Decimal("4.5e11")
    assert ev.price_to_tangible_book == mcap / Decimal("4.45e12")
    assert ev.ev_to_ebitda(Decimal("1e12")) is None and ev.ev_to_sales(Decimal("1e12")) is None


def test_financials_require_tangible_book_and_negative_tbv_has_no_multiple() -> None:
    args = dict(market_cap=Decimal("1e9"), total_debt=Decimal("5e9"), lease_liabilities=Decimal("0"),
                preferred_stock=Decimal("0"), minority_interest=Decimal("0"), cash_and_st_investments=Decimal("0"))
    with pytest.raises(ValueError, match="tangible_book_value"):
        calculate_institutional_ev(**args, sector=SectorClass.NBFC)  # type: ignore[arg-type]
    ev = calculate_institutional_ev(**args, sector=SectorClass.BANK, tangible_book_value=Decimal("-2e8"))  # type: ignore[arg-type]
    assert ev.price_to_tangible_book is None


def test_ev_inputs_validation() -> None:
    base = dict(total_debt=Decimal("0"), lease_liabilities=Decimal("0"), preferred_stock=Decimal("0"),
                minority_interest=Decimal("-1e6"), cash_and_st_investments=Decimal("0"), sector="GENERAL")
    with pytest.raises(ValueError):
        calculate_institutional_ev(market_cap=Decimal("0"), **base)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="lease"):
        calculate_institutional_ev(market_cap=Decimal("1"), **{**base, "lease_liabilities": Decimal("-1")})  # type: ignore[arg-type]
    # Deficit minority interest (negative NCI) is legitimate.
    assert calculate_institutional_ev(market_cap=Decimal("1e7"), **base).enterprise_value == Decimal("9e6")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        tangible_book_value(Decimal("1"), goodwill=Decimal("-1"))


def test_beta_matches_statsmodels_ols() -> None:
    rng = np.random.default_rng(11)
    x = rng.normal(0.002, 0.025, 104)
    y = 0.001 + 1.15 * x + rng.normal(0, 0.02, 104)
    ours = calculate_bloomberg_beta(y, x)
    fit = sm.OLS(y, sm.add_constant(x)).fit()
    assert ours.raw_beta == pytest.approx(fit.params[1], rel=1e-12)
    assert ours.alpha == pytest.approx(fit.params[0], rel=1e-10)
    assert ours.beta_std_error == pytest.approx(fit.bse[1], rel=1e-10)
    assert ours.alpha_std_error == pytest.approx(fit.bse[0], rel=1e-10)
    assert ours.r_squared == pytest.approx(fit.rsquared, rel=1e-12)
    assert ours.t_stat == pytest.approx(fit.tvalues[1], rel=1e-10)
    assert ours.adjusted_beta == pytest.approx(0.67 * fit.params[1] + 0.33)


def test_beta_drops_incomplete_pairs_and_validates_shape() -> None:
    x = np.array([0.01, -0.02, np.nan, 0.015, 0.005, -0.01])
    y = np.array([0.012, -0.025, 0.03, np.nan, 0.006, -0.011])
    est = calculate_bloomberg_beta(pl.Series(y), list(x))
    assert est.n_observations == 4
    with pytest.raises(ValueError, match="aligned"):
        calculate_bloomberg_beta([0.1, 0.2, 0.3], [0.1, 0.2])
    with pytest.raises(ValueError, match="zero variance"):
        calculate_bloomberg_beta([0.1, 0.2, 0.3], [0.01, 0.01, 0.01])
