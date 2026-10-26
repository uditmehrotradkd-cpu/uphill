"""Accuracy benchmark on three listings that together exercise every analytics branch.

* AAPL     - NASDAQ vs SPX; corporate EV with leases; American options on the NYSE calendar.
* RELIANCE - NSE vs NIFTY; corporate EV with heavy debt and minority interest; European stock options.
* HDFCBANK - NSE vs NIFTY; bank -> tangible-book valuation basis; European stock options.

Layer 1 (always runs) uses real inputs: 2023-2024 daily closes (split/bonus-adjusted, not dividend-
adjusted, as ``BETA <GO>`` uses) and filed balance sheets (SEC 10-K, NSE results XBRL). Outputs are
checked against independent implementations: statsmodels OLS on separately built weekly returns,
hand-derived EV arithmetic, closed-form Black-Scholes, and a QuantLib CRR binomial tree for
American exercise.

Layer 2 (opt-in) checks Bloomberg parity against values typed into
``benchmarks/bloomberg_reference.json``; empty entries are skipped.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
import QuantLib as ql
import statsmodels.api as sm
from scipy.stats import norm

from quantterm.models import Exchange, SectorClass
from quantterm.quant.derivatives_ovme import (
    ExerciseStyle,
    MarketInputs,
    OptionContract,
    OptionType,
    OptionValuation,
    OptionValuationEngine,
)
from quantterm.quant.equity_metrics import (
    InstitutionalEV,
    ReturnFrequency,
    ValuationBasis,
    calculate_institutional_ev,
    compute_beta,
    tangible_book_value,
)

DATA = Path(__file__).parent / "benchmarks"
REFERENCE: dict[str, Any] = json.loads((DATA / "bloomberg_reference.json").read_text(encoding="utf-8"))
TOL: dict[str, float] = REFERENCE["tolerances"]
BETA_START, BETA_END = date(2023, 1, 1), date(2024, 12, 31)


@dataclass(frozen=True)
class Benchmark:
    symbol: str
    exchange: Exchange
    index: str
    sector: SectorClass


BENCHMARKS = [
    Benchmark("AAPL", Exchange.NASDAQ, "SPX", SectorClass.GENERAL),
    Benchmark("RELIANCE", Exchange.NSE, "NIFTY", SectorClass.GENERAL),
    Benchmark("HDFCBANK", Exchange.NSE, "NIFTY", SectorClass.BANK),
]
IDS = [b.symbol for b in BENCHMARKS]


def _closes(name: str) -> pl.DataFrame:
    return pl.read_csv(DATA / f"{name}_daily_close.csv", try_parse_dates=True)


def _bloomberg(symbol: str, section: str, field: str) -> float | None:
    value = REFERENCE[symbol][section][field]
    return None if value is None else float(value)


# ===================================================================================== BETA
def _independent_weekly_returns(stock: str, index: str) -> tuple[np.ndarray, np.ndarray]:
    """Plain-Python rebuild: common dates, last close of each ISO week, simple returns."""
    def load(name: str) -> dict[date, float]:
        with open(DATA / f"{name}_daily_close.csv", newline="") as fh:
            return {date.fromisoformat(r["date"]): float(r["close"]) for r in csv.DictReader(fh)}

    s, b = load(stock), load(index)
    weekly: dict[tuple[int, int], tuple[float, float]] = {}
    for d in sorted(set(s) & set(b)):
        if BETA_START <= d <= BETA_END:
            weekly[d.isocalendar()[:2]] = (s[d], b[d])  # later days overwrite -> last close of the week
    closes = np.array(list(weekly.values()))
    rets = closes[1:] / closes[:-1] - 1.0
    return rets[:, 0], rets[:, 1]


# Locked from the first validated run; any change means the beta pipeline's numerics moved.
BETA_BASELINE = {"AAPL": 1.0112348399318414, "RELIANCE": 1.1813549396153662, "HDFCBANK": 1.0286206638456012}


@pytest.mark.parametrize("bm", BENCHMARKS, ids=IDS)
def test_beta_matches_statsmodels_on_real_prices(bm: Benchmark) -> None:
    ours = compute_beta(_closes(bm.symbol), _closes(bm.index), ReturnFrequency.WEEKLY, BETA_START, BETA_END)
    y, x = _independent_weekly_returns(bm.symbol, bm.index)
    fit = sm.OLS(y, sm.add_constant(x)).fit()

    assert ours.n_observations == len(y) == 104
    assert ours.raw_beta == pytest.approx(fit.params[1], rel=1e-10)
    assert ours.beta_std_error == pytest.approx(fit.bse[1], rel=1e-10)
    assert ours.r_squared == pytest.approx(fit.rsquared, rel=1e-10)
    assert ours.adjusted_beta == pytest.approx(0.67 * fit.params[1] + 0.33, rel=1e-12)
    assert abs(ours.adjusted_beta - 1.0) < abs(ours.raw_beta - 1.0)  # Blume shrinks toward the market
    assert ours.raw_beta == pytest.approx(BETA_BASELINE[bm.symbol], abs=1e-9)


@pytest.mark.parametrize("bm", BENCHMARKS, ids=IDS)
def test_beta_bloomberg_parity(bm: Benchmark) -> None:
    raw, adjusted = _bloomberg(bm.symbol, "beta", "raw"), _bloomberg(bm.symbol, "beta", "adjusted")
    if raw is None and adjusted is None:
        pytest.skip("no BETA <GO> reference recorded")
    ours = compute_beta(_closes(bm.symbol), _closes(bm.index), ReturnFrequency.WEEKLY, BETA_START, BETA_END)
    if raw is not None:
        assert ours.raw_beta == pytest.approx(raw, abs=TOL["beta_abs"])
    if adjusted is not None:
        assert ours.adjusted_beta == pytest.approx(adjusted, abs=TOL["beta_abs"])


# ===================================================================================== EV
def _price_on(symbol: str, d: date) -> Decimal:
    row = _closes(symbol).filter(pl.col("date") == d)
    return Decimal(str(row["close"].item()))


def _ev(symbol: str) -> InstitutionalEV:
    if symbol == "AAPL":
        # FY2024 10-K, accession 0000320193-24-000123 (USD). Shares: cover page, 18 Oct 2024.
        return calculate_institutional_ev(
            market_cap=_price_on("AAPL", date(2024, 9, 27)) * Decimal("15115823000"),
            total_debt=Decimal("85750e6") + Decimal("10912e6") + Decimal("9967e6"),  # LT debt + current + CP
            lease_liabilities=Decimal("11534e6") + Decimal("896e6"),  # operating + finance leases
            preferred_stock=Decimal(0),
            minority_interest=Decimal(0),
            cash_and_st_investments=Decimal("29943e6") + Decimal("35228e6"),  # cash + current securities
            sector=SectorClass.GENERAL,
        )
    if symbol == "RELIANCE":
        # FY2024 consolidated results XBRL, NSE filing 1169208 (INR). Price/shares on the 1:1 bonus-adjusted basis.
        return calculate_institutional_ev(
            market_cap=_price_on("RELIANCE", date(2024, 3, 28)) * Decimal("6766000000") * 2,
            total_debt=Decimal("2227120000000") + Decimal("1019100000000"),
            lease_liabilities=Decimal(0),  # not broken out in the results XBRL
            preferred_stock=Decimal(0),
            minority_interest=Decimal("1323070000000"),
            cash_and_st_investments=Decimal("972250000000") + Decimal("1061700000000"),
            sector=SectorClass.GENERAL,
            tangible_book_value=tangible_book_value(
                Decimal("7934810000000"),
                goodwill=Decimal("149890000000"),
                other_intangibles=Decimal("1589120000000") + Decimal("1864730000000"),
            ),
        )
    # HDFCBANK FY2024 consolidated banking-format XBRL, NSE filing 1169188 (INR); 1:1 bonus-adjusted basis.
    return calculate_institutional_ev(
        market_cap=_price_on("HDFCBANK", date(2024, 3, 28)) * Decimal("7596900000") * 2,
        total_debt=Decimal("7306154600000"),  # borrowings; deposits (2.38e13) are not debt either
        lease_liabilities=Decimal(0),
        preferred_stock=Decimal(0),
        minority_interest=Decimal(0),  # not reported in the banking-format XBRL
        cash_and_st_investments=Decimal(0),
        sector=SectorClass.BANK,
        tangible_book_value=tangible_book_value(Decimal("7596900000") + Decimal("4529828400000")),
    )


def test_ev_aapl_from_10k() -> None:
    ev = _ev("AAPL")
    assert ev.market_cap == Decimal("3443233321170.00")  # 227.79 x 15,115,823,000
    assert ev.enterprise_value == Decimal("3497121321170.00")
    ebitda = Decimal("123216e6") + Decimal("11445e6")  # FY24 operating income + D&A
    ratio = ev.ev_to_ebitda(ebitda)
    assert ratio is not None and float(ratio) == pytest.approx(25.9698, abs=1e-4)


def test_ev_reliance_counts_debt_and_minority_nets_cash() -> None:
    ev = _ev("RELIANCE")
    assert ev.valuation_basis is ValuationBasis.ENTERPRISE_VALUE
    assert ev.market_cap == Decimal("20106522200000.00")
    assert ev.enterprise_value == Decimal("22641862200000.00")
    assert ev.price_to_tangible_book == ev.market_cap / Decimal("4331070000000")


def test_ev_hdfcbank_switches_to_tangible_book() -> None:
    ev = _ev("HDFCBANK")
    assert ev.valuation_basis is ValuationBasis.TANGIBLE_BOOK
    assert ev.debt_included == 0 and ev.cash_netted == 0  # 7.3 trn of borrowings not added back
    assert ev.enterprise_value == ev.market_cap == Decimal("10999551510000.00")
    assert ev.price_to_tangible_book is not None
    assert float(ev.price_to_tangible_book) == pytest.approx(2.424183, abs=1e-6)
    assert ev.ev_to_ebitda(Decimal("1e12")) is None


@pytest.mark.parametrize("bm", BENCHMARKS, ids=IDS)
def test_ev_bloomberg_parity(bm: Benchmark) -> None:
    ev = _ev(bm.symbol)
    if bm.sector.is_financial:
        ref = _bloomberg(bm.symbol, "ev", "price_to_tangible_book")
        if ref is None:
            pytest.skip("no PX_TO_TANG_BV reference recorded")
        assert ev.price_to_tangible_book is not None
        assert float(ev.price_to_tangible_book) == pytest.approx(ref, rel=TOL["ptbv_rel"])
    else:
        ref = _bloomberg(bm.symbol, "ev", "enterprise_value")
        if ref is None:
            pytest.skip("no CURR_ENTP_VAL reference recorded")
        assert float(ev.enterprise_value) == pytest.approx(ref, rel=TOL["ev_rel"])


# ===================================================================================== OVME
VALUATION_DATE = date(2024, 12, 31)
OVME_CASES: dict[str, tuple[OptionContract, MarketInputs]] = {
    # Jan-25 monthly put; NYSE closed 1 Jan and 9 Jan 2025 (national day of mourning).
    "AAPL": (
        OptionContract(underlying="AAPL", exchange=Exchange.NASDAQ, option_type=OptionType.PUT,
                       strike=250.0, expiry=date(2025, 1, 17)),
        MarketInputs(valuation_date=VALUATION_DATE, spot=250.42, volatility=0.22, risk_free_rate=0.0425,
                     dividend_yield=0.004),
    ),
    # NSE monthly stock options expire on the last Thursday; European exercise.
    "RELIANCE": (
        OptionContract(underlying="RELIANCE", exchange=Exchange.NSE, option_type=OptionType.CALL,
                       strike=1220.0, expiry=date(2025, 1, 30)),
        MarketInputs(valuation_date=VALUATION_DATE, spot=1215.45, volatility=0.24, risk_free_rate=0.066,
                     dividend_yield=0.004),
    ),
    # Spot on the pre-bonus basis that listed strikes used in Dec 2024.
    "HDFCBANK": (
        OptionContract(underlying="HDFCBANK", exchange=Exchange.NSE, option_type=OptionType.PUT,
                       strike=1780.0, expiry=date(2025, 1, 30)),
        MarketInputs(valuation_date=VALUATION_DATE, spot=1772.85, volatility=0.18, risk_free_rate=0.066,
                     dividend_yield=0.011),
    ),
}
ENGINES = {Exchange.NASDAQ: OptionValuationEngine(Exchange.NASDAQ), Exchange.NSE: OptionValuationEngine(Exchange.NSE)}


def _value(symbol: str, exercise: ExerciseStyle | None = None) -> tuple[OptionContract, MarketInputs, OptionValuation]:
    contract, market = OVME_CASES[symbol]
    if exercise is not None:
        contract = contract.model_copy(update={"exercise": exercise})
    return contract, market, ENGINES[contract.exchange].value(contract, market)


def _black_scholes(c: OptionContract, m: MarketInputs, t: float) -> dict[str, float]:
    s, k, r, q, v = m.spot, c.strike, m.risk_free_rate, m.dividend_yield, m.volatility
    d1 = (math.log(s / k) + (r - q + 0.5 * v * v) * t) / (v * math.sqrt(t))
    d2 = d1 - v * math.sqrt(t)
    dq, dr = math.exp(-q * t), math.exp(-r * t)
    common_theta = -s * dq * norm.pdf(d1) * v / (2 * math.sqrt(t))
    if c.option_type is OptionType.CALL:
        npv = s * dq * norm.cdf(d1) - k * dr * norm.cdf(d2)
        delta = dq * norm.cdf(d1)
        theta = common_theta - r * k * dr * norm.cdf(d2) + q * s * dq * norm.cdf(d1)
        rho = k * t * dr * norm.cdf(d2)
    else:
        npv = k * dr * norm.cdf(-d2) - s * dq * norm.cdf(-d1)
        delta = -dq * norm.cdf(-d1)
        theta = common_theta + r * k * dr * norm.cdf(-d2) - q * s * dq * norm.cdf(-d1)
        rho = -k * t * dr * norm.cdf(-d2)
    return {
        "npv": npv, "delta": delta, "gamma": dq * norm.pdf(d1) / (s * v * math.sqrt(t)),
        "theta_per_day": theta / 365, "vega_per_vol_point": s * dq * norm.pdf(d1) * math.sqrt(t) / 100,
        "rho_per_pct": rho / 100,
    }


@pytest.mark.parametrize("symbol", IDS)
def test_european_greeks_equal_closed_form(symbol: str) -> None:
    contract, market, val = _value(symbol, ExerciseStyle.EUROPEAN)
    ref = _black_scholes(contract, market, val.time_to_expiry_years)
    for greek, expected in ref.items():
        assert getattr(val, greek) == pytest.approx(expected, rel=1e-8, abs=1e-12), greek


def _binomial(contract: OptionContract, market: MarketInputs, vol: float, rate: float, steps: int = 5000) -> ql.VanillaOption:
    cal = ENGINES[contract.exchange].calendar
    dc = ql.Actual365Fixed()
    today = ql.Date(market.valuation_date.day, market.valuation_date.month, market.valuation_date.year)
    expiry = cal.adjust(ql.Date(contract.expiry.day, contract.expiry.month, contract.expiry.year), ql.Preceding)
    ql.Settings.instance().evaluationDate = today
    process = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(market.spot)),
        ql.YieldTermStructureHandle(ql.FlatForward(today, market.dividend_yield, dc)),
        ql.YieldTermStructureHandle(ql.FlatForward(today, rate, dc)),
        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, cal, vol, dc)),
    )
    kind = ql.Option.Put if contract.option_type is OptionType.PUT else ql.Option.Call
    option = ql.VanillaOption(ql.PlainVanillaPayoff(kind, contract.strike), ql.AmericanExercise(today, expiry))
    option.setPricingEngine(ql.BinomialVanillaEngine(process, "crr", steps))
    return option


def test_american_greeks_agree_with_independent_binomial_tree() -> None:
    contract, market, val = _value("AAPL")
    assert val.exercise is ExerciseStyle.AMERICAN
    v, r = market.volatility, market.risk_free_rate
    tree = _binomial(contract, market, v, r)
    vega = (_binomial(contract, market, v + 0.01, r).NPV() - _binomial(contract, market, v - 0.01, r).NPV()) / 2
    rho = (_binomial(contract, market, v, r + 1e-4).NPV() - _binomial(contract, market, v, r - 1e-4).NPV()) / 2e-4 / 100

    assert val.npv == pytest.approx(tree.NPV(), rel=1e-3)
    assert val.delta == pytest.approx(tree.delta(), abs=2e-3)
    assert val.gamma == pytest.approx(tree.gamma(), rel=3e-2)
    assert val.theta_per_day == pytest.approx(tree.theta() / 365, rel=5e-2)
    assert val.vega_per_vol_point == pytest.approx(vega, rel=2e-2)
    assert val.rho_per_pct == pytest.approx(rho, rel=5e-2)
    european = _value("AAPL", ExerciseStyle.EUROPEAN)[2]
    assert val.npv >= european.npv  # early-exercise premium is never negative


def test_trading_days_follow_exchange_calendars() -> None:
    assert _value("AAPL")[2].trading_days_to_expiry == 11  # 1 Jan and 9 Jan 2025 closures excluded
    assert _value("RELIANCE")[2].trading_days_to_expiry == 22
    assert _value("HDFCBANK")[2].time_to_expiry_years == pytest.approx(30 / 365)  # Actual/365 Fixed


@pytest.mark.parametrize("symbol", IDS)
def test_implied_vol_round_trip(symbol: str) -> None:
    contract, market, val = _value(symbol)
    engine = ENGINES[contract.exchange]
    solved = engine.value_from_premium(contract, market.model_copy(update={"volatility": 0.9}), val.npv)
    assert solved.volatility == pytest.approx(market.volatility, abs=1e-6)
    assert solved.delta == pytest.approx(val.delta, abs=1e-6)


@pytest.mark.parametrize("symbol", IDS)
def test_ovme_bloomberg_parity(symbol: str) -> None:
    refs = {k: v for k, v in REFERENCE[symbol]["ovme"].items() if v is not None}
    if not refs:
        pytest.skip("no OVME reference recorded")
    _, _, val = _value(symbol)
    for field, expected in refs.items():
        tol = TOL["option_price_rel"] if field == "npv" else TOL["greek_rel"]
        assert getattr(val, field) == pytest.approx(float(expected), rel=tol), field
