from __future__ import annotations

import math
from datetime import date

import pytest
from scipy.stats import norm

pytest.importorskip("QuantLib")

from quantterm.models import Exchange  # noqa: E402
from quantterm.quant.derivatives_ovme import (  # noqa: E402
    ExerciseStyle,
    MarketInputs,
    OptionContract,
    OptionType,
    OptionValuationEngine,
)

NSE = OptionValuationEngine(Exchange.NSE)
NYSE = OptionValuationEngine(Exchange.NASDAQ)


def _bs(s: float, k: float, t: float, r: float, q: float, v: float, call: bool) -> dict[str, float]:
    d1 = (math.log(s / k) + (r - q + 0.5 * v * v) * t) / (v * math.sqrt(t))
    d2 = d1 - v * math.sqrt(t)
    dq, dr = math.exp(-q * t), math.exp(-r * t)
    if call:
        price = s * dq * norm.cdf(d1) - k * dr * norm.cdf(d2)
        delta = dq * norm.cdf(d1)
        theta = -s * dq * norm.pdf(d1) * v / (2 * math.sqrt(t)) - r * k * dr * norm.cdf(d2) + q * s * dq * norm.cdf(d1)
    else:
        price = k * dr * norm.cdf(-d2) - s * dq * norm.cdf(-d1)
        delta = -dq * norm.cdf(-d1)
        theta = -s * dq * norm.pdf(d1) * v / (2 * math.sqrt(t)) + r * k * dr * norm.cdf(-d2) - q * s * dq * norm.cdf(-d1)
    gamma = dq * norm.pdf(d1) / (s * v * math.sqrt(t))
    vega = s * dq * norm.pdf(d1) * math.sqrt(t)
    return {"price": price, "delta": delta, "gamma": gamma, "theta": theta / 365, "vega": vega / 100}


NIFTY_CALL = OptionContract(
    underlying="NIFTY",
    exchange=Exchange.NSE,
    option_type=OptionType.CALL,
    strike=24500.0,
    expiry=date(2024, 8, 15),  # Independence Day holiday -> trades expire 14 Aug
)
NIFTY_MKT = MarketInputs(
    valuation_date=date(2024, 8, 1), spot=24950.0, volatility=0.14, risk_free_rate=0.068, dividend_yield=0.012
)


def test_nse_holiday_expiry_rolls_back() -> None:
    assert NSE.adjusted_expiry(NIFTY_CALL) == date(2024, 8, 14)
    assert NSE.calendar.name() == "National Stock Exchange of India"


def test_nyse_good_friday_expiry_rolls_back() -> None:
    c = OptionContract(
        underlying="AAPL", exchange=Exchange.NASDAQ, option_type=OptionType.PUT, strike=200.0, expiry=date(2025, 4, 18)
    )
    assert NYSE.adjusted_expiry(c) == date(2025, 4, 17)
    assert NYSE.calendar.name() == "New York stock exchange"
    assert c.exercise_style is ExerciseStyle.AMERICAN
    assert NIFTY_CALL.exercise_style is ExerciseStyle.EUROPEAN


def test_official_holiday_list_overrides_quantlib_table() -> None:
    # A special closure absent from QuantLib's table (e.g. an election-day holiday).
    engine = OptionValuationEngine(Exchange.NSE, official_holidays=[date(2024, 8, 14), date(2024, 8, 15)])
    assert engine.adjusted_expiry(NIFTY_CALL) == date(2024, 8, 13)
    assert NSE.adjusted_expiry(NIFTY_CALL) == date(2024, 8, 14)  # built-in calendar unaffected


def test_engine_rejects_other_market() -> None:
    with pytest.raises(ValueError, match="engine"):
        NYSE.value(NIFTY_CALL, NIFTY_MKT)


def test_european_greeks_match_closed_form() -> None:
    res = NSE.value(NIFTY_CALL, NIFTY_MKT)
    t = (date(2024, 8, 14) - date(2024, 8, 1)).days / 365.0
    ref = _bs(24950.0, 24500.0, t, 0.068, 0.012, 0.14, call=True)
    assert res.time_to_expiry_years == pytest.approx(t)
    assert res.npv == pytest.approx(ref["price"], rel=1e-8)
    assert res.delta == pytest.approx(ref["delta"], rel=1e-8)
    assert res.gamma == pytest.approx(ref["gamma"], rel=1e-8)
    assert res.vega_per_vol_point == pytest.approx(ref["vega"], rel=1e-8)
    assert res.theta_per_day == pytest.approx(ref["theta"], rel=1e-6)
    assert res.trading_days_to_expiry == 9


def test_put_call_parity() -> None:
    put = NIFTY_CALL.model_copy(update={"option_type": OptionType.PUT})
    c, p = NSE.value(NIFTY_CALL, NIFTY_MKT), NSE.value(put, NIFTY_MKT)
    t = c.time_to_expiry_years
    fwd_diff = 24950.0 * math.exp(-0.012 * t) - 24500.0 * math.exp(-0.068 * t)
    assert c.npv - p.npv == pytest.approx(fwd_diff, rel=1e-9)
    assert c.delta - p.delta == pytest.approx(math.exp(-0.012 * t), rel=1e-9)


def test_american_put_early_exercise_premium() -> None:
    eu = OptionContract(
        underlying="AAPL", exchange=Exchange.NASDAQ, option_type=OptionType.PUT,
        strike=230.0, expiry=date(2025, 1, 17), exercise=ExerciseStyle.EUROPEAN,
    )
    am = eu.model_copy(update={"exercise": ExerciseStyle.AMERICAN})
    mkt = MarketInputs(valuation_date=date(2024, 10, 1), spot=200.0, volatility=0.25, risk_free_rate=0.05)
    r_eu, r_am = NYSE.value(eu, mkt), NYSE.value(am, mkt)
    assert r_am.npv > r_eu.npv
    assert r_am.npv >= 30.0  # never below intrinsic
    assert -1.0 <= r_am.delta < r_eu.delta < 0
    assert r_am.vega_per_vol_point > 0
    assert r_am.rho_per_pct < 0


def test_american_call_no_dividend_equals_european() -> None:
    am = OptionContract(
        underlying="MSFT", exchange=Exchange.NASDAQ, option_type=OptionType.CALL, strike=420.0, expiry=date(2024, 12, 20)
    )
    eu = am.model_copy(update={"exercise": ExerciseStyle.EUROPEAN})
    mkt = MarketInputs(valuation_date=date(2024, 9, 3), spot=410.0, volatility=0.22, risk_free_rate=0.05)
    r_am, r_eu = NYSE.value(am, mkt), NYSE.value(eu, mkt)
    assert r_am.npv == pytest.approx(r_eu.npv, rel=2e-3)
    assert r_am.vega_per_vol_point == pytest.approx(r_eu.vega_per_vol_point, rel=1e-2)
    assert r_am.rho_per_pct == pytest.approx(r_eu.rho_per_pct, rel=1e-2)


def test_implied_vol_round_trip_and_value_from_premium() -> None:
    target = NSE.value(NIFTY_CALL, NIFTY_MKT).npv
    seeded = NIFTY_MKT.model_copy(update={"volatility": 0.30})
    assert NSE.implied_volatility(NIFTY_CALL, seeded, target) == pytest.approx(0.14, abs=1e-6)
    solved = NSE.value_from_premium(NIFTY_CALL, seeded, target)
    assert solved.volatility == pytest.approx(0.14, abs=1e-6) and solved.npv == pytest.approx(target, rel=1e-8)


def test_american_implied_vol_round_trip() -> None:
    am = OptionContract(underlying="AAPL", exchange=Exchange.NASDAQ, option_type=OptionType.PUT,
                        strike=230.0, expiry=date(2025, 1, 17))
    mkt = MarketInputs(valuation_date=date(2024, 10, 1), spot=225.0, volatility=0.27, risk_free_rate=0.05)
    premium = NYSE.value(am, mkt).npv
    assert NYSE.implied_volatility(am, mkt.model_copy(update={"volatility": 0.5}), premium) == pytest.approx(0.27, abs=1e-5)


@pytest.mark.parametrize("premium", [0.0, 300.0, 25000.0])
def test_premium_outside_arbitrage_bounds_has_no_implied_vol(premium: float) -> None:
    # Deep ITM call: intrinsic ~450, so 300 is below the lower bound; 25 000 exceeds the spot.
    with pytest.raises(ValueError, match="no-arbitrage"):
        NSE.implied_volatility(NIFTY_CALL, NIFTY_MKT, premium)


def test_expired_option_rejected() -> None:
    late = NIFTY_MKT.model_copy(update={"valuation_date": date(2024, 8, 14)})
    with pytest.raises(ValueError, match="expiry"):
        NSE.value(NIFTY_CALL, late)
