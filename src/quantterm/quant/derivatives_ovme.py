"""Bloomberg ``OVME``-style single-option valuation on QuantLib.

Conventions: Actual/365 Fixed for every curve and for time to expiry; the exchange holiday calendar
(``UnitedStates(NYSE)`` / ``India(NSE)``) rolls expiries back to the last trading day and counts trading
days; flat continuously compounded rate and dividend yield. Greeks are shown in OVME units: theta per
calendar day, vega per vol point, rho per 1 % rate. European options use the closed form, so every Greek
is analytic. American options use a finite-difference grid: delta, gamma and theta come off the grid,
vega and rho by central bump-and-reprice on the same grid.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, timedelta
from enum import StrEnum

import QuantLib as ql
from pydantic import BaseModel, ConfigDict, Field
from scipy.optimize import brentq

from quantterm.models import Exchange, Market

_QL_LOCK = threading.Lock()  # QuantLib's evaluation date is process-global
_VOL_BUMP = 0.01
_RATE_BUMP = 0.0001


class OptionType(StrEnum):
    CALL = "CALL"
    PUT = "PUT"


class ExerciseStyle(StrEnum):
    EUROPEAN = "EUROPEAN"
    AMERICAN = "AMERICAN"


def exchange_calendar(exchange: Exchange, official_holidays: Iterable[date] | None = None) -> ql.Calendar:
    """QuantLib's built-in exchange calendar, or an exact copy of an exchange-published holiday list.

    QuantLib's hard-coded NSE table can lag or differ from the circular NSE publishes each December;
    passing ``official_holidays`` builds an isolated calendar from that list (weekends + holidays).
    """
    if official_holidays is None:
        return ql.UnitedStates(ql.UnitedStates.NYSE) if exchange.market is Market.US else ql.India(ql.India.NSE)
    cal = ql.BespokeCalendar(f"{exchange.value} official")
    cal.addWeekend(ql.Saturday)
    cal.addWeekend(ql.Sunday)
    for d in official_holidays:
        cal.addHoliday(_to_ql(d))
    return cal


def default_exercise(exchange: Exchange) -> ExerciseStyle:
    """US single-stock options are American; NSE/BSE stock and index options are European."""
    return ExerciseStyle.AMERICAN if exchange.market is Market.US else ExerciseStyle.EUROPEAN


# SEBI's expiry-day rationalisation moved NSE monthly contracts from last Thursday to last Tuesday.
NSE_TUESDAY_EXPIRY_FROM = date(2025, 9, 1)


def standard_monthly_expiry(exchange: Exchange, year: int, month: int) -> date:
    """Unadjusted listed monthly expiry: US third Friday; NSE last Thursday (last Tuesday from Sep 2025)."""
    first = date(year, month, 1)
    if exchange.market is Market.US:
        return first + timedelta(days=(4 - first.weekday()) % 7 + 14)
    if exchange is Exchange.NSE:
        weekday = 1 if first >= NSE_TUESDAY_EXPIRY_FROM else 3
        last = date(year + month // 12, month % 12 + 1, 1) - timedelta(days=1)
        return last - timedelta(days=(last.weekday() - weekday) % 7)
    raise ValueError("BSE monthly expiry weekday has changed repeatedly; pass the expiry explicitly")


class OptionContract(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    underlying: str = Field(min_length=1)
    exchange: Exchange
    option_type: OptionType
    strike: float = Field(gt=0)
    expiry: date
    exercise: ExerciseStyle | None = None

    @property
    def exercise_style(self) -> ExerciseStyle:
        return self.exercise or default_exercise(self.exchange)


class MarketInputs(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    valuation_date: date
    spot: float = Field(gt=0)
    volatility: float = Field(gt=0, le=5.0)
    risk_free_rate: float = Field(ge=-0.05, le=1.0)  # continuously compounded
    dividend_yield: float = Field(default=0.0, ge=0, le=1.0)


class OptionValuation(BaseModel):
    model_config = ConfigDict(frozen=True)

    npv: float
    delta: float
    gamma: float
    theta_per_day: float
    vega_per_vol_point: float
    rho_per_pct: float
    volatility: float
    exercise: ExerciseStyle
    adjusted_expiry: date
    time_to_expiry_years: float
    trading_days_to_expiry: int


def _to_ql(d: date) -> ql.Date:
    return ql.Date(d.day, d.month, d.year)


def _from_ql(d: ql.Date) -> date:
    return date(d.year(), d.month(), d.dayOfMonth())


@contextmanager
def _evaluation_date(d: ql.Date) -> Iterator[None]:
    with _QL_LOCK:
        settings = ql.Settings.instance()
        previous = settings.evaluationDate
        settings.evaluationDate = d
        try:
            yield
        finally:
            settings.evaluationDate = previous


@dataclass
class _Setup:
    option: ql.VanillaOption
    process: ql.BlackScholesMertonProcess
    vol_quote: ql.SimpleQuote
    rate_quote: ql.SimpleQuote
    eval_date: ql.Date
    expiry: ql.Date
    time_to_expiry: float


def no_arbitrage_bounds(
    option_type: OptionType, exercise: ExerciseStyle, spot: float, strike: float, rate: float, q: float, t: float
) -> tuple[float, float]:
    """(lower, upper) premium bounds; a quote outside them has no implied volatility."""
    s_fwd, k_disc = spot * math.exp(-q * t), strike * math.exp(-rate * t)
    if option_type is OptionType.CALL:
        lower, upper = max(s_fwd - k_disc, 0.0), s_fwd
        if exercise is ExerciseStyle.AMERICAN:
            lower, upper = max(lower, spot - strike), spot
    else:
        lower, upper = max(k_disc - s_fwd, 0.0), k_disc
        if exercise is ExerciseStyle.AMERICAN:
            lower, upper = max(lower, strike - spot), strike
    return lower, upper


class OptionValuationEngine:
    """One engine per market; reusable and thread-safe (QuantLib global state is locked per call)."""

    def __init__(
        self,
        exchange: Exchange,
        *,
        official_holidays: Iterable[date] | None = None,
        fd_time_steps: int = 400,
        fd_space_steps: int = 400,
    ) -> None:
        if fd_time_steps < 50 or fd_space_steps < 50:
            raise ValueError("finite-difference grids below 50 steps give unstable American Greeks")
        self.exchange = exchange
        self.calendar = exchange_calendar(exchange, official_holidays)
        self.day_count = ql.Actual365Fixed()
        self._t_steps = fd_time_steps
        self._x_steps = fd_space_steps

    def _check(self, contract: OptionContract) -> None:
        if contract.exchange.market is not self.exchange.market:
            raise ValueError(f"{contract.exchange.value} contract on a {self.exchange.market.value} engine")

    def adjusted_expiry(self, contract: OptionContract) -> date:
        """An expiry on an exchange holiday moves to the preceding trading day (NSE and OCC practice)."""
        self._check(contract)
        return _from_ql(self.calendar.adjust(_to_ql(contract.expiry), ql.Preceding))

    def next_monthly_expiry(self, after: date, min_days: int = 7) -> date:
        """First holiday-adjusted standard monthly expiry at least ``min_days`` after ``after``."""
        year, month = after.year, after.month
        for _ in range(24):
            raw = standard_monthly_expiry(self.exchange, year, month)
            adjusted = _from_ql(self.calendar.adjust(_to_ql(raw), ql.Preceding))
            if (adjusted - after).days >= min_days:
                return adjusted
            year, month = (year + 1, 1) if month == 12 else (year, month + 1)
        raise AssertionError("unreachable: a monthly expiry exists within 24 months")

    def _build(self, contract: OptionContract, market: MarketInputs) -> _Setup:
        """Must be called inside ``_evaluation_date``."""
        self._check(contract)
        eval_ql = _to_ql(market.valuation_date)
        expiry_ql = self.calendar.adjust(_to_ql(contract.expiry), ql.Preceding)
        if expiry_ql <= eval_ql:
            raise ValueError("holiday-adjusted expiry must be after valuation_date")

        dc = self.day_count
        vol_q = ql.SimpleQuote(market.volatility)
        rate_q = ql.SimpleQuote(market.risk_free_rate)
        r_ts = ql.YieldTermStructureHandle(ql.FlatForward(eval_ql, ql.QuoteHandle(rate_q), dc))
        q_ts = ql.YieldTermStructureHandle(ql.FlatForward(eval_ql, market.dividend_yield, dc))
        vol_ts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(eval_ql, self.calendar, ql.QuoteHandle(vol_q), dc))
        process = ql.BlackScholesMertonProcess(ql.QuoteHandle(ql.SimpleQuote(market.spot)), q_ts, r_ts, vol_ts)
        payoff = ql.PlainVanillaPayoff(
            ql.Option.Call if contract.option_type is OptionType.CALL else ql.Option.Put, contract.strike
        )
        if contract.exercise_style is ExerciseStyle.EUROPEAN:
            option = ql.VanillaOption(payoff, ql.EuropeanExercise(expiry_ql))
            option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
        else:
            option = ql.VanillaOption(payoff, ql.AmericanExercise(eval_ql, expiry_ql))
            option.setPricingEngine(ql.FdBlackScholesVanillaEngine(process, self._t_steps, self._x_steps))
        return _Setup(option, process, vol_q, rate_q, eval_ql, expiry_ql, float(dc.yearFraction(eval_ql, expiry_ql)))

    def value(self, contract: OptionContract, market: MarketInputs) -> OptionValuation:
        with _evaluation_date(_to_ql(market.valuation_date)):
            s = self._build(contract, market)
            opt = s.option
            if contract.exercise_style is ExerciseStyle.EUROPEAN:
                vega, rho = float(opt.vega()), float(opt.rho())
            else:
                vega = _central_bump(opt, s.vol_quote, market.volatility, _VOL_BUMP)
                rho = _central_bump(opt, s.rate_quote, market.risk_free_rate, _RATE_BUMP)
            return OptionValuation(
                npv=float(opt.NPV()),
                delta=float(opt.delta()),
                gamma=float(opt.gamma()),
                theta_per_day=float(opt.theta()) / 365.0,
                vega_per_vol_point=vega / 100.0,
                rho_per_pct=rho / 100.0,
                volatility=market.volatility,
                exercise=contract.exercise_style,
                adjusted_expiry=_from_ql(s.expiry),
                time_to_expiry_years=s.time_to_expiry,
                trading_days_to_expiry=int(self.calendar.businessDaysBetween(s.eval_date, s.expiry, False, True)),
            )

    def implied_volatility(
        self,
        contract: OptionContract,
        market: MarketInputs,
        option_price: float,
        *,
        accuracy: float = 1e-7,
        max_evaluations: int = 500,
        min_vol: float = 1e-4,
        max_vol: float = 5.0,
    ) -> float:
        """Vol at which this engine reprices ``option_price`` (same grid as ``value``, so round trips are exact).

        ``market.volatility`` is ignored.
        """
        with _evaluation_date(_to_ql(market.valuation_date)):
            s = self._build(contract, market)
            lower, upper = no_arbitrage_bounds(
                contract.option_type, contract.exercise_style, market.spot, contract.strike,
                market.risk_free_rate, market.dividend_yield, s.time_to_expiry,
            )
            if not lower < option_price < upper:
                raise ValueError(
                    f"premium {option_price} outside no-arbitrage bounds ({lower:.6f}, {upper:.6f}); no implied vol"
                )

            def pricing_error(vol: float) -> float:
                s.vol_quote.setValue(vol)
                return float(s.option.NPV()) - option_price

            lo_err, hi_err = pricing_error(min_vol), pricing_error(max_vol)
            if lo_err > 0 or hi_err < 0:
                raise ValueError(f"premium {option_price} not attainable for vol in [{min_vol}, {max_vol}]")
            return float(brentq(pricing_error, min_vol, max_vol, xtol=accuracy, maxiter=max_evaluations))

    def value_from_premium(self, contract: OptionContract, market: MarketInputs, option_price: float) -> OptionValuation:
        """OVME "solve for volatility": back out IV from a quoted premium, then report Greeks at that vol."""
        iv = self.implied_volatility(contract, market, option_price)
        return self.value(contract, market.model_copy(update={"volatility": iv}))


def _central_bump(option: ql.VanillaOption, quote: ql.SimpleQuote, base: float, h: float) -> float:
    quote.setValue(base + h)
    up = option.NPV()
    quote.setValue(base - h)
    down = option.NPV()
    quote.setValue(base)
    return float((up - down) / (2.0 * h))
