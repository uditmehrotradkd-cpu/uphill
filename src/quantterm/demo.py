"""Deterministic demo market for a fresh install (no vendor keys, no internet).

RELIANCE, HDFCBANK, TCS (NSE) and AAPL (NASDAQ) plus the configured beta benchmarks:
  * one year of raw daily OHLCV bars on each exchange's holiday calendar,
  * one stock split (AAPL 4-for-1) and two cash dividends per stock, applied to the raw prints,
  * point-in-time FY fundamentals from a primary filer and a vendor copy (one deliberate >0.5% break,
    one restatement), with realistic filing timestamps,
  * risk-free rate prints and a listed option chain priced off a volatility smile.

Everything is synthetic and labelled with ``DEMO_*`` sources: none of it is market data. Random draws are
keyed by (listing, date) from a fixed origin, so re-seeding on a later day appends new sessions with the
same history instead of rewriting it.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import QuantLib as ql
from pydantic import BaseModel, ConfigDict, SecretStr

from quantterm.connectors.base import Metric
from quantterm.corporate_actions import CorporateAction, CorporateActionType, split_factor
from quantterm.db.repositories import (
    DAILY,
    CorporateActionRepository,
    Database,
    FundamentalsRepository,
    MacroRepository,
    OptionQuote,
    OptionQuoteRepository,
    PriceRepository,
    SecurityRepository,
)
from quantterm.models import (
    Exchange,
    FundamentalFact,
    MacroObservation,
    Market,
    OhlcvBar,
    PeriodType,
    SectorClass,
    session_bar_ts,
)
from quantterm.quant.derivatives_ovme import MarketInputs, OptionContract, OptionType, exchange_calendar
from quantterm.quant.technicals import realized_volatility
from quantterm.settings import Settings
from quantterm.workspace import OVMERequest, Workspace, strike_step

DEMO_PRIMARY = "DEMO_SEED"
DEMO_VENDOR = "DEMO_VENDOR"
DEMO_OPTIONS = "DEMO_OPTIONS"
DEMO_MACRO = "DEMO_MACRO"
DEMO_SOURCES = frozenset({DEMO_PRIMARY, DEMO_VENDOR, DEMO_OPTIONS, DEMO_MACRO})
DEFAULT_DATABASE = "quantterm_demo"
PATH_ORIGIN = date(2025, 1, 1)
WINDOW = timedelta(days=365)
CRORE = Decimal(10_000_000)
MILLION = Decimal(1_000_000)
TRADING_DAYS = 252

# Distinct sources in every store that records provenance; anything outside DEMO_SOURCES is real data.
SOURCES_SQL = """
SELECT source FROM raw_market_bars UNION SELECT source FROM corporate_actions
UNION SELECT source_authority FROM financial_metrics_pit UNION SELECT source FROM macro_observations
UNION SELECT source FROM option_quotes
"""


class DemoSeedRefused(RuntimeError):
    """The target database holds non-demo data; synthetic rows must never be mixed into it."""


@dataclass(frozen=True)
class DemoListing:
    symbol: str
    exchange: Exchange
    sector: SectorClass
    industry: str | None
    origin_price: float  # raw close at PATH_ORIGIN
    drift: float  # annualised
    vol: float  # annualised total volatility
    beta: float
    avg_volume: int
    seed: int


STOCKS: tuple[DemoListing, ...] = (
    DemoListing("RELIANCE", Exchange.NSE, SectorClass.GENERAL, "Oil & Gas", 1_240.0, 0.10, 0.22, 1.05, 9_000_000, 101),
    DemoListing("HDFCBANK", Exchange.NSE, SectorClass.BANK, "Banks", 880.0, 0.12, 0.19, 0.90, 14_000_000, 102),
    DemoListing("TCS", Exchange.NSE, SectorClass.GENERAL, "IT Services", 4_100.0, -0.12, 0.24, 0.75, 2_600_000, 103),
    DemoListing("AAPL", Exchange.NASDAQ, SectorClass.GENERAL, "Technology Hardware", 245.0, 0.14, 0.27, 1.15,
                52_000_000, 104),
)
_MARKET = {Market.US: (0.08, 0.15, 7001), Market.IN: (0.10, 0.13, 7002)}  # drift, vol, seed of the market factor
_BENCHMARK_LEVELS = {"NIFTY": 23_600.0, "SPX": 5_900.0, "SPY": 590.0}
_INDICES = frozenset({"NIFTY", "SPX"})


@dataclass(frozen=True)
class DemoSplit:
    symbol: str
    ex_date: date
    new_shares: int
    old_shares: int
    announced_at: datetime


@dataclass(frozen=True)
class DemoDividend:
    symbol: str
    ex_date: date
    amount: Decimal  # per share as traded on the ex-date
    declared_at: datetime


SPLITS = (DemoSplit("AAPL", date(2026, 6, 8), 4, 1, datetime(2026, 4, 30, 20, 30, tzinfo=UTC)),)
DIVIDENDS = (
    DemoDividend("RELIANCE", date(2025, 12, 19), Decimal("2.50"), datetime(2025, 11, 28, 12, 0, tzinfo=UTC)),
    DemoDividend("RELIANCE", date(2026, 8, 14), Decimal("5.50"), datetime(2026, 4, 24, 14, 30, tzinfo=UTC)),
    DemoDividend("HDFCBANK", date(2025, 12, 12), Decimal("2.50"), datetime(2025, 11, 20, 11, 0, tzinfo=UTC)),
    DemoDividend("HDFCBANK", date(2026, 6, 26), Decimal("11.00"), datetime(2026, 4, 18, 10, 0, tzinfo=UTC)),
    DemoDividend("TCS", date(2026, 1, 16), Decimal("11.00"), datetime(2026, 1, 9, 11, 0, tzinfo=UTC)),
    DemoDividend("TCS", date(2026, 6, 5), Decimal("30.00"), datetime(2026, 4, 9, 11, 0, tzinfo=UTC)),
    DemoDividend("AAPL", date(2026, 2, 9), Decimal("0.26"), datetime(2026, 1, 29, 21, 30, tzinfo=UTC)),
    DemoDividend("AAPL", date(2026, 8, 10), Decimal("0.07"), datetime(2026, 7, 30, 20, 30, tzinfo=UTC)),
)


# ------------------------------------------------------------------------------------- fundamentals
def _in(**v: float) -> dict[str, Decimal]:
    """INR crore -> INR; share counts and EPS are already in units."""
    return {k: Decimal(str(x)) if k in (Metric.DILUTED_SHARES, Metric.EPS_DILUTED) else Decimal(str(x)) * CRORE
            for k, x in v.items()}


def _us(**v: float) -> dict[str, Decimal]:
    """USD millions -> USD; share counts and EPS are already in units."""
    return {k: Decimal(str(x)) if k in (Metric.DILUTED_SHARES, Metric.EPS_DILUTED) else Decimal(str(x)) * MILLION
            for k, x in v.items()}


@dataclass(frozen=True)
class DemoFiling:
    symbol: str
    period_end: date
    filed_at: datetime
    values: Mapping[str, Decimal]


# Banks report no EBITDA (EV for BANK/NBFC excludes operating debt and uses P/TBV instead).
FILINGS: tuple[DemoFiling, ...] = (
    DemoFiling("RELIANCE", date(2025, 3, 31), datetime(2025, 4, 25, 14, 30, tzinfo=UTC), _in(
        revenue=964_693, ebitda=183_422, total_debt=369_575, cash_and_equivalents=223_000, minority_interest=120_000,
        diluted_shares=13_532_000_000, net_income=69_648, eps_diluted=51.47, total_equity=840_000)),
    DemoFiling("RELIANCE", date(2026, 3, 31), datetime(2026, 4, 24, 14, 30, tzinfo=UTC), _in(
        revenue=1_031_000, ebitda=198_500, total_debt=372_000, cash_and_equivalents=238_000, minority_interest=126_000,
        diluted_shares=13_532_000_000, net_income=76_000, eps_diluted=56.16, total_equity=905_000)),
    DemoFiling("HDFCBANK", date(2025, 3, 31), datetime(2025, 4, 19, 10, 0, tzinfo=UTC), _in(
        revenue=336_367, total_debt=634_606, cash_and_equivalents=223_000, minority_interest=18_500,
        diluted_shares=15_300_000_000, net_income=70_792, eps_diluted=46.30, total_equity=500_000,
        intangible_assets=1_200)),
    DemoFiling("HDFCBANK", date(2026, 3, 31), datetime(2026, 4, 18, 10, 0, tzinfo=UTC), _in(
        revenue=362_000, total_debt=560_000, cash_and_equivalents=240_000, minority_interest=24_000,
        diluted_shares=15_380_000_000, net_income=77_600, eps_diluted=50.50, total_equity=565_000,
        intangible_assets=1_350)),
    DemoFiling("TCS", date(2025, 3, 31), datetime(2025, 4, 10, 11, 0, tzinfo=UTC), _in(
        revenue=255_324, ebitda=68_400, total_debt=8_200, cash_and_equivalents=54_000, minority_interest=1_000,
        diluted_shares=3_618_000_000, net_income=48_553, eps_diluted=134.19, total_equity=94_756)),
    DemoFiling("TCS", date(2026, 3, 31), datetime(2026, 4, 9, 11, 0, tzinfo=UTC), _in(
        revenue=266_000, ebitda=71_900, total_debt=8_700, cash_and_equivalents=51_500, minority_interest=1_050,
        diluted_shares=3_618_000_000, net_income=50_900, eps_diluted=140.70, total_equity=97_000)),
    DemoFiling("AAPL", date(2024, 9, 28), datetime(2024, 11, 1, 10, 1, tzinfo=UTC), _us(
        revenue=391_035, ebitda=134_661, total_debt=106_629, cash_and_equivalents=65_171, minority_interest=0,
        diluted_shares=15_408_095_000, net_income=93_736, eps_diluted=6.08, total_equity=56_950)),
    DemoFiling("AAPL", date(2025, 9, 27), datetime(2025, 10, 31, 10, 1, tzinfo=UTC), _us(
        revenue=416_161, ebitda=144_748, total_debt=98_657, cash_and_equivalents=54_697, minority_interest=0,
        diluted_shares=15_004_697_000, net_income=112_010, eps_diluted=7.46, total_equity=73_733)),
)
# RELIANCE restates FY2025 revenue in its FY2026 results: a second vintage, never an overwrite.
RESTATEMENTS = (DemoFiling("RELIANCE", date(2025, 3, 31), datetime(2026, 4, 24, 14, 30, tzinfo=UTC),
                           _in(revenue=968_550)),)
VENDOR_LAG = timedelta(days=2)
VENDOR_BREAKS = {("TCS", date(2026, 3, 31), Metric.EBITDA): Decimal("1.012")}  # 1.2% > 0.5% tolerance
VENDOR_OMITS = {("RELIANCE", Metric.MINORITY_INTEREST)}
RISK_FREE_LEVELS = {Market.US: 3.85, Market.IN: 5.45}  # annual %, 3M bill


# ------------------------------------------------------------------------------------- generation
def _rng(*key: int) -> np.random.Generator:
    return np.random.default_rng(list(key))


def _student_t(rng: np.random.Generator) -> float:
    return float(rng.standard_t(5) * math.sqrt(3 / 5))  # unit variance, fat tails


def sessions(exchange: Exchange, start: date, end: date) -> list[date]:
    cal = exchange_calendar(exchange)
    out, d = [], start
    while d <= end:
        if cal.isBusinessDay(ql.Date(d.day, d.month, d.year)):
            out.append(d)
        d += timedelta(days=1)
    return out


def last_completed_session(exchange: Exchange, as_of: datetime) -> date:
    local = as_of.astimezone(exchange.timezone)
    day = local.date() if local.time() >= exchange.session_close else local.date() - timedelta(days=1)
    return sessions(exchange, day - timedelta(days=10), day)[-1]


def _tick(exchange: Exchange) -> float:
    return 0.05 if exchange.market is Market.IN else 0.01


def _floor(x: float, tick: float) -> float:
    return round(math.floor(x / tick + 1e-9) * tick, 4)


def _ceil(x: float, tick: float) -> float:
    return round(math.ceil(x / tick - 1e-9) * tick, 4)


def _round(x: float, tick: float) -> float:
    return round(round(x / tick) * tick, 4)


def generate_bars(listing: DemoListing, end: date) -> list[OhlcvBar]:
    """Raw daily bars for ``(end - 1 year, end]``: a one-factor fat-tailed walk with splits and dividends
    applied to the prints exactly as a venue would publish them."""
    m_drift, m_vol, m_seed = _MARKET[listing.exchange.market]
    dt = 1 / TRADING_DAYS
    sigma_d = listing.vol * math.sqrt(dt)
    idio = math.sqrt(max(listing.vol**2 - (listing.beta * m_vol) ** 2, (0.25 * listing.vol) ** 2))
    tick = _tick(listing.exchange)
    splits = {s.ex_date: s for s in SPLITS if s.symbol == listing.symbol}
    dividends = {d.ex_date: d for d in DIVIDENDS if d.symbol == listing.symbol}
    cutoff = end - WINDOW
    close, volume_scale = listing.origin_price, 1.0
    bars: list[OhlcvBar] = []
    for day in sessions(listing.exchange, PATH_ORIGIN, end):
        z_m = _student_t(_rng(m_seed, day.toordinal()))
        rng = _rng(listing.seed, day.toordinal())
        r = ((listing.drift - 0.5 * listing.vol**2) * dt + listing.beta * m_vol * math.sqrt(dt) * z_m
             + idio * math.sqrt(dt) * _student_t(rng))
        reference = close
        if day in splits:
            s = splits[day]
            reference /= s.new_shares / s.old_shares
            volume_scale *= s.new_shares / s.old_shares
        if day in dividends:
            reference -= float(dividends[day].amount)
        gap = float(rng.normal(0.0, 0.3 * sigma_d))
        open_ = _round(reference * math.exp(gap), tick)
        close = _round(reference * math.exp(r), tick)
        high = _ceil(max(open_, close) * math.exp(abs(float(rng.normal(0.0, 0.6 * sigma_d)))), tick)
        low = _floor(min(open_, close) * math.exp(-abs(float(rng.normal(0.0, 0.6 * sigma_d)))), tick)
        shock = 1 + 1.5 * abs(r) / sigma_d
        volume = 0 if listing.avg_volume == 0 else int(
            listing.avg_volume * volume_scale * shock * math.exp(float(rng.normal(-0.3, 0.35))))
        vwap = min(max((high + low + 2 * close) / 4, low), high)
        if day > cutoff:
            bars.append(OhlcvBar(ts=session_bar_ts(listing.exchange, day), open=open_, high=high, low=low,
                                 close=close, volume=volume, vwap=None if volume == 0 else round(vwap, 4)))
    return bars


def benchmark_listings(settings: Settings) -> list[DemoListing]:
    out = []
    for i, (_, ref) in enumerate(sorted(settings.beta_benchmarks.items())):
        exchange, _, ticker = ref.partition(":")
        out.append(DemoListing(ticker, Exchange(exchange), SectorClass.GENERAL, None,
                               _BENCHMARK_LEVELS.get(ticker, 1_000.0), 0.09, 0.14, 1.0,
                               0 if ticker in _INDICES else 60_000_000, 900 + i))
    return out


def corporate_actions() -> list[CorporateAction]:
    actions = [CorporateAction(symbol=s.symbol, exchange=_listing(s.symbol).exchange, action_type=CorporateActionType.SPLIT,
                               ex_date=s.ex_date, announced_at=s.announced_at,
                               adjustment_factor=split_factor(Decimal(s.new_shares), Decimal(s.old_shares)),
                               source=DEMO_PRIMARY) for s in SPLITS]
    actions += [CorporateAction(symbol=d.symbol, exchange=_listing(d.symbol).exchange,
                                action_type=CorporateActionType.CASH_DIVIDEND, ex_date=d.ex_date,
                                announced_at=d.declared_at, raw_dividend_amount=d.amount, source=DEMO_PRIMARY)
                for d in DIVIDENDS]
    return actions


def _listing(symbol: str) -> DemoListing:
    return next(s for s in STOCKS if s.symbol == symbol)


def fundamental_facts() -> dict[Exchange, list[FundamentalFact]]:
    out: dict[Exchange, list[FundamentalFact]] = {}

    def add(f: DemoFiling, source: str, confidence: float, filed: datetime, scale: Mapping[str, Decimal]) -> None:
        listing = _listing(f.symbol)
        for metric, value in f.values.items():
            if source == DEMO_VENDOR and (f.symbol, metric) in VENDOR_OMITS:
                continue
            out.setdefault(listing.exchange, []).append(FundamentalFact(
                symbol=f.symbol, metric=metric, period_type=PeriodType.FISCAL_YEAR, period_end_date=f.period_end,
                filing_timestamp=filed, value=(value * scale.get(metric, Decimal(1))).quantize(Decimal("0.000001")),
                currency="XXX" if metric == Metric.DILUTED_SHARES else listing.exchange.currency, source=source,
                source_document_id=f"{source}:{f.symbol}:{f.period_end}:{filed:%Y%m%d}", confidence_score=confidence,
            ))

    for f in (*FILINGS, *RESTATEMENTS):
        add(f, DEMO_PRIMARY, 1.0, f.filed_at, {})
    for f in FILINGS:
        breaks: dict[str, Decimal] = {m: k for (s, p, m), k in VENDOR_BREAKS.items()
                                      if s == f.symbol and p == f.period_end}
        add(f, DEMO_VENDOR, 0.8, f.filed_at + VENDOR_LAG, breaks)
    return out


def macro_observations(settings: Settings, end: date) -> list[MacroObservation]:
    """Weekly 3M bill prints for 13 weeks to ``end``, each released the next day."""
    out = []
    for market, level in RISK_FREE_LEVELS.items():
        series = settings.risk_free_series.get(market.value)
        if series is None:
            continue
        for week in range(13, 0, -1):
            observed = end - timedelta(days=7 * week - 6)
            wiggle = 0.03 * math.sin(observed.toordinal() / 9)
            out.append(MacroObservation(
                series_id=series, observation_date=observed,
                release_timestamp=datetime.combine(observed + timedelta(days=1), time(12, 0), tzinfo=UTC),
                value=Decimal(f"{level + wiggle:.2f}"), source=DEMO_MACRO))
    return out


def smile(atm: float, strike: float, forward: float, t: float) -> float:
    """Equity skew: richer downside vol, mild wings; standardised by ATM vol and time."""
    x = math.log(strike / forward) / (atm * math.sqrt(t))
    return min(max(atm * (1 - 0.10 * x + 0.04 * x * x), 0.5 * atm), 3.0 * atm)


async def option_chain(ws: Workspace, listing: DemoListing, as_of: datetime, *,
                       strikes_each_side: int = 8) -> list[OptionQuote]:
    """Two monthly expiries of quotes at the last close, priced by the OVME engine off ``smile``."""
    sec = await ws.securities.get(listing.symbol, listing.exchange)
    engine = ws.engine(sec.exchange)
    view = await ws.ovme(sec, as_of, OVMERequest(volatility=0.3), strikes_each_side=0)
    spot_d, _, rate_d, div_d = view.inputs
    assert spot_d.value is not None and rate_d.value is not None and div_d.value is not None
    assert spot_d.timestamp is not None
    spot, rate, div = spot_d.value, rate_d.value, div_d.value
    valuation_date = spot_d.timestamp.astimezone(sec.exchange.timezone).date()
    bars = await ws.daily_bars(sec, valuation_date - timedelta(days=100), as_of, adjust_dividends=False)
    atm = max(0.12, 1.08 * realized_volatility(bars["close"].tail(60)))  # implied trades over realised
    first = engine.next_monthly_expiry(valuation_date, min_days=2)
    expiries = [first, engine.next_monthly_expiry(first, min_days=1)]
    tick, step = _tick(sec.exchange), strike_step(spot)
    centre = round(spot / step) * step
    market = MarketInputs(valuation_date=valuation_date, spot=spot, volatility=atm, risk_free_rate=rate,
                          dividend_yield=div)
    quotes: list[OptionQuote] = []
    for expiry in expiries:
        t = (expiry - valuation_date).days / 365
        forward = spot * math.exp((rate - div) * t)
        for k in range(-strikes_each_side, strikes_each_side + 1):
            strike = centre + k * step
            if strike <= 0:
                continue
            iv = smile(atm, strike, forward, t)
            for kind in OptionType:
                contract = OptionContract(underlying=sec.symbol, exchange=sec.exchange, option_type=kind,
                                          strike=strike, expiry=expiry)
                fair = engine.value(contract, market.model_copy(update={"volatility": iv})).npv
                half = (tick + 0.015 * fair) / 2
                bid, ask = max(_floor(fair - half, tick), 0.0), max(_ceil(fair + half, tick), tick)
                rng = _rng(listing.seed, expiry.toordinal(), int(strike * 100), 0 if kind is OptionType.CALL else 1)
                moneyness = math.exp(-abs(math.log(strike / spot)) * 8)
                quotes.append(OptionQuote(
                    symbol=sec.symbol, exchange=sec.exchange, expiry=expiry, strike=Decimal(str(strike)),
                    option_type=kind, timestamp=spot_d.timestamp, bid=bid, ask=ask, last=_round(fair, tick) or None,
                    volume=int(5_000 * moneyness * float(rng.uniform(0.5, 1.5))),
                    open_interest=int(40_000 * moneyness * float(rng.uniform(0.5, 1.5))), source=DEMO_OPTIONS,
                ))
    return quotes


# ------------------------------------------------------------------------------------- orchestration
class SeedReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    as_of: datetime
    securities: int
    bars: int
    splits: int
    dividends: int
    fundamentals: int
    macro: int
    option_quotes: int


def demo_settings(base: Settings, database_url: str) -> Settings:
    """Settings that read the demo database and treat DEMO_SEED / DEMO_VENDOR as the filer and the vendor."""
    return base.model_copy(update={
        "database_url": SecretStr(database_url),
        "fundamentals_primary_sources": {e.value: DEMO_PRIMARY for e in Exchange},
        "fundamentals_secondary_source": DEMO_VENDOR,
        "watchlist": [f"{s.exchange.value}:{s.symbol}" for s in STOCKS],
    })


async def assert_demo_only(db: Database) -> None:
    foreign = sorted({r["source"] for r in await db.fetch(SOURCES_SQL)} - DEMO_SOURCES)
    if foreign:
        raise DemoSeedRefused(f"database already holds non-demo data (sources: {', '.join(foreign)}); "
                              "refusing to mix synthetic rows into it - seed a separate database")


async def seed_demo(db: Database, settings: Settings, as_of: datetime | None = None) -> SeedReport:
    """Idempotent: known bars, actions, filing vintages, prints and quotes are skipped on re-runs."""
    as_of = as_of or datetime.now(UTC)
    securities = SecurityRepository(db)
    listings = [*STOCKS, *benchmark_listings(settings)]
    for s in listings:
        await securities.upsert(s.symbol, s.exchange, sector_class=s.sector, industry=s.industry)

    prices = PriceRepository(db, securities)
    bars = 0
    for s in listings:
        bars += await prices.append_bars(s.symbol, s.exchange, DAILY,
                                         generate_bars(s, last_completed_session(s.exchange, as_of)), DEMO_PRIMARY)
    actions = [a for a in corporate_actions() if a.announced_at <= as_of]
    await CorporateActionRepository(db, securities).append(actions)

    fundamentals = FundamentalsRepository(db, securities)
    facts = 0
    for exchange, rows in fundamental_facts().items():
        facts += await fundamentals.append(exchange, [f for f in rows if f.filing_timestamp <= as_of])

    end = min(last_completed_session(e, as_of) for e in (Exchange.NSE, Exchange.NYSE))
    macro = await MacroRepository(db).append([m for m in macro_observations(settings, end)
                                               if m.release_timestamp <= as_of])

    ws = Workspace(db, demo_settings(settings, settings.database_url.get_secret_value()), clock=lambda: as_of)
    quote_repo = OptionQuoteRepository(db, securities)
    quotes = 0
    for s in STOCKS:
        quotes += await quote_repo.append(await option_chain(ws, s, as_of))
    return SeedReport(as_of=as_of, securities=len(listings), bars=bars,
                      splits=sum(a.action_type is CorporateActionType.SPLIT for a in actions),
                      dividends=sum(a.action_type is CorporateActionType.CASH_DIVIDEND for a in actions),
                      fundamentals=facts, macro=macro, option_quotes=quotes)


# ------------------------------------------------------------------------------------- database bootstrap
_DB_NAME = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")


class DatabaseUnavailable(RuntimeError):
    pass


def with_database(dsn: str, name: str) -> str:
    if not _DB_NAME.fullmatch(name):
        raise ValueError(f"invalid database name {name!r} (lowercase letters, digits, underscore)")
    parts = urlsplit(dsn)
    return urlunsplit(parts._replace(path=f"/{name}"))


async def ensure_database(server_dsn: str, name: str, *, timeout: float = 5.0) -> str:
    """Create database ``name`` on the server behind ``server_dsn`` if missing; returns its DSN."""
    import asyncpg

    target = with_database(server_dsn, name)
    try:
        conn = await asyncpg.connect(server_dsn, timeout=timeout)
    except (OSError, TimeoutError, asyncpg.PostgresError) as exc:
        host = urlsplit(server_dsn).hostname
        raise DatabaseUnavailable(f"PostgreSQL not reachable at {host}: {type(exc).__name__}: {exc}") from exc
    try:
        exists = await conn.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", name)
        if not exists:
            await conn.execute(f'CREATE DATABASE "{name}"')  # name validated by _DB_NAME
    finally:
        await conn.close()
    return target
