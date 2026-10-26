"""A small but realistic market for terminal/agent tests: NSE oil & gas pair, an NSE bank, NIFTY, AAPL, SPX.

Prices are seeded geometric random walks (stocks driven by their index with a known beta) on weekday
sessions; fundamentals have a primary XBRL filing, a vendor copy (with one >0.5% break) and a later
restatement, so point-in-time behaviour is observable.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

import numpy as np
from fakes import FakeDatabase

from quantterm.connectors.base import Metric
from quantterm.corporate_actions import CorporateAction, CorporateActionType, bonus_factor
from quantterm.db.repositories import (
    DAILY,
    CorporateActionRepository,
    Database,
    FundamentalsRepository,
    MacroRepository,
    PriceRepository,
    SecurityRepository,
)
from quantterm.models import Exchange, FundamentalFact, MacroObservation, OhlcvBar, PeriodType, SectorClass, session_bar_ts
from quantterm.settings import Settings
from quantterm.workspace import Workspace

IST = ZoneInfo("Asia/Kolkata")
AS_OF = datetime(2024, 12, 31, 18, 0, tzinfo=IST)
FIRST_SESSION = date(2023, 6, 1)
CRORE = Decimal(10_000_000)
BONUS_EX = date(2024, 10, 28)
FY24_FILED = datetime(2024, 5, 15, 12, 30, tzinfo=UTC)
RESTATED = datetime(2024, 9, 2, 11, 0, tzinfo=UTC)


def sessions(start: date = FIRST_SESSION, end: date = AS_OF.date()) -> list[date]:
    out, d = [], start
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def walk(index_returns: np.ndarray, beta: float, sigma: float, seed: int, start: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    r = beta * index_returns + rng.normal(0.0, sigma, index_returns.size)
    return start * np.exp(np.cumsum(r))


def settings() -> Settings:
    return Settings(api_token="a" * 32, ingest_token="i" * 32, _env_file=None)


def _bars(exchange: Exchange, days: list[date], closes: np.ndarray, volume: int) -> list[OhlcvBar]:
    return [OhlcvBar(ts=session_bar_ts(exchange, d), open=float(c), high=float(c) * 1.01, low=float(c) * 0.99,
                     close=float(c), volume=volume + 1000 * (i % 7), vwap=float(c)) for i, (d, c) in enumerate(zip(days, closes))]


def _facts(symbol: str, period_end: date, filed: datetime, values: dict[str, Decimal], source: str,
           currency: str = "INR", confidence: float = 1.0) -> list[FundamentalFact]:
    return [FundamentalFact(symbol=symbol, metric=m, period_type=PeriodType.FISCAL_YEAR, period_end_date=period_end,
                            filing_timestamp=filed, value=v, currency="XXX" if m == Metric.DILUTED_SHARES else currency,
                            source=source, source_document_id=f"{source}:{symbol}:{period_end}",
                            confidence_score=confidence) for m, v in values.items()]


def _cr(**values: float) -> dict[str, Decimal]:
    return {k: (Decimal(str(v)) * CRORE if k not in (Metric.DILUTED_SHARES, Metric.EPS_DILUTED) else Decimal(str(v)))
            for k, v in values.items()}


RELIANCE_FY24 = _cr(revenue=901064, ebitda=178677, operating_income=127000, net_income=69621, eps_diluted=102.9,
                    total_assets=1755986, total_equity=793481, total_debt=346142, cash_and_equivalents=217000,
                    diluted_shares=6766000000, minority_interest=113000, lease_liabilities=22000,
                    goodwill=13000, intangible_assets=150000)
RELIANCE_FY23 = _cr(revenue=877835, ebitda=153920, operating_income=110000, net_income=66702, eps_diluted=98.6,
                    total_assets=1607431, total_equity=715872, total_debt=314000, cash_and_equivalents=190000,
                    diluted_shares=6766000000, minority_interest=100000, lease_liabilities=20000,
                    goodwill=12000, intangible_assets=140000)
ONGC_FY24 = _cr(revenue=643037, ebitda=105000, operating_income=70000, net_income=40526, eps_diluted=32.2,
                total_assets=700000, total_equity=310000, total_debt=130000, cash_and_equivalents=40000,
                diluted_shares=12580000000, minority_interest=30000)
HDFC_FY24 = _cr(revenue=283649, operating_income=90000, net_income=60812, eps_diluted=80.0,
                total_assets=3617623, total_equity=455636, total_debt=730615, cash_and_equivalents=215000,
                diluted_shares=7600000000, minority_interest=20000, goodwill=7000, intangible_assets=1000)
AAPL_FY24 = {k: Decimal(str(v)) for k, v in dict(revenue=391035e6, ebitda=134661e6, operating_income=123216e6,
                                                 net_income=93736e6, eps_diluted=6.08, total_assets=364980e6,
                                                 total_equity=56950e6, total_debt=106629e6,
                                                 cash_and_equivalents=65171e6, diluted_shares=15408095000).items()}


async def seed(db: Database) -> None:
    securities = SecurityRepository(db)
    for symbol, exchange, sector, industry in (
        ("RELIANCE", Exchange.NSE, SectorClass.GENERAL, "Oil & Gas"),
        ("ONGC", Exchange.NSE, SectorClass.GENERAL, "Oil & Gas"),
        ("HDFCBANK", Exchange.NSE, SectorClass.BANK, "Banks"),
        ("NIFTY", Exchange.NSE, SectorClass.GENERAL, None),
        ("AAPL", Exchange.NASDAQ, SectorClass.GENERAL, "Technology Hardware"),
        ("SPX", Exchange.NYSE, SectorClass.GENERAL, None),
    ):
        await securities.upsert(symbol, exchange, sector_class=sector, industry=industry)

    prices = PriceRepository(db, securities)
    days = sessions()
    rng = np.random.default_rng(7)
    nifty_r = rng.normal(0.0004, 0.009, len(days))
    spx_r = rng.normal(0.0005, 0.010, len(days))
    nifty = 18500 * np.exp(np.cumsum(nifty_r))
    reliance = walk(nifty_r, 1.10, 0.008, 11, 2600.0)
    # Raw (unadjusted) prints halve on the 1:1 bonus ex-date; the engine restores continuity at query time.
    bonus_mask = np.array([d >= BONUS_EX for d in days])
    reliance_raw = np.where(bonus_mask, reliance / 2, reliance)
    series = {
        ("NIFTY", Exchange.NSE): (nifty, 0),
        ("RELIANCE", Exchange.NSE): (reliance_raw, 5_000_000),
        ("ONGC", Exchange.NSE): (walk(nifty_r, 0.85, 0.011, 12, 170.0), 12_000_000),
        ("HDFCBANK", Exchange.NSE): (walk(nifty_r, 0.95, 0.007, 13, 1600.0), 9_000_000),
        ("SPX", Exchange.NYSE): (4200 * np.exp(np.cumsum(spx_r)), 0),
        ("AAPL", Exchange.NASDAQ): (walk(spx_r, 1.20, 0.010, 14, 180.0), 50_000_000),
    }
    for (symbol, exchange), (closes, volume) in series.items():
        await prices.append_bars(symbol, exchange, DAILY, _bars(exchange, days, closes, volume), "EXCHANGE_EOD")

    await CorporateActionRepository(db, securities).append([
        CorporateAction(symbol="RELIANCE", exchange=Exchange.NSE, action_type=CorporateActionType.BONUS,
                        ex_date=BONUS_EX, announced_at=datetime(2024, 9, 5, 12, 0, tzinfo=UTC),
                        adjustment_factor=bonus_factor(Decimal(1), Decimal(1)), source="NSE_CA"),
        CorporateAction(symbol="RELIANCE", exchange=Exchange.NSE, action_type=CorporateActionType.CASH_DIVIDEND,
                        ex_date=date(2024, 8, 19), announced_at=datetime(2024, 4, 22, 14, 0, tzinfo=UTC),
                        raw_dividend_amount=Decimal("10"), source="NSE_CA"),
    ])

    facts = FundamentalsRepository(db, securities)
    fy23_filed = datetime(2023, 5, 20, 12, 0, tzinfo=UTC)
    nse: list[FundamentalFact] = []
    nse += _facts("RELIANCE", date(2023, 3, 31), fy23_filed, RELIANCE_FY23, "NSE_XBRL")
    nse += _facts("RELIANCE", date(2024, 3, 31), FY24_FILED, RELIANCE_FY24, "NSE_XBRL")
    vendor = dict(RELIANCE_FY24)
    vendor[Metric.REVENUE] = vendor[Metric.REVENUE] * Decimal("1.012")  # 1.2% vendor break
    nse += _facts("RELIANCE", date(2024, 3, 31), FY24_FILED + timedelta(days=2), vendor, "FMP", confidence=0.8)
    nse += _facts("RELIANCE", date(2024, 3, 31), RESTATED,
                  {Metric.REVENUE: RELIANCE_FY24[Metric.REVENUE] + 1500 * CRORE}, "NSE_XBRL")
    nse += _facts("ONGC", date(2024, 3, 31), datetime(2024, 5, 25, 9, 0, tzinfo=UTC), ONGC_FY24, "NSE_XBRL")
    nse += _facts("HDFCBANK", date(2024, 3, 31), datetime(2024, 4, 20, 9, 0, tzinfo=UTC), HDFC_FY24, "NSE_XBRL")
    await facts.append(Exchange.NSE, nse)
    await facts.append(Exchange.NASDAQ, _facts("AAPL", date(2024, 9, 28), datetime(2024, 11, 1, 10, 1, tzinfo=UTC),
                                               AAPL_FY24, "SEC_EDGAR", currency="USD"))

    await MacroRepository(db).append([
        MacroObservation(series_id="RBI:TBILL91", observation_date=date(2024, 12, 24),
                         release_timestamp=datetime(2024, 12, 24, 13, 0, tzinfo=UTC), value=Decimal("6.6"),
                         source="RBI"),
        MacroObservation(series_id="FRED:DTB3", observation_date=date(2024, 12, 27),
                         release_timestamp=datetime(2024, 12, 30, 21, 0, tzinfo=UTC), value=Decimal("4.31"),
                         source="FRED"),
    ])


async def seeded_workspace() -> tuple[FakeDatabase, Workspace]:
    db = FakeDatabase()
    await seed(db)
    return db, Workspace(db, settings(), clock=lambda: AS_OF)
