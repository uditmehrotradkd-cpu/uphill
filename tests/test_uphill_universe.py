from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import UTC, date, datetime
from pathlib import Path

from quantterm.db.repositories import SecurityRecord
from quantterm.models import Exchange, SectorClass
from quantterm.workspace import Datum, GPView, WorkspaceError
from uphill.market.screener import Screener, stats
from uphill.market.universe import (
    UniverseStock,
    load_cached,
    news_aliases,
    parse_constituents,
    save_cache,
    sector_class_for,
)

CSV = (
    "\ufeffCompany Name,Industry,Symbol,Series,ISIN Code\n"
    "HDFC Bank Ltd.,Financial Services,HDFCBANK,EQ,INE040A01034\n"
    "Bajaj Finance Ltd.,Financial Services,BAJFINANCE,EQ,INE296A01032\n"
    "SBI Life Insurance Company Ltd.,Financial Services,SBILIFE,EQ,INE123W01016\n"
    "Indian Hotels Co. Ltd.,Consumer Services,INDHOTEL,EQ,INE053A01029\n"
    "Bank of India,Financial Services,BANKINDIA,EQ,INE084A01016\n"
    "Some Bond,Financial Services,XBOND,N1,INE000000000\n"
)


def _stock(name: str, industry: str = "Financial Services", symbol: str = "X") -> UniverseStock:
    return UniverseStock(symbol=symbol, name=name, industry=industry, isin="", cap="large")


def test_parse_constituents_keeps_equity_series_only() -> None:
    stocks = parse_constituents(CSV, "mid")
    assert [s.symbol for s in stocks] == ["HDFCBANK", "BAJFINANCE", "SBILIFE", "INDHOTEL", "BANKINDIA"]
    assert stocks[0].isin == "INE040A01034" and all(s.cap == "mid" for s in stocks)
    assert stocks[0].listing == "NSE:HDFCBANK"


def test_sector_class_and_news_aliases() -> None:
    by_symbol = {s.symbol: s for s in parse_constituents(CSV, "large")}
    assert sector_class_for(by_symbol["HDFCBANK"]) is SectorClass.BANK
    assert sector_class_for(by_symbol["BAJFINANCE"]) is SectorClass.NBFC
    assert sector_class_for(by_symbol["SBILIFE"]) is SectorClass.INSURANCE
    assert sector_class_for(by_symbol["INDHOTEL"]) is SectorClass.GENERAL
    assert news_aliases(by_symbol["INDHOTEL"]) == ["Indian Hotels"]
    assert news_aliases(by_symbol["BANKINDIA"]) == ["Bank of India"]  # "India" is part of the name, kept
    assert news_aliases(_stock("ABB Ltd.", symbol="ABB")) == []


def test_universe_cache_round_trip(tmp_path: Path) -> None:
    stocks = parse_constituents(CSV, "small")
    save_cache(stocks, tmp_path / "u.json")
    loaded, fetched = load_cached(tmp_path / "u.json")
    assert loaded == stocks and fetched is not None
    assert load_cached(tmp_path / "missing.json") == ([], None)


def test_stats_horizons_and_52w_range() -> None:
    closes = [100.0 + i for i in range(250)]  # 100 .. 349
    s = stats(closes, [1000.0] * 250)
    assert s["last_close"] == 349 and s["prev_close"] == 348
    assert round(s["chg_1d"] or 0, 4) == round((349 / 348 - 1) * 100, 4)
    assert round(s["chg_1m"] or 0, 4) == round((349 / 328 - 1) * 100, 4)
    assert s["high_52w"] == 349 and s["low_52w"] == 100 and s["from_high"] == 0
    assert s["chg_1y"] is not None and s["avg_volume_20d"] == 1000
    short = stats([10.0, 11.0], [5.0, 5.0])
    assert short["chg_1m"] is None and short["chg_1y"] is None
    assert stats([], []) == {}


class _Workspace:
    async def resolve(self, ticker: str, exchanges: Sequence[Exchange]) -> SecurityRecord:
        if ticker == "NODATA":
            raise WorkspaceError("no price history")
        return SecurityRecord(symbol=ticker, exchange=Exchange.NSE, isin=None, currency="INR",
                              sector_class=SectorClass.GENERAL)

    async def gp(self, sec: SecurityRecord, as_of: datetime, *, lookback_days: int, adjust_splits: bool,
                 adjust_dividends: bool) -> GPView:
        rsi = Datum(label="RSI(14)", value=42.0, source="P", confidence_score=1, timestamp=None, provenance="t")
        return GPView(security=sec, as_of=as_of, adjust_splits=True, adjust_dividends=False,
                      dates=[date(2026, 10, 22), date(2026, 10, 23)], close=[100.0, 110.0], volume=[1.0, 3.0],
                      rsi=[None, 42.0], macd=[None, None], signal=[None, None], summary=[rsi])


def test_screener_refresh_isolates_missing_history() -> None:
    universe = [_stock("Good Ltd.", "IT", "GOOD"), _stock("No Data Ltd.", "IT", "NODATA")]
    screener = Screener(_Workspace(), universe, clock=lambda: datetime(2026, 10, 26, tzinfo=UTC))  # type: ignore[arg-type]
    assert screener.rows["NSE:GOOD"]["error"] == "loading"
    assert asyncio.run(screener.refresh()) == 1
    good, bad = screener.rows["NSE:GOOD"], screener.rows["NSE:NODATA"]
    assert good["last_close"] == 110 and round(good["chg_1d"], 6) == 10 and good["rsi"] == 42
    assert good["last_date"] == "2026-10-23" and good["cap"] == "large" and good["error"] is None
    assert bad["error"].startswith("WorkspaceError")
