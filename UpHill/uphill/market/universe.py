"""NSE stock universe by market-cap bucket, from the official Nifty index constituent lists.

SEBI's categorisation (rank by full market cap): large cap = top 100, mid cap = 101-250, small cap = 251+.
Nifty 100 / Midcap 150 / Smallcap 250 track exactly those bands and are rebalanced semi-annually.

    python -m uphill.market.universe              # refresh lists, register stocks, backfill ~400 days of EOD bars
    python -m uphill.market.universe --days 800   # longer history
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import io
import json
import logging
import re
import sys
from collections.abc import Iterable, Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict

from quantterm.connectors.http import HttpFetcher
from quantterm.connectors.prices import NseBhavcopySource
from quantterm.db.pool import create_pool
from quantterm.db.repositories import DAILY, PriceRepository, SecurityRepository
from quantterm.models import Exchange, OhlcvBar, SectorClass
from quantterm.settings import Settings

log = logging.getLogger(__name__)

Cap = Literal["large", "mid", "small"]
INDEX_LISTS: dict[Cap, str] = {
    "large": "https://www.niftyindices.com/IndexConstituent/ind_nifty100list.csv",
    "mid": "https://www.niftyindices.com/IndexConstituent/ind_niftymidcap150list.csv",
    "small": "https://www.niftyindices.com/IndexConstituent/ind_niftysmallcap250list.csv",
}
CAP_INDEX_NAME: dict[Cap, str] = {"large": "Nifty 100", "mid": "Nifty Midcap 150", "small": "Nifty Smallcap 250"}
CACHE_PATH = Path(__file__).resolve().parents[2] / "data" / "universe.json"
CACHE_MAX_AGE = timedelta(days=7)
BROWSER_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) UpHill/0.1"
_SUFFIX = re.compile(r"\s+(?:Ltd\.?|Limited|Co\.?|Corporation|Corp\.?|\(India\))$", re.IGNORECASE)


class UniverseStock(BaseModel):
    model_config = ConfigDict(frozen=True)

    symbol: str
    name: str
    industry: str
    isin: str
    cap: Cap

    @property
    def listing(self) -> str:
        return f"NSE:{self.symbol}"


def parse_constituents(csv_text: str, cap: Cap) -> list[UniverseStock]:
    stocks = []
    for row in csv.DictReader(io.StringIO(csv_text.lstrip("\ufeff"))):
        symbol = (row.get("Symbol") or "").strip()
        if not symbol or (row.get("Series") or "EQ").strip() != "EQ":
            continue
        stocks.append(UniverseStock(symbol=symbol, name=(row.get("Company Name") or symbol).strip(),
                                    industry=(row.get("Industry") or "Unclassified").strip(),
                                    isin=(row.get("ISIN Code") or "").strip(), cap=cap))
    return stocks


def sector_class_for(stock: UniverseStock) -> SectorClass:
    """EV/valuation rules differ for lenders and insurers; the Nifty industry alone does not separate them."""
    if stock.industry != "Financial Services":
        return SectorClass.GENERAL
    name = stock.name.lower()
    if "bank" in name:
        return SectorClass.BANK
    if "insurance" in name or "assurance" in name:
        return SectorClass.INSURANCE
    if any(k in name for k in ("finance", "financial", "fincorp", "credit", "capital", "housing", "leasing")):
        return SectorClass.NBFC
    return SectorClass.GENERAL  # exchanges, AMCs, brokers, depositories


def news_aliases(stock: UniverseStock) -> list[str]:
    """Company name without legal suffixes, e.g. 'Reliance Industries Ltd.' -> 'Reliance Industries'."""
    name = stock.name
    for _ in range(3):
        name = _SUFFIX.sub("", name).strip(" .,")
    return [name] if len(name) >= 4 and name.upper() != stock.symbol else []


async def fetch_universe(client: httpx.AsyncClient) -> list[UniverseStock]:
    stocks: dict[str, UniverseStock] = {}
    for cap, url in INDEX_LISTS.items():
        response = await client.get(url, headers={"User-Agent": BROWSER_UA})
        response.raise_for_status()
        for s in parse_constituents(response.text, cap):
            stocks.setdefault(s.symbol, s)  # a stock belongs to exactly one band; first (largest) wins
    return list(stocks.values())


def load_cached(path: Path = CACHE_PATH) -> tuple[list[UniverseStock], datetime | None]:
    if not path.exists():
        return [], None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ([UniverseStock.model_validate(s) for s in payload["stocks"]],
            datetime.fromisoformat(payload["fetched"]))


def save_cache(stocks: Sequence[UniverseStock], path: Path = CACHE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"fetched": datetime.now(UTC).isoformat(),
                                "stocks": [s.model_dump() for s in stocks]}, indent=1), encoding="utf-8")


async def load_universe(client: httpx.AsyncClient, path: Path = CACHE_PATH, refresh: bool = False) -> list[UniverseStock]:
    cached, fetched = load_cached(path)
    if cached and fetched and not refresh and datetime.now(UTC) - fetched < CACHE_MAX_AGE:
        return cached
    try:
        stocks = await fetch_universe(client)
    except httpx.HTTPError as exc:
        if cached:
            log.warning("index lists unavailable (%s); using cached universe from %s", type(exc).__name__, fetched)
            return cached
        raise
    save_cache(stocks, path)
    return stocks


async def register(securities: SecurityRepository, stocks: Iterable[UniverseStock]) -> int:
    n = 0
    for s in stocks:
        await securities.upsert(s.symbol, Exchange.NSE, sector_class=sector_class_for(s), isin=s.isin or None,
                                industry=s.industry)
        n += 1
    return n


async def backfill(prices: PriceRepository, bhavcopy: NseBhavcopySource, symbols: set[str], start: date,
                   end: date) -> tuple[int, int]:
    """Raw NSE EOD bars for every symbol; one bhavcopy download per session covers the whole universe."""
    per_symbol: dict[str, list[OhlcvBar]] = {s: [] for s in symbols}
    sessions = 0
    day = start
    while day <= end:
        if day.weekday() < 5:
            bars = await bhavcopy.fetch_session(day)
            if bars:
                sessions += 1
                for s in symbols:
                    if s in bars:
                        per_symbol[s].append(bars[s])
            print(f"\r{day} sessions={sessions}", end="", flush=True)
        day += timedelta(days=1)
    print()
    inserted = 0
    for s, series in per_symbol.items():
        inserted += await prices.append_bars(s, Exchange.NSE, DAILY, series, NseBhavcopySource.source)
    return sessions, inserted


async def _main(args: argparse.Namespace) -> None:
    settings = Settings()
    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        stocks = await load_universe(client, refresh=True)
        counts = {c: sum(s.cap == c for s in stocks) for c in INDEX_LISTS}
        print(f"universe: {len(stocks)} stocks {counts}")
        pool = await create_pool(settings)
        try:
            securities = SecurityRepository(pool)
            print(f"registered {await register(securities, stocks)} listings")
            if args.days > 0:
                end = date.fromisoformat(args.end) if args.end else datetime.now(UTC).date() - timedelta(days=1)
                sessions, inserted = await backfill(
                    PriceRepository(pool, securities), NseBhavcopySource(HttpFetcher(client, rate_per_second=2)),
                    {s.symbol for s in stocks}, end - timedelta(days=args.days), end)
                print(f"backfill: {sessions} sessions, {inserted} new bars")
        finally:
            await pool.close()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="uphill-universe")
    parser.add_argument("--days", type=int, default=400, help="calendar days of EOD history to backfill (0 = none)")
    parser.add_argument("--end", help="last session YYYY-MM-DD (default yesterday)")
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    asyncio.run(_main(parser.parse_args(argv)))


if __name__ == "__main__":
    main()
