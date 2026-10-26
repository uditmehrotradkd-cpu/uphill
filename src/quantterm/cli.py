"""Operational CLI: migrations, security registration, ingestion jobs and reconciliation runs.

    python -m quantterm.cli migrate
    python -m quantterm.cli register NASDAQ AAPL --isin US0378331005
    python -m quantterm.cli ingest-fundamentals SEC_EDGAR NASDAQ AAPL
    python -m quantterm.cli ingest-bhavcopy 2025-01-16 INFY TCS
    python -m quantterm.cli ingest-bhavcopy 2024-10-28 INFY TCS --through 2026-10-23
    python -m quantterm.cli ingest-polygon-bars NASDAQ 2024-01-01 2024-12-31 AAPL MSFT
    python -m quantterm.cli ingest-polygon-actions NASDAQ AAPL MSFT
    python -m quantterm.cli reconcile NASDAQ AAPL SEC_EDGAR FMP Q revenue net_income
    python -m quantterm.cli kite-login            # daily: refreshes QT_KITE_ACCESS_TOKEN in .env
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from quantterm.connectors.base import FundamentalsSource
from quantterm.connectors.edgar import EdgarFundamentalsSource
from quantterm.connectors.http import HttpFetcher
from quantterm.connectors.india_xbrl import IndianXbrlFundamentalsSource, NseFilingIndex
from quantterm.connectors.kite import create_kite_session, kite_login_url, parse_request_token
from quantterm.connectors.prices import NseBhavcopySource, PolygonCorporateActionsSource, PolygonDailyBarsSource
from quantterm.connectors.vendors import FmpFundamentalsSource, PolygonFundamentalsSource
from quantterm.db.pool import apply_schema, create_pool
from quantterm.db.repositories import (
    DAILY,
    CorporateActionRepository,
    FundamentalsRepository,
    PriceRepository,
    ReconciliationRepository,
    SecurityRepository,
)
from quantterm.models import Exchange, PeriodType, SectorClass
from quantterm.pit import PointInTimeFundamentalsRepository
from quantterm.services import FundamentalsIngestionService, ReconciliationService
from quantterm.settings import Settings

FUNDAMENTAL_SOURCES = ("SEC_EDGAR", "NSE_XBRL", "FMP", "POLYGON")


def _require(value: Any, name: str) -> Any:
    if value is None:
        raise SystemExit(f"{name} is not configured")
    return value


def _fundamentals_source(name: str, settings: Settings, client: httpx.AsyncClient) -> FundamentalsSource:
    if name == "SEC_EDGAR":
        return EdgarFundamentalsSource(HttpFetcher(client, rate_per_second=8), settings.sec_user_agent)
    if name == "NSE_XBRL":
        http = HttpFetcher(client, rate_per_second=2)
        return IndianXbrlFundamentalsSource(http, NseFilingIndex(http), Exchange.NSE)
    if name == "FMP":
        return FmpFundamentalsSource(
            HttpFetcher(client, rate_per_second=5), _require(settings.fmp_api_key, "QT_FMP_API_KEY")
        )
    if name == "POLYGON":
        return PolygonFundamentalsSource(
            HttpFetcher(client, rate_per_second=5),
            _require(settings.polygon_api_key, "QT_POLYGON_API_KEY"),
            base_url=settings.polygon_base_url,
        )
    raise SystemExit(f"unknown source {name}")


def set_env_var(path: Path, name: str, value: str) -> None:
    """Replaces (or appends) ``name=value`` in a dotenv file, leaving every other line untouched."""
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    entry = f"{name}={value}"
    for i, line in enumerate(lines):
        if line.partition("=")[0].strip() == name:
            lines[i] = entry
            break
    else:
        lines.append(entry)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def _kite_login(settings: Settings, request_token: str | None, env_file: Path) -> None:
    api_key = _require(settings.kite_api_key, "QT_KITE_API_KEY")
    api_secret = _require(settings.kite_api_secret, "QT_KITE_API_SECRET")
    if request_token is None:
        print(f"Log in at:\n  {kite_login_url(api_key)}\nthen paste the redirect URL (or request_token):")
        request_token = input("> ")
    try:
        token = parse_request_token(request_token)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    async with httpx.AsyncClient(timeout=30) as client:
        access_token = await create_kite_session(client, api_key, api_secret, token)
    set_env_var(env_file, "QT_KITE_ACCESS_TOKEN", access_token.get_secret_value())
    print(f"QT_KITE_ACCESS_TOKEN written to {env_file}; restart the API service to pick it up")


async def _main(args: argparse.Namespace) -> None:
    settings = Settings()
    if args.cmd == "kite-login":
        await _kite_login(settings, args.request_token, Path(args.env_file))
        return
    pool = await create_pool(settings)
    try:
        securities = SecurityRepository(pool)
        if args.cmd == "migrate":
            await apply_schema(pool)
            print("schema applied")
            return
        if args.cmd == "register":
            rec = await securities.upsert(
                args.symbol, Exchange(args.exchange), sector_class=SectorClass(args.sector), isin=args.isin,
                industry=args.industry,
            )
            print(rec.model_dump_json())
            return
        if args.cmd == "reconcile":
            service = ReconciliationService(
                PointInTimeFundamentalsRepository(pool), securities, ReconciliationRepository(pool),
                settings.reconciliation_tolerance,
            )
            as_of = datetime.fromisoformat(args.as_of) if args.as_of else datetime.now(UTC)
            recon = await service.run(
                args.symbol, Exchange(args.exchange), args.metrics, PeriodType(args.period_type),
                args.primary, args.secondary, as_of,
            )
            frame = recon.to_frame()
            print(frame.filter(frame["is_break"]) if args.breaks_only else frame)
            for alert in recon.alerts:
                print(alert.message, file=sys.stderr)
            return

        async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
            prices = PriceRepository(pool, securities)
            if args.cmd == "ingest-fundamentals":
                source = _fundamentals_source(args.source, settings, client)
                report = await FundamentalsIngestionService(FundamentalsRepository(pool, securities)).ingest(
                    source, args.symbol, Exchange(args.exchange)
                )
                print(report.model_dump_json())
            elif args.cmd == "ingest-bhavcopy":
                bhavcopy = NseBhavcopySource(HttpFetcher(client, rate_per_second=2))
                session = date.fromisoformat(args.session)
                through = date.fromisoformat(args.through) if args.through else session
                while session <= through:
                    if session.weekday() < 5:
                        bars = await bhavcopy.fetch_session(session)
                        if not bars:
                            print(f"{session}: no bhavcopy (holiday)", file=sys.stderr)
                        for symbol in args.symbols:
                            if bars and symbol not in bars:
                                print(f"{session} {symbol}: not in bhavcopy", file=sys.stderr)
                            elif bars:
                                n = await prices.append_bars(
                                    symbol, Exchange.NSE, DAILY, [bars[symbol]], NseBhavcopySource.source
                                )
                                print(f"{session} {symbol}: inserted={n}", flush=True)
                    session += timedelta(days=1)
            elif args.cmd == "ingest-polygon-bars":
                polygon_http = HttpFetcher(client, rate_per_second=settings.polygon_requests_per_minute / 60)
                source_bars = PolygonDailyBarsSource(
                    polygon_http, _require(settings.polygon_api_key, "QT_POLYGON_API_KEY"), settings.polygon_base_url
                )
                exchange = Exchange(args.exchange)
                for symbol in args.symbols:
                    fetched = await source_bars.fetch_range(
                        symbol, exchange, date.fromisoformat(args.start), date.fromisoformat(args.end)
                    )
                    n = await prices.append_bars(symbol, exchange, DAILY, fetched, PolygonDailyBarsSource.source)
                    print(f"{symbol}: fetched={len(fetched)} inserted={n}", flush=True)
            elif args.cmd == "ingest-polygon-actions":
                source_actions = PolygonCorporateActionsSource(
                    HttpFetcher(client, rate_per_second=settings.polygon_requests_per_minute / 60),
                    _require(settings.polygon_api_key, "QT_POLYGON_API_KEY"),
                    settings.polygon_base_url,
                )
                for symbol in args.symbols:
                    actions = await source_actions.fetch(symbol, Exchange(args.exchange))
                    n = await CorporateActionRepository(pool, securities).append(actions)
                    print(f"{symbol}: fetched={len(actions)} inserted={n}", flush=True)
    finally:
        await pool.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quantterm")
    sub = parser.add_subparsers(dest="cmd", required=True)
    exchanges = [e.value for e in Exchange]

    sub.add_parser("migrate")

    p = sub.add_parser("register")
    p.add_argument("exchange", choices=exchanges)
    p.add_argument("symbol")
    p.add_argument("--sector", choices=[s.value for s in SectorClass], default=SectorClass.GENERAL.value)
    p.add_argument("--isin")
    p.add_argument("--industry", help="peer-group key used by RV, e.g. 'Banks' or 'Oil & Gas Refining'")

    p = sub.add_parser("ingest-fundamentals")
    p.add_argument("source", choices=FUNDAMENTAL_SOURCES)
    p.add_argument("exchange", choices=exchanges)
    p.add_argument("symbol")

    p = sub.add_parser("ingest-bhavcopy")
    p.add_argument("session", help="YYYY-MM-DD")
    p.add_argument("symbols", nargs="+")
    p.add_argument("--through", help="YYYY-MM-DD; ingest every weekday session from SESSION to this date")

    p = sub.add_parser("kite-login", help="exchange a Kite login request_token for today's access token")
    p.add_argument("request_token", nargs="?", help="request_token or full redirect URL; prompted if omitted")
    p.add_argument("--env-file", default=".env")

    p = sub.add_parser("ingest-polygon-bars")
    p.add_argument("exchange", choices=["NYSE", "NASDAQ"])
    p.add_argument("start")
    p.add_argument("end")
    p.add_argument("symbols", nargs="+")

    p = sub.add_parser("ingest-polygon-actions", help="splits + cash dividends (US)")
    p.add_argument("exchange", choices=["NYSE", "NASDAQ"])
    p.add_argument("symbols", nargs="+")

    p = sub.add_parser("reconcile")
    p.add_argument("exchange", choices=exchanges)
    p.add_argument("symbol")
    p.add_argument("primary")
    p.add_argument("secondary")
    p.add_argument("period_type", choices=[t.value for t in PeriodType])
    p.add_argument("metrics", nargs="+")
    p.add_argument("--as-of", help="ISO-8601 with offset; default now")
    p.add_argument("--breaks-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    asyncio.run(_main(build_parser().parse_args(argv)))


if __name__ == "__main__":
    main()
