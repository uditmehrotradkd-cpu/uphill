"""Raw (unadjusted) end-of-day bars: NSE UDiFF bhavcopy and Polygon aggregates with ``adjusted=false``.

Vendor-adjusted series (e.g. Kite historical, Polygon ``adjusted=true``) must never be fed to
``raw_market_bars``; adjustment is applied at query time from ``corporate_actions``.
"""

from __future__ import annotations

import csv
import io
import logging
import zipfile
from collections.abc import Iterable, Mapping
from datetime import UTC, date, datetime, time
from decimal import Decimal, InvalidOperation
from typing import Any

from pydantic import SecretStr, ValidationError

from quantterm.connectors.http import HttpFetcher, SourceHTTPError
from quantterm.corporate_actions import CorporateAction, CorporateActionType, split_factor
from quantterm.models import Exchange, Market, OhlcvBar, session_bar_ts

log = logging.getLogger(__name__)

NSE_BHAVCOPY_URL = (
    "https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{yyyymmdd}_F_0000.csv.zip"
)
NSE_EQUITY_SERIES = frozenset({"EQ", "BE", "BZ", "SM", "ST"})


def parse_nse_bhavcopy(
    csv_text: str, series: Iterable[str] = NSE_EQUITY_SERIES
) -> dict[str, OhlcvBar]:
    """UDiFF CM bhavcopy CSV -> {symbol: raw daily bar}. Rows failing OHLC sanity are logged and dropped."""
    wanted = frozenset(series)
    bars: dict[str, OhlcvBar] = {}
    for row in csv.DictReader(io.StringIO(csv_text)):
        if row["SctySrs"] not in wanted or row["FinInstrmTp"] != "STK":
            continue
        try:
            volume = int(row["TtlTradgVol"])
            bars[row["TckrSymb"]] = OhlcvBar(
                ts=session_bar_ts(Exchange.NSE, date.fromisoformat(row["TradDt"])),
                open=float(row["OpnPric"]),
                high=float(row["HghPric"]),
                low=float(row["LwPric"]),
                close=float(row["ClsPric"]),
                volume=volume,
                vwap=float(row["TtlTrfVal"]) / volume if volume > 0 else None,
            )
        except (ValidationError, ValueError) as exc:
            log.warning("bhavcopy %s %s rejected: %s", row.get("TradDt"), row.get("TckrSymb"), exc)
    return bars


class NseBhavcopySource:
    source = "NSE_BHAVCOPY"

    def __init__(self, http: HttpFetcher, url_template: str = NSE_BHAVCOPY_URL) -> None:
        self._http = http
        self._template = url_template

    async def fetch_session(self, session: date) -> dict[str, OhlcvBar]:
        """Empty dict for non-trading days (NSE answers 404 when no bhavcopy exists)."""
        url = self._template.format(yyyymmdd=session.strftime("%Y%m%d"))
        try:
            response = await self._http.get(url, headers={"User-Agent": "Mozilla/5.0"})
        except SourceHTTPError as exc:
            if exc.status == 404:
                return {}
            raise
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            members = [n for n in archive.namelist() if n.lower().endswith(".csv")]
            if len(members) != 1:
                raise ValueError(f"bhavcopy {session}: expected one CSV, found {members}")
            return parse_nse_bhavcopy(archive.read(members[0]).decode("utf-8-sig"))


def parse_polygon_aggs(results: Iterable[Mapping[str, Any]], exchange: Exchange) -> list[OhlcvBar]:
    """Polygon daily aggs (``t`` = ms epoch of the session start) -> raw bars keyed by session date."""
    bars: list[OhlcvBar] = []
    tz = exchange.timezone
    for r in results:
        session = datetime.fromtimestamp(r["t"] / 1000, tz=UTC).astimezone(tz).date()
        vwap = r.get("vw")
        if vwap is not None and not r["l"] <= vwap <= r["h"]:
            # Polygon's daily vw includes extended-hours prints (e.g. post-earnings); OHLC are regular session.
            log.info("polygon %s vwap %s outside session range [%s, %s]; stored without vwap", session, vwap,
                     r["l"], r["h"])
            vwap = None
        try:
            bars.append(
                OhlcvBar(
                    ts=session_bar_ts(exchange, session),
                    open=r["o"],
                    high=r["h"],
                    low=r["l"],
                    close=r["c"],
                    volume=round(r["v"]),
                    vwap=vwap,
                )
            )
        except ValidationError as exc:
            log.warning("polygon bar %s rejected: %s", session, exc)
    return bars


class PolygonDailyBarsSource:
    source = "POLYGON"

    def __init__(self, http: HttpFetcher, api_key: SecretStr, base_url: str = "https://api.polygon.io") -> None:
        self._http = http
        self._key = api_key
        self._base = base_url.rstrip("/")

    async def fetch_range(self, symbol: str, exchange: Exchange, start: date, end: date) -> list[OhlcvBar]:
        if exchange.market is not Market.US:
            raise ValueError("Polygon aggregates are used for US listings only")
        url: str | None = (
            f"{self._base}/v2/aggs/ticker/{symbol}/range/1/day/{start.isoformat()}/{end.isoformat()}"
        )
        params: dict[str, str | int] = {
            "adjusted": "false",
            "sort": "asc",
            "limit": 50000,
            "apiKey": self._key.get_secret_value(),
        }
        bars: list[OhlcvBar] = []
        while url is not None:
            payload = await self._http.get_json(url, params=params)
            bars.extend(parse_polygon_aggs(payload.get("results") or [], exchange))
            url = payload.get("next_url")
            params = {"apiKey": self._key.get_secret_value()}
        return bars


def _start_of(day: date, exchange: Exchange) -> datetime:
    return datetime.combine(day, time.min, tzinfo=exchange.timezone)


def parse_polygon_splits(results: Iterable[Mapping[str, Any]], symbol: str, exchange: Exchange) -> list[CorporateAction]:
    """Polygon has no split announcement time: treat the split as known from its ex-date (never too early)."""
    out: list[CorporateAction] = []
    for r in results:
        try:
            ex = date.fromisoformat(r["execution_date"])
            out.append(CorporateAction(
                symbol=symbol, exchange=exchange, action_type=CorporateActionType.SPLIT, ex_date=ex,
                announced_at=_start_of(ex, exchange),
                adjustment_factor=split_factor(Decimal(str(r["split_to"])), Decimal(str(r["split_from"]))),
                source=PolygonCorporateActionsSource.source,
            ))
        except (KeyError, ValueError, InvalidOperation, ValidationError) as exc:
            log.warning("polygon split %s rejected: %s", r, exc)
    return out


def parse_polygon_dividends(
    results: Iterable[Mapping[str, Any]], symbol: str, exchange: Exchange, currency: str
) -> list[CorporateAction]:
    """Cash distributions in the listing currency, known from the end of the declaration day (else ex-date)."""
    out: list[CorporateAction] = []
    for r in results:
        try:
            if r.get("currency", currency) != currency:
                log.warning("polygon dividend %s %s in %s skipped (listing trades in %s)",
                            symbol, r.get("ex_dividend_date"), r.get("currency"), currency)
                continue
            ex = date.fromisoformat(r["ex_dividend_date"])
            declared = r.get("declaration_date")
            announced = (datetime.combine(date.fromisoformat(declared), time.max, tzinfo=exchange.timezone)
                         if declared else _start_of(ex, exchange))
            out.append(CorporateAction(
                symbol=symbol, exchange=exchange, action_type=CorporateActionType.CASH_DIVIDEND, ex_date=ex,
                announced_at=min(announced, _start_of(ex, exchange)),
                raw_dividend_amount=Decimal(str(r["cash_amount"])), source=PolygonCorporateActionsSource.source,
            ))
        except (KeyError, ValueError, InvalidOperation, ValidationError) as exc:
            log.warning("polygon dividend %s rejected: %s", r, exc)
    # One row per ex-date in corporate_actions: a special paid alongside the regular dividend is summed.
    merged: dict[date, CorporateAction] = {}
    for a in out:
        prior = merged.get(a.ex_date)
        merged[a.ex_date] = a if prior is None else prior.model_copy(update={
            "raw_dividend_amount": (prior.raw_dividend_amount or Decimal(0)) + (a.raw_dividend_amount or Decimal(0)),
            "announced_at": max(prior.announced_at, a.announced_at),
        })
    return list(merged.values())


class PolygonCorporateActionsSource:
    """Splits and cash dividends from Polygon reference data (raw amounts; factors derived at query time)."""

    source = "POLYGON"

    def __init__(self, http: HttpFetcher, api_key: SecretStr, base_url: str = "https://api.polygon.io") -> None:
        self._http = http
        self._key = api_key
        self._base = base_url.rstrip("/")

    async def _paged(self, path: str, symbol: str) -> list[Mapping[str, Any]]:
        url: str | None = f"{self._base}{path}"
        params: dict[str, str | int] = {"ticker": symbol, "limit": 1000, "apiKey": self._key.get_secret_value()}
        rows: list[Mapping[str, Any]] = []
        while url is not None:
            payload = await self._http.get_json(url, params=params)
            rows.extend(payload.get("results") or [])
            url = payload.get("next_url")
            params = {"apiKey": self._key.get_secret_value()}
        return rows

    async def fetch(self, symbol: str, exchange: Exchange) -> list[CorporateAction]:
        if exchange.market is not Market.US:
            raise ValueError("Polygon reference data is used for US listings only")
        splits = parse_polygon_splits(await self._paged("/v3/reference/splits", symbol), symbol, exchange)
        dividends = parse_polygon_dividends(await self._paged("/v3/reference/dividends", symbol), symbol, exchange,
                                            exchange.currency)
        return splits + dividends
