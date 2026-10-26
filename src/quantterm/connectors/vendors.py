"""Vendor REST fundamentals (secondary sources for reconciliation): FMP and Polygon.

Both are latest-snapshot feeds whose history can be silently restated, so their knowledge time is
the moment we observed the value (``KnowledgeTime.OBSERVED``), never the original filing date.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

from pydantic import SecretStr

from quantterm.connectors.base import KnowledgeTime, Metric, SourceError, metric_currency
from quantterm.connectors.http import HttpFetcher
from quantterm.models import Exchange, FundamentalFact, Market, PeriodType

Clock = Callable[[], datetime]
# Vendor values are normalised re-statements of filings; regulatory sources score 1.0.
VENDOR_CONFIDENCE = 0.8


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _decimal(v: Any) -> Decimal | None:
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float, str)):
        return Decimal(str(v))
    raise SourceError(f"unexpected numeric value {v!r}")


# --------------------------------------------------------------------------------------------- FMP
FMP_INCOME_FIELDS: dict[Metric, tuple[str, ...]] = {
    Metric.REVENUE: ("revenue",),
    Metric.OPERATING_INCOME: ("operatingIncome",),
    Metric.PRETAX_INCOME: ("incomeBeforeTax",),
    Metric.NET_INCOME: ("netIncome",),
    Metric.EPS_DILUTED: ("epsDiluted", "epsdiluted"),  # stable vs legacy v3 spelling
    Metric.EBITDA: ("ebitda",),
    Metric.DILUTED_SHARES: ("weightedAverageShsOutDil",),
}
FMP_BALANCE_FIELDS: dict[Metric, tuple[str, ...]] = {
    Metric.TOTAL_ASSETS: ("totalAssets",),
    Metric.TOTAL_LIABILITIES: ("totalLiabilities",),
    Metric.TOTAL_EQUITY: ("totalStockholdersEquity",),
    Metric.CASH_AND_EQUIVALENTS: ("cashAndCashEquivalents",),
    Metric.GOODWILL: ("goodwill",),
    Metric.INTANGIBLE_ASSETS: ("intangibleAssets",),
    Metric.TOTAL_DEBT: ("totalDebt",),
    Metric.MINORITY_INTEREST: ("minorityInterest",),
    Metric.PREFERRED_STOCK: ("preferredStock",),
}
_FMP_PERIODS = {"Q1": PeriodType.QUARTER, "Q2": PeriodType.QUARTER, "Q3": PeriodType.QUARTER,
                "Q4": PeriodType.QUARTER, "FY": PeriodType.FISCAL_YEAR}


def fmp_symbol(symbol: str, exchange: Exchange) -> str:
    if exchange is Exchange.NSE:
        return f"{symbol}.NS"
    if exchange is Exchange.BSE:
        return f"{symbol}.BO"
    return symbol.replace(".", "-")


def parse_fmp_statements(
    rows: Sequence[Mapping[str, Any]],
    symbol: str,
    fields: Mapping[Metric, tuple[str, ...]],
    observed_at: datetime,
    source: str = "FMP",
) -> list[FundamentalFact]:
    out: list[FundamentalFact] = []
    for row in rows:
        period_type = _FMP_PERIODS.get(str(row.get("period")))
        if period_type is None:
            continue
        end = date.fromisoformat(str(row["date"]))
        if end > observed_at.date():
            continue
        currency = str(row.get("reportedCurrency") or "")
        if len(currency) != 3:
            raise SourceError(f"FMP {symbol} {end}: missing reportedCurrency")
        for metric, keys in fields.items():
            value = next((_decimal(row[k]) for k in keys if row.get(k) is not None), None)
            if value is None:
                continue
            out.append(
                FundamentalFact(
                    symbol=symbol,
                    metric=metric.value,
                    period_type=period_type,
                    period_end_date=end,
                    filing_timestamp=observed_at,
                    value=value,
                    currency=metric_currency(metric.value, currency),
                    source=source,
                    confidence_score=VENDOR_CONFIDENCE,
                )
            )
    return out


class FmpFundamentalsSource:
    source = "FMP"
    knowledge_time = KnowledgeTime.OBSERVED

    def __init__(
        self,
        http: HttpFetcher,
        api_key: SecretStr,
        *,
        base_url: str = "https://financialmodelingprep.com/stable",
        limit: int = 40,
        clock: Clock = _utc_now,
    ) -> None:
        self._http = http
        self._key = api_key
        self._base = base_url.rstrip("/")
        self._limit = limit
        self._clock = clock

    async def _statement(self, endpoint: str, vendor_symbol: str, period: str) -> list[Mapping[str, Any]]:
        payload = await self._http.get_json(
            f"{self._base}/{endpoint}",
            params={
                "symbol": vendor_symbol,
                "period": period,
                "limit": self._limit,
                "apikey": self._key.get_secret_value(),
            },
        )
        if not isinstance(payload, list):
            raise SourceError(f"FMP {endpoint} for {vendor_symbol}: expected list, got {type(payload).__name__}")
        return payload

    async def fetch_fundamentals(self, symbol: str, exchange: Exchange) -> list[FundamentalFact]:
        observed_at = self._clock()
        vendor_symbol = fmp_symbol(symbol, exchange)
        facts: list[FundamentalFact] = []
        for period in ("quarter", "annual"):
            income = await self._statement("income-statement", vendor_symbol, period)
            balance = await self._statement("balance-sheet-statement", vendor_symbol, period)
            facts += parse_fmp_statements(income, symbol, FMP_INCOME_FIELDS, observed_at)
            facts += parse_fmp_statements(balance, symbol, FMP_BALANCE_FIELDS, observed_at)
        return facts


# ----------------------------------------------------------------------------------------- Polygon
POLYGON_FIELDS: dict[Metric, tuple[str, str]] = {
    Metric.REVENUE: ("income_statement", "revenues"),
    Metric.OPERATING_INCOME: ("income_statement", "operating_income_loss"),
    Metric.PRETAX_INCOME: ("income_statement", "income_loss_from_continuing_operations_before_tax"),
    Metric.NET_INCOME: ("income_statement", "net_income_loss_attributable_to_parent"),
    Metric.EPS_DILUTED: ("income_statement", "diluted_earnings_per_share"),
    Metric.TOTAL_ASSETS: ("balance_sheet", "assets"),
    Metric.TOTAL_LIABILITIES: ("balance_sheet", "liabilities"),
    Metric.TOTAL_EQUITY: ("balance_sheet", "equity_attributable_to_parent"),
}
_POLYGON_TIMEFRAMES = {"quarterly": PeriodType.QUARTER, "annual": PeriodType.FISCAL_YEAR}


def parse_polygon_financials(
    results: Sequence[Mapping[str, Any]], symbol: str, observed_at: datetime
) -> list[FundamentalFact]:
    out: list[FundamentalFact] = []
    for result in results:
        period_type = _POLYGON_TIMEFRAMES.get(str(result.get("timeframe")))
        if period_type is None:
            continue
        end = date.fromisoformat(str(result["end_date"]))
        if end > observed_at.date():
            continue
        financials = result.get("financials") or {}
        for metric, (statement, key) in POLYGON_FIELDS.items():
            item = (financials.get(statement) or {}).get(key)
            if not item or item.get("value") is None:
                continue
            # Units look like "USD" or "USD / shares".
            currency = str(item.get("unit", "")).split("/")[0].strip()
            if len(currency) != 3:
                raise SourceError(f"Polygon {symbol} {end} {key}: unexpected unit {item.get('unit')!r}")
            value = _decimal(item["value"])
            assert value is not None
            out.append(
                FundamentalFact(
                    symbol=symbol,
                    metric=metric.value,
                    period_type=period_type,
                    period_end_date=end,
                    filing_timestamp=observed_at,
                    value=value,
                    currency=currency,
                    source="POLYGON",
                    confidence_score=VENDOR_CONFIDENCE,
                )
            )
    return out


class PolygonFundamentalsSource:
    source = "POLYGON"
    knowledge_time = KnowledgeTime.OBSERVED

    def __init__(
        self,
        http: HttpFetcher,
        api_key: SecretStr,
        *,
        base_url: str = "https://api.polygon.io",
        max_pages: int = 20,
        clock: Clock = _utc_now,
    ) -> None:
        self._http = http
        self._key = api_key
        self._base = base_url.rstrip("/")
        self._max_pages = max_pages
        self._clock = clock

    async def _paged(self, url: str, params: dict[str, str | int]) -> list[Mapping[str, Any]]:
        results: list[Mapping[str, Any]] = []
        next_url: str | None = url
        query: dict[str, str | int] = {**params, "apiKey": self._key.get_secret_value()}
        for _ in range(self._max_pages):
            if next_url is None:
                break
            payload = await self._http.get_json(next_url, params=query)
            results.extend(payload.get("results") or [])
            next_url = payload.get("next_url")
            # ``next_url`` embeds the cursor but not the key.
            query = {"apiKey": self._key.get_secret_value()}
        return results

    async def fetch_fundamentals(self, symbol: str, exchange: Exchange) -> list[FundamentalFact]:
        if exchange.market is not Market.US:
            raise ValueError("Polygon financials cover US listings only")
        observed_at = self._clock()
        facts: list[FundamentalFact] = []
        for timeframe in ("quarterly", "annual"):
            results = await self._paged(
                f"{self._base}/vX/reference/financials",
                {"ticker": symbol, "timeframe": timeframe, "limit": 100, "order": "desc",
                 "sort": "period_of_report_date"},
            )
            facts += parse_polygon_financials(results, symbol, observed_at)
        return facts
