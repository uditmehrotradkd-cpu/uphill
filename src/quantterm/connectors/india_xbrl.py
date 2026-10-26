"""NSE/BSE Ind-AS financial-results XBRL (``in-bse-fin`` taxonomy) -> bitemporal facts.

Context period dates in exchange-generated instances are not reliable for duration contexts (the
year-to-date context can carry the quarter's dates), so each duration context's true span is read
from the ``DateOfStartOfReportingPeriod`` / ``DateOfEndOfReportingPeriod`` facts tagged with it.
Knowledge time is the exchange dissemination timestamp of the filing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Protocol
from xml.etree.ElementTree import Element
from zoneinfo import ZoneInfo

from defusedxml import DefusedXmlException
from defusedxml import ElementTree as SafeET

from quantterm.connectors.base import KnowledgeTime, Metric, SourceError
from quantterm.connectors.http import HttpFetcher
from quantterm.models import Exchange, FundamentalFact, Market, PeriodType

_IST = ZoneInfo("Asia/Kolkata")
_XBRLI = "{http://www.xbrl.org/2003/instance}"
_XSI_NIL = "{http://www.w3.org/2001/XMLSchema-instance}nil"

IND_AS_TAGS: dict[Metric, tuple[str, ...]] = {
    Metric.REVENUE: ("RevenueFromOperations",),
    Metric.PRETAX_INCOME: ("ProfitBeforeTax",),
    Metric.NET_INCOME: ("ProfitOrLossAttributableToOwnersOfParent", "ProfitLossForPeriod"),
    Metric.EPS_DILUTED: ("DilutedEarningsLossPerShareFromContinuingAndDiscontinuedOperations",),
    Metric.TOTAL_ASSETS: ("Assets",),
    Metric.TOTAL_LIABILITIES: ("Liabilities",),
    Metric.TOTAL_EQUITY: ("EquityAttributableToOwnersOfParent", "Equity"),
    Metric.CASH_AND_EQUIVALENTS: ("CashAndCashEquivalents",),
    Metric.GOODWILL: ("Goodwill",),
    Metric.INTANGIBLE_ASSETS: ("OtherIntangibleAssets",),
}


@dataclass(frozen=True)
class XbrlFilingRef:
    url: str
    symbol: str
    exchange: Exchange
    disseminated_at: datetime
    consolidated: bool
    document_id: str


@dataclass(frozen=True)
class _Context:
    start: date | None
    end: date | None
    instant: date | None
    dimensional: bool


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _span_type(start: date, end: date) -> PeriodType | None:
    days = (end - start).days + 1
    if 80 <= days <= 100:
        return PeriodType.QUARTER
    if 170 <= days <= 195:
        return PeriodType.HALF_YEAR
    if 350 <= days <= 380:
        return PeriodType.FISCAL_YEAR
    return None


def _contexts(root: Element) -> dict[str, _Context]:
    out: dict[str, _Context] = {}
    for ctx in root.iter(f"{_XBRLI}context"):
        period = ctx.find(f"{_XBRLI}period")
        if period is None:
            continue

        def _d(name: str, p: Element = period) -> date | None:
            el = p.find(f"{_XBRLI}{name}")
            return date.fromisoformat(el.text.strip()) if el is not None and el.text else None

        dimensional = ctx.find(f".//{_XBRLI}scenario") is not None or ctx.find(f".//{_XBRLI}segment") is not None
        out[ctx.attrib["id"]] = _Context(_d("startDate"), _d("endDate"), _d("instant"), dimensional)
    return out


def _unit_currency(root: Element) -> dict[str, str]:
    """unit id -> ISO currency for monetary and per-share units."""
    out: dict[str, str] = {}
    for unit in root.iter(f"{_XBRLI}unit"):
        measure = unit.find(f".//{_XBRLI}measure")  # first measure = numerator for divide units
        if measure is not None and measure.text and measure.text.startswith("iso4217:"):
            out[unit.attrib["id"]] = measure.text.split(":", 1)[1]
    return out


def parse_ind_as_results(document: bytes, ref: XbrlFilingRef) -> list[FundamentalFact]:
    try:
        root = SafeET.fromstring(document, forbid_dtd=True)
    except (SafeET.ParseError, DefusedXmlException) as exc:
        raise SourceError(f"{ref.document_id}: rejected XBRL document ({type(exc).__name__})") from exc

    contexts = _contexts(root)
    currencies = _unit_currency(root)
    facts = [el for el in root if "in-bse-fin" in el.tag and el.get(_XSI_NIL) != "true"]

    text: dict[tuple[str, str], str] = {}
    for el in facts:
        if el.text is not None and el.get("contextRef"):
            text.setdefault((_local(el.tag), el.attrib["contextRef"]), el.text.strip())

    nature = next((v for (name, _), v in text.items() if name == "NatureOfReportStandaloneConsolidated"), None)
    if nature is not None and (nature.lower() == "consolidated") != ref.consolidated:
        raise SourceError(f"{ref.document_id}: filing is {nature}, expected consolidated={ref.consolidated}")
    fy_end_text = next((v for (name, _), v in text.items() if name == "DateOfEndOfFinancialYear"), None)
    fy_end = date.fromisoformat(fy_end_text) if fy_end_text else None

    # True span of each non-dimensional duration context.
    durations: dict[str, tuple[PeriodType, date]] = {}
    for cid, ctx in contexts.items():
        if ctx.dimensional or ctx.instant is not None:
            continue
        start_s = text.get(("DateOfStartOfReportingPeriod", cid))
        end_s = text.get(("DateOfEndOfReportingPeriod", cid))
        if start_s is None or end_s is None:
            continue
        start, end = date.fromisoformat(start_s), date.fromisoformat(end_s)
        span = _span_type(start, end)
        if span is not None:
            durations[cid] = (span, end)
    if not durations:
        raise SourceError(f"{ref.document_id}: no reporting-period contexts found")

    def instant_type(d: date) -> PeriodType:
        if fy_end is not None and d == fy_end:
            return PeriodType.FISCAL_YEAR
        spans = {pt for pt, end in durations.values() if end == d}
        return PeriodType.HALF_YEAR if PeriodType.HALF_YEAR in spans else PeriodType.QUARTER

    tag_rank = {tag: (metric, rank) for metric, tags in IND_AS_TAGS.items() for rank, tag in enumerate(tags)}
    best: dict[tuple[Metric, PeriodType, date], tuple[int, Decimal, str]] = {}
    for el in facts:
        hit = tag_rank.get(_local(el.tag))
        context_id = el.get("contextRef")
        unit = el.get("unitRef")
        if hit is None or context_id is None or unit is None or el.text is None:
            continue
        context = contexts.get(context_id)
        if context is None or context.dimensional:
            continue
        if context.instant is not None:
            key_period: tuple[PeriodType, date] = (instant_type(context.instant), context.instant)
        elif context_id in durations:
            key_period = durations[context_id]
        else:
            continue
        currency = currencies.get(unit)
        if currency is None:
            continue
        try:
            value = Decimal(el.text.strip())
        except InvalidOperation as exc:
            raise SourceError(f"{ref.document_id}: non-numeric {_local(el.tag)}={el.text!r}") from exc
        metric, rank = hit
        key = (metric, *key_period)
        if key not in best or rank < best[key][0]:
            best[key] = (rank, value, currency)

    return [
        FundamentalFact(
            symbol=ref.symbol,
            metric=metric.value,
            period_type=period_type,
            period_end_date=end,
            filing_timestamp=ref.disseminated_at,
            value=value,
            currency=currency,
            source=f"{ref.exchange.value}_XBRL",
            source_document_id=ref.document_id,
        )
        for (metric, period_type, end), (_, value, currency) in sorted(best.items())
    ]


class FilingIndex(Protocol):
    async def list_results(self, symbol: str, *, consolidated: bool) -> list[XbrlFilingRef]: ...


def _nse_timestamp(value: str | None) -> datetime | None:
    if not value or not value.strip() or value.strip() == "-":
        return None
    return datetime.strptime(value.strip(), "%d-%b-%Y %H:%M:%S").replace(tzinfo=_IST)


def parse_nse_results_index(
    items: Sequence[Mapping[str, Any]], symbol: str, *, consolidated: bool
) -> list[XbrlFilingRef]:
    """NSE ``corporates-financial-results`` rows -> XBRL refs (rows without an XBRL instance are skipped)."""
    refs: list[XbrlFilingRef] = []
    for item in items:
        url = item.get("xbrl") or ""
        if not url.endswith(".xml"):
            continue
        if (item.get("consolidated") == "Consolidated") != consolidated:
            continue
        if item.get("cumulative") not in (None, "Non-cumulative"):
            continue
        # Exchange dissemination time is when the market saw it; broadcast time is the fallback.
        disseminated = _nse_timestamp(item.get("exchdisstime")) or _nse_timestamp(item.get("broadCastDate"))
        if disseminated is None:
            continue
        refs.append(
            XbrlFilingRef(
                url=url,
                symbol=symbol,
                exchange=Exchange.NSE,
                disseminated_at=disseminated,
                consolidated=consolidated,
                document_id=f"NSE:{item['seqNumber']}",
            )
        )
    return refs


class NseFilingIndex:
    BASE = "https://www.nseindia.com"
    _HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)",
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.nseindia.com/companies-listing/corporate-filings-financial-results",
    }

    def __init__(self, http: HttpFetcher) -> None:
        self._http = http
        self._warmed = False

    async def list_results(self, symbol: str, *, consolidated: bool) -> list[XbrlFilingRef]:
        if not self._warmed:
            # NSE's API rejects clients that have not first obtained session cookies from the site.
            await self._http.get(self.BASE, headers=self._HEADERS)
            self._warmed = True
        items = await self._http.get_json(
            f"{self.BASE}/api/corporates-financial-results",
            params={"index": "equities", "symbol": symbol, "period": "Quarterly"},
            headers=self._HEADERS,
        )
        if not isinstance(items, list):
            raise SourceError(f"NSE results index for {symbol} returned {type(items).__name__}, expected list")
        return parse_nse_results_index(items, symbol, consolidated=consolidated)


class IndianXbrlFundamentalsSource:
    knowledge_time = KnowledgeTime.FILING

    def __init__(
        self, http: HttpFetcher, index: FilingIndex, exchange: Exchange, *, consolidated: bool = True
    ) -> None:
        if exchange.market is not Market.IN:
            raise ValueError("Indian XBRL source requires NSE or BSE")
        self._http = http
        self._index = index
        self._exchange = exchange
        self._consolidated = consolidated

    @property
    def source(self) -> str:
        return f"{self._exchange.value}_XBRL"

    async def fetch_fundamentals(self, symbol: str, exchange: Exchange) -> list[FundamentalFact]:
        if exchange is not self._exchange:
            raise ValueError(f"source configured for {self._exchange.value}, got {exchange.value}")
        facts: list[FundamentalFact] = []
        for ref in await self._index.list_results(symbol, consolidated=self._consolidated):
            response = await self._http.get(ref.url, headers={"User-Agent": NseFilingIndex._HEADERS["User-Agent"]})
            facts.extend(parse_ind_as_results(response.content, ref))
        return facts
