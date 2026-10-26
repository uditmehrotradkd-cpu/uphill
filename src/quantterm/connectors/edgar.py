"""SEC EDGAR XBRL ``companyfacts`` -> bitemporal facts (primary regulatory source for US listings).

Knowledge time is the EDGAR acceptance timestamp of the filing that carried each value, so every
comparative restated in a later 10-K/10-Q becomes its own vintage. When the acceptance time is not
available the end of the filing date (US/Eastern) is used, which is never earlier than acceptance.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Any
from zoneinfo import ZoneInfo

from quantterm.connectors.base import FLOW_METRICS, KnowledgeTime, Metric, SourceError, metric_currency
from quantterm.connectors.http import HttpFetcher
from quantterm.models import Exchange, FundamentalFact, Market, PeriodType

SOURCE = "SEC_EDGAR"
# FY - (Q1+Q2+Q3) inherits any tag inconsistency between the 10-K and the 10-Qs.
DERIVED_CONFIDENCE = 0.95
_ET = ZoneInfo("America/New_York")
ANNUAL_FORMS = frozenset({"10-K", "10-K/A", "10-KT", "10-KT/A"})
QUARTERLY_FORMS = frozenset({"10-Q", "10-Q/A"})

# Priority order: the first tag present in a filing wins for that (metric, period).
US_GAAP_TAGS: dict[Metric, tuple[str, ...]] = {
    Metric.REVENUE: (
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
    ),
    Metric.OPERATING_INCOME: ("OperatingIncomeLoss",),
    Metric.PRETAX_INCOME: (
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
    ),
    Metric.NET_INCOME: ("NetIncomeLoss",),
    Metric.EPS_DILUTED: ("EarningsPerShareDiluted",),
    Metric.TOTAL_ASSETS: ("Assets",),
    Metric.TOTAL_LIABILITIES: ("Liabilities",),
    Metric.TOTAL_EQUITY: ("StockholdersEquity",),
    Metric.CASH_AND_EQUIVALENTS: ("CashAndCashEquivalentsAtCarryingValue",),
    Metric.MINORITY_INTEREST: ("MinorityInterest",),
    Metric.PREFERRED_STOCK: ("PreferredStockValue",),
    Metric.DILUTED_SHARES: ("WeightedAverageNumberOfDilutedSharesOutstanding",),
    Metric.LEASE_LIABILITIES: ("OperatingLeaseLiability",),
    Metric.GOODWILL: ("Goodwill",),
    Metric.INTANGIBLE_ASSETS: ("IntangibleAssetsNetExcludingGoodwill",),
}


def _unit(metric: Metric) -> str:
    if metric is Metric.DILUTED_SHARES:
        return "shares"
    return "USD/shares" if metric is Metric.EPS_DILUTED else "USD"


@dataclass(frozen=True)
class _Obs:
    metric: Metric
    period_type: PeriodType
    start: date | None
    end: date
    value: Decimal
    accn: str
    filed_at: datetime
    priority: int


def _period_type(start: date | None, end: date, form: str) -> PeriodType | None:
    if start is None:  # instant (balance sheet)
        return PeriodType.FISCAL_YEAR if form in ANNUAL_FORMS else PeriodType.QUARTER
    days = (end - start).days + 1
    if 80 <= days <= 100:
        return PeriodType.QUARTER
    if 170 <= days <= 195:
        return PeriodType.HALF_YEAR
    if 350 <= days <= 380:
        return PeriodType.FISCAL_YEAR
    return None  # 9-month YTD and other partial spans


def _end_of_day_et(d: date) -> datetime:
    return datetime.combine(d, time(23, 59, 59), tzinfo=_ET)


def _derive_fourth_quarters(observations: list[_Obs]) -> list[_Obs]:
    """Q4 = FY - (Q1 + Q2 + Q3) for flow metrics, using only quarters public when the 10-K was accepted."""
    quarters = [
        o for o in observations
        if o.period_type is PeriodType.QUARTER and o.start is not None and o.metric in FLOW_METRICS
    ]
    derived: list[_Obs] = []
    for fy in observations:
        if fy.period_type is not PeriodType.FISCAL_YEAR or fy.start is None or fy.metric not in FLOW_METRICS:
            continue
        window_start = fy.start - timedelta(days=3)
        known = [
            q for q in quarters
            if q.metric is fy.metric
            and q.start is not None
            and q.start >= window_start
            and q.end <= fy.end
            and q.filed_at <= fy.filed_at
        ]
        if any(q.end == fy.end for q in known):
            continue  # Q4 reported directly
        latest: dict[date, _Obs] = {}
        for q in known:
            if q.end not in latest or q.filed_at > latest[q.end].filed_at:
                latest[q.end] = q
        if len(latest) != 3:
            continue
        derived.append(
            replace(
                fy,
                period_type=PeriodType.QUARTER,
                start=max(latest) + timedelta(days=1),
                value=fy.value - sum((q.value for q in latest.values()), Decimal(0)),
                accn=f"{fy.accn}:Q4_DERIVED",
            )
        )
    return derived


def parse_companyfacts(
    payload: Mapping[str, Any], symbol: str, acceptance: Mapping[str, datetime]
) -> list[FundamentalFact]:
    facts = payload.get("facts")
    gaap = facts.get("us-gaap") if isinstance(facts, Mapping) else None
    if not isinstance(gaap, Mapping):
        raise SourceError(f"{symbol}: companyfacts payload has no us-gaap facts")

    best: dict[tuple[Metric, PeriodType, date, str], _Obs] = {}
    for metric, tags in US_GAAP_TAGS.items():
        for priority, tag in enumerate(tags):
            entries = gaap.get(tag, {}).get("units", {}).get(_unit(metric), [])
            for e in entries:
                form = e.get("form")
                if form not in ANNUAL_FORMS and form not in QUARTERLY_FORMS:
                    continue
                end = date.fromisoformat(e["end"])
                start = date.fromisoformat(e["start"]) if "start" in e else None
                period_type = _period_type(start, end, form)
                if period_type is None:
                    continue
                accn = e["accn"]
                filed_at = acceptance.get(accn) or _end_of_day_et(date.fromisoformat(e["filed"]))
                if filed_at.date() < end:
                    continue
                key = (metric, period_type, end, accn)
                current = best.get(key)
                if current is None or priority < current.priority:
                    best[key] = _Obs(
                        metric, period_type, start, end, Decimal(str(e["val"])), accn, filed_at, priority
                    )

    observations = list(best.values())
    observations.extend(_derive_fourth_quarters(observations))
    return [
        FundamentalFact(
            symbol=symbol,
            metric=o.metric.value,
            period_type=o.period_type,
            period_end_date=o.end,
            filing_timestamp=o.filed_at,
            value=o.value,
            currency=metric_currency(o.metric.value, "USD"),
            source=SOURCE,
            source_document_id=o.accn,
            confidence_score=DERIVED_CONFIDENCE if o.accn.endswith(":Q4_DERIVED") else 1.0,
        )
        for o in sorted(observations, key=lambda o: (o.metric, o.period_type, o.end, o.filed_at))
    ]


def parse_acceptance_times(filings: Mapping[str, Any]) -> dict[str, datetime]:
    """``filings.recent`` (or an older ``filings.files`` page) -> accession -> acceptance (UTC)."""
    return {
        accn: datetime.fromisoformat(accepted)
        for accn, accepted in zip(filings["accessionNumber"], filings["acceptanceDateTime"], strict=True)
        if accepted
    }


class EdgarFundamentalsSource:
    source = SOURCE
    knowledge_time = KnowledgeTime.FILING

    def __init__(
        self,
        http: HttpFetcher,
        user_agent: str,
        *,
        data_base_url: str = "https://data.sec.gov",
        tickers_url: str = "https://www.sec.gov/files/company_tickers.json",
    ) -> None:
        if "@" not in user_agent:
            raise ValueError("SEC fair-access policy requires a User-Agent containing a contact e-mail")
        self._http = http
        self._headers = {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}
        self._base = data_base_url.rstrip("/")
        self._tickers_url = tickers_url
        self._ciks: dict[str, int] | None = None

    async def cik_for(self, symbol: str) -> int:
        if self._ciks is None:
            payload = await self._http.get_json(self._tickers_url, headers=self._headers)
            self._ciks = {row["ticker"].upper(): int(row["cik_str"]) for row in payload.values()}
        key = symbol.upper().replace(".", "-")
        if key not in self._ciks:
            raise SourceError(f"{symbol} not found in SEC company_tickers.json")
        return self._ciks[key]

    async def _acceptance_times(self, cik: int) -> dict[str, datetime]:
        submissions = await self._http.get_json(
            f"{self._base}/submissions/CIK{cik:010d}.json", headers=self._headers
        )
        filings = submissions["filings"]
        accepted = parse_acceptance_times(filings["recent"])
        for page in filings.get("files", []):
            older = await self._http.get_json(f"{self._base}/submissions/{page['name']}", headers=self._headers)
            accepted.update(parse_acceptance_times(older))
        return accepted

    async def fetch_fundamentals(self, symbol: str, exchange: Exchange) -> list[FundamentalFact]:
        if exchange.market is not Market.US:
            raise ValueError(f"SEC EDGAR covers US listings only, got {exchange.value}")
        cik = await self.cik_for(symbol)
        acceptance = await self._acceptance_times(cik)
        payload = await self._http.get_json(
            f"{self._base}/api/xbrl/companyfacts/CIK{cik:010d}.json", headers=self._headers
        )
        return parse_companyfacts(payload, symbol, acceptance)
