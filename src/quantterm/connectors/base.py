"""Contracts shared by all fundamentals connectors."""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol

from quantterm.models import Exchange, FundamentalFact


class KnowledgeTime(StrEnum):
    # Source exposes the public-dissemination time of every value (each filing is a vintage).
    FILING = "FILING"
    # Source is a latest-snapshot vendor: a value is only knowable from the moment we observed it.
    OBSERVED = "OBSERVED"


class Metric(StrEnum):
    """Canonical metric names shared by every connector so reconciliation keys line up."""

    REVENUE = "revenue"
    OPERATING_INCOME = "operating_income"
    PRETAX_INCOME = "pretax_income"
    NET_INCOME = "net_income"  # attributable to owners of the parent
    EPS_DILUTED = "eps_diluted"
    TOTAL_ASSETS = "total_assets"
    TOTAL_LIABILITIES = "total_liabilities"
    TOTAL_EQUITY = "total_equity"  # attributable to owners of the parent
    CASH_AND_EQUIVALENTS = "cash_and_equivalents"
    EBITDA = "ebitda"
    TOTAL_DEBT = "total_debt"
    MINORITY_INTEREST = "minority_interest"
    PREFERRED_STOCK = "preferred_stock"
    DILUTED_SHARES = "diluted_shares"  # weighted-average diluted share count
    LEASE_LIABILITIES = "lease_liabilities"
    GOODWILL = "goodwill"
    INTANGIBLE_ASSETS = "intangible_assets"  # excluding goodwill


# Additive across quarters (a fiscal Q4 can be derived as FY - Q1 - Q2 - Q3). EPS is not.
FLOW_METRICS: frozenset[Metric] = frozenset(
    {Metric.REVENUE, Metric.OPERATING_INCOME, Metric.PRETAX_INCOME, Metric.NET_INCOME, Metric.EBITDA}
)
SHARE_COUNT_METRICS: frozenset[str] = frozenset({Metric.DILUTED_SHARES.value})
NO_CURRENCY = "XXX"  # ISO 4217 code for "no currency involved"


def metric_currency(metric: str, reporting_currency: str) -> str:
    return NO_CURRENCY if metric in SHARE_COUNT_METRICS else reporting_currency


class SourceError(RuntimeError):
    """A source responded with unusable data (schema drift, wrong entity, malformed document)."""


class FundamentalsSource(Protocol):
    @property
    def source(self) -> str: ...

    @property
    def knowledge_time(self) -> KnowledgeTime: ...

    async def fetch_fundamentals(self, symbol: str, exchange: Exchange) -> list[FundamentalFact]: ...
