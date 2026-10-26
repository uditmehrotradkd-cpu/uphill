"""Validated service functions: the only entry points used by the API (and later the agent layer)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from quantterm.connectors.base import FundamentalsSource, KnowledgeTime
from quantterm.db.repositories import FundamentalsRepository, ReconciliationRepository, SecurityRepository
from quantterm.hygiene.reconciler import (
    RECONCILED_LINE_ITEMS,
    ConfidencePolicy,
    FundamentalReconciler,
    ReconciliationReport,
)
from quantterm.models import Exchange, PeriodType
from quantterm.pit import PointInTimeFundamentalsRepository


class IngestReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    source: str
    symbol: str
    exchange: Exchange
    fetched: int
    inserted: int


class FundamentalsIngestionService:
    def __init__(self, facts: FundamentalsRepository) -> None:
        self._facts = facts

    async def ingest(self, source: FundamentalsSource, symbol: str, exchange: Exchange) -> IngestReport:
        facts = await source.fetch_fundamentals(symbol, exchange)
        inserted = await self._facts.append(
            exchange, facts, only_if_changed=source.knowledge_time is KnowledgeTime.OBSERVED
        )
        return IngestReport(
            source=source.source, symbol=symbol, exchange=exchange, fetched=len(facts), inserted=inserted
        )


class ReconciliationService:
    def __init__(
        self,
        pit: PointInTimeFundamentalsRepository,
        securities: SecurityRepository,
        breaks: ReconciliationRepository,
        tolerance: float,
    ) -> None:
        self._pit = pit
        self._securities = securities
        self._breaks = breaks
        self._reconciler = FundamentalReconciler(ConfidencePolicy(tolerance=tolerance), line_items=None)

    async def run(
        self,
        symbol: str,
        exchange: Exchange,
        metrics: Sequence[str],
        period_type: PeriodType,
        primary_source: str,
        secondary_source: str,
        as_of: datetime,
    ) -> ReconciliationReport:
        """Both sides resolved to the same ``as_of``; every break is persisted for audit.

        ``primary_source`` is the regulatory filing (XBRL); its value is kept whenever sources disagree.
        """
        if primary_source == secondary_source:
            raise ValueError("primary and secondary sources must differ")
        await self._securities.get(symbol, exchange)
        primary = await self._pit.fetch_as_of([symbol], exchange, metrics, period_type, primary_source, as_of)
        secondary = await self._pit.fetch_as_of([symbol], exchange, metrics, period_type, secondary_source, as_of)
        report = self._reconciler.reconcile(
            primary, secondary, primary_source=primary_source, secondary_source=secondary_source
        )
        await self._breaks.record_breaks(
            symbol, exchange, as_of, primary_source, secondary_source, report.to_frame()
        )
        return report

    async def run_line_items(
        self,
        symbol: str,
        exchange: Exchange,
        period_type: PeriodType,
        primary_source: str,
        secondary_source: str,
        as_of: datetime,
    ) -> ReconciliationReport:
        """Revenue, EBITDA, debt, cash, minority interest, preferred and diluted shares."""
        return await self.run(
            symbol, exchange, RECONCILED_LINE_ITEMS, period_type, primary_source, secondary_source, as_of
        )
