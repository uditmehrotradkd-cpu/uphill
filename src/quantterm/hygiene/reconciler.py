"""Line-item reconciliation: primary regulatory XBRL (source A) vs secondary vendor API (source B).

Every item is classified by ``reconciliation.reconcile`` (match / variance / scale / sign / currency /
missing). Within 0.5 % the item scores 1.0. Beyond it a discrepancy alert is raised, the XBRL value
is kept, and the confidence is degraded according to how badly and in what way the sources disagree.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from quantterm.connectors.base import Metric, metric_currency
from quantterm.models import FundamentalFact, PeriodType
from quantterm.pit import facts_to_frame
from quantterm.reconciliation import ReconStatus, reconcile

log = logging.getLogger(__name__)

# Inputs to Bloomberg-style EV and per-share valuation.
RECONCILED_LINE_ITEMS: tuple[str, ...] = (
    Metric.REVENUE.value,
    Metric.EBITDA.value,
    Metric.TOTAL_DEBT.value,
    Metric.CASH_AND_EQUIVALENTS.value,
    Metric.MINORITY_INTEREST.value,
    Metric.PREFERRED_STOCK.value,
    Metric.DILUTED_SHARES.value,
)


class Resolution(StrEnum):
    CONSENSUS = "CONSENSUS"  # sources agree; primary value used
    PRIMARY_OVERRIDE = "PRIMARY_OVERRIDE"  # sources disagree; primary XBRL value kept
    PRIMARY_UNVERIFIED = "PRIMARY_UNVERIFIED"  # secondary has no value to check against
    SECONDARY_FALLBACK = "SECONDARY_FALLBACK"  # primary filing lacks the item


class ConfidencePolicy(BaseModel):
    """Confidence for a variance breach decays with each decade of variance beyond the tolerance:
    ``1 - decay_per_decade * log10(variance / tolerance)``, clipped to [breach_floor, breach_ceiling].
    With the defaults: 1 % -> 0.925, 5 % -> 0.75, >= 50 % -> 0.5."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tolerance: float = Field(default=0.005, gt=0)
    breach_ceiling: float = Field(default=0.95, gt=0, lt=1)
    breach_floor: float = Field(default=0.5, ge=0, lt=1)
    decay_per_decade: float = Field(default=0.25, gt=0)
    # A clean power-of-ten ratio is a vendor unit error; the filing itself is not in doubt.
    scale_mismatch: float = Field(default=0.9, ge=0, le=1)
    sign_mismatch: float = Field(default=0.5, ge=0, le=1)
    currency_mismatch: float = Field(default=0.5, ge=0, le=1)
    primary_unverified: float = Field(default=0.9, ge=0, le=1)
    secondary_fallback: float = Field(default=0.6, ge=0, le=1)

    def score(self, status: ReconStatus, relative_variance: float | None) -> float:
        if status is ReconStatus.MATCH:
            return 1.0
        if status is ReconStatus.VARIANCE_BREACH:
            if relative_variance is None or not math.isfinite(relative_variance):
                return self.breach_floor  # primary is zero and the vendor is not
            raw = 1.0 - self.decay_per_decade * math.log10(relative_variance / self.tolerance)
            return min(self.breach_ceiling, max(self.breach_floor, raw))
        return {
            ReconStatus.SCALE_MISMATCH: self.scale_mismatch,
            ReconStatus.SIGN_MISMATCH: self.sign_mismatch,
            ReconStatus.CURRENCY_MISMATCH: self.currency_mismatch,
            ReconStatus.MISSING_SECONDARY: self.primary_unverified,
            ReconStatus.MISSING_PRIMARY: self.secondary_fallback,
        }[status]


class ReconciledLineItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    symbol: str
    metric: str
    period_type: PeriodType
    period_end_date: date
    primary_value: Decimal | None
    secondary_value: Decimal | None
    resolved_value: Decimal
    currency: str
    filing_timestamp: datetime  # knowledge time of the resolved value
    relative_variance: float | None
    status: ReconStatus
    resolution: Resolution
    confidence_score: float
    primary_source: str
    secondary_source: str

    @property
    def is_break(self) -> bool:
        return self.status is not ReconStatus.MATCH


class DiscrepancyAlert(BaseModel):
    model_config = ConfigDict(frozen=True)

    item: ReconciledLineItem
    message: str


class ReconciliationReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    primary_source: str
    secondary_source: str
    items: tuple[ReconciledLineItem, ...]
    alerts: tuple[DiscrepancyAlert, ...]

    @property
    def breaks(self) -> int:
        return sum(i.is_break for i in self.items)

    def resolved(self, metric: str, period_end_date: date | None = None) -> ReconciledLineItem | None:
        """The item for ``period_end_date``, or for the latest period when omitted."""
        matches = [i for i in self.items if i.metric == metric
                   and (period_end_date is None or i.period_end_date == period_end_date)]
        return max(matches, key=lambda i: i.period_end_date) if matches else None

    def to_frame(self) -> pl.DataFrame:
        """Shape expected by ``ReconciliationRepository.record_breaks``."""
        return pl.DataFrame(
            {
                "symbol": [i.symbol for i in self.items],
                "metric": [i.metric for i in self.items],
                "period_type": [i.period_type.value for i in self.items],
                "period_end_date": [i.period_end_date for i in self.items],
                "value_primary": [None if i.primary_value is None else float(i.primary_value) for i in self.items],
                "value_secondary": [
                    None if i.secondary_value is None else float(i.secondary_value) for i in self.items
                ],
                "resolved_value": [float(i.resolved_value) for i in self.items],
                "relative_variance": [i.relative_variance for i in self.items],
                "status": [i.status.value for i in self.items],
                "is_break": [i.is_break for i in self.items],
                "confidence_score": [i.confidence_score for i in self.items],
            },
            schema={
                "symbol": pl.Utf8, "metric": pl.Utf8, "period_type": pl.Utf8, "period_end_date": pl.Date,
                "value_primary": pl.Float64, "value_secondary": pl.Float64, "resolved_value": pl.Float64,
                "relative_variance": pl.Float64, "status": pl.Utf8, "is_break": pl.Boolean,
                "confidence_score": pl.Float64,
            },
        )


_FactKey = tuple[str, str, str, date]


def _single_source(facts: Sequence[FundamentalFact], side: str) -> str | None:
    sources = {f.source for f in facts}
    if len(sources) > 1:
        raise ValueError(f"{side} side mixes sources {sorted(sources)}; reconcile one source per side")
    return next(iter(sources), None)


def _index(facts: Sequence[FundamentalFact], side: str) -> dict[_FactKey, FundamentalFact]:
    out: dict[_FactKey, FundamentalFact] = {}
    for f in facts:
        key = (f.symbol, f.metric, f.period_type.value, f.period_end_date)
        if key in out:
            raise ValueError(f"{side} side has two values for {key}; resolve point-in-time first")
        out[key] = f
    return out


class FundamentalReconciler:
    def __init__(
        self,
        policy: ConfidencePolicy | None = None,
        line_items: Sequence[str] | None = RECONCILED_LINE_ITEMS,
        on_alert: Callable[[DiscrepancyAlert], None] | None = None,
    ) -> None:
        """``line_items=None`` reconciles every metric present on either side."""
        self.policy = policy or ConfidencePolicy()
        self._items = None if line_items is None else frozenset(line_items)
        self._on_alert = on_alert

    def reconcile(
        self,
        primary: Sequence[FundamentalFact],
        secondary: Sequence[FundamentalFact],
        *,
        primary_source: str | None = None,
        secondary_source: str | None = None,
    ) -> ReconciliationReport:
        """Both sides must already be point-in-time resolved to the same ``as_of``."""
        if self._items is not None:
            primary = [f for f in primary if f.metric in self._items]
            secondary = [f for f in secondary if f.metric in self._items]
        a_source = primary_source or _single_source(primary, "primary") or "PRIMARY"
        b_source = secondary_source or _single_source(secondary, "secondary") or "SECONDARY"
        a, b = _index(primary, "primary"), _index(secondary, "secondary")

        frame = reconcile(facts_to_frame(primary), facts_to_frame(secondary), tolerance=self.policy.tolerance)
        items: list[ReconciledLineItem] = []
        alerts: list[DiscrepancyAlert] = []
        for row in frame.iter_rows(named=True):
            key: _FactKey = (row["symbol"], row["metric"], row["period_type"], row["period_end_date"])
            pa, sb = a.get(key), b.get(key)
            status = ReconStatus(row["status"])
            chosen = pa if pa is not None else sb
            assert chosen is not None
            if status is ReconStatus.MATCH:
                resolution = Resolution.CONSENSUS
            elif pa is None:
                resolution = Resolution.SECONDARY_FALLBACK
            elif sb is None:
                resolution = Resolution.PRIMARY_UNVERIFIED
            else:
                resolution = Resolution.PRIMARY_OVERRIDE
            item = ReconciledLineItem(
                symbol=key[0],
                metric=key[1],
                period_type=PeriodType(key[2]),
                period_end_date=key[3],
                primary_value=pa.value if pa else None,
                secondary_value=sb.value if sb else None,
                resolved_value=chosen.value,
                filing_timestamp=chosen.filing_timestamp,
                currency=chosen.currency,
                relative_variance=row["relative_variance"],
                status=status,
                resolution=resolution,
                confidence_score=self.policy.score(status, row["relative_variance"]),
                primary_source=a_source,
                secondary_source=b_source,
            )
            items.append(item)
            if item.is_break and resolution is not Resolution.PRIMARY_UNVERIFIED:
                alerts.append(self._alert(item))
            elif resolution is Resolution.PRIMARY_UNVERIFIED:
                log.info("%s %s %s: no %s value to verify against", item.symbol, item.metric,
                         item.period_end_date, b_source)
        return ReconciliationReport(
            primary_source=a_source, secondary_source=b_source, items=tuple(items), alerts=tuple(alerts)
        )

    def reconcile_line_items(
        self,
        symbol: str,
        period_type: PeriodType,
        period_end_date: date,
        primary: Mapping[str, Decimal],
        secondary: Mapping[str, Decimal],
        *,
        currency: str,
        as_of: datetime,
        primary_source: str = "XBRL",
        secondary_source: str = "API",
    ) -> ReconciliationReport:
        """One period's line items keyed by metric name; ``as_of`` is when both values were known."""

        def facts(values: Mapping[str, Decimal], source: str) -> list[FundamentalFact]:
            return [
                FundamentalFact(
                    symbol=symbol, metric=metric, period_type=period_type, period_end_date=period_end_date,
                    filing_timestamp=as_of, value=value, currency=metric_currency(metric, currency), source=source,
                )
                for metric, value in values.items()
            ]

        return self.reconcile(facts(primary, primary_source), facts(secondary, secondary_source),
                              primary_source=primary_source, secondary_source=secondary_source)

    def _alert(self, item: ReconciledLineItem) -> DiscrepancyAlert:
        variance = "n/a" if item.relative_variance is None else f"{item.relative_variance:.4%}"
        message = (
            f"{item.symbol} {item.metric} {item.period_type.value} {item.period_end_date}: {item.status.value} "
            f"{item.primary_source}={item.primary_value} {item.secondary_source}={item.secondary_value} "
            f"variance={variance} -> {item.resolution.value} {item.resolved_value} "
            f"(confidence {item.confidence_score:.3f})"
        )
        log.warning("reconciliation discrepancy: %s", message)
        alert = DiscrepancyAlert(item=item, message=message)
        if self._on_alert is not None:
            self._on_alert(alert)
        return alert
