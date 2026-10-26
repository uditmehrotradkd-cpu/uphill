"""Deterministic analytics views behind the terminal functions GP, FA, RV and OVME.

Every number leaves this module as a ``Datum`` carrying where it came from (source system and the
producing function), how far it can be trusted (``confidence_score``) and when it became knowable
(``timestamp``: filing / dissemination / session-close time). The command router, the TUI and the AI
agent all consume these views, so a figure shown on screen and a figure cited by the agent are the
same object. All reads are point-in-time at the requested ``as_of``.
"""

from __future__ import annotations

import asyncio
import math
import statistics
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from quantterm.connectors.base import Metric
from quantterm.db.repositories import (
    CorporateActionRepository,
    Database,
    MacroRepository,
    OptionQuote,
    OptionQuoteRepository,
    OutlierTickRepository,
    SecurityRecord,
    SecurityRepository,
)
from quantterm.engine.adjustments import get_adjusted_series
from quantterm.engine.pit_query import (
    MetricSnapshot,
    fetch_latest_snapshot,
    fetch_pit_snapshot,
    fetch_revision_history,
)
from quantterm.corporate_actions import CAPITAL_CHANGE_TYPES, CorporateActionType
from quantterm.hygiene.reconciler import FundamentalReconciler, ReconciledLineItem, ReconciliationReport
from quantterm.models import Exchange, Market, PeriodType
from quantterm.pit import PointInTimeFundamentalsRepository
from quantterm.quant.derivatives_ovme import (
    ExerciseStyle,
    MarketInputs,
    OptionContract,
    OptionType,
    OptionValuation,
    OptionValuationEngine,
)
from quantterm.quant.equity_metrics import (
    ReturnFrequency,
    calculate_institutional_ev,
    compute_beta,
    tangible_book_value,
)
from quantterm.quant.technicals import macd, realized_volatility, rsi
from quantterm.settings import Settings

DERIVED = "DERIVED"
PRICES = "raw_market_bars"
ABSENT_CONFIDENCE = 0.9  # a component no source reports (e.g. preferred stock) is taken as zero

FA_METRICS: tuple[str, ...] = (
    Metric.REVENUE, Metric.EBITDA, Metric.OPERATING_INCOME, Metric.NET_INCOME, Metric.EPS_DILUTED,
    Metric.TOTAL_ASSETS, Metric.TOTAL_EQUITY, Metric.TOTAL_DEBT, Metric.CASH_AND_EQUIVALENTS,
    Metric.DILUTED_SHARES,
)
EV_LINE_ITEMS: tuple[str, ...] = (
    Metric.REVENUE, Metric.EBITDA, Metric.TOTAL_DEBT, Metric.LEASE_LIABILITIES, Metric.CASH_AND_EQUIVALENTS,
    Metric.MINORITY_INTEREST, Metric.PREFERRED_STOCK, Metric.DILUTED_SHARES, Metric.TOTAL_EQUITY,
    Metric.GOODWILL, Metric.INTANGIBLE_ASSETS, Metric.NET_INCOME, Metric.EPS_DILUTED,
)
RV_COLUMNS: tuple[str, ...] = ("Price", "Mkt Cap", "EV", "EV/EBITDA", "EV/Sales", "P/E", "P/TBV", "Beta (adj)")


class WorkspaceError(RuntimeError):
    """The request is valid but the data needed to answer it is not in the store."""


class Datum(BaseModel):
    model_config = ConfigDict(frozen=True)

    label: str
    value: float | None
    unit: str = ""
    source: str
    confidence_score: float = Field(ge=0, le=1)
    timestamp: datetime | None
    provenance: str
    note: str | None = None


def derive(label: str, value: float | None, unit: str, inputs: Sequence[Datum], provenance: str,
           note: str | None = None) -> Datum:
    """A number computed from other data: as trustworthy as its weakest input, knowable once all are."""
    stamps = [d.timestamp for d in inputs if d.timestamp is not None]
    missing = [d.label for d in inputs if d.value is None]
    if missing:
        value, note = None, f"unavailable: no {', '.join(missing)}"
    return Datum(
        label=label, value=value, unit=unit, source=DERIVED,
        confidence_score=min((d.confidence_score for d in inputs), default=1.0),
        timestamp=max(stamps) if stamps else None, provenance=provenance, note=note,
    )


def _ratio(a: Datum, b: Datum) -> float | None:
    if a.value is None or b.value is None or b.value == 0:
        return None
    return a.value / b.value


def format_value(d: Datum) -> str:
    """Display form shared by the TUI and the agent renderer, so both show the same figure."""
    v = d.value
    if v is None:
        return "n/a"
    if d.unit == "%":
        return f"{v:,.2f}%"
    if d.unit == "x":
        return f"{v:,.2f}x"
    if d.unit == "MAD":
        return f"{v:,.1f} MAD"
    if d.unit == "shares" or len(d.unit) == 3:
        a = abs(v)
        body = (f"{v / 1e12:,.2f}tn" if a >= 1e12 else f"{v / 1e9:,.2f}bn" if a >= 1e9
                else f"{v / 1e6:,.2f}mn" if a >= 1e6 else f"{v:,.2f}")
        return f"{body} {d.unit}"
    return f"{v:,.4f}" if abs(v) < 10 else f"{v:,.2f}"


def reconciled_datum(it: ReconciledLineItem, label: str) -> Datum:
    variance = "" if it.relative_variance is None else f"; variance {it.relative_variance:.2%}"
    return Datum(label=label, value=float(it.resolved_value), unit="shares" if it.currency == "XXX" else it.currency,
                 source=it.primary_source if it.primary_value is not None else it.secondary_source,
                 confidence_score=it.confidence_score, timestamp=it.filing_timestamp,
                 provenance="hygiene.reconciler.FundamentalReconciler",
                 note=f"period {it.period_end_date}; {it.status.value}/{it.resolution.value}{variance}")


def _session_close(exchange: Exchange, session: date) -> datetime:
    return datetime.combine(session, exchange.session_close, tzinfo=exchange.timezone)


def _parse_listing(ref: str) -> tuple[Exchange, str]:
    exchange, _, ticker = ref.partition(":")
    return Exchange(exchange), ticker


def strike_step(spot: float) -> float:
    for limit, step in ((25, 0.5), (100, 1.0), (250, 2.5), (500, 5.0), (1000, 10.0), (2500, 20.0), (5000, 50.0)):
        if spot < limit:
            return step
    return 100.0


# ------------------------------------------------------------------------------------------ views
class GPView(BaseModel):
    model_config = ConfigDict(frozen=True)

    security: SecurityRecord
    as_of: datetime
    adjust_splits: bool
    adjust_dividends: bool
    dates: list[date]
    close: list[float]
    volume: list[float]
    rsi: list[float | None]
    macd: list[float | None]
    signal: list[float | None]
    summary: list[Datum]


class FARow(BaseModel):
    model_config = ConfigDict(frozen=True)

    label: str
    cells: list[Datum]


class FAView(BaseModel):
    model_config = ConfigDict(frozen=True)

    security: SecurityRecord
    as_of: datetime
    period_type: PeriodType
    periods: list[date]
    rows: list[FARow]
    ratios: list[FARow]
    ev_table: list[Datum]
    reconciliation: list[ReconciledLineItem]
    notes: list[str]


class RVRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    security: SecurityRecord
    is_subject: bool
    cells: dict[str, Datum]


class RVView(BaseModel):
    model_config = ConfigDict(frozen=True)

    security: SecurityRecord
    industry: str
    as_of: datetime
    columns: list[str]
    rows: list[RVRow]
    medians: dict[str, Datum]


class OVMERequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    option_type: OptionType | None = None
    strike: float | None = Field(default=None, gt=0)
    expiry: date | None = None
    volatility: float | None = Field(default=None, gt=0, le=5)
    rate: float | None = Field(default=None, ge=-0.05, le=1)
    dividend_yield: float | None = Field(default=None, ge=0, le=1)
    premium: float | None = Field(default=None, gt=0)


class QuotedLeg(BaseModel):
    model_config = ConfigDict(frozen=True)

    bid: float
    ask: float
    mid: float
    implied_vol: float | None  # None when the mid is outside no-arbitrage bounds
    timestamp: datetime
    source: str


class ChainRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    strike: float
    call: OptionValuation
    put: OptionValuation
    call_quote: QuotedLeg | None = None
    put_quote: QuotedLeg | None = None


class OVMEView(BaseModel):
    model_config = ConfigDict(frozen=True)

    security: SecurityRecord
    as_of: datetime
    expiry: date
    exercise: ExerciseStyle
    inputs: list[Datum]
    chain: list[ChainRow]
    calculator: OptionValuation | None
    calculator_contract: OptionContract | None
    notes: list[str]


# -------------------------------------------------------------------------------------- workspace
class Workspace:
    def __init__(self, db: Database, settings: Settings, *, clock: Callable[[], datetime] = lambda: datetime.now(UTC)):
        self._db = db
        self._settings = settings
        self.clock = clock
        self.securities = SecurityRepository(db)
        self._pit = PointInTimeFundamentalsRepository(db)
        self._macro = MacroRepository(db)
        self._actions = CorporateActionRepository(db, self.securities)
        self._outliers = OutlierTickRepository(db)
        self._quotes = OptionQuoteRepository(db, self.securities)
        self._engines: dict[Exchange, OptionValuationEngine] = {}

    def engine(self, exchange: Exchange) -> OptionValuationEngine:
        # Per exchange, not per market: NSE and BSE share a calendar but not expiry rules.
        engine = self._engines.get(exchange)
        if engine is None:
            engine = self._engines[exchange] = OptionValuationEngine(exchange)
        return engine

    async def resolve(self, ticker: str, exchanges: Sequence[Exchange]) -> SecurityRecord:
        return await self.securities.resolve(ticker, exchanges)

    # ------------------------------------------------------------------ prices
    async def daily_bars(
        self, sec: SecurityRecord, start: date, as_of: datetime, *, adjust_splits: bool = True,
        adjust_dividends: bool = True,
    ) -> pl.DataFrame:
        end = as_of.astimezone(sec.exchange.timezone).date()
        return await get_adjusted_series(
            sec.symbol, start, end, as_of, adjust_splits, adjust_dividends, exchange=sec.exchange, db=self._db
        )

    async def last_price(self, sec: SecurityRecord, as_of: datetime) -> Datum:
        start = as_of.astimezone(sec.exchange.timezone).date() - timedelta(days=14)
        bars = await self.daily_bars(sec, start, as_of, adjust_dividends=False)
        if bars.is_empty():
            return Datum(label="Last price", value=None, unit=sec.currency, source=PRICES, confidence_score=0.0,
                         timestamp=None, provenance="engine.adjustments.get_adjusted_series",
                         note="no completed session in the 14 days before as-of")
        session = bars["session_date"][-1]
        return Datum(label="Last price", value=float(bars["close"][-1]), unit=sec.currency, source=PRICES,
                     confidence_score=1.0, timestamp=_session_close(sec.exchange, session),
                     provenance="engine.adjustments.get_adjusted_series", note=f"close {session}")

    async def gp(
        self, sec: SecurityRecord, as_of: datetime, *, lookback_days: int = 365, adjust_splits: bool = True,
        adjust_dividends: bool = True,
    ) -> GPView:
        local = as_of.astimezone(sec.exchange.timezone).date()
        # Extra history so RSI/MACD are already warmed up at the start of the visible window.
        bars = await self.daily_bars(sec, local - timedelta(days=lookback_days + 60), as_of,
                                     adjust_splits=adjust_splits, adjust_dividends=adjust_dividends)
        if bars.is_empty():
            raise WorkspaceError(f"no price history for {sec.exchange.value}:{sec.symbol} before {local}")
        close = bars["close"].to_numpy()
        r, m, s, _ = rsi(close), *macd(close)
        visible = bars["session_date"] >= local - timedelta(days=lookback_days)
        idx = [i for i, v in enumerate(visible.to_list()) if v]
        last = bars.height - 1
        stamp = _session_close(sec.exchange, bars["session_date"][last])
        basis = ("split/bonus" if adjust_splits else "") + (" + dividend" if adjust_dividends else "")
        prov = "engine.adjustments.get_adjusted_series"

        def nan(x: float) -> float | None:
            return None if math.isnan(x) else float(x)

        def datum(label: str, value: float | None, unit: str, provenance: str, note: str | None = None) -> Datum:
            return Datum(label=label, value=value, unit=unit, source=PRICES, confidence_score=1.0, timestamp=stamp,
                         provenance=provenance, note=note)

        first_close = float(close[idx[0]])
        window_close = close[idx[0]:]
        summary = [
            datum("Last close", float(close[last]), sec.currency, prov, f"adjusted: {basis.strip() or 'none'}"),
            datum("Period change", (float(close[last]) / first_close - 1) * 100, "%", prov,
                  f"since {bars['session_date'][idx[0]]}"),
            datum("Period high", float(window_close.max()), sec.currency, prov),
            datum("Period low", float(window_close.min()), sec.currency, prov),
            datum("RSI(14)", nan(r[last]), "", "quant.technicals.rsi", "Wilder smoothing"),
            datum("MACD(12,26)", nan(m[last]), sec.currency, "quant.technicals.macd"),
            datum("MACD signal(9)", nan(s[last]), sec.currency, "quant.technicals.macd"),
        ]
        return GPView(
            security=sec, as_of=as_of, adjust_splits=adjust_splits, adjust_dividends=adjust_dividends,
            dates=[bars["session_date"][i] for i in idx], close=[float(close[i]) for i in idx],
            volume=[float(bars["volume"][i]) for i in idx], rsi=[nan(r[i]) for i in idx],
            macd=[nan(m[i]) for i in idx], signal=[nan(s[i]) for i in idx], summary=summary,
        )

    # ------------------------------------------------------------------ fundamentals
    @staticmethod
    def _fact_datum(s: MetricSnapshot, label: str | None = None) -> Datum:
        return Datum(
            label=label or s.metric_name, value=float(s.metric_value),
            unit="shares" if s.currency == "XXX" else s.currency, source=s.source_authority,
            confidence_score=float(s.confidence_score), timestamp=s.filing_timestamp,
            provenance="engine.pit_query.fetch_pit_snapshot",
            note=f"period {s.period_end_date} rev {s.revision_number}",
        )

    def _primary_source(self, exchange: Exchange) -> str:
        return self._settings.fundamentals_primary_sources.get(exchange.value, f"{exchange.value}_XBRL")

    async def reconcile(self, sec: SecurityRecord, as_of: datetime, period_type: PeriodType) -> ReconciliationReport:
        """Primary regulatory XBRL vs vendor, read-only (audit persistence is ReconciliationService's job)."""
        primary_source = self._primary_source(sec.exchange)
        secondary_source = self._settings.fundamentals_secondary_source
        primary = await self._pit.fetch_as_of([sec.symbol], sec.exchange, EV_LINE_ITEMS, period_type,
                                              primary_source, as_of)
        secondary = await self._pit.fetch_as_of([sec.symbol], sec.exchange, EV_LINE_ITEMS, period_type,
                                                secondary_source, as_of)
        return FundamentalReconciler(line_items=EV_LINE_ITEMS).reconcile(
            primary, secondary, primary_source=primary_source, secondary_source=secondary_source
        )

    async def enterprise_value(
        self, sec: SecurityRecord, as_of: datetime, period_type: PeriodType = PeriodType.FISCAL_YEAR
    ) -> tuple[list[Datum], ReconciliationReport]:
        report = await self.reconcile(sec, as_of, period_type)
        price = await self.last_price(sec, as_of)

        def item(metric: str, label: str, required: bool = False) -> Datum:
            it = report.resolved(metric)
            if it is None:
                return Datum(label=label, value=None if required else 0.0, unit=sec.currency,
                             source=DERIVED, confidence_score=0.0 if required else ABSENT_CONFIDENCE,
                             timestamp=None, provenance="hygiene.reconciler.FundamentalReconciler",
                             note="not reported by any source" + ("" if required else "; treated as 0"))
            return reconciled_datum(it, label)

        shares = (await self.capital_adjuster(sec, as_of))(item(Metric.DILUTED_SHARES, "Diluted shares", required=True),
                                                           False)
        debt = item(Metric.TOTAL_DEBT, "Total debt")
        leases = item(Metric.LEASE_LIABILITIES, "Lease liabilities")
        preferred = item(Metric.PREFERRED_STOCK, "Preferred stock")
        minority = item(Metric.MINORITY_INTEREST, "Minority interest")
        cash = item(Metric.CASH_AND_EQUIVALENTS, "Cash & ST investments")
        equity = item(Metric.TOTAL_EQUITY, "Common equity", required=True)
        goodwill = item(Metric.GOODWILL, "Goodwill")
        intangibles = item(Metric.INTANGIBLE_ASSETS, "Other intangibles")
        ebitda = item(Metric.EBITDA, "EBITDA", required=True)
        revenue = item(Metric.REVENUE, "Revenue", required=True)
        mcap = derive("Market cap", None if price.value is None or shares.value is None else price.value * shares.value,
                      sec.currency, [price, shares], "workspace.enterprise_value",
                      "last price x diluted weighted-average shares")

        financial = sec.sector_class.is_financial
        components = [mcap] + ([preferred, minority] if financial else [debt, leases, preferred, minority, cash])
        tbv_inputs = [equity, goodwill, intangibles]
        if mcap.value is None or (financial and equity.value is None):
            ev = derive("Enterprise value", None, sec.currency, components + (tbv_inputs if financial else []),
                        "quant.equity_metrics.calculate_institutional_ev")
            ptbv = derive("P/TBV", None, "x", [mcap] + tbv_inputs, "quant.equity_metrics.calculate_institutional_ev")
        else:
            def dec(d: Datum) -> Decimal:
                return Decimal(repr(d.value if d.value is not None else 0.0))

            tbv = (tangible_book_value(dec(equity), dec(goodwill), dec(intangibles))
                   if equity.value is not None else None)
            result = calculate_institutional_ev(
                market_cap=dec(mcap), total_debt=dec(debt), lease_liabilities=dec(leases),
                preferred_stock=dec(preferred), minority_interest=dec(minority),
                cash_and_st_investments=dec(cash), sector=sec.sector_class, tangible_book_value=tbv,
            )
            basis = "bank/NBFC basis: debt, leases and cash excluded" if financial else None
            ev = derive("Enterprise value", float(result.enterprise_value), sec.currency, components,
                        "quant.equity_metrics.calculate_institutional_ev", basis)
            ptbv = derive("P/TBV", None if result.price_to_tangible_book is None else float(result.price_to_tangible_book),
                          "x", [mcap] + tbv_inputs, "quant.equity_metrics.calculate_institutional_ev")
        multiples = [ptbv]
        if not financial:
            multiples = [
                derive("EV/EBITDA", _ratio(ev, ebitda) if (ebitda.value or 0) > 0 else None, "x", [ev, ebitda],
                       "workspace.enterprise_value"),
                derive("EV/Sales", _ratio(ev, revenue) if (revenue.value or 0) > 0 else None, "x", [ev, revenue],
                       "workspace.enterprise_value"),
                ptbv,
            ]
        table = [price, shares, mcap] + ([] if financial else [debt, leases, cash]) + [preferred, minority, ev, *multiples]
        return table, report

    async def capital_adjuster(self, sec: SecurityRecord, as_of: datetime) -> Callable[[Datum, bool], Datum]:
        """Restate a filed share count / per-share figure for capital changes after its filing.

        Filings already reflect splits effective before issuance (retrospective restatement), so only
        split/bonus/rights ex-dates after the filing date and on or before ``as_of`` apply, using the same
        factors the price engine applies. The stored figure is never modified.
        """
        local = as_of.astimezone(sec.exchange.timezone).date()
        changes = [a for a in await self._actions.fetch_known(sec.symbol, sec.exchange, as_of)
                   if a.action_type in CAPITAL_CHANGE_TYPES and a.adjustment_factor is not None and a.ex_date <= local]

        def adjust(d: Datum, per_share: bool) -> Datum:
            if d.value is None or d.timestamp is None:
                return d
            filed = d.timestamp.astimezone(sec.exchange.timezone).date()
            applied = [a for a in changes if a.ex_date > filed]
            if not applied:
                return d
            factor = math.prod(float(a.adjustment_factor or 1) for a in applied)
            value = d.value * factor if per_share else d.value / factor
            kinds = ", ".join(f"{a.action_type.value.lower()} {a.ex_date}" for a in applied)
            return d.model_copy(update={
                "value": value, "timestamp": max([d.timestamp, *(a.announced_at for a in applied)]),
                "note": f"{d.note or ''}; adjusted for {kinds} (factor {factor:.6g})".lstrip("; "),
            })

        return adjust

    async def fa(self, sec: SecurityRecord, as_of: datetime, *, period_type: PeriodType = PeriodType.FISCAL_YEAR,
                 periods: int = 5) -> FAView:
        span_days = (periods + 1) * (366 if period_type is PeriodType.FISCAL_YEAR else 92)
        snaps = await fetch_pit_snapshot(
            self._db, [sec.symbol], sec.exchange, FA_METRICS, period_type, as_of, None,
            periods_from=as_of.date() - timedelta(days=span_days),
        )
        if not snaps:
            raise WorkspaceError(f"no {period_type.value} fundamentals for {sec.exchange.value}:{sec.symbol} as of {as_of}")
        ends = sorted({s.period_end_date for s in snaps}, reverse=True)[:periods]
        by_key = {(s.metric_name, s.period_end_date): s for s in snaps}
        adjust = await self.capital_adjuster(sec, as_of)
        per_share: dict[str, bool] = {Metric.EPS_DILUTED: True, Metric.DILUTED_SHARES: False}

        def cell(metric: str, end: date) -> Datum:
            s = by_key.get((metric, end))
            if s is None:
                return Datum(label=metric, value=None, source=DERIVED, confidence_score=0.0, timestamp=None,
                             provenance="engine.pit_query.fetch_pit_snapshot", note="not filed")
            d = self._fact_datum(s)
            return adjust(d, per_share[metric]) if metric in per_share else d

        rows = [FARow(label=m, cells=[cell(m, e) for e in ends]) for m in FA_METRICS]

        def ratio_row(label: str, num: Callable[[date], Datum], den: str, scale: float, unit: str) -> FARow:
            cells = []
            for e in ends:
                n, d = num(e), cell(den, e)
                value = _ratio(n, d)
                cells.append(derive(label, None if value is None else value * scale, unit, [n, d], "workspace.fa"))
            return FARow(label=label, cells=cells)

        def metric(name: str) -> Callable[[date], Datum]:
            return lambda e: cell(name, e)

        def net_debt(e: date) -> Datum:
            debt, cash = cell(Metric.TOTAL_DEBT, e), cell(Metric.CASH_AND_EQUIVALENTS, e)
            value = None if debt.value is None or cash.value is None else debt.value - cash.value
            return derive("Net debt", value, sec.currency, [debt, cash], "workspace.fa")

        ratios = [
            ratio_row("EBITDA margin", metric(Metric.EBITDA), Metric.REVENUE, 100.0, "%"),
            ratio_row("Net margin", metric(Metric.NET_INCOME), Metric.REVENUE, 100.0, "%"),
            ratio_row("ROE", metric(Metric.NET_INCOME), Metric.TOTAL_EQUITY, 100.0, "%"),
            ratio_row("Net debt / EBITDA", net_debt, Metric.EBITDA, 1.0, "x"),
        ]
        ev_table, report = await self.enterprise_value(sec, as_of, period_type)
        notes = [a.message for a in report.alerts]
        return FAView(security=sec, as_of=as_of, period_type=period_type, periods=ends, rows=rows, ratios=ratios,
                      ev_table=ev_table, reconciliation=list(report.items), notes=notes)

    # ------------------------------------------------------------------ audit trails
    async def revisions(self, sec: SecurityRecord, metric: str, period_type: PeriodType, period_end: date,
                        as_of: datetime, source: str | None = None) -> list[Datum]:
        """Every vintage of one reported figure that was public at ``as_of``, oldest first."""
        authority = source or self._primary_source(sec.exchange)
        snaps = await fetch_revision_history(self._db, sec.symbol, sec.exchange, metric, period_type, period_end,
                                             authority, as_of)
        if not snaps:
            raise WorkspaceError(f"no {authority} {metric} for period {period_end} public as of {as_of}")
        return [self._fact_datum(s, f"{metric} {period_end} rev {s.revision_number}") for s in snaps]

    async def tick_hygiene(self, sec: SecurityRecord, as_of: datetime, hours: int = 24) -> list[Datum]:
        """What the MAD bad-tick filter quarantined in the ``hours`` before ``as_of``."""
        groups = await self._outliers.summary(sec.symbol, sec.exchange, as_of - timedelta(hours=hours), as_of)
        prov = "hygiene.tick_filter.TickHygieneFilter"
        out = [Datum(label=f"Rejected ticks ({hours}h)", value=float(sum(g.rejected for g in groups)), source="outlier_ticks",
                     confidence_score=1.0, timestamp=max((g.last_rejected_at for g in groups), default=None),
                     provenance=prov, note=", ".join(f"{g.reason}: {g.rejected}" for g in groups) or "none")]
        for g in groups:
            out.append(Datum(label=f"Max deviation, {g.reason}", value=g.max_deviation_mads, unit="MAD",
                             source="outlier_ticks", confidence_score=1.0, timestamp=g.last_rejected_at,
                             provenance=prov, note=f"{g.rejected} prints"))
        return out

    # ------------------------------------------------------------------ beta
    async def beta(self, sec: SecurityRecord, as_of: datetime, *, years: int = 2,
                   frequency: ReturnFrequency = ReturnFrequency.WEEKLY) -> list[Datum]:
        """``BETA <GO>``: price (capital-change adjusted, not dividend) returns vs the market benchmark."""
        ref = self._settings.beta_benchmarks.get(sec.exchange.market.value)
        if ref is None:
            raise WorkspaceError(f"no beta benchmark configured for {sec.exchange.market.value}")
        bexch, bticker = _parse_listing(ref)
        bench = await self.securities.get(bticker, bexch)
        end = as_of.astimezone(sec.exchange.timezone).date()
        start = end - timedelta(days=366 * years)
        frames = []
        for s in (sec, bench):
            bars = await self.daily_bars(s, start, as_of, adjust_dividends=False)
            frames.append(bars.select(pl.col("session_date").alias("date"), "close"))
        try:
            result = compute_beta(frames[0], frames[1], frequency, start, end)
        except ValueError as exc:
            raise WorkspaceError(f"beta unavailable for {sec.symbol}: {exc}") from exc
        stamp = _session_close(sec.exchange, result.end)
        note = f"{years}Y {frequency.value.lower()} vs {ref}, n={result.n_observations}, {result.start}..{result.end}"
        prov = "quant.equity_metrics.compute_beta"

        def d(label: str, value: float) -> Datum:
            return Datum(label=label, value=value, source=PRICES, confidence_score=1.0, timestamp=stamp,
                         provenance=prov, note=note)

        return [d("Adjusted beta", result.adjusted_beta), d("Raw beta", result.raw_beta),
                d("Beta std error", result.beta_std_error), d("R-squared", result.r_squared)]

    # ------------------------------------------------------------------ relative valuation
    async def rv(self, sec: SecurityRecord, as_of: datetime, *, max_peers: int = 25) -> RVView:
        if not sec.industry:
            raise WorkspaceError(f"{sec.exchange.value}:{sec.symbol} has no industry; register it with --industry")
        same_market = [e for e in Exchange if e.market is sec.exchange.market]
        peers = [p for p in await self.securities.peers(sec.industry, same_market)
                 if not (p.symbol == sec.symbol and p.exchange is sec.exchange)]
        peers = [sec, *peers[:max_peers - 1]]
        rows = [RVRow(security=p, is_subject=p is sec, cells=await self._rv_cells(p, as_of)) for p in peers]

        medians: dict[str, Datum] = {}
        comps = [r for r in rows if not r.is_subject]
        for col in RV_COLUMNS[1:]:
            vals = [r.cells[col] for r in comps if r.cells[col].value is not None]
            medians[col] = derive(f"Peer median {col}", statistics.median(v.value for v in vals) if vals else None,  # type: ignore[type-var]
                                  rows[0].cells[col].unit, vals, "workspace.rv",
                                  f"{len(vals)} of {len(comps)} peers, subject excluded")
        return RVView(security=sec, industry=sec.industry, as_of=as_of, columns=list(RV_COLUMNS), rows=rows,
                      medians=medians)

    async def _rv_cells(self, sec: SecurityRecord, as_of: datetime) -> dict[str, Datum]:
        price = await self.last_price(sec, as_of)
        snaps = await fetch_latest_snapshot(self._db, [sec.symbol], sec.exchange, EV_LINE_ITEMS,
                                            PeriodType.FISCAL_YEAR, as_of)
        facts = {s.metric_name: self._fact_datum(s) for s in snaps}

        def get(metric: str, required: bool = False) -> Datum:
            if metric in facts:
                return facts[metric]
            return Datum(label=metric, value=None if required else 0.0, source=DERIVED,
                         confidence_score=0.0 if required else ABSENT_CONFIDENCE, timestamp=None,
                         provenance="engine.pit_query.fetch_latest_snapshot",
                         note="not filed" + ("" if required else "; treated as 0"))

        shares, ebitda, revenue, eps = (get(Metric.DILUTED_SHARES, True), get(Metric.EBITDA, True),
                                        get(Metric.REVENUE, True), get(Metric.EPS_DILUTED, True))
        adjust = await self.capital_adjuster(sec, as_of)
        shares, eps = adjust(shares, False), adjust(eps, True)
        equity = get(Metric.TOTAL_EQUITY, True)
        mcap = derive("Mkt Cap", None if price.value is None or shares.value is None else price.value * shares.value,
                      sec.currency, [price, shares], "workspace.rv")
        financial = sec.sector_class.is_financial
        parts = [mcap] + ([get(Metric.PREFERRED_STOCK), get(Metric.MINORITY_INTEREST)] if financial else
                          [get(Metric.TOTAL_DEBT), get(Metric.LEASE_LIABILITIES), get(Metric.PREFERRED_STOCK),
                           get(Metric.MINORITY_INTEREST), get(Metric.CASH_AND_EQUIVALENTS)])
        tbv_parts = [equity, get(Metric.GOODWILL), get(Metric.INTANGIBLE_ASSETS)]
        ev_value = ptbv_value = None
        if mcap.value is not None and (not financial or equity.value is not None):
            def dec(d: Datum) -> Decimal:
                return Decimal(repr(d.value if d.value is not None else 0.0))

            tbv = tangible_book_value(dec(tbv_parts[0]), dec(tbv_parts[1]), dec(tbv_parts[2])) \
                if equity.value is not None else None
            if financial:
                result = calculate_institutional_ev(dec(mcap), Decimal(0), Decimal(0), dec(parts[1]), dec(parts[2]),
                                                    Decimal(0), sec.sector_class, tangible_book_value=tbv)
            else:
                result = calculate_institutional_ev(dec(mcap), dec(parts[1]), dec(parts[2]), dec(parts[3]),
                                                    dec(parts[4]), dec(parts[5]), sec.sector_class,
                                                    tangible_book_value=tbv)
            ev_value = float(result.enterprise_value)
            ptbv_value = None if result.price_to_tangible_book is None else float(result.price_to_tangible_book)
        ev = derive("EV", ev_value, sec.currency, parts, "quant.equity_metrics.calculate_institutional_ev")

        def positive(d: Datum) -> bool:
            return d.value is not None and d.value > 0

        cells = {
            "Price": price,
            "Mkt Cap": mcap,
            "EV": ev,
            "EV/EBITDA": derive("EV/EBITDA", _ratio(ev, ebitda) if positive(ebitda) and not financial else None,
                                "x", [ev, ebitda], "workspace.rv"),
            "EV/Sales": derive("EV/Sales", _ratio(ev, revenue) if positive(revenue) and not financial else None,
                               "x", [ev, revenue], "workspace.rv"),
            "P/E": derive("P/E", _ratio(price, eps) if positive(eps) else None, "x", [price, eps], "workspace.rv"),
            "P/TBV": derive("P/TBV", ptbv_value, "x", [mcap, *tbv_parts], "quant.equity_metrics.calculate_institutional_ev"),
        }
        try:
            cells["Beta (adj)"] = (await self.beta(sec, as_of))[0]
        except (WorkspaceError, LookupError) as exc:
            cells["Beta (adj)"] = Datum(label="Beta (adj)", value=None, source=DERIVED, confidence_score=0.0,
                                        timestamp=None, provenance="quant.equity_metrics.compute_beta", note=str(exc))
        return cells

    @staticmethod
    def _user(label: str, value: float, now: datetime) -> Datum:
        return Datum(label=label, value=value, source="USER_INPUT", confidence_score=1.0, timestamp=now,
                     provenance="terminal command / agent tool argument")

    async def _realized_vol(self, sec: SecurityRecord, as_of: datetime, spot: Datum, reason: str) -> Datum:
        local = as_of.astimezone(sec.exchange.timezone).date()
        bars = await self.daily_bars(sec, local - timedelta(days=366), as_of, adjust_dividends=False)
        if bars.height < 30:
            raise WorkspaceError("fewer than 30 sessions for realised volatility; pass vol= explicitly")
        return Datum(label="Volatility", value=realized_volatility(bars["close"]), source=PRICES,
                     confidence_score=1.0, timestamp=spot.timestamp, provenance="quant.technicals.realized_volatility",
                     note=f"1Y realised volatility used as proxy; {reason}")

    async def _risk_free(self, sec: SecurityRecord, as_of: datetime) -> Datum:
        series = self._settings.risk_free_series.get(sec.exchange.market.value)
        frame = (await self._macro.fetch_as_of(series, as_of.date() - timedelta(days=30), as_of.date(), as_of)
                 if series else pl.DataFrame())
        if frame.is_empty():
            raise WorkspaceError(f"no risk-free observation for {series} in the 30 days to as-of; pass rate=")
        pct = float(frame["value"][-1])
        return Datum(label="Risk-free rate", value=math.log1p(pct / 100), source=series or "",
                     confidence_score=1.0, timestamp=frame["release_timestamp"][-1],
                     provenance="db.repositories.MacroRepository.fetch_as_of",
                     note=f"{pct}% annual -> continuous; observation {frame['observation_date'][-1]}")

    async def _dividend_yield(self, sec: SecurityRecord, as_of: datetime, spot: float) -> Datum:
        local = as_of.astimezone(sec.exchange.timezone).date()
        known = await self._actions.fetch_known(sec.symbol, sec.exchange, as_of)
        paid = [a for a in known if a.action_type is CorporateActionType.CASH_DIVIDEND
                and local - timedelta(days=365) < a.ex_date <= local]
        changes = [a for a in known if a.action_type in CAPITAL_CHANGE_TYPES and a.adjustment_factor is not None
                   and a.ex_date <= local]

        def per_current_share(div: Decimal, ex: date) -> float:
            # A dividend paid before a split/bonus is per old share: restate it on today's share basis.
            return float(div) * math.prod(float(c.adjustment_factor or 1) for c in changes if c.ex_date > ex)

        total = sum(per_current_share(a.raw_dividend_amount or Decimal(0), a.ex_date) for a in paid)
        return Datum(label="Dividend yield", value=math.log1p(total / spot), source="corporate_actions",
                     confidence_score=1.0, timestamp=max((a.announced_at for a in paid), default=None),
                     provenance="db.repositories.CorporateActionRepository.fetch_known",
                     note=f"trailing 12M cash dividends {total:g} {sec.currency} per current share "
                          f"over {len(paid)} ex-dates")

    async def ovme(self, sec: SecurityRecord, as_of: datetime, request: OVMERequest | None = None,
                   *, strikes_each_side: int = 5) -> OVMEView:
        request = request or OVMERequest()
        engine = self.engine(sec.exchange)
        valuation_date = as_of.astimezone(sec.exchange.timezone).date()
        now = self.clock()
        max_age = timedelta(days=self._settings.option_quote_max_age_days)
        quotes = await self._quotes.latest_chain(sec.symbol, sec.exchange, as_of - max_age, as_of, valuation_date)

        def live(expiry: date) -> bool:
            probe = OptionContract(underlying=sec.symbol, exchange=sec.exchange, option_type=OptionType.CALL,
                                   strike=1.0, expiry=expiry)
            return engine.adjusted_expiry(probe) > valuation_date

        try:
            if request.expiry is not None:
                expiry = request.expiry
            else:
                quoted = sorted({q.expiry for q in quotes if live(q.expiry)})
                expiry = quoted[0] if quoted else engine.next_monthly_expiry(valuation_date)
        except ValueError as exc:
            raise WorkspaceError(f"OVME: {exc}") from exc
        legs: dict[tuple[float, OptionType], OptionQuote] = {
            (float(q.strike), q.option_type): q for q in quotes if q.expiry == expiry}

        spot_d = await self.last_price(sec, as_of)
        if spot_d.value is None:
            raise WorkspaceError(f"no spot price for {sec.exchange.value}:{sec.symbol} as of {as_of}")
        spot = spot_d.value
        rate_d = (self._user("Risk-free rate", request.rate, now) if request.rate is not None
                  else await self._risk_free(sec, as_of))
        div_d = (self._user("Dividend yield", request.dividend_yield, now) if request.dividend_yield is not None
                 else await self._dividend_yield(sec, as_of, spot))
        assert rate_d.value is not None and div_d.value is not None
        base = MarketInputs(valuation_date=valuation_date, spot=spot, volatility=0.2, risk_free_rate=rate_d.value,
                            dividend_yield=div_d.value)

        def contract(kind: OptionType, strike: float) -> OptionContract:
            return OptionContract(underlying=sec.symbol, exchange=sec.exchange, option_type=kind, strike=strike,
                                  expiry=expiry)

        quoted_strikes = sorted({k for k, _ in legs})
        if quoted_strikes:
            nearest = sorted(quoted_strikes, key=lambda k: abs(k - spot))[: 2 * strikes_each_side + 1]
            strikes = sorted(nearest)
        else:
            step = strike_step(spot)
            centre = round(spot / step) * step
            strikes = [centre + k * step for k in range(-strikes_each_side, strikes_each_side + 1)
                       if centre + k * step > 0]
        calc_contract = (contract(request.option_type or OptionType.CALL, request.strike)
                         if request.strike is not None else None)

        def implied(kind: OptionType, strike: float) -> float | None:
            q = legs.get((strike, kind))
            if q is None:
                return None
            try:
                return engine.implied_volatility(contract(kind, strike), base, q.mid)
            except ValueError:
                return None

        def price(vol: float) -> tuple[list[ChainRow], OptionValuation | None, dict[tuple[float, OptionType], float]]:
            # CPU-bound (American finite differences, IV root-finding): runs off the event loop.
            ivs = {(k, t): iv for k in strikes for t in OptionType if (iv := implied(t, k)) is not None}
            flat = base.model_copy(update={"volatility": vol})

            def leg(kind: OptionType, strike: float) -> tuple[OptionValuation, QuotedLeg | None]:
                iv = ivs.get((strike, kind))
                q = legs.get((strike, kind))
                market = flat if iv is None else base.model_copy(update={"volatility": iv})
                quote = None if q is None else QuotedLeg(bid=q.bid, ask=q.ask, mid=q.mid, implied_vol=iv,
                                                         timestamp=q.timestamp, source=q.source)
                return engine.value(contract(kind, strike), market), quote

            rows = []
            for k in strikes:
                (call, cq), (put, pq) = leg(OptionType.CALL, k), leg(OptionType.PUT, k)
                rows.append(ChainRow(strike=k, call=call, put=put, call_quote=cq, put_quote=pq))
            calc: OptionValuation | None = None
            if calc_contract is not None:
                calc = (engine.value_from_premium(calc_contract, flat, request.premium) if request.premium is not None
                        else engine.value(calc_contract, flat))
            return rows, calc, ivs

        notes: list[str] = []
        if request.volatility is not None:
            vol_d = self._user("Volatility", request.volatility, now)
        else:
            atm = min(strikes, key=lambda k: abs(k - spot)) if strikes else None
            atm_ivs = [] if atm is None else await asyncio.to_thread(
                lambda: [iv for t in OptionType if (iv := implied(t, atm)) is not None])
            if atm_ivs:
                assert atm is not None
                used = [legs[(atm, t)] for t in OptionType if (atm, t) in legs]
                vol_d = Datum(label="Volatility", value=sum(atm_ivs) / len(atm_ivs), source=used[0].source,
                              confidence_score=1.0, timestamp=max(q.timestamp for q in used),
                              provenance="quant.derivatives_ovme.OptionValuationEngine.implied_volatility",
                              note=f"ATM {atm:g} listed mid implied vol, expiry {expiry}")
            else:
                reason = ("no usable ATM quote" if legs
                          else f"no listed option quotes in the {max_age.days} days to as-of")
                vol_d = await self._realized_vol(sec, as_of, spot_d, reason)
        assert vol_d.value is not None
        try:
            chain, calc, ivs = await asyncio.to_thread(price, vol_d.value)
        except ValueError as exc:  # e.g. a premium outside no-arbitrage bounds has no implied volatility
            raise WorkspaceError(f"OVME: {exc}") from exc
        inputs = [spot_d, vol_d, rate_d, div_d]
        notes += [d.note for d in inputs if d.note and d.source != "USER_INPUT"]
        if legs:
            notes.append(f"{len(ivs)} of {sum(1 for k in strikes for t in OptionType if (k, t) in legs)} quoted legs "
                         f"valued at their own mid implied vol; unquoted legs at the volatility input")
        notes = [d.note for d in inputs if d.note and d.source != "USER_INPUT"]
        if calc is not None and request.premium is not None:
            notes.append(f"calculator volatility implied from premium {request.premium:g}")
        return OVMEView(security=sec, as_of=as_of, expiry=chain[0].call.adjusted_expiry if chain else expiry,
                        exercise=chain[0].call.exercise if chain else ExerciseStyle.EUROPEAN, inputs=inputs,
                        chain=chain, calculator=calc, calculator_contract=calc_contract, notes=notes)


def option_datums(v: OptionValuation, currency: str, inputs: Iterable[Datum], label: str) -> list[Datum]:
    """Greeks as Datums: as trustworthy as the weakest market input, knowable when all inputs were."""
    ins = list(inputs)
    prov = "quant.derivatives_ovme.OptionValuationEngine"
    return [
        derive(f"{label} premium", v.npv, currency, ins, prov, f"{v.exercise.value.lower()}, expiry {v.adjusted_expiry}"),
        derive(f"{label} delta", v.delta, "", ins, prov),
        derive(f"{label} gamma", v.gamma, "", ins, prov),
        derive(f"{label} theta/day", v.theta_per_day, currency, ins, prov),
        derive(f"{label} vega/vol pt", v.vega_per_vol_point, currency, ins, prov),
        derive(f"{label} rho/1%", v.rho_per_pct, currency, ins, prov),
        derive(f"{label} volatility", v.volatility, "", ins, prov),
    ]
