"""In-memory stand-ins that emulate the SQL semantics the repositories and engines rely on.

Dispatch is by identity with the SQL constants, so an unexpected query fails loudly. The
``financial_metrics_pit`` revision trigger is emulated (numbering, idempotency, out-of-order rejection).
Real SQL is exercised by the ``integration`` tests against PostgreSQL/TimescaleDB.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Iterable, Mapping, Sequence
from contextlib import asynccontextmanager
from datetime import date, datetime
from typing import Any

from quantterm import demo
from quantterm.db.repositories import (
    CorporateActionRepository,
    FundamentalsRepository,
    MacroRepository,
    OptionQuoteRepository,
    OutlierRow,
    OutlierTickRepository,
    PriceRepository,
    ReconciliationRepository,
    SecurityRepository,
)
from quantterm.engine import adjustments, pit_query

_METRIC_KEY = ("ticker", "exchange", "metric_name", "period_type", "period_end_date", "source_authority")
_METRIC_COLS = ("ticker", "exchange", "metric_name", "period_type", "period_end_date", "filing_timestamp",
                "metric_value", "currency", "source_authority", "source_document_id", "confidence_score")
_ACTION_COLS = ("ticker", "exchange", "ex_date", "action_type", "adjustment_factor", "raw_dividend_amount",
                "announced_at", "source")


class FakeCheckViolation(Exception):
    """Stands in for ``asyncpg.exceptions.CheckViolationError`` raised by the revision trigger."""


class FakeForeignKeyViolation(Exception):
    pass


def _latest_by(rows: Iterable[dict[str, Any]], key: Callable[[dict[str, Any]], Any],
               rank: Callable[[dict[str, Any]], Any]) -> list[dict[str, Any]]:
    best: dict[Any, dict[str, Any]] = {}
    for r in rows:
        k = key(r)
        if k not in best or rank(r) > rank(best[k]):
            best[k] = r
    return [best[k] for k in sorted(best)]


class FakeDatabase:
    def __init__(self) -> None:
        self.securities: dict[tuple[str, str], dict[str, Any]] = {}
        self.metrics: list[dict[str, Any]] = []
        self.actions: list[dict[str, Any]] = []
        self.bars: dict[tuple[Any, ...], dict[str, Any]] = {}
        self.ticks: list[Sequence[Any]] = []
        self.outliers: list[Sequence[Any]] = []
        self.breaks: list[dict[str, Any]] = []
        self.macro: list[dict[str, Any]] = []
        self.quotes: dict[tuple[Any, ...], dict[str, Any]] = {}
        self.fail_copy = False
        self.queries: list[str] = []

    # ------------------------------------------------------------------ asyncpg surface
    async def execute(self, query: str, *args: Any) -> str:
        raise AssertionError(f"unexpected execute: {query[:80]}")

    async def copy_records_to_table(
        self, table_name: str, *, records: Iterable[Sequence[Any]], columns: Sequence[str]
    ) -> str:
        if self.fail_copy:
            raise ConnectionError("database unavailable")
        rows = list(records)
        target = {"ticks_raw": self.ticks, "outlier_ticks": self.outliers}[table_name]
        target.extend(rows)
        return f"COPY {len(rows)}"

    async def fetch(self, query: str, *args: Any) -> Sequence[Mapping[str, Any]]:
        self.queries.append(query)
        handler = self._handlers().get(query)
        if handler is None:
            raise AssertionError(f"unexpected query: {query[:120]}")
        return handler(*args)

    def _handlers(self) -> dict[str, Callable[..., list[dict[str, Any]]]]:
        return {
            SecurityRepository.UPSERT: self._upsert_security,
            SecurityRepository.GET: lambda t, e: [dict(self.securities[(t, e)])] if (t, e) in self.securities else [],
            SecurityRepository.PEERS: self._peers,
            FundamentalsRepository.INSERT: lambda *a: self._insert_metrics(a, only_if_changed=False),
            FundamentalsRepository.INSERT_IF_CHANGED: lambda *a: self._insert_metrics(a, only_if_changed=True),
            pit_query.SNAPSHOT_SQL: self._snapshot,
            pit_query.BEST_SNAPSHOT_SQL: self._best_snapshot,
            pit_query.LATEST_SQL: lambda *a: self._latest_period(self._snapshot(*a)),
            pit_query.BEST_LATEST_SQL: lambda *a: self._latest_period(self._best_snapshot(*a)),
            pit_query.REVISION_HISTORY_SQL: self._revision_history,
            CorporateActionRepository.INSERT: self._insert_actions,
            CorporateActionRepository.SELECT_KNOWN: lambda t, e, as_of: self._actions(t, e, as_of, None),
            adjustments.KNOWN_ACTIONS_SQL: lambda t, e, as_of, ex: self._actions(t, e, as_of, ex),
            PriceRepository.INSERT_BARS: self._insert_bars,
            adjustments.RAW_BARS_SQL: self._raw_bars,
            ReconciliationRepository.INSERT: self._insert_breaks,
            MacroRepository.INSERT: self._insert_macro,
            MacroRepository.SELECT_AS_OF: self._macro_as_of,
            OutlierTickRepository.SUMMARY: self._outlier_summary,
            OptionQuoteRepository.INSERT: self._insert_quotes,
            OptionQuoteRepository.LATEST_CHAIN: self._latest_chain,
            demo.SOURCES_SQL: self._sources,
        }

    def _require_listing(self, ticker: str, exchange: str) -> None:
        if (ticker, exchange) not in self.securities:
            raise FakeForeignKeyViolation(f"{exchange}:{ticker} not in securities")

    # ------------------------------------------------------------------ securities
    def _upsert_security(self, ticker: str, exchange: str, isin: str | None, currency: str, sector: str,
                         industry: str | None) -> list[dict[str, Any]]:
        row = self.securities.setdefault((ticker, exchange), {"ticker": ticker, "exchange": exchange, "isin": isin,
                                                              "industry": industry})
        row.update(isin=isin or row["isin"], currency=currency, sector_class=sector,
                   industry=industry or row["industry"])
        return [dict(row)]

    def _peers(self, industry: str, exchanges: list[str]) -> list[dict[str, Any]]:
        return [dict(r) for k, r in sorted(self.securities.items())
                if r["industry"] == industry and r["exchange"] in exchanges]

    # ------------------------------------------------------------------ financial_metrics_pit
    def _insert_metrics(self, args: tuple[Any, ...], *, only_if_changed: bool) -> list[dict[str, Any]]:
        incoming = sorted((dict(zip(_METRIC_COLS, row, strict=True)) for row in zip(*args, strict=True)),
                          key=lambda r: r["filing_timestamp"])
        snapshot = {tuple(m[k] for k in _METRIC_KEY): m for m in _latest_by(
            self.metrics, lambda m: tuple(m[k] for k in _METRIC_KEY), lambda m: m["revision_number"])}
        inserted: list[dict[str, Any]] = []
        for row in incoming:
            self._require_listing(row["ticker"], row["exchange"])
            key = tuple(row[k] for k in _METRIC_KEY)
            if only_if_changed and key in snapshot:
                prior = snapshot[key]
                if prior["metric_value"] == row["metric_value"] and prior["currency"] == row["currency"]:
                    continue
            same_key = [m for m in self.metrics if tuple(m[k] for k in _METRIC_KEY) == key]
            if any(m["filing_timestamp"] == row["filing_timestamp"] for m in same_key):
                continue  # trigger returns NULL for a known vintage
            if same_key:
                last = max(same_key, key=lambda m: m["revision_number"])
                if row["filing_timestamp"] < last["filing_timestamp"]:
                    raise FakeCheckViolation(f"out-of-order vintage for {key}")
            row["revision_number"] = len(same_key) + 1
            self.metrics.append(row)
            inserted.append({"revision_number": row["revision_number"]})
        return inserted

    def _visible(self, tickers: list[str], exchange: str, metrics: list[str], period_type: str,
                 as_of: datetime, periods_from: date) -> list[dict[str, Any]]:
        return [
            m for m in self.metrics
            if m["ticker"] in tickers and m["exchange"] == exchange and m["metric_name"] in metrics
            and m["period_type"] == period_type and m["filing_timestamp"] <= as_of
            and m["period_end_date"] >= periods_from
        ]

    def _snapshot(self, tickers: list[str], exchange: str, metrics: list[str], period_type: str,
                  as_of: datetime, periods_from: date, source: str) -> list[dict[str, Any]]:
        rows = [m for m in self._visible(tickers, exchange, metrics, period_type, as_of, periods_from)
                if m["source_authority"] == source]
        return _latest_by(rows, lambda m: (m["ticker"], m["metric_name"], m["period_end_date"]),
                          lambda m: m["revision_number"])

    def _best_snapshot(self, *args: Any) -> list[dict[str, Any]]:
        per_source = _latest_by(
            self._visible(*args),
            lambda m: (m["ticker"], m["metric_name"], m["period_end_date"], m["source_authority"]),
            lambda m: m["revision_number"],
        )
        best: dict[Any, dict[str, Any]] = {}
        for m in sorted(per_source, key=lambda m: (-m["confidence_score"], -m["filing_timestamp"].timestamp(),
                                                   m["source_authority"])):
            best.setdefault((m["ticker"], m["metric_name"], m["period_end_date"]), m)
        return [best[k] for k in sorted(best)]

    @staticmethod
    def _latest_period(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return _latest_by(rows, lambda m: (m["ticker"], m["metric_name"]), lambda m: m["period_end_date"])

    def _revision_history(self, ticker: str, exchange: str, metric: str, period_type: str, end: date,
                          source: str, as_of: datetime) -> list[dict[str, Any]]:
        key = (ticker, exchange, metric, period_type, end, source)
        rows = [m for m in self.metrics if tuple(m[k] for k in _METRIC_KEY) == key and m["filing_timestamp"] <= as_of]
        return sorted(rows, key=lambda m: m["revision_number"])

    # ------------------------------------------------------------------ corporate actions
    def _insert_actions(self, *cols: list[Any]) -> list[dict[str, Any]]:
        inserted = []
        for row in zip(*cols, strict=True):
            action = dict(zip(_ACTION_COLS, row, strict=True))
            self._require_listing(action["ticker"], action["exchange"])
            key = (action["ticker"], action["exchange"], action["action_type"], action["ex_date"])
            if any((a["ticker"], a["exchange"], a["action_type"], a["ex_date"]) == key for a in self.actions):
                continue
            self.actions.append(action)
            inserted.append({"?column?": 1})
        return inserted

    def _actions(self, ticker: str, exchange: str, as_of: datetime, ex_cutoff: date | None) -> list[dict[str, Any]]:
        rows = [
            a for a in self.actions
            if a["ticker"] == ticker and a["exchange"] == exchange and a["announced_at"] <= as_of
            and (ex_cutoff is None or a["ex_date"] <= ex_cutoff)
        ]
        return sorted(rows, key=lambda r: (r["ex_date"], r["action_type"]))

    # ------------------------------------------------------------------ raw bars
    def _insert_bars(self, ticker: str, exchange: str, interval: Any, source: str, *cols: list[Any]) -> list[dict[str, Any]]:
        self._require_listing(ticker, exchange)
        inserted = []
        for ts, o, h, low, c, v, vw in zip(*cols, strict=True):
            key = (ticker, exchange, interval, ts)
            if key not in self.bars:
                self.bars[key] = {"timestamp": ts, "open": o, "high": h, "low": low, "close": c, "volume": v,
                                  "vwap": vw, "source": source}
                inserted.append({"?column?": 1})
        return inserted

    def _raw_bars(self, ticker: str, exchange: str, interval: Any, start: datetime, end: datetime) -> list[dict[str, Any]]:
        rows = [b for (t, e, iv, ts), b in self.bars.items()
                if t == ticker and e == exchange and iv == interval and start <= ts < end]
        return sorted(rows, key=lambda r: r["timestamp"])

    # ------------------------------------------------------------------ audit + macro
    def _insert_breaks(self, ticker: str, exchange: str, as_of: datetime, primary: str, secondary: str,
                       *cols: list[Any]) -> list[dict[str, Any]]:
        for metric, pt, end, pv, sv, rv, st in zip(*cols, strict=True):
            self.breaks.append(dict(ticker=ticker, exchange=exchange, as_of=as_of, primary_source=primary,
                                    secondary_source=secondary, metric_name=metric, period_type=pt,
                                    period_end_date=end, primary_value=pv, secondary_value=sv,
                                    relative_variance=rv, status=st))
        return [{"?column?": 1}] * len(cols[0])

    def _insert_macro(self, *cols: list[Any]) -> list[dict[str, Any]]:
        inserted = []
        for sid, od, rt, v, src in zip(*cols, strict=True):
            if not any((m["series_id"], m["observation_date"], m["release_timestamp"]) == (sid, od, rt)
                       for m in self.macro):
                self.macro.append(dict(series_id=sid, observation_date=od, release_timestamp=rt, value=v, source=src))
                inserted.append({"?column?": 1})
        return inserted

    def _macro_as_of(self, series_id: str, start: date, end: date, as_of: datetime) -> list[dict[str, Any]]:
        rows = [m for m in self.macro
                if m["series_id"] == series_id and start <= m["observation_date"] <= end
                and m["release_timestamp"] <= as_of]
        return _latest_by(rows, lambda m: m["observation_date"], lambda m: m["release_timestamp"])

    def _outlier_summary(self, ticker: str, exchange: str, start: datetime, end: datetime) -> list[dict[str, Any]]:
        groups: dict[str, list[OutlierRow]] = {}
        for raw in self.outliers:
            o = OutlierRow(*raw)
            if o.ticker == ticker and o.exchange == exchange and start < o.timestamp <= end:
                groups.setdefault(o.reason, []).append(o)
        rows = [{"reason": reason, "n": len(g),
                 "max_dev": max((o.deviation_mads for o in g if o.deviation_mads is not None), default=None),
                 "last_ts": max(o.timestamp for o in g)} for reason, g in groups.items()]
        return sorted(rows, key=lambda r: (-r["n"], r["reason"]))

    def _sources(self) -> list[dict[str, Any]]:
        stores = [*self.bars.values(), *self.actions, *self.macro, *self.quotes.values()]
        found = {r["source"] for r in stores} | {m["source_authority"] for m in self.metrics}
        return [{"source": s} for s in sorted(found)]

    def _insert_quotes(self, ticker: str, exchange: str, *cols: list[Any]) -> list[dict[str, Any]]:
        self._require_listing(ticker, exchange)
        inserted = []
        for expiry, strike, kind, ts, bid, ask, last, volume, oi, source in zip(*cols, strict=True):
            key = (ticker, exchange, expiry, strike, kind, ts)
            if key not in self.quotes:
                self.quotes[key] = {"expiry": expiry, "strike": strike, "option_type": kind, "timestamp": ts,
                                    "bid": bid, "ask": ask, "last": last, "volume": volume, "open_interest": oi,
                                    "source": source}
                inserted.append({"?column?": 1})
        return inserted

    def _latest_chain(self, ticker: str, exchange: str, since: datetime, as_of: datetime,
                      after: date) -> list[dict[str, Any]]:
        rows = [q for (t, e, *_), q in self.quotes.items()
                if t == ticker and e == exchange and since < q["timestamp"] <= as_of and q["expiry"] > after]
        return _latest_by(rows, lambda q: (q["expiry"], q["strike"], q["option_type"]), lambda q: q["timestamp"])


class InMemoryBroker:
    def __init__(self) -> None:
        self.published: list[tuple[str, str]] = []
        self._subscribers: list[tuple[frozenset[str], asyncio.Queue[tuple[str, str]]]] = []

    async def publish(self, channel: str, message: str) -> None:
        self.published.append((channel, message))
        for channels, queue in self._subscribers:
            if channel in channels:
                queue.put_nowait((channel, message))

    @asynccontextmanager
    async def subscribe(self, channels: Sequence[str]) -> AsyncIterator[AsyncIterator[tuple[str, str]]]:
        entry: tuple[frozenset[str], asyncio.Queue[tuple[str, str]]] = (frozenset(channels), asyncio.Queue())
        self._subscribers.append(entry)

        async def messages() -> AsyncIterator[tuple[str, str]]:
            while True:
                yield await entry[1].get()

        try:
            yield messages()
        finally:
            self._subscribers.remove(entry)


class DictCache:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    async def get(self, key: str) -> str | None:
        return self.store.get(key)

    async def set(self, key: str, value: str, ttl_seconds: int) -> None:
        assert ttl_seconds > 0
        self.store[key] = value
