"""Async repositories over the bitemporal, append-only schema (``schema.sql``).

Write paths never UPDATE fundamentals, macro vintages or raw prices; conflicting raw rows are skipped,
not overwritten. Every table is keyed by (ticker, exchange). Adjusted prices and PIT reads live in
``quantterm.engine``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any, NamedTuple, Protocol

import polars as pl
from pydantic import BaseModel, ConfigDict

from quantterm.corporate_actions import CorporateAction, CorporateActionType
from quantterm.engine.adjustments import RAW_BARS_SQL
from quantterm.models import Exchange, FundamentalFact, MacroObservation, OhlcvBar, SectorClass
from quantterm.quant.derivatives_ovme import OptionType

DAILY = timedelta(days=1)

RAW_BAR_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "timestamp": pl.Datetime("us", "UTC"),
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Int64,
    "vwap": pl.Float64,
}


class Database(Protocol):
    """Subset of ``asyncpg.Pool`` used by the repositories."""

    async def fetch(self, query: str, *args: Any) -> Sequence[Mapping[str, Any]]: ...

    async def execute(self, query: str, *args: Any) -> str: ...

    async def copy_records_to_table(
        self, table_name: str, *, records: Iterable[Sequence[Any]], columns: Sequence[str]
    ) -> str: ...


class UnknownSecurityError(LookupError):
    pass


def _require_aware(ts: datetime, name: str) -> None:
    if ts.tzinfo is None or ts.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")


def _decimal_or_none(x: float | None) -> Decimal | None:
    return None if x is None else Decimal(repr(x))


class SecurityRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    symbol: str
    exchange: Exchange
    isin: str | None
    currency: str
    sector_class: SectorClass
    industry: str | None = None


_SEC_COLS = "ticker, exchange, isin, currency, sector_class, industry"


def _security(row: Mapping[str, Any]) -> SecurityRecord:
    return SecurityRecord(
        symbol=row["ticker"],
        exchange=Exchange(row["exchange"]),
        isin=row["isin"],
        currency=row["currency"],
        sector_class=SectorClass(row["sector_class"]),
        industry=row["industry"],
    )


class SecurityRepository:
    """Reference data. Mutable by design (sector reclassification), unlike every fact table."""

    UPSERT = f"""
    INSERT INTO securities (ticker, exchange, isin, currency, sector_class, industry)
    VALUES ($1, $2, $3, $4, $5, $6)
    ON CONFLICT (ticker, exchange) DO UPDATE
       SET isin = COALESCE(EXCLUDED.isin, securities.isin),
           currency = EXCLUDED.currency,
           sector_class = EXCLUDED.sector_class,
           industry = COALESCE(EXCLUDED.industry, securities.industry)
    RETURNING {_SEC_COLS}
    """
    GET = f"SELECT {_SEC_COLS} FROM securities WHERE ticker = $1 AND exchange = $2"
    PEERS = f"""
    SELECT {_SEC_COLS} FROM securities
    WHERE industry = $1 AND exchange = ANY($2::text[]) AND delisted_on IS NULL
    ORDER BY ticker, exchange
    """

    def __init__(self, db: Database) -> None:
        self._db = db
        self._cache: dict[tuple[str, Exchange], SecurityRecord] = {}

    async def upsert(
        self,
        symbol: str,
        exchange: Exchange,
        *,
        sector_class: SectorClass = SectorClass.GENERAL,
        isin: str | None = None,
        currency: str | None = None,
        industry: str | None = None,
    ) -> SecurityRecord:
        rows = await self._db.fetch(
            self.UPSERT, symbol, exchange.value, isin, currency or exchange.currency, sector_class.value, industry
        )
        record = _security(rows[0])
        self._cache[(symbol, exchange)] = record
        return record

    async def get(self, symbol: str, exchange: Exchange) -> SecurityRecord:
        key = (symbol, exchange)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        rows = await self._db.fetch(self.GET, symbol, exchange.value)
        if not rows:
            raise UnknownSecurityError(f"{exchange.value}:{symbol} is not registered in securities")
        record = _security(rows[0])
        self._cache[key] = record
        return record

    async def require(self, exchange: Exchange, symbols: Iterable[str]) -> None:
        for s in set(symbols):
            await self.get(s, exchange)

    async def resolve(self, symbol: str, exchanges: Sequence[Exchange]) -> SecurityRecord:
        """First listing of ``symbol`` among ``exchanges`` (Bloomberg composite codes span several venues)."""
        for exchange in exchanges:
            try:
                return await self.get(symbol, exchange)
            except UnknownSecurityError:
                continue
        raise UnknownSecurityError(f"{symbol} is not registered on {', '.join(e.value for e in exchanges)}")

    async def peers(self, industry: str, exchanges: Sequence[Exchange]) -> list[SecurityRecord]:
        rows = await self._db.fetch(self.PEERS, industry, [e.value for e in exchanges])
        return [_security(r) for r in rows]


class FundamentalsRepository:
    """Append-only writer for ``financial_metrics_pit``; ``revision_number`` is assigned by the DB trigger.

    Rows are sent in knowledge-time order because the trigger rejects a vintage older than one stored.
    Reads go through ``engine.pit_query``.
    """

    _INCOMING = """
    SELECT * FROM unnest($1::text[], $2::text[], $3::text[], $4::text[], $5::date[], $6::timestamptz[],
                         $7::numeric[], $8::text[], $9::text[], $10::text[], $11::numeric[])
        AS i(ticker, exchange, metric_name, period_type, period_end_date, filing_timestamp,
             metric_value, currency, source_authority, source_document_id, confidence_score)
    """
    _COLS = (
        "ticker, exchange, metric_name, period_type, period_end_date, filing_timestamp, "
        "metric_value, currency, source_authority, source_document_id, confidence_score"
    )
    _KEY = "ticker, exchange, metric_name, period_type, period_end_date, source_authority"
    INSERT = f"""
    INSERT INTO financial_metrics_pit ({_COLS})
    SELECT {_COLS} FROM ({_INCOMING}) incoming
    ORDER BY filing_timestamp
    ON CONFLICT DO NOTHING
    RETURNING revision_number
    """
    # Vendor feeds are snapshots, not vintages: only record a new revision when the value moved.
    INSERT_IF_CHANGED = f"""
    WITH incoming AS ({_INCOMING}),
    latest AS (
        SELECT DISTINCT ON ({_KEY}) {_KEY}, p.metric_value, p.currency
        FROM financial_metrics_pit p
        JOIN (SELECT DISTINCT {_KEY} FROM incoming) k USING ({_KEY})
        ORDER BY {_KEY}, p.revision_number DESC
    )
    INSERT INTO financial_metrics_pit ({_COLS})
    SELECT i.* FROM incoming i
    LEFT JOIN latest l USING ({_KEY})
    WHERE l.metric_value IS DISTINCT FROM i.metric_value OR l.currency IS DISTINCT FROM i.currency
    ORDER BY i.filing_timestamp
    ON CONFLICT DO NOTHING
    RETURNING revision_number
    """

    def __init__(self, db: Database, securities: SecurityRepository) -> None:
        self._db = db
        self._securities = securities

    async def append(
        self, exchange: Exchange, facts: Sequence[FundamentalFact], *, only_if_changed: bool = False
    ) -> int:
        """Returns the number of revisions actually inserted (known vintages/unchanged values are skipped)."""
        if not facts:
            return 0
        await self._securities.require(exchange, (f.symbol for f in facts))
        ordered = sorted(facts, key=lambda f: f.filing_timestamp)
        rows = await self._db.fetch(
            self.INSERT_IF_CHANGED if only_if_changed else self.INSERT,
            [f.symbol for f in ordered],
            [exchange.value] * len(ordered),
            [f.metric for f in ordered],
            [f.period_type.value for f in ordered],
            [f.period_end_date for f in ordered],
            [f.filing_timestamp for f in ordered],
            [f.value for f in ordered],
            [f.currency for f in ordered],
            [f.source for f in ordered],
            [f.source_document_id for f in ordered],
            [Decimal(f"{f.confidence_score:.3f}") for f in ordered],
        )
        return len(rows)


class CorporateActionRepository:
    _COLS = "ticker, exchange, ex_date, action_type, adjustment_factor, raw_dividend_amount, announced_at, source"
    INSERT = f"""
    INSERT INTO corporate_actions ({_COLS})
    SELECT * FROM unnest($1::text[], $2::text[], $3::date[], $4::text[],
                         $5::numeric[], $6::numeric[], $7::timestamptz[], $8::text[])
    ON CONFLICT (ticker, exchange, action_type, ex_date) DO NOTHING
    RETURNING 1
    """
    SELECT_KNOWN = f"""
    SELECT {_COLS}
    FROM corporate_actions
    WHERE ticker = $1 AND exchange = $2 AND announced_at <= $3
    ORDER BY ex_date, action_type
    """

    def __init__(self, db: Database, securities: SecurityRepository) -> None:
        self._db = db
        self._securities = securities

    async def append(self, actions: Sequence[CorporateAction]) -> int:
        """An action already stored for (listing, type, ex-date) is kept; corrections need explicit review."""
        if not actions:
            return 0
        for a in actions:
            await self._securities.get(a.symbol, a.exchange)
        rows = await self._db.fetch(
            self.INSERT,
            [a.symbol for a in actions],
            [a.exchange.value for a in actions],
            [a.ex_date for a in actions],
            [a.action_type.value for a in actions],
            [a.adjustment_factor for a in actions],
            [a.raw_dividend_amount for a in actions],
            [a.announced_at for a in actions],
            [a.source for a in actions],
        )
        return len(rows)

    async def fetch_known(self, symbol: str, exchange: Exchange, as_of: datetime) -> list[CorporateAction]:
        """Actions announced on or before ``as_of``, whatever their ex-date."""
        _require_aware(as_of, "as_of")
        rows = await self._db.fetch(self.SELECT_KNOWN, symbol, exchange.value, as_of)
        return [
            CorporateAction(
                symbol=r["ticker"],
                exchange=Exchange(r["exchange"]),
                action_type=CorporateActionType(r["action_type"]),
                ex_date=r["ex_date"],
                announced_at=r["announced_at"],
                adjustment_factor=r["adjustment_factor"],
                raw_dividend_amount=r["raw_dividend_amount"],
                source=r["source"],
            )
            for r in rows
        ]


class PriceRepository:
    INSERT_BARS = """
    INSERT INTO raw_market_bars
        (ticker, exchange, bar_interval, "timestamp", open, high, low, close, volume, vwap, source)
    SELECT $1::text, $2::text, $3::interval, t.ts, t.o, t.h, t.l, t.c, t.v, t.vw, $4::text
    FROM unnest($5::timestamptz[], $6::float8[], $7::float8[], $8::float8[], $9::float8[], $10::bigint[],
                $11::float8[]) AS t(ts, o, h, l, c, v, vw)
    ON CONFLICT (ticker, exchange, bar_interval, "timestamp") DO NOTHING
    RETURNING 1
    """

    def __init__(self, db: Database, securities: SecurityRepository) -> None:
        self._db = db
        self._securities = securities

    async def append_bars(
        self, symbol: str, exchange: Exchange, interval: timedelta, bars: Sequence[OhlcvBar], source: str
    ) -> int:
        """Insert raw bars; an existing bar for the same (listing, interval, timestamp) is kept, never replaced."""
        if not bars:
            return 0
        await self._securities.get(symbol, exchange)
        rows = await self._db.fetch(
            self.INSERT_BARS,
            symbol,
            exchange.value,
            interval,
            source,
            [b.ts for b in bars],
            [b.open for b in bars],
            [b.high for b in bars],
            [b.low for b in bars],
            [b.close for b in bars],
            [b.volume for b in bars],
            [b.vwap for b in bars],
        )
        return len(rows)

    async def fetch_raw(
        self, symbol: str, exchange: Exchange, interval: timedelta, start: datetime, end: datetime
    ) -> pl.DataFrame:
        """Unadjusted bars with ``start <= timestamp < end``."""
        _require_aware(start, "start")
        _require_aware(end, "end")
        rows = await self._db.fetch(RAW_BARS_SQL, symbol, exchange.value, interval, start, end)
        return pl.DataFrame({col: [r[col] for r in rows] for col in RAW_BAR_SCHEMA}, schema=RAW_BAR_SCHEMA)


class TickRow(NamedTuple):
    ticker: str
    exchange: str
    timestamp: datetime
    price: float
    size: int
    accepted: bool
    reject_reason: str | None
    source: str


class TickRepository:
    COLUMNS: tuple[str, ...] = TickRow._fields

    def __init__(self, db: Database) -> None:
        self._db = db

    async def append(self, rows: Sequence[TickRow]) -> int:
        """Bulk COPY of every print, accepted or rejected, with the MAD verdict."""
        if not rows:
            return 0
        await self._db.copy_records_to_table("ticks_raw", records=rows, columns=self.COLUMNS)
        return len(rows)


class OutlierRow(NamedTuple):
    ticker: str
    exchange: str
    timestamp: datetime
    price: float
    size: int
    reason: str
    rolling_median: float | None
    rolling_mad: float | None
    deviation_mads: float | None
    best_bid: float | None
    best_ask: float | None
    book_age_ms: float | None
    source: str


class OutlierSummary(NamedTuple):
    reason: str
    rejected: int
    max_deviation_mads: float | None
    last_rejected_at: datetime


class OutlierTickRepository:
    COLUMNS: tuple[str, ...] = OutlierRow._fields
    SUMMARY = """
    SELECT reason, count(*) AS n, max(deviation_mads) AS max_dev, max("timestamp") AS last_ts
    FROM outlier_ticks
    WHERE ticker = $1 AND exchange = $2 AND "timestamp" > $3 AND "timestamp" <= $4
    GROUP BY reason
    ORDER BY n DESC, reason
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    async def append(self, rows: Sequence[OutlierRow]) -> int:
        if not rows:
            return 0
        await self._db.copy_records_to_table("outlier_ticks", records=rows, columns=self.COLUMNS)
        return len(rows)

    async def summary(self, symbol: str, exchange: Exchange, start: datetime, end: datetime) -> list[OutlierSummary]:
        """Prints the MAD filter quarantined in ``(start, end]``, grouped by rejection reason."""
        rows = await self._db.fetch(self.SUMMARY, symbol, exchange.value, start, end)
        return [OutlierSummary(r["reason"], int(r["n"]), None if r["max_dev"] is None else float(r["max_dev"]),
                               r["last_ts"]) for r in rows]


class MacroRepository:
    INSERT = """
    INSERT INTO macro_observations (series_id, observation_date, release_timestamp, value, source)
    SELECT * FROM unnest($1::text[], $2::date[], $3::timestamptz[], $4::numeric[], $5::text[])
    ON CONFLICT DO NOTHING
    RETURNING 1
    """
    SELECT_AS_OF = """
    SELECT DISTINCT ON (observation_date) observation_date, value, release_timestamp, source
    FROM macro_observations
    WHERE series_id = $1 AND observation_date BETWEEN $2 AND $3 AND release_timestamp <= $4
    ORDER BY observation_date, release_timestamp DESC
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    async def append(self, observations: Sequence[MacroObservation]) -> int:
        if not observations:
            return 0
        rows = await self._db.fetch(
            self.INSERT,
            [o.series_id for o in observations],
            [o.observation_date for o in observations],
            [o.release_timestamp for o in observations],
            [o.value for o in observations],
            [o.source for o in observations],
        )
        return len(rows)

    async def fetch_as_of(self, series_id: str, start: date, end: date, as_of: datetime) -> pl.DataFrame:
        """Latest vintage of each observation released on or before ``as_of`` (first-release safe)."""
        _require_aware(as_of, "as_of")
        rows = await self._db.fetch(self.SELECT_AS_OF, series_id, start, end, as_of)
        return pl.DataFrame(
            {
                "observation_date": [r["observation_date"] for r in rows],
                "value": [float(r["value"]) for r in rows],
                "release_timestamp": [r["release_timestamp"] for r in rows],
                "source": [r["source"] for r in rows],
            },
            schema={
                "observation_date": pl.Date,
                "value": pl.Float64,
                "release_timestamp": pl.Datetime("us", "UTC"),
                "source": pl.Utf8,
            },
        )


class ReconciliationRepository:
    INSERT = """
    INSERT INTO reconciliation_breaks
        (ticker, exchange, metric_name, period_type, period_end_date, as_of, primary_source, secondary_source,
         primary_value, secondary_value, relative_variance, status)
    SELECT $1::text, $2::text, t.metric, t.period_type, t.period_end_date, $3::timestamptz, $4::text, $5::text,
           t.pv, t.sv, t.rv, t.status
    FROM unnest($6::text[], $7::text[], $8::date[], $9::numeric[], $10::numeric[], $11::float8[], $12::text[])
        AS t(metric, period_type, period_end_date, pv, sv, rv, status)
    RETURNING 1
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    async def record_breaks(
        self,
        symbol: str,
        exchange: Exchange,
        as_of: datetime,
        primary_source: str,
        secondary_source: str,
        reconciled: pl.DataFrame,
    ) -> int:
        """Persist every non-MATCH row of a ``reconciliation.reconcile`` result for audit."""
        _require_aware(as_of, "as_of")
        breaks = reconciled.filter(pl.col("is_break"))
        if breaks.is_empty():
            return 0
        rows = await self._db.fetch(
            self.INSERT,
            symbol,
            exchange.value,
            as_of,
            primary_source,
            secondary_source,
            breaks["metric"].to_list(),
            breaks["period_type"].to_list(),
            breaks["period_end_date"].to_list(),
            [_decimal_or_none(v) for v in breaks["value_primary"].to_list()],
            [_decimal_or_none(v) for v in breaks["value_secondary"].to_list()],
            breaks["relative_variance"].to_list(),
            breaks["status"].to_list(),
        )
        return len(rows)


class OptionQuote(BaseModel):
    model_config = ConfigDict(frozen=True)

    symbol: str
    exchange: Exchange
    expiry: date
    strike: Decimal
    option_type: OptionType
    timestamp: datetime
    bid: float
    ask: float
    last: float | None = None
    volume: int | None = None
    open_interest: int | None = None
    source: str

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2 if self.bid > 0 else self.ask / 2


class OptionQuoteRepository:
    INSERT = """
    INSERT INTO option_quotes (ticker, exchange, expiry, strike, option_type, "timestamp", bid, ask, last, volume,
                               open_interest, source)
    SELECT $1::text, $2::text, t.expiry, t.strike, t.option_type, t.ts, t.bid, t.ask, t.last, t.volume, t.oi, t.source
    FROM unnest($3::date[], $4::numeric[], $5::text[], $6::timestamptz[], $7::float8[], $8::float8[], $9::float8[],
                $10::bigint[], $11::bigint[], $12::text[])
        AS t(expiry, strike, option_type, ts, bid, ask, last, volume, oi, source)
    ON CONFLICT (ticker, exchange, expiry, strike, option_type, "timestamp") DO NOTHING
    RETURNING 1
    """
    # Latest quote per contract within the staleness window; expired contracts excluded.
    LATEST_CHAIN = """
    SELECT DISTINCT ON (expiry, strike, option_type)
           expiry, strike, option_type, "timestamp", bid, ask, last, volume, open_interest, source
    FROM option_quotes
    WHERE ticker = $1 AND exchange = $2 AND "timestamp" > $3 AND "timestamp" <= $4 AND expiry > $5
    ORDER BY expiry, strike, option_type, "timestamp" DESC
    """

    def __init__(self, db: Database, securities: SecurityRepository) -> None:
        self._db = db
        self._securities = securities

    async def append(self, quotes: Sequence[OptionQuote]) -> int:
        if not quotes:
            return 0
        listings = {(q.symbol, q.exchange) for q in quotes}
        if len(listings) != 1:
            raise ValueError("append quotes for one underlying at a time")
        symbol, exchange = listings.pop()
        await self._securities.get(symbol, exchange)
        rows = await self._db.fetch(
            self.INSERT, symbol, exchange.value,
            [q.expiry for q in quotes], [q.strike for q in quotes], [q.option_type.value for q in quotes],
            [q.timestamp for q in quotes], [q.bid for q in quotes], [q.ask for q in quotes],
            [q.last for q in quotes], [q.volume for q in quotes], [q.open_interest for q in quotes],
            [q.source for q in quotes],
        )
        return len(rows)

    async def latest_chain(self, symbol: str, exchange: Exchange, since: datetime, as_of: datetime,
                           after: date) -> list[OptionQuote]:
        """Most recent quote per live contract (expiry after ``after``) published in ``(since, as_of]``."""
        _require_aware(as_of, "as_of")
        rows = await self._db.fetch(self.LATEST_CHAIN, symbol, exchange.value, since, as_of, after)
        return [OptionQuote(symbol=symbol, exchange=exchange, expiry=r["expiry"], strike=Decimal(r["strike"]),
                            option_type=OptionType(r["option_type"]), timestamp=r["timestamp"], bid=r["bid"],
                            ask=r["ask"], last=r["last"], volume=r["volume"], open_interest=r["open_interest"],
                            source=r["source"]) for r in rows]
