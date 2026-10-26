"""Shared domain models and enumerations."""

from __future__ import annotations

import math
from datetime import date, datetime, time
from decimal import Decimal
from enum import StrEnum
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

VWAP_TOLERANCE = 1e-6  # matches the raw_market_bars vwap CHECK


class Market(StrEnum):
    US = "US"
    IN = "IN"


class Exchange(StrEnum):
    NYSE = "NYSE"
    NASDAQ = "NASDAQ"
    NSE = "NSE"
    BSE = "BSE"

    @property
    def market(self) -> Market:
        return Market.US if self in (Exchange.NYSE, Exchange.NASDAQ) else Market.IN

    @property
    def timezone(self) -> ZoneInfo:
        return ZoneInfo("America/New_York" if self.market is Market.US else "Asia/Kolkata")

    @property
    def currency(self) -> str:
        return "USD" if self.market is Market.US else "INR"

    @property
    def session_close(self) -> time:
        """Regular-session close in exchange local time; an EOD bar is not knowable before it."""
        return time(16, 0) if self.market is Market.US else time(15, 30)


def session_bar_ts(exchange: Exchange, session: date) -> datetime:
    """Canonical ``ts`` of a daily bar: local midnight of the session date (Polygon/NSE convention)."""
    return datetime.combine(session, time(0), tzinfo=exchange.timezone)


class SectorClass(StrEnum):
    """Coarse sector buckets that change street-convention formulas."""

    GENERAL = "GENERAL"
    BANK = "BANK"
    NBFC = "NBFC"
    INSURANCE = "INSURANCE"

    @property
    def is_financial(self) -> bool:
        return self is not SectorClass.GENERAL


class PeriodType(StrEnum):
    QUARTER = "Q"
    HALF_YEAR = "H"
    FISCAL_YEAR = "FY"
    TRAILING_12M = "TTM"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class FundamentalFact(_Frozen):
    """One bitemporal observation: what value was publicly known for a period, and since when."""

    symbol: str = Field(min_length=1)
    metric: str = Field(min_length=1)
    period_type: PeriodType
    period_end_date: date
    filing_timestamp: datetime
    value: Decimal
    currency: str = Field(min_length=3, max_length=3)
    source: str = Field(min_length=1)
    source_document_id: str | None = None
    # 1.0 = value read directly from a regulatory filing; lower for derived or vendor-sourced values.
    confidence_score: float = Field(default=1.0, ge=0, le=1)

    @field_validator("filing_timestamp")
    @classmethod
    def _require_tz(cls, v: datetime) -> datetime:
        if v.tzinfo is None or v.utcoffset() is None:
            raise ValueError("filing_timestamp must be timezone-aware")
        return v

    @model_validator(mode="after")
    def _filed_after_period(self) -> FundamentalFact:
        if self.filing_timestamp.date() < self.period_end_date:
            raise ValueError("filing_timestamp cannot precede period_end_date")
        return self


class Tick(_Frozen):
    """Raw trade print as received from a venue/vendor feed. Untrusted."""

    symbol: str = Field(min_length=1)
    exchange: Exchange
    timestamp: datetime
    price: float
    size: int = Field(ge=0)

    @field_validator("timestamp")
    @classmethod
    def _require_tz(cls, v: datetime) -> datetime:
        if v.tzinfo is None or v.utcoffset() is None:
            raise ValueError("tick timestamp must be timezone-aware")
        return v

    @field_validator("price")
    @classmethod
    def _finite(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("tick price must be finite")
        return v


class BookLevel(_Frozen):
    price: float = Field(gt=0, allow_inf_nan=False)
    quantity: int = Field(ge=0)
    orders: int = Field(default=0, ge=0)


class OrderBookSnapshot(_Frozen):
    """Top-of-book depth (e.g. Kite full mode: 5 levels a side). Bids best-first descending, asks ascending."""

    symbol: str = Field(min_length=1)
    exchange: Exchange
    timestamp: datetime
    bids: tuple[BookLevel, ...] = ()
    asks: tuple[BookLevel, ...] = ()

    @field_validator("timestamp")
    @classmethod
    def _require_tz(cls, v: datetime) -> datetime:
        if v.tzinfo is None or v.utcoffset() is None:
            raise ValueError("book timestamp must be timezone-aware")
        return v

    @model_validator(mode="after")
    def _ordered(self) -> OrderBookSnapshot:
        bid_prices = [lvl.price for lvl in self.bids]
        ask_prices = [lvl.price for lvl in self.asks]
        if bid_prices != sorted(bid_prices, reverse=True) or ask_prices != sorted(ask_prices):
            raise ValueError("bids must be descending and asks ascending")
        return self

    @property
    def best_bid(self) -> float | None:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> float | None:
        return self.asks[0].price if self.asks else None

    @property
    def crossed(self) -> bool:
        """Crossed/locked books occur in auctions and on stale feeds; they confirm nothing."""
        return self.best_bid is not None and self.best_ask is not None and self.best_bid >= self.best_ask

    def depth_between(self, low: float, high: float) -> int:
        return sum(lvl.quantity for lvl in (*self.bids, *self.asks) if low <= lvl.price <= high)


class OhlcvBar(_Frozen):
    """Raw, unadjusted bar exactly as printed by the venue. Mirrors the ``raw_market_bars`` CHECKs."""

    ts: datetime
    open: float = Field(gt=0, allow_inf_nan=False)
    high: float = Field(gt=0, allow_inf_nan=False)
    low: float = Field(gt=0, allow_inf_nan=False)
    close: float = Field(gt=0, allow_inf_nan=False)
    volume: int = Field(ge=0)
    vwap: float | None = Field(default=None, gt=0, allow_inf_nan=False)

    @field_validator("ts")
    @classmethod
    def _require_tz(cls, v: datetime) -> datetime:
        if v.tzinfo is None or v.utcoffset() is None:
            raise ValueError("bar timestamp must be timezone-aware")
        return v

    @model_validator(mode="after")
    def _consistent(self) -> OhlcvBar:
        if self.high < max(self.open, self.close) or self.low > min(self.open, self.close):
            raise ValueError("bar violates low <= open, close <= high")
        if self.vwap is not None and not (
            self.low * (1 - VWAP_TOLERANCE) <= self.vwap <= self.high * (1 + VWAP_TOLERANCE)
        ):
            raise ValueError("vwap outside [low, high]")
        return self


class MacroObservation(_Frozen):
    """One vintage of a macro series value: ``observation_date`` as published at ``release_timestamp``."""

    series_id: str = Field(min_length=1)
    observation_date: date
    release_timestamp: datetime
    value: Decimal
    source: str = Field(min_length=1)

    @field_validator("release_timestamp")
    @classmethod
    def _require_tz(cls, v: datetime) -> datetime:
        if v.tzinfo is None or v.utcoffset() is None:
            raise ValueError("release_timestamp must be timezone-aware")
        return v
