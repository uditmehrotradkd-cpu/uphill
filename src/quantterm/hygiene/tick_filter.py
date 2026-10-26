"""Rolling Median-Absolute-Deviation filter for live ticks with order-book confirmation.

A print deviating more than ``mad_multiplier`` x MAD from the rolling median is quarantined unless
the current order book corroborates it (fresh, uncrossed book quoting around the print with visible
depth at that price). Quarantined prints go to ``outlier_ticks`` and never reach the bar builder,
which only accepts ``ValidatedTick``. For feeds without depth, a run of mutually consistent outliers
is accepted as a genuine level change (news gap, circuit release) so the filter cannot lock out.

Per-tick cost is O(log w) search + O(w) list shift for the window update and O(w/2) for the MAD.
"""

from __future__ import annotations

from bisect import bisect_left, insort
from collections import deque
from collections.abc import Iterable, Iterator, Mapping
from datetime import datetime, timedelta
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from quantterm.models import Exchange, Market, OrderBookSnapshot, Tick
from quantterm.ticks import ValidatedTick


class QuarantineReason(StrEnum):
    NON_POSITIVE_PRICE = "NON_POSITIVE_PRICE"
    OUT_OF_ORDER = "OUT_OF_ORDER"
    OUTSIDE_PRICE_BAND = "OUTSIDE_PRICE_BAND"
    UNCONFIRMED_MAD_OUTLIER = "UNCONFIRMED_MAD_OUTLIER"


class Acceptance(StrEnum):
    WARMUP = "WARMUP"
    WITHIN_MAD = "WITHIN_MAD"
    BOOK_CONFIRMED = "BOOK_CONFIRMED"
    REGIME_SHIFT = "REGIME_SHIFT"


class HygieneConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    window: int = Field(default=50, ge=5)
    min_samples: int = Field(default=10, ge=3)
    mad_multiplier: float = Field(default=5.0, gt=0)
    # MAD floors so a flat tape (MAD = 0) does not quarantine a one-tick move.
    tick_size: float = Field(default=0.05, gt=0)
    min_mad_ticks: float = Field(default=1.0, ge=0)
    min_relative_mad: float = Field(default=0.0005, ge=0)
    book_max_age: timedelta = timedelta(seconds=2)
    book_price_tolerance: float = Field(default=0.001, ge=0)
    min_confirming_depth: int = Field(default=1, ge=1)
    regime_confirmations: int = Field(default=5, ge=3)
    regime_tolerance: float = Field(default=0.005, gt=0)
    max_backward_skew: timedelta = timedelta(0)
    quarantine_log_size: int = Field(default=10_000, ge=1)

    @model_validator(mode="after")
    def _consistent(self) -> HygieneConfig:
        if self.min_samples > self.window:
            raise ValueError("min_samples cannot exceed window")
        return self


class OutlierTick(BaseModel):
    """Quarantine record with the statistics that condemned the print."""

    model_config = ConfigDict(frozen=True)

    tick: Tick
    reason: QuarantineReason
    rolling_median: float | None = None
    rolling_mad: float | None = None
    deviation_mads: float | None = None
    best_bid: float | None = None
    best_ask: float | None = None
    book_age_ms: float | None = None


class HygieneVerdict(BaseModel):
    model_config = ConfigDict(frozen=True)

    tick: Tick
    acceptance: Acceptance | None = None
    validated: ValidatedTick | None = None
    outlier: OutlierTick | None = None

    @property
    def accepted(self) -> bool:
        return self.validated is not None

    @property
    def reason(self) -> QuarantineReason | None:
        return self.outlier.reason if self.outlier else None


class RollingWindow:
    """Fixed-size FIFO window kept simultaneously in arrival order and sorted order."""

    def __init__(self, size: int) -> None:
        self._size = size
        self._fifo: deque[float] = deque()
        self._sorted: list[float] = []

    def __len__(self) -> int:
        return len(self._fifo)

    def push(self, value: float) -> None:
        if len(self._fifo) == self._size:
            oldest = self._fifo.popleft()
            del self._sorted[bisect_left(self._sorted, oldest)]
        self._fifo.append(value)
        insort(self._sorted, value)

    def reset(self, values: Iterable[float]) -> None:
        self._fifo.clear()
        self._sorted.clear()
        for v in values:
            self.push(v)

    def median(self) -> float:
        s, n = self._sorted, len(self._sorted)
        if n == 0:
            raise ValueError("empty window")
        mid = n // 2
        return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2

    def mad(self, median: float) -> float:
        """Median of |x - median|: the deviations on each side of the median are already sorted in the
        sorted window, so a two-pointer merge reaches the middle element without sorting."""
        s, n = self._sorted, len(self._sorted)
        j = bisect_left(s, median)
        i = j - 1
        lo_k, hi_k = (n - 1) // 2, n // 2
        lo_v = d = 0.0
        for k in range(hi_k + 1):
            if i >= 0 and (j >= n or median - s[i] <= s[j] - median):
                d = median - s[i]
                i -= 1
            else:
                d = s[j] - median
                j += 1
            if k == lo_k:
                lo_v = d
        return (lo_v + d) / 2


class TickHygieneFilter:
    """Per-listing filter. Not thread-safe: one instance per (exchange, symbol) per event loop."""

    def __init__(
        self,
        symbol: str,
        exchange: Exchange,
        config: HygieneConfig | None = None,
        price_band: tuple[float, float] | None = None,
    ) -> None:
        self.symbol = symbol
        self.exchange = exchange
        self.config = config or HygieneConfig()
        self._band: tuple[float, float] | None = None
        if price_band is not None:
            self.set_price_band(*price_band)
        self._window = RollingWindow(self.config.window)
        self._pending: deque[float] = deque(maxlen=self.config.regime_confirmations)
        self._last_ts: datetime | None = None
        self._min_required = self.config.min_samples
        self._book: OrderBookSnapshot | None = None
        self.outlier_ticks: deque[OutlierTick] = deque(maxlen=self.config.quarantine_log_size)

    def set_price_band(self, lower: float, upper: float) -> None:
        """Exchange price band: NSE/BSE daily circuit, US LULD."""
        if not (0 < lower < upper):
            raise ValueError("price band must satisfy 0 < lower < upper")
        self._band = (lower, upper)

    def update_book(self, book: OrderBookSnapshot) -> None:
        if (book.symbol, book.exchange) != (self.symbol, self.exchange):
            raise ValueError(f"book for {book.exchange}:{book.symbol} sent to {self.exchange}:{self.symbol} filter")
        if self._book is None or book.timestamp >= self._book.timestamp:
            self._book = book

    def _book_context(self, tick: Tick) -> tuple[OrderBookSnapshot | None, float | None]:
        if self._book is None:
            return None, None
        age = tick.timestamp - self._book.timestamp
        return self._book, age.total_seconds() * 1000

    def _book_confirms(self, tick: Tick) -> bool:
        book, age_ms = self._book_context(tick)
        if book is None or age_ms is None or book.crossed:
            return False
        if abs(age_ms) > self.config.book_max_age.total_seconds() * 1000:
            return False
        if book.best_bid is None or book.best_ask is None:
            return False
        tol = self.config.book_price_tolerance
        price = tick.price
        if not book.best_bid * (1 - tol) <= price <= book.best_ask * (1 + tol):
            return False
        return book.depth_between(price * (1 - tol), price * (1 + tol)) >= self.config.min_confirming_depth

    def _quarantine(
        self,
        tick: Tick,
        reason: QuarantineReason,
        median: float | None = None,
        mad: float | None = None,
        deviation: float | None = None,
    ) -> HygieneVerdict:
        book, age_ms = self._book_context(tick)
        outlier = OutlierTick(
            tick=tick,
            reason=reason,
            rolling_median=median,
            rolling_mad=mad,
            deviation_mads=deviation,
            best_bid=book.best_bid if book else None,
            best_ask=book.best_ask if book else None,
            book_age_ms=age_ms,
        )
        self.outlier_ticks.append(outlier)
        return HygieneVerdict(tick=tick, outlier=outlier)

    def _accept(self, tick: Tick, how: Acceptance, median: float, deviation: float) -> HygieneVerdict:
        self._window.push(tick.price)
        self._last_ts = tick.timestamp
        self._pending.clear()
        return HygieneVerdict(
            tick=tick,
            acceptance=how,
            validated=ValidatedTick(tick=tick, median=median, deviation_mads=deviation),
        )

    def process(self, tick: Tick) -> HygieneVerdict:
        if (tick.symbol, tick.exchange) != (self.symbol, self.exchange):
            raise ValueError(f"tick for {tick.exchange}:{tick.symbol} sent to {self.exchange}:{self.symbol} filter")
        price = tick.price
        if price <= 0:
            return self._quarantine(tick, QuarantineReason.NON_POSITIVE_PRICE)
        if self._last_ts is not None and tick.timestamp < self._last_ts - self.config.max_backward_skew:
            return self._quarantine(tick, QuarantineReason.OUT_OF_ORDER)
        if self._band is not None and not (self._band[0] <= price <= self._band[1]):
            return self._quarantine(tick, QuarantineReason.OUTSIDE_PRICE_BAND)

        if len(self._window) < self._min_required:
            return self._accept(tick, Acceptance.WARMUP, price, 0.0)

        cfg = self.config
        median = self._window.median()
        mad = self._window.mad(median)
        scale = max(mad, cfg.min_mad_ticks * cfg.tick_size, median * cfg.min_relative_mad)
        deviation = abs(price - median) / scale
        if deviation <= cfg.mad_multiplier:
            return self._accept(tick, Acceptance.WITHIN_MAD, median, deviation)
        if self._book_confirms(tick):
            return self._accept(tick, Acceptance.BOOK_CONFIRMED, median, deviation)

        self._pending.append(price)
        if len(self._pending) == cfg.regime_confirmations:
            centre = sorted(self._pending)[len(self._pending) // 2]
            if max(abs(p - centre) for p in self._pending) / centre <= cfg.regime_tolerance:
                levels = list(self._pending)
                self._window.reset(levels)
                # Stay armed on the new level instead of re-entering an unfiltered warm-up.
                self._min_required = min(cfg.min_samples, cfg.regime_confirmations)
                self._last_ts = tick.timestamp
                self._pending.clear()
                return HygieneVerdict(
                    tick=tick,
                    acceptance=Acceptance.REGIME_SHIFT,
                    validated=ValidatedTick(tick=tick, median=centre, deviation_mads=0.0),
                )
        return self._quarantine(tick, QuarantineReason.UNCONFIRMED_MAD_OUTLIER, median, mad, deviation)


DEFAULT_TICK_SIZES: dict[Market, float] = {Market.US: 0.01, Market.IN: 0.05}


class TickHygieneBank:
    """Routes a multi-listing stream (ticks and depth) to per-(exchange, symbol) filters."""

    def __init__(
        self, config: HygieneConfig | None = None, tick_sizes: Mapping[Market, float] = DEFAULT_TICK_SIZES
    ) -> None:
        base = config or HygieneConfig()
        self._configs = {m: base.model_copy(update={"tick_size": size}) for m, size in tick_sizes.items()}
        self._default = base
        self._filters: dict[tuple[Exchange, str], TickHygieneFilter] = {}

    def filter_for(self, exchange: Exchange, symbol: str) -> TickHygieneFilter:
        key = (exchange, symbol)
        f = self._filters.get(key)
        if f is None:
            config = self._configs.get(exchange.market, self._default)
            f = self._filters[key] = TickHygieneFilter(symbol, exchange, config)
        return f

    def process(self, tick: Tick) -> HygieneVerdict:
        return self.filter_for(tick.exchange, tick.symbol).process(tick)

    def update_book(self, book: OrderBookSnapshot) -> None:
        self.filter_for(book.exchange, book.symbol).update_book(book)

    def outlier_ticks(self) -> Iterator[OutlierTick]:
        for f in self._filters.values():
            yield from f.outlier_ticks
