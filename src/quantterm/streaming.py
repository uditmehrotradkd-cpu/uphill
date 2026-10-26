"""Real-time tick ingestion: tick hygiene -> raw tick store + bar builder -> Redis fan-out.

Every print is persisted with its hygiene verdict; quarantined prints also go to ``outlier_ticks``.
Only validated ticks reach candles and subscribers. Depth snapshots feed the filter's book confirmation.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import timedelta

from quantterm.db.repositories import (
    OutlierRow,
    OutlierTickRepository,
    PriceRepository,
    SecurityRepository,
    TickRepository,
    TickRow,
)
from quantterm.hygiene.tick_filter import HygieneConfig, HygieneVerdict, OutlierTick, TickHygieneBank
from quantterm.models import Exchange, OhlcvBar, OrderBookSnapshot, Tick
from quantterm.pubsub import Broker
from quantterm.ticks import Candle, CandleAggregator

log = logging.getLogger(__name__)

REJECT_CHANNEL = "qt:rejects"


def tick_channel(exchange: Exchange, symbol: str) -> str:
    return f"qt:ticks:{exchange.value}:{symbol}"


def candle_channel(exchange: Exchange, symbol: str) -> str:
    return f"qt:candles:{exchange.value}:{symbol}"


class TickIngestionPipeline:
    """Single-process owner of per-symbol filter and candle state. Share one instance per worker."""

    def __init__(
        self,
        *,
        securities: SecurityRepository,
        ticks: TickRepository,
        outliers: OutlierTickRepository,
        prices: PriceRepository,
        broker: Broker,
        candle_interval: timedelta,
        hygiene_config: HygieneConfig | None = None,
        flush_size: int = 500,
    ) -> None:
        if flush_size < 1:
            raise ValueError("flush_size must be >= 1")
        self._securities = securities
        self._ticks = ticks
        self._outliers = outliers
        self._prices = prices
        self._broker = broker
        self._interval = candle_interval
        self.hygiene = TickHygieneBank(hygiene_config)
        self._aggregators: dict[Exchange, CandleAggregator] = {}
        self._flush_size = flush_size
        self._buffer: list[TickRow] = []
        self._quarantine: list[OutlierRow] = []
        self._flush_lock = asyncio.Lock()

    def _aggregator(self, exchange: Exchange) -> CandleAggregator:
        agg = self._aggregators.get(exchange)
        if agg is None:
            agg = self._aggregators[exchange] = CandleAggregator(self._interval)
        return agg

    async def submit_book(self, book: OrderBookSnapshot) -> None:
        """Depth snapshot used to confirm or refute large moves. Raises ``UnknownSecurityError``."""
        await self._securities.get(book.symbol, book.exchange)
        self.hygiene.update_book(book)

    async def submit(self, tick: Tick, source: str) -> HygieneVerdict:
        """Raises ``UnknownSecurityError`` for unregistered instruments (nothing is stored)."""
        await self._securities.get(tick.symbol, tick.exchange)
        # Filter + candle update run without an await in between so concurrent feeds cannot interleave.
        verdict = self.hygiene.process(tick)
        closed: Candle | None = None
        if verdict.validated is not None:
            closed = self._aggregator(tick.exchange).update(verdict.validated)
        self._buffer.append(
            TickRow(
                ticker=tick.symbol,
                exchange=tick.exchange.value,
                timestamp=tick.timestamp,
                price=tick.price,
                size=tick.size,
                accepted=verdict.accepted,
                reject_reason=verdict.reason.value if verdict.reason else None,
                source=source,
            )
        )
        if verdict.outlier is not None:
            self._quarantine.append(_outlier_row(verdict.outlier, source))

        if verdict.validated is not None:
            await self._broker.publish(
                tick_channel(tick.exchange, tick.symbol), verdict.validated.model_dump_json()
            )
        elif verdict.outlier is not None:
            await self._broker.publish(REJECT_CHANNEL, verdict.outlier.model_dump_json())
        if closed is not None:
            await self._persist_candle(tick.exchange, closed, source)
        if len(self._buffer) >= self._flush_size:
            await self.flush()
        return verdict

    async def _persist_candle(self, exchange: Exchange, candle: Candle, source: str) -> None:
        bar = OhlcvBar(
            ts=candle.start,
            open=candle.open,
            high=candle.high,
            low=candle.low,
            close=candle.close,
            volume=candle.volume,
            vwap=candle.vwap,
        )
        await self._prices.append_bars(candle.symbol, exchange, self._interval, [bar], f"{source}:LIVE")
        await self._broker.publish(candle_channel(exchange, candle.symbol), candle.model_dump_json())

    async def flush(self) -> int:
        """Writes buffered prints and quarantine records; returns the number of prints written."""
        async with self._flush_lock:
            rows, self._buffer = self._buffer, []
            try:
                written = await self._ticks.append(rows)
            except BaseException:
                # Keep raw prints for the next attempt; they are never silently dropped.
                self._buffer[:0] = rows
                raise
            outliers, self._quarantine = self._quarantine, []
            try:
                await self._outliers.append(outliers)
            except BaseException:
                self._quarantine[:0] = outliers
                raise
            return written

    async def run_periodic_flush(self, interval_seconds: float) -> None:
        while True:
            await asyncio.sleep(interval_seconds)
            try:
                await self.flush()
            except Exception:
                log.exception("tick flush failed; %d rows retained for retry", len(self._buffer))

    @property
    def pending(self) -> int:
        return len(self._buffer)

    @property
    def pending_outliers(self) -> int:
        return len(self._quarantine)


def _outlier_row(outlier: OutlierTick, source: str) -> OutlierRow:
    t = outlier.tick
    return OutlierRow(
        ticker=t.symbol,
        exchange=t.exchange.value,
        timestamp=t.timestamp,
        price=t.price,
        size=t.size,
        reason=outlier.reason.value,
        rolling_median=outlier.rolling_median,
        rolling_mad=outlier.rolling_mad,
        deviation_mads=outlier.deviation_mads,
        best_bid=outlier.best_bid,
        best_ask=outlier.best_ask,
        book_age_ms=outlier.book_age_ms,
        source=source,
    )
