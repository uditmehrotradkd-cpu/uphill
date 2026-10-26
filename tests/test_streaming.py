from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta

import pytest
from fakes import FakeDatabase, InMemoryBroker

from quantterm.db.repositories import (
    OutlierRow,
    OutlierTickRepository,
    PriceRepository,
    SecurityRepository,
    TickRepository,
    TickRow,
    UnknownSecurityError,
)
from quantterm.hygiene.tick_filter import Acceptance, HygieneConfig
from quantterm.models import BookLevel, Exchange, OrderBookSnapshot, Tick
from quantterm.streaming import REJECT_CHANNEL, TickIngestionPipeline, candle_channel, tick_channel

T0 = datetime(2025, 1, 16, 4, 0, tzinfo=UTC)  # 09:30 IST


def _tick(i: int, price: float, size: int = 100, symbol: str = "INFY") -> Tick:
    return Tick(symbol=symbol, exchange=Exchange.NSE, timestamp=T0 + timedelta(seconds=i), price=price, size=size)


def _pipeline(db: FakeDatabase, broker: InMemoryBroker, flush_size: int = 1000) -> TickIngestionPipeline:
    securities = SecurityRepository(db)
    asyncio.run(securities.upsert("INFY", Exchange.NSE))
    return TickIngestionPipeline(
        securities=securities,
        ticks=TickRepository(db),
        outliers=OutlierTickRepository(db),
        prices=PriceRepository(db, securities),
        broker=broker,
        candle_interval=timedelta(minutes=1),
        hygiene_config=HygieneConfig(min_samples=10, window=50),
        flush_size=flush_size,
    )


def test_fat_finger_print_is_stored_but_not_published_or_candled() -> None:
    db, broker = FakeDatabase(), InMemoryBroker()
    pipeline = _pipeline(db, broker)

    async def scenario() -> None:
        for i in range(30):
            await pipeline.submit(_tick(i, 1928.0 + (i % 5) * 0.05), "KITE")
        bad = await pipeline.submit(_tick(30, 19.28), "KITE")  # decimal-shift fat finger
        assert not bad.accepted
        await pipeline.flush()

    asyncio.run(scenario())
    rows = [TickRow(*r) for r in db.ticks]
    assert len(rows) == 31
    assert [r.accepted for r in rows].count(False) == 1
    rejected = next(r for r in rows if not r.accepted)
    assert rejected.reject_reason == "UNCONFIRMED_MAD_OUTLIER" and rejected.price == 19.28
    assert (rejected.ticker, rejected.exchange) == ("INFY", "NSE")
    (quarantined,) = [OutlierRow(*r) for r in db.outliers]
    assert quarantined.price == 19.28 and quarantined.source == "KITE"
    assert quarantined.rolling_median == pytest.approx(1928.1) and quarantined.deviation_mads is not None
    published_prices = [json.loads(m)["tick"]["price"] for c, m in broker.published if c == tick_channel(Exchange.NSE, "INFY")]
    assert 19.28 not in published_prices and len(published_prices) == 30
    assert sum(1 for c, _ in broker.published if c == REJECT_CHANNEL) == 1


def test_closed_candle_is_persisted_and_published() -> None:
    db, broker = FakeDatabase(), InMemoryBroker()
    pipeline = _pipeline(db, broker)

    async def scenario() -> None:
        await pipeline.submit(_tick(0, 1928.0, 10), "KITE")
        await pipeline.submit(_tick(30, 1931.0, 20), "KITE")
        await pipeline.submit(_tick(45, 1926.5, 5), "KITE")
        await pipeline.submit(_tick(61, 1929.0, 7), "KITE")  # opens the next minute -> closes 04:00 bar

    asyncio.run(scenario())
    (bar,) = db.bars.values()
    assert (bar["open"], bar["high"], bar["low"], bar["close"], bar["volume"]) == (1928.0, 1931.0, 1926.5, 1926.5, 35)
    assert bar["vwap"] == pytest.approx((1928.0 * 10 + 1931.0 * 20 + 1926.5 * 5) / 35)
    assert bar["source"] == "KITE:LIVE"
    candles = [json.loads(m) for c, m in broker.published if c == candle_channel(Exchange.NSE, "INFY")]
    assert candles[0]["close"] == 1926.5 and candles[0]["trades"] == 3


def test_unknown_security_is_refused_without_side_effects() -> None:
    db, broker = FakeDatabase(), InMemoryBroker()
    pipeline = _pipeline(db, broker)
    with pytest.raises(UnknownSecurityError):
        asyncio.run(pipeline.submit(_tick(0, 100.0, symbol="UNLISTED"), "KITE"))
    assert pipeline.pending == 0 and broker.published == []


def test_failed_flush_retains_raw_ticks_for_retry() -> None:
    db, broker = FakeDatabase(), InMemoryBroker()
    pipeline = _pipeline(db, broker)

    async def scenario() -> None:
        for i in range(5):
            await pipeline.submit(_tick(i, 1928.0), "KITE")
        db.fail_copy = True
        with pytest.raises(ConnectionError):
            await pipeline.flush()
        assert pipeline.pending == 5
        await pipeline.submit(_tick(5, 1928.1), "KITE")
        db.fail_copy = False
        assert await pipeline.flush() == 6

    asyncio.run(scenario())
    assert [TickRow(*row).timestamp for row in db.ticks] == [T0 + timedelta(seconds=i) for i in range(6)]


def test_auto_flush_at_threshold() -> None:
    db, broker = FakeDatabase(), InMemoryBroker()
    pipeline = _pipeline(db, broker, flush_size=3)

    async def scenario() -> None:
        for i in range(4):
            await pipeline.submit(_tick(i, 1928.0), "KITE")

    asyncio.run(scenario())
    assert len(db.ticks) == 3 and pipeline.pending == 1


def test_depth_snapshot_lets_a_genuine_gap_through() -> None:
    db, broker = FakeDatabase(), InMemoryBroker()
    pipeline = _pipeline(db, broker)
    book = OrderBookSnapshot(
        symbol="INFY", exchange=Exchange.NSE, timestamp=T0 + timedelta(seconds=30),
        bids=(BookLevel(price=1734.9, quantity=1200),), asks=(BookLevel(price=1735.2, quantity=800),),
    )

    async def scenario() -> Acceptance | None:
        for i in range(30):
            await pipeline.submit(_tick(i, 1928.0 + (i % 5) * 0.05), "KITE")
        await pipeline.submit_book(book)  # -10 % on guidance cut, book already repriced
        verdict = await pipeline.submit(_tick(30, 1735.0), "KITE")
        await pipeline.flush()
        return verdict.acceptance

    assert asyncio.run(scenario()) is Acceptance.BOOK_CONFIRMED
    assert db.outliers == []
