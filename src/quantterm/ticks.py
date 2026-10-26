"""Live OHLCV bar builder. It only accepts ``ValidatedTick``, which only ``hygiene.tick_filter`` emits."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from pydantic import BaseModel, ConfigDict

from quantterm.models import Tick


class ValidatedTick(BaseModel):
    """A tick that has passed tick hygiene. Only this type may update candles/indicators."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tick: Tick
    median: float
    deviation_mads: float


class Candle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    start: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    trades: int
    notional: float = 0.0

    @property
    def vwap(self) -> float | None:
        return self.notional / self.volume if self.volume > 0 else None


class CandleAggregator:
    """Time-bucketed OHLCV from validated ticks. Emits the finished candle when a tick opens a new bucket."""

    def __init__(self, interval: timedelta) -> None:
        if interval <= timedelta(0):
            raise ValueError("interval must be positive")
        self._interval = interval
        self._current: dict[str, Candle] = {}

    def _bucket(self, ts: datetime) -> datetime:
        epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
        n = (ts.astimezone(timezone.utc) - epoch) // self._interval
        return epoch + n * self._interval

    def update(self, vt: ValidatedTick) -> Candle | None:
        t = vt.tick
        start = self._bucket(t.timestamp)
        cur = self._current.get(t.symbol)
        if cur is not None and start < cur.start:
            return None  # late tick for an already-closed bucket; raw store keeps it
        if cur is None or start > cur.start:
            self._current[t.symbol] = Candle(
                symbol=t.symbol, start=start, open=t.price, high=t.price, low=t.price,
                close=t.price, volume=t.size, trades=1, notional=t.price * t.size,
            )
            return cur
        cur.high = max(cur.high, t.price)
        cur.low = min(cur.low, t.price)
        cur.close = t.price
        cur.volume += t.size
        cur.notional += t.price * t.size
        cur.trades += 1
        return None

    def current(self, symbol: str) -> Candle | None:
        return self._current.get(symbol)
