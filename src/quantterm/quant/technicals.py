"""Technical indicators on adjusted closes, using the conventions of Bloomberg GP studies.

RSI: Wilder (1978) - seed with the simple mean of the first ``period`` gains/losses, then smooth with
alpha = 1 / period. MACD: EMA(fast) - EMA(slow) with each EMA seeded by the SMA of its first window,
signal = EMA(signal) of MACD seeded the same way. Values before an indicator is defined are NaN.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import polars as pl

FloatArray = npt.NDArray[np.float64]
SeriesLike = Sequence[float] | FloatArray | pl.Series


def _array(values: SeriesLike, *, allow_leading_nan: bool = False) -> FloatArray:
    if isinstance(values, pl.Series):
        values = values.cast(pl.Float64).to_numpy()
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("expected a 1-D series")
    finite = np.isfinite(arr)
    if allow_leading_nan and finite.any():
        finite = finite[int(np.argmax(finite)):]
    if not np.all(finite):
        raise ValueError("series contains non-finite values")
    return arr


def rsi(close: SeriesLike, period: int = 14) -> FloatArray:
    if period < 2:
        raise ValueError("period must be >= 2")
    c = _array(close)
    out = np.full(c.size, np.nan)
    if c.size <= period:
        return out
    change = np.diff(c)
    gains, losses = np.clip(change, 0, None), np.clip(-change, 0, None)
    avg_gain, avg_loss = gains[:period].mean(), losses[:period].mean()

    def value(g: float, l_: float) -> float:
        if l_ == 0:
            return 100.0 if g > 0 else 50.0
        return 100.0 - 100.0 / (1.0 + g / l_)

    out[period] = value(avg_gain, avg_loss)
    for i in range(period, change.size):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        out[i + 1] = value(avg_gain, avg_loss)
    return out


def ema(values: SeriesLike, span: int) -> FloatArray:
    """EMA with alpha = 2 / (span + 1), seeded by the SMA of the first ``span`` defined values.

    Leading NaNs (e.g. an indicator still warming up) are skipped.
    """
    if span < 1:
        raise ValueError("span must be >= 1")
    v = _array(values, allow_leading_nan=True)
    out = np.full(v.size, np.nan)
    defined = np.flatnonzero(np.isfinite(v))
    if defined.size < span:
        return out
    start = int(defined[0])
    seed_end = start + span
    out[seed_end - 1] = v[start:seed_end].mean()
    alpha = 2.0 / (span + 1)
    for i in range(seed_end, v.size):
        out[i] = alpha * v[i] + (1 - alpha) * out[i - 1]
    return out


def macd(
    close: SeriesLike, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Returns (macd, signal, histogram)."""
    if not 0 < fast < slow:
        raise ValueError("require 0 < fast < slow")
    c = _array(close)
    line = ema(c, fast) - ema(c, slow)
    sig = ema(line, signal)
    return line, sig, line - sig


def realized_volatility(close: SeriesLike, periods_per_year: int = 252) -> float:
    """Annualised sample standard deviation of log returns."""
    c = _array(close)
    if c.size < 3:
        raise ValueError("need at least 3 closes")
    if np.any(c <= 0):
        raise ValueError("closes must be positive")
    r = np.diff(np.log(c))
    return float(r.std(ddof=1) * math.sqrt(periods_per_year))
