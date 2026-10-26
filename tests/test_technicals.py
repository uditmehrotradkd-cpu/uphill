from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from quantterm.models import Exchange
from quantterm.quant.derivatives_ovme import OptionValuationEngine, standard_monthly_expiry
from quantterm.quant.technicals import ema, macd, realized_volatility, rsi


def _wilder_rsi_reference(close: list[float], period: int = 14) -> list[float | None]:
    """Textbook Wilder RSI written independently of the vectorised implementation."""
    out: list[float | None] = [None] * len(close)
    gains = [max(close[i] - close[i - 1], 0.0) for i in range(1, len(close))]
    losses = [max(close[i - 1] - close[i], 0.0) for i in range(1, len(close))]
    ag, al = sum(gains[:period]) / period, sum(losses[:period]) / period
    out[period] = 100 - 100 / (1 + ag / al)
    for i in range(period, len(gains)):
        ag = (ag * (period - 1) + gains[i]) / period
        al = (al * (period - 1) + losses[i]) / period
        out[i + 1] = 100 - 100 / (1 + ag / al)
    return out


# Wilder's worked example (New Concepts in Technical Trading Systems, 1978) closing prices.
WILDER = [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28,
          46.00, 46.03, 46.41, 46.22, 45.64, 46.21, 46.25, 45.71, 46.45, 45.78, 45.35, 44.03, 44.18, 44.22, 44.57,
          43.42, 42.66, 43.13]


def test_rsi_matches_wilder_reference_series() -> None:
    got = rsi(WILDER)
    ref = _wilder_rsi_reference(WILDER)
    assert np.isnan(got[:14]).all()
    np.testing.assert_allclose(got[14:], [x for x in ref[14:] if x is not None], rtol=1e-12)
    # First RSI = 100 - 100 / (1 + 0.238571 / 0.099857) in exact arithmetic (StockCharts' 70.53 rounds
    # the intermediate averages).
    assert float(got[14]) == pytest.approx(70.4641, abs=1e-4)


def test_rsi_saturates_on_monotone_series() -> None:
    assert rsi(list(range(1, 40)))[-1] == 100.0
    assert rsi(list(range(40, 1, -1)))[-1] == 0.0


def test_ema_seeded_with_sma_and_skips_leading_nan() -> None:
    v = [np.nan, np.nan, 1.0, 2.0, 3.0, 4.0]
    out = ema(v, 3)
    assert np.isnan(out[:4]).all()
    assert out[4] == pytest.approx(2.0)
    assert out[5] == pytest.approx(0.5 * 4.0 + 0.5 * 2.0)


def test_macd_line_signal_and_histogram_identities() -> None:
    rng = np.random.default_rng(3)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 200)))
    line, signal, hist = macd(close)
    assert np.isnan(line[:25]).all() and np.isfinite(line[25:]).all()
    assert np.isnan(signal[:33]).all() and np.isfinite(signal[33:]).all()
    np.testing.assert_allclose(hist[33:], line[33:] - signal[33:])
    np.testing.assert_allclose(line[25:], (ema(close, 12) - ema(close, 26))[25:])


def test_realized_volatility_recovers_generating_sigma() -> None:
    rng = np.random.default_rng(5)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 5000)))
    assert realized_volatility(close) == pytest.approx(0.02 * np.sqrt(252), rel=0.03)
    with pytest.raises(ValueError):
        realized_volatility([1.0, -1.0, 2.0])


@pytest.mark.parametrize(("exchange", "year", "month", "expected"), [
    (Exchange.NYSE, 2025, 1, date(2025, 1, 17)),      # third Friday
    (Exchange.NASDAQ, 2024, 3, date(2024, 3, 15)),
    (Exchange.NSE, 2024, 12, date(2024, 12, 26)),     # last Thursday
    (Exchange.NSE, 2025, 1, date(2025, 1, 30)),
    (Exchange.NSE, 2025, 9, date(2025, 9, 30)),       # last Tuesday after SEBI rationalisation
])
def test_standard_monthly_expiry(exchange: Exchange, year: int, month: int, expected: date) -> None:
    assert standard_monthly_expiry(exchange, year, month) == expected


def test_bse_monthly_expiry_must_be_explicit() -> None:
    with pytest.raises(ValueError):
        standard_monthly_expiry(Exchange.BSE, 2025, 1)


def test_next_monthly_expiry_rolls_and_adjusts_for_holidays() -> None:
    nse = OptionValuationEngine(Exchange.NSE)
    assert nse.next_monthly_expiry(date(2024, 12, 23)) == date(2025, 1, 30)  # Dec 26 is < 7 days away
    assert nse.next_monthly_expiry(date(2024, 12, 2)) == date(2024, 12, 26)
    us = OptionValuationEngine(Exchange.NYSE)
    # 2025-04-18 (third Friday) is Good Friday: expiry moves to Thursday 2025-04-17.
    assert us.next_monthly_expiry(date(2025, 4, 1)) == date(2025, 4, 17)
