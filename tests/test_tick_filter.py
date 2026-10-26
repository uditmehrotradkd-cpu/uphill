from __future__ import annotations

import random
import statistics
from datetime import UTC, datetime, timedelta

import pytest

from quantterm.hygiene.tick_filter import (
    Acceptance,
    HygieneConfig,
    QuarantineReason,
    RollingWindow,
    TickHygieneBank,
    TickHygieneFilter,
)
from quantterm.models import BookLevel, Exchange, OrderBookSnapshot, Tick

T0 = datetime(2024, 11, 4, 3, 45, tzinfo=UTC)  # 09:15 IST, NSE open


def _tick(i: int, price: float, symbol: str = "HDFCBANK", size: int = 10) -> Tick:
    return Tick(symbol=symbol, exchange=Exchange.NSE, timestamp=T0 + timedelta(seconds=i), price=price, size=size)


def _book(i: int, bids: list[tuple[float, int]], asks: list[tuple[float, int]], symbol: str = "HDFCBANK") -> OrderBookSnapshot:
    return OrderBookSnapshot(
        symbol=symbol, exchange=Exchange.NSE, timestamp=T0 + timedelta(seconds=i),
        bids=tuple(BookLevel(price=p, quantity=q) for p, q in bids),
        asks=tuple(BookLevel(price=p, quantity=q) for p, q in asks),
    )


def _filter(**cfg: object) -> TickHygieneFilter:
    return TickHygieneFilter("HDFCBANK", Exchange.NSE, HygieneConfig.model_validate(cfg))


def _warm(f: TickHygieneFilter, n: int = 30, base: float = 1750.0) -> int:
    for i in range(n):
        assert f.process(_tick(i, base + (0.05 if i % 2 else -0.05))).accepted
    return n


# ------------------------------------------------------------------------------ rolling statistics
def test_rolling_window_matches_brute_force_median_and_mad() -> None:
    rng = random.Random(7)
    window = RollingWindow(50)
    history: list[float] = []
    for _ in range(500):
        x = round(rng.gauss(1750, 3), 2) if rng.random() > 0.05 else round(rng.uniform(100, 5000), 2)
        window.push(x)
        history.append(x)
        current = history[-50:]
        m = statistics.median(current)
        assert window.median() == pytest.approx(m)
        assert window.mad(m) == pytest.approx(statistics.median(abs(v - m) for v in current))


def test_rolling_window_handles_duplicates_and_reset() -> None:
    window = RollingWindow(5)
    for v in [10.0, 10.0, 10.0, 11.0, 9.0, 10.0]:
        window.push(v)
    assert len(window) == 5 and window.median() == 10.0 and window.mad(10.0) == 0.0
    window.reset([1.0, 2.0, 3.0])
    assert window.median() == 2.0 and window.mad(2.0) == 1.0


# ------------------------------------------------------------------------------ MAD rule
def test_fat_finger_is_quarantined_with_diagnostics() -> None:
    f = _filter()
    i = _warm(f)
    bad = f.process(_tick(i, 17500.0))  # misplaced decimal
    assert not bad.accepted and bad.validated is None
    assert bad.reason is QuarantineReason.UNCONFIRMED_MAD_OUTLIER
    assert bad.outlier is not None
    assert bad.outlier.rolling_median == pytest.approx(1750.0)
    assert bad.outlier.rolling_mad == pytest.approx(0.05)
    assert bad.outlier.deviation_mads is not None and bad.outlier.deviation_mads > 1000
    assert list(f.outlier_ticks) == [bad.outlier]
    assert f.process(_tick(i + 1, 1750.10)).acceptance is Acceptance.WITHIN_MAD


def test_threshold_is_five_mads_from_rolling_median() -> None:
    f = _filter(tick_size=0.05)
    offsets = [-4.0, -2.0, 0.0, 2.0, 4.0]  # median 1000, MAD 2 (above both floors)
    for i in range(30):
        assert f.process(_tick(i, 1000.0 + offsets[i % 5])).accepted
    over = f.process(_tick(30, 1010.5))
    at_limit = f.process(_tick(31, 1010.0))
    assert over.reason is QuarantineReason.UNCONFIRMED_MAD_OUTLIER and over.outlier is not None
    assert over.outlier.deviation_mads == pytest.approx(5.25)
    assert at_limit.acceptance is Acceptance.WITHIN_MAD
    assert at_limit.validated is not None and at_limit.validated.deviation_mads == pytest.approx(5.0)


def test_flat_tape_uses_mad_floor() -> None:
    f = _filter()
    for i in range(20):
        f.process(_tick(i, 1750.0))
    # MAD = 0; floor = max(1 tick = 0.05, 5 bp of 1750 = 0.875) -> 5 x 0.875 = 4.375 band.
    assert f.process(_tick(21, 1750.05)).accepted
    assert f.process(_tick(22, 1754.0)).accepted
    assert not f.process(_tick(23, 1755.0)).accepted


# ------------------------------------------------------------------------------ book confirmation
def test_genuine_gap_is_confirmed_by_order_book() -> None:
    f = _filter()
    i = _warm(f)
    f.update_book(_book(i, bids=[(1924.9, 200), (1924.5, 400)], asks=[(1925.1, 150), (1925.5, 300)]))
    v = f.process(_tick(i, 1925.0))  # +10 % on results, book already re-centred
    assert v.acceptance is Acceptance.BOOK_CONFIRMED and v.validated is not None
    assert v.validated.deviation_mads > 5
    assert not f.outlier_ticks


@pytest.mark.parametrize(
    ("book_offset_s", "bids", "asks", "why"),
    [
        (-5, [(1924.9, 200)], [(1925.1, 150)], "stale"),
        (0, [(1926.0, 200)], [(1924.0, 150)], "crossed"),
        (0, [(1749.95, 900)], [(1750.05, 800)], "book still at old level"),
        (0, [(1924.9, 200)], [], "one-sided"),
    ],
)
def test_book_that_does_not_corroborate_leaves_print_quarantined(
    book_offset_s: int, bids: list[tuple[float, int]], asks: list[tuple[float, int]], why: str
) -> None:
    f = _filter()
    i = _warm(f)
    f.update_book(_book(i + book_offset_s, bids=bids, asks=asks))
    v = f.process(_tick(i, 1925.0))
    assert v.reason is QuarantineReason.UNCONFIRMED_MAD_OUTLIER, why
    assert v.outlier is not None and v.outlier.best_bid == bids[0][0]


def test_thin_book_does_not_confirm() -> None:
    f = _filter(min_confirming_depth=500)
    i = _warm(f)
    f.update_book(_book(i, bids=[(1924.9, 200)], asks=[(1925.1, 150)]))
    assert not f.process(_tick(i, 1925.0)).accepted


def test_older_book_snapshot_does_not_replace_newer() -> None:
    f = _filter()
    i = _warm(f)
    f.update_book(_book(i, bids=[(1924.9, 200)], asks=[(1925.1, 150)]))
    f.update_book(_book(i - 1, bids=[(1749.95, 900)], asks=[(1750.05, 800)]))  # late, out-of-order
    assert f.process(_tick(i, 1925.0)).acceptance is Acceptance.BOOK_CONFIRMED


# ------------------------------------------------------------------------------ depth-less fallback
def test_regime_shift_accepted_after_consistent_outliers_without_book() -> None:
    f = _filter(regime_confirmations=4)
    i = _warm(f)
    verdicts = [f.process(_tick(i + k, 1925.0 + 0.1 * k)) for k in range(4)]
    assert [v.accepted for v in verdicts] == [False, False, False, True]
    assert verdicts[-1].acceptance is Acceptance.REGIME_SHIFT
    assert len(f.outlier_ticks) == 3
    assert f.process(_tick(i + 10, 1925.5)).accepted
    # Still armed on the new level: a bad print right after the shift is caught.
    assert not f.process(_tick(i + 11, 192.5)).accepted


def test_inconsistent_outliers_do_not_trigger_regime_shift() -> None:
    f = _filter(regime_confirmations=3)
    i = _warm(f)
    for k, p in enumerate([2500.0, 900.0, 3100.0, 50.0]):
        assert not f.process(_tick(i + k, p)).accepted


# ------------------------------------------------------------------------------ structural rejects
@pytest.mark.parametrize("price", [0.0, -1.0])
def test_non_positive_price(price: float) -> None:
    assert _filter().process(_tick(0, price)).reason is QuarantineReason.NON_POSITIVE_PRICE


def test_out_of_order_tick() -> None:
    f = _filter()
    f.process(_tick(10, 1750.0))
    assert f.process(_tick(5, 1750.0)).reason is QuarantineReason.OUT_OF_ORDER


def test_circuit_band() -> None:
    f = TickHygieneFilter("HDFCBANK", Exchange.NSE, price_band=(1575.0, 1925.0))  # 10 % band
    assert f.process(_tick(0, 1930.0)).reason is QuarantineReason.OUTSIDE_PRICE_BAND


def test_filter_refuses_other_listings() -> None:
    f = _filter()
    with pytest.raises(ValueError):
        f.process(_tick(0, 3000.0, symbol="TCS"))
    with pytest.raises(ValueError):
        f.update_book(_book(0, [(1.0, 1)], [(2.0, 1)], symbol="TCS"))


def test_quarantine_log_is_bounded() -> None:
    f = _filter(quarantine_log_size=3)
    i = _warm(f)
    for k in range(5):
        f.process(_tick(i + k, 17500.0 + k * 1000))
    assert [o.tick.price for o in f.outlier_ticks] == [19500.0, 20500.0, 21500.0]


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        HygieneConfig(window=5, min_samples=10)


def test_book_model_validation() -> None:
    with pytest.raises(ValueError, match="descending"):
        _book(0, bids=[(10.0, 1), (11.0, 1)], asks=[(12.0, 1)])
    assert _book(0, bids=[(12.0, 1)], asks=[(11.0, 1)]).crossed


# ------------------------------------------------------------------------------ bank
def test_bank_routes_books_and_uses_market_tick_sizes() -> None:
    bank = TickHygieneBank()
    assert bank.filter_for(Exchange.NASDAQ, "AAPL").config.tick_size == 0.01
    assert bank.filter_for(Exchange.NSE, "HDFCBANK").config.tick_size == 0.05
    for i in range(30):
        bank.process(_tick(i, 1750.0 + 0.05 * (i % 3)))
    bank.update_book(_book(30, bids=[(1924.9, 200)], asks=[(1925.1, 150)]))
    assert bank.process(_tick(30, 1925.0)).acceptance is Acceptance.BOOK_CONFIRMED
    assert not bank.process(_tick(31, 19.25)).accepted
    assert [o.tick.price for o in bank.outlier_ticks()] == [19.25]
