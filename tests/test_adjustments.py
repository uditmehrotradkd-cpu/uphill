from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any
from zoneinfo import ZoneInfo

import polars as pl
import pytest
from fakes import FakeDatabase

from quantterm.corporate_actions import (
    CorporateAction,
    CorporateActionType,
    bonus_factor,
    rights_factor,
    split_factor,
)
from quantterm.db.repositories import DAILY, CorporateActionRepository, PriceRepository, SecurityRepository
from quantterm.engine.adjustments import actions_frame, apply_adjustments, get_adjusted_series
from quantterm.models import Exchange, OhlcvBar, session_bar_ts

ET = ZoneInfo("America/New_York")
IST = ZoneInfo("Asia/Kolkata")
ANNOUNCED = datetime(2024, 5, 22, 20, 0, tzinfo=UTC)


def _bars(dates: list[date], closes: list[float], volumes: list[int], vwap: bool = False) -> pl.DataFrame:
    data: dict[str, Any] = {
        "session_date": dates,
        "open": closes,
        "high": [c * 1.01 for c in closes],
        "low": [c * 0.99 for c in closes],
        "close": closes,
        "volume": volumes,
    }
    if vwap:
        data["vwap"] = [c * 1.001 for c in closes]
    return pl.DataFrame(data)


def _act(kind: CorporateActionType, ex: date, *, factor: Decimal | None = None, dividend: str | None = None,
         symbol: str = "NVDA", exchange: Exchange = Exchange.NASDAQ, announced: datetime = ANNOUNCED) -> CorporateAction:
    return CorporateAction(
        symbol=symbol, exchange=exchange, action_type=kind, ex_date=ex, announced_at=announced,
        adjustment_factor=factor, raw_dividend_amount=None if dividend is None else Decimal(dividend), source="TEST",
    )


def _split(ex: date, new: int, old: int) -> CorporateAction:
    return _act(CorporateActionType.SPLIT, ex, factor=split_factor(Decimal(new), Decimal(old)))


def _dividend(ex: date, amount: str) -> CorporateAction:
    return _act(CorporateActionType.CASH_DIVIDEND, ex, dividend=amount)


@pytest.fixture
def nvda_split_window() -> pl.DataFrame:
    # NVDA 10-for-1, ex-date 2024-06-10; raw closes and volumes as printed.
    return _bars(
        [date(2024, 6, 6), date(2024, 6, 7), date(2024, 6, 10), date(2024, 6, 11)],
        [1209.98, 1208.88, 121.79, 120.91],
        [41_000_000, 41_400_000, 314_000_000, 222_000_000],
        vwap=True,
    )


# ----------------------------------------------------------------------------- pure transform
def test_split_adjusts_history_prices_volume_vwap_without_mutating_raw(nvda_split_window: pl.DataFrame) -> None:
    raw_copy = nvda_split_window.clone()
    adj = apply_adjustments(nvda_split_window, actions_frame([_split(date(2024, 6, 10), 10, 1)]))
    assert adj["close"].to_list() == pytest.approx([120.998, 120.888, 121.79, 120.91])
    assert adj["vwap"][0] == pytest.approx(1209.98 * 1.001 * 0.1)
    assert adj["volume"].to_list() == pytest.approx([410_000_000, 414_000_000, 314_000_000, 222_000_000])
    assert nvda_split_window.equals(raw_copy)


def test_gp_toggles_are_independent() -> None:
    raw = _bars([date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)], [200.0, 99.0, 99.5], [100, 200, 200])
    actions = actions_frame([_split(date(2024, 1, 3), 2, 1), _dividend(date(2024, 1, 4), "0.99")])
    both = apply_adjustments(raw, actions)
    splits_only = apply_adjustments(raw, actions, adjust_dividends=False)
    dividends_only = apply_adjustments(raw, actions, adjust_splits=False)
    none = apply_adjustments(raw, actions, adjust_splits=False, adjust_dividends=False)

    div_factor = (99.0 - 0.99) / 99.0
    assert both["close"].to_list() == pytest.approx([200 * 0.5 * div_factor, 99.0 * div_factor, 99.5])
    assert splits_only["close"].to_list() == pytest.approx([100.0, 99.0, 99.5])
    assert dividends_only["close"].to_list() == pytest.approx([200 * div_factor, 99.0 * div_factor, 99.5])
    assert dividends_only["volume"].to_list() == pytest.approx([100, 200, 200])  # cash never touches volume
    assert none["close"].to_list() == pytest.approx([200.0, 99.0, 99.5])


def test_indian_bonus_one_for_one() -> None:
    # RELIANCE 1:1 bonus, ex/record date 2024-10-28.
    raw = _bars([date(2024, 10, 25), date(2024, 10, 28)], [2655.0, 1338.0], [1_000, 2_000])
    bonus = _act(CorporateActionType.BONUS, date(2024, 10, 28), factor=bonus_factor(Decimal(1), Decimal(1)),
                 symbol="RELIANCE", exchange=Exchange.NSE)
    adj = apply_adjustments(raw, actions_frame([bonus]))
    assert adj["close"].to_list() == pytest.approx([1327.5, 1338.0])
    assert adj["volume"].to_list() == pytest.approx([2_000, 2_000])


def test_rights_issue_is_a_capital_change() -> None:
    raw = _bars([date(2020, 5, 13), date(2020, 5, 14)], [1445.0, 1433.0], [10, 10])
    factor = rights_factor(Decimal(1), Decimal(15), Decimal(1257), Decimal(1445))
    rights = actions_frame([_act(CorporateActionType.RIGHTS, date(2020, 5, 14), factor=factor,
                                 symbol="RELIANCE", exchange=Exchange.NSE)])
    assert apply_adjustments(raw, rights)["close"][0] == pytest.approx(1433.25)
    assert apply_adjustments(raw, rights, adjust_splits=False)["close"][0] == pytest.approx(1445.0)


def test_same_day_split_and_dividend_uses_post_split_prior_close() -> None:
    raw = _bars([date(2024, 1, 2), date(2024, 1, 3)], [200.0, 99.0], [100, 200])
    adj = apply_adjustments(raw, actions_frame([_split(date(2024, 1, 3), 2, 1), _dividend(date(2024, 1, 3), "1")]))
    # Post-split prior close 100, dividend 1 -> 0.99; combined with the 0.5 split factor.
    assert adj["close"][0] == pytest.approx(200.0 * 0.5 * 0.99)
    assert adj["volume"][0] == pytest.approx(200)


def test_multiple_actions_compound_backwards() -> None:
    raw = _bars([date(2020, 1, 2), date(2021, 1, 4), date(2022, 1, 3)], [400.0, 200.0, 100.0], [1, 1, 1])
    adj = apply_adjustments(raw, actions_frame([_split(date(2021, 1, 4), 2, 1), _split(date(2022, 1, 3), 2, 1)]))
    assert adj["close"].to_list() == pytest.approx([100.0, 100.0, 100.0])
    assert adj["price_factor"].to_list() == pytest.approx([0.25, 0.5, 1.0])


def test_reverse_split() -> None:
    raw = _bars([date(2024, 1, 2), date(2024, 1, 3)], [0.5, 5.0], [1_000_000, 100_000])
    adj = apply_adjustments(raw, actions_frame([_split(date(2024, 1, 3), 1, 10)]))
    assert adj["close"].to_list() == pytest.approx([5.0, 5.0])
    assert adj["volume"].to_list() == pytest.approx([100_000, 100_000])


def test_dividend_exceeding_prior_close_is_a_data_error() -> None:
    raw = _bars([date(2024, 1, 2), date(2024, 1, 3)], [1.0, 0.5], [1, 1])
    with pytest.raises(ValueError, match="prior close"):
        apply_adjustments(raw, actions_frame([_dividend(date(2024, 1, 3), "1.5")]))


def test_ex_date_on_first_bar_is_ignored_not_fatal() -> None:
    raw = _bars([date(2024, 1, 3), date(2024, 1, 4)], [10.0, 10.1], [1, 1])
    adj = apply_adjustments(raw, actions_frame([_dividend(date(2024, 1, 3), "0.5")]))
    assert adj["price_factor"].to_list() == [1.0, 1.0]


# ------------------------------------------------------------------------- storage-backed series
def _run(coro: Any) -> Any:
    return asyncio.run(coro)


@pytest.fixture
def db() -> FakeDatabase:
    db = FakeDatabase()
    securities = SecurityRepository(db)
    _run(securities.upsert("NVDA", Exchange.NASDAQ))
    prices = PriceRepository(db, securities)
    sessions = [date(2024, 6, 6), date(2024, 6, 7), date(2024, 6, 10), date(2024, 6, 11)]
    closes = [1209.98, 1208.88, 121.79, 120.91]
    bars = [OhlcvBar(ts=session_bar_ts(Exchange.NASDAQ, d), open=c, high=c * 1.01, low=c * 0.99, close=c,
                     volume=1_000_000, vwap=c) for d, c in zip(sessions, closes)]
    _run(prices.append_bars("NVDA", Exchange.NASDAQ, DAILY, bars, "POLYGON"))
    _run(CorporateActionRepository(db, securities).append([
        _split(date(2024, 6, 10), 10, 1),
        # Dividend announced after the split; raw amount is per post-split share.
        _act(CorporateActionType.CASH_DIVIDEND, date(2024, 6, 11), dividend="0.01",
             announced=datetime(2024, 5, 29, 20, 0, tzinfo=UTC)),
    ]))
    return db


def _series(db: FakeDatabase, as_of: date | datetime, start: date = date(2024, 6, 6), end: date = date(2024, 6, 11),
            **toggles: bool) -> pl.DataFrame:
    return _run(get_adjusted_series("NVDA", start, end, as_of, exchange=Exchange.NASDAQ, db=db, **toggles))


def test_series_applies_actions_with_ex_date_after_end_date(db: FakeDatabase) -> None:
    out = _series(db, date(2024, 6, 12), end=date(2024, 6, 7))
    div = (121.79 - 0.01) / 121.79
    assert out["close"].to_list() == pytest.approx([120.998 * div, 120.888 * div])
    assert out["session_date"].to_list() == [date(2024, 6, 6), date(2024, 6, 7)]


def test_series_ignores_future_ex_dates_and_unannounced_actions(db: FakeDatabase) -> None:
    after_split_close = _series(db, datetime(2024, 6, 10, 21, 0, tzinfo=UTC))
    assert after_split_close["close"].to_list() == pytest.approx([120.998, 120.888, 121.79])  # no dividend yet

    db.actions.append({"ticker": "NVDA", "exchange": "NASDAQ", "ex_date": date(2024, 6, 7),
                       "action_type": "SPLIT", "adjustment_factor": Decimal("0.5"), "raw_dividend_amount": None,
                       "announced_at": datetime(2024, 6, 20, tzinfo=UTC), "source": "LATE_VENDOR_FIX"})
    before_booking = _series(db, datetime(2024, 6, 10, 21, 0, tzinfo=UTC))
    assert before_booking["close"][0] == pytest.approx(120.998)  # late-booked action invisible at as-of


def test_series_excludes_session_still_trading(db: FakeDatabase) -> None:
    intraday = _series(db, datetime(2024, 6, 11, 11, 0, tzinfo=ET))
    assert intraday["session_date"].to_list()[-1] == date(2024, 6, 10)
    # 2024-06-11 ex-dividend is <= as-of date, so bars before it are dividend-adjusted already.
    assert intraday["close"][-1] == pytest.approx(121.79 * (121.79 - 0.01) / 121.79)


def test_series_rejects_naive_datetime(db: FakeDatabase) -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        _series(db, datetime(2024, 6, 12))


def test_series_toggle_off_returns_raw_prices(db: FakeDatabase) -> None:
    raw = _series(db, date(2024, 6, 12), adjust_splits=False, adjust_dividends=False)
    assert raw["close"].to_list() == pytest.approx([1209.98, 1208.88, 121.79, 120.91])
    assert sorted(b["close"] for b in db.bars.values()) == sorted([1209.98, 1208.88, 121.79, 120.91])


def test_intraday_series_uses_daily_close_for_dividend_factor() -> None:
    db = FakeDatabase()
    securities = SecurityRepository(db)
    _run(securities.upsert("INFY", Exchange.NSE))
    prices = PriceRepository(db, securities)
    minute = timedelta(minutes=1)
    d1, d2 = date(2024, 5, 30), date(2024, 5, 31)
    _run(prices.append_bars("INFY", Exchange.NSE, DAILY, [
        OhlcvBar(ts=session_bar_ts(Exchange.NSE, d1), open=1420, high=1430, low=1415, close=1425.0, volume=1),
    ], "NSE_BHAVCOPY"))
    t1 = datetime(2024, 5, 30, 15, 29, tzinfo=IST)
    t2 = datetime(2024, 5, 31, 9, 15, tzinfo=IST)
    _run(prices.append_bars("INFY", Exchange.NSE, minute, [
        OhlcvBar(ts=t1, open=1426, high=1427, low=1424, close=1426.0, volume=10),
        OhlcvBar(ts=t2, open=1400, high=1401, low=1399, close=1400.0, volume=10),
    ], "KITE:LIVE"))
    _run(CorporateActionRepository(db, securities).append([
        _act(CorporateActionType.CASH_DIVIDEND, d2, dividend="28", symbol="INFY", exchange=Exchange.NSE,
             announced=datetime(2024, 4, 18, 16, 0, tzinfo=IST)),
    ]))
    out = _run(get_adjusted_series("INFY", d1, d2, datetime(2024, 5, 31, 9, 17, tzinfo=IST),
                                   exchange=Exchange.NSE, db=db, interval=minute))
    factor = (1425.0 - 28) / 1425.0  # official close, not the 15:29 print
    assert out["close"].to_list() == pytest.approx([1426.0 * factor, 1400.0])
    assert out["session_date"].to_list() == [d1, d2]


def test_intraday_bar_not_returned_before_it_completes() -> None:
    db = FakeDatabase()
    securities = SecurityRepository(db)
    _run(securities.upsert("INFY", Exchange.NSE))
    minute = timedelta(minutes=1)
    t = datetime(2024, 5, 31, 9, 15, tzinfo=IST)
    _run(PriceRepository(db, securities).append_bars("INFY", Exchange.NSE, minute, [
        OhlcvBar(ts=t, open=1400, high=1401, low=1399, close=1400.0, volume=10)], "KITE:LIVE"))
    during = _run(get_adjusted_series("INFY", t.date(), t.date(), t + timedelta(seconds=30),
                                      exchange=Exchange.NSE, db=db, interval=minute))
    after = _run(get_adjusted_series("INFY", t.date(), t.date(), t + minute,
                                     exchange=Exchange.NSE, db=db, interval=minute))
    assert during.is_empty() and after.height == 1
