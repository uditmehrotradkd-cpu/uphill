from __future__ import annotations

import asyncio
import math
import statistics
from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest
from fakes import FakeDatabase
from market_seed import settings as base_settings

from quantterm.connectors.base import Metric
from quantterm.demo import (
    DEMO_OPTIONS,
    DIVIDENDS,
    STOCKS,
    DemoSeedRefused,
    SeedReport,
    assert_demo_only,
    demo_settings,
    generate_bars,
    seed_demo,
    smile,
    with_database,
)
from quantterm.models import Exchange, OhlcvBar, session_bar_ts
from quantterm.db.repositories import DAILY, PriceRepository, SecurityRepository
from quantterm.terminal.command_router import CommandRouter
from quantterm.workspace import FAView, GPView, OVMEView, Workspace

AS_OF = datetime(2026, 10, 23, 21, 0, tzinfo=UTC)  # after the NSE and US closes
AAPL = next(s for s in STOCKS if s.symbol == "AAPL")
RELIANCE = next(s for s in STOCKS if s.symbol == "RELIANCE")


def run(coro: Any) -> Any:
    return asyncio.run(coro)


@pytest.fixture(scope="module")
def seeded() -> Iterator[tuple[FakeDatabase, SeedReport, Workspace]]:
    db = FakeDatabase()
    settings = demo_settings(base_settings(), "postgresql://demo@127.0.0.1/quantterm_demo")
    report = run(seed_demo(db, settings, AS_OF))
    yield db, report, Workspace(db, settings, clock=lambda: AS_OF)


def test_bars_cover_one_year_and_are_reproducible_across_seed_dates() -> None:
    end = date(2026, 10, 23)
    bars = generate_bars(RELIANCE, end)
    first = bars[0].ts.astimezone(Exchange.NSE.timezone).date()
    assert end - timedelta(days=365) < first <= end - timedelta(days=360) and 240 <= len(bars) <= 252
    later = {b.ts: b for b in generate_bars(RELIANCE, end + timedelta(days=14))}
    assert all(later[b.ts] == b for b in bars if b.ts in later)  # re-seeding later never rewrites history
    assert all(b.low <= min(b.open, b.close) <= max(b.open, b.close) <= b.high for b in bars)
    assert all(round(b.close / 0.05, 6).is_integer() for b in bars)  # NSE tick


def test_split_and_dividends_are_in_the_raw_prints() -> None:
    by_day = {b.ts.astimezone(Exchange.NASDAQ.timezone).date(): b for b in generate_bars(AAPL, date(2026, 10, 23))}
    before, after = by_day[date(2026, 6, 5)], by_day[date(2026, 6, 8)]
    assert 0.2 < after.close / before.close < 0.3  # 4-for-1 on the raw tape
    days = sorted(by_day)
    split = days.index(date(2026, 6, 8))
    med = [statistics.median(by_day[d].volume for d in w) for w in (days[split - 60:split], days[split:split + 60])]
    assert 2.5 < med[1] / med[0] < 6  # share count, and so traded volume, quadruples (volume also tracks |return|)


def test_seed_counts_and_idempotency(seeded: tuple[FakeDatabase, SeedReport, Workspace]) -> None:
    db, report, ws = seeded
    assert report.securities == 6 and report.splits == 1 and report.dividends == len(DIVIDENDS) == 8
    assert report.bars > 6 * 240 and report.fundamentals > 100 and report.macro == 26 and report.option_quotes > 200
    again = run(seed_demo(db, demo_settings(base_settings(), "postgresql://x@h/quantterm_demo"), AS_OF))
    assert (again.bars, again.fundamentals, again.macro, again.option_quotes) == (0, 0, 0, 0)


def test_terminal_functions_work_out_of_the_box(seeded: tuple[FakeDatabase, SeedReport, Workspace]) -> None:
    _, _, ws = seeded
    router = CommandRouter(ws, clock=lambda: AS_OF)

    async def views() -> list[Any]:
        return [await router.execute(c) for c in ("AAPL US Equity GP 1Y", "FA", "TCS IN Equity FA",
                                                  "HDFCBANK IN Equity FA", "RELIANCE IN Equity FA")]

    gp, aapl_fa, tcs_fa, bank_fa, rel_fa = run(views())
    assert all(v.kind == "VIEW" for v in (gp, aapl_fa, tcs_fa, bank_fa, rel_fa)), [v.message for v in (gp, aapl_fa)]
    assert isinstance(gp.view, GPView)
    closes = gp.view.close
    assert max(b / a for a, b in zip(closes, closes[1:])) < 1.25  # split removed by query-time adjustment

    assert isinstance(aapl_fa.view, FAView)
    shares = {d.label: d for d in aapl_fa.view.ev_table}["Diluted shares"]
    assert shares.value == pytest.approx(15_004_697_000 * 4) and "split 2026-06-08" in (shares.note or "")

    assert isinstance(tcs_fa.view, FAView)
    assert any(Metric.EBITDA in n and "DEMO_VENDOR" in n for n in tcs_fa.view.notes)  # 1.2% vendor break

    assert isinstance(bank_fa.view, FAView)
    bank_ev = {d.label: d for d in bank_fa.view.ev_table}
    assert "Total debt" not in bank_ev and bank_ev["P/TBV"].value is not None

    assert isinstance(rel_fa.view, FAView)
    revenue_fy25 = {r.label: r for r in rel_fa.view.rows}[Metric.REVENUE].cells[1]
    assert revenue_fy25.value == pytest.approx(968_550 * 1e7) and "rev 2" in (revenue_fy25.note or "")


def test_ovme_prices_off_the_listed_chain(seeded: tuple[FakeDatabase, SeedReport, Workspace]) -> None:
    _, _, ws = seeded

    async def view(symbol: str, exchange: Exchange) -> OVMEView:
        return await ws.ovme(await ws.securities.get(symbol, exchange), AS_OF, strikes_each_side=4)

    for symbol, exchange in (("RELIANCE", Exchange.NSE), ("AAPL", Exchange.NASDAQ)):
        v = run(view(symbol, exchange))
        spot, vol, rate, div = v.inputs
        assert vol.source == DEMO_OPTIONS and "listed mid implied vol" in (vol.note or "")
        assert rate.source.startswith(("RBI:", "FRED:")) and div.value is not None and div.value > 0
        assert len(v.chain) == 9 and all(r.call_quote and r.put_quote for r in v.chain)
        ivs = [(r.strike, r.call_quote.implied_vol, r.put_quote.implied_vol) for r in v.chain  # type: ignore[union-attr]
               if r.call_quote and r.put_quote]
        assert all(c is not None and p is not None for _, c, p in ivs)
        assert ivs[0][2] > ivs[-1][1]  # downside skew: low-strike puts richer than high-strike calls
        for row in v.chain:
            assert row.call_quote is not None and row.call_quote.bid <= row.call.npv <= row.call_quote.ask
        atm = min(v.chain, key=lambda r: abs(r.strike - (spot.value or 0)))
        assert atm.call.volatility == pytest.approx(atm.put.volatility, abs=0.02)


def test_smile_is_skewed_and_floored() -> None:
    assert smile(0.25, 90, 100, 0.1) > smile(0.25, 100, 100, 0.1) == pytest.approx(0.25)
    assert smile(0.25, 1, 100, 0.1) == pytest.approx(0.75)  # capped at 3x ATM


def test_guard_refuses_databases_with_real_data() -> None:
    db = FakeDatabase()

    async def scenario() -> None:
        securities = SecurityRepository(db)
        await securities.upsert("AAPL", Exchange.NASDAQ)
        await PriceRepository(db, securities).append_bars("AAPL", Exchange.NASDAQ, DAILY, [OhlcvBar(
            ts=session_bar_ts(Exchange.NASDAQ, date(2026, 10, 23)), open=1, high=1, low=1, close=1, volume=1)],
            "POLYGON")
        await assert_demo_only(db)

    with pytest.raises(DemoSeedRefused, match="POLYGON"):
        run(scenario())
    run(assert_demo_only(FakeDatabase()))


def test_database_name_is_validated_before_use() -> None:
    assert with_database("postgresql://u:p@127.0.0.1:5432/quantterm", "quantterm_demo") == \
        "postgresql://u:p@127.0.0.1:5432/quantterm_demo"
    with pytest.raises(ValueError):
        with_database("postgresql://u:p@h/x", 'demo"; DROP DATABASE quantterm; --')


def test_dividend_yield_restates_pre_split_dividend(seeded: tuple[FakeDatabase, SeedReport, Workspace]) -> None:
    _, _, ws = seeded
    v = run(ws.ovme(run(ws.securities.get("AAPL", Exchange.NASDAQ)), AS_OF, strikes_each_side=0))
    spot, _, _, div = v.inputs
    # 0.26 paid before the 4-for-1 split is 0.065 per current share; 0.07 after.
    assert div.value == pytest.approx(math.log1p((0.065 + 0.07) / (spot.value or 1)))
