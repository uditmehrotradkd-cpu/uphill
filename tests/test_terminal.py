from __future__ import annotations

import asyncio
import math
from collections.abc import Iterator
from datetime import UTC, date, datetime, time
from typing import Any

import pytest
from fakes import FakeDatabase
from market_seed import AS_OF, BONUS_EX, CRORE, RELIANCE_FY24, RESTATED, seeded_workspace

from quantterm.connectors.base import Metric
from quantterm.db.repositories import OutlierRow
from quantterm.models import Exchange, PeriodType
from quantterm.quant.derivatives_ovme import ExerciseStyle, OptionType
from quantterm.terminal.command_router import (
    CommandError,
    CommandRouter,
    FAArgs,
    Function,
    GPArgs,
    display_name,
    parse,
)
from quantterm.workspace import (
    ABSENT_CONFIDENCE,
    DERIVED,
    FAView,
    GPView,
    OVMERequest,
    OVMEView,
    RVView,
    Workspace,
    format_value,
)


def run(coro: Any) -> Any:
    return asyncio.run(coro)


@pytest.fixture(scope="module")
def market() -> Iterator[tuple[FakeDatabase, Workspace]]:
    yield run(seeded_workspace())


@pytest.fixture
def router(market: tuple[FakeDatabase, Workspace]) -> CommandRouter:
    return CommandRouter(market[1], clock=lambda: AS_OF)


# ------------------------------------------------------------------------------------------ parser
def test_parse_security_with_function_and_go_key() -> None:
    cmd = parse("aapl us equity gp 2y <GO>")
    assert cmd.kind == "RUN" and cmd.security is not None
    assert cmd.security.ticker == "AAPL" and cmd.security.exchanges == (Exchange.NASDAQ, Exchange.NYSE)
    assert cmd.function is Function.GP and cmd.args == GPArgs(lookback_days=731)


@pytest.mark.parametrize(("text", "exchanges"), [
    ("RELIANCE IN Equity", (Exchange.NSE,)),
    ("RELIANCE IB Equity", (Exchange.BSE,)),
    ("IBM UN Equity", (Exchange.NYSE,)),
    ("MSFT UW Equity", (Exchange.NASDAQ,)),
    ("M&M NSE Equity", (Exchange.NSE,)),
])
def test_parse_exchange_codes(text: str, exchanges: tuple[Exchange, ...]) -> None:
    cmd = parse(text)
    assert cmd.security is not None and cmd.security.exchanges == exchanges and cmd.function is None


def test_parse_function_arguments() -> None:
    assert parse("FA Q 8").args == FAArgs(period_type=PeriodType.QUARTER, periods=8)
    assert parse("GP 6M NODIV").args == GPArgs(lookback_days=183, adjust_dividends=False)
    assert parse("GP RAW").args == GPArgs(adjust_splits=False, adjust_dividends=False)
    ovme = parse("OVME P 250 2025-01-17 vol=24% rate=0.043 div=1.5% px=5.10").args
    assert ovme == OVMERequest(option_type=OptionType.PUT, strike=250.0, expiry=date(2025, 1, 17), volatility=0.24,
                               rate=0.043, dividend_yield=0.015, premium=5.10)


def test_parse_asof_is_end_of_day() -> None:
    cmd = parse("ASOF 2024-06-28")
    assert cmd.as_of == datetime.combine(date(2024, 6, 28), time.max, tzinfo=UTC)
    assert parse("ASOF NOW").as_of is None


@pytest.mark.parametrize("text", ["FOO XX Equity", "GP 7W", "OVME vol=abc", "OVME foo=1", "ASOF 2024-13-01",
                                  "ZZZ", "RV extra", "WATCH ADD", "OVME vol=900%", ""])
def test_parse_errors_are_explicit(text: str) -> None:
    with pytest.raises(CommandError):
        parse(text)


# ------------------------------------------------------------------------------------------ router
def test_function_without_security_is_an_error(router: CommandRouter) -> None:
    result = run(router.execute("GP"))
    assert result.kind == "ERROR" and "no active security" in result.message


def test_unknown_listing_is_reported_not_raised(router: CommandRouter) -> None:
    result = run(router.execute("TCS IN Equity"))
    assert result.kind == "ERROR" and "TCS" in result.message


def test_security_then_functions_keep_context_and_back(router: CommandRouter) -> None:
    async def scenario() -> list[Any]:
        return [await router.execute(t) for t in
                ("RELIANCE IN Equity", "GP 6M", "FA", "ONGC IN Equity", "BACK")]

    loaded, gp, fa, ongc, back = run(scenario())
    assert loaded.kind == "STATUS" and loaded.message == "loaded RELIANCE IN Equity"
    assert gp.kind == "VIEW" and isinstance(gp.view, GPView) and gp.view.security.symbol == "RELIANCE"
    assert isinstance(fa.view, FAView) and fa.view.security.symbol == "RELIANCE"
    assert isinstance(ongc.view, FAView) and ongc.view.security.symbol == "ONGC"  # function kept
    assert isinstance(back.view, FAView) and back.view.security.symbol == "RELIANCE"
    assert router.context.startswith("RELIANCE IN Equity | FA | live")


def test_us_composite_code_resolves_nasdaq_listing(router: CommandRouter) -> None:
    result = run(router.execute("AAPL US Equity GP 3M"))
    assert result.kind == "VIEW" and result.security is not None
    assert result.security.exchange is Exchange.NASDAQ and display_name(result.security) == "AAPL US Equity"


# ------------------------------------------------------------------------------------------ GP
def test_gp_is_corporate_action_adjusted_with_indicators(market: tuple[FakeDatabase, Workspace]) -> None:
    _, ws = market

    async def views() -> tuple[GPView, GPView]:
        sec = await ws.resolve("RELIANCE", [Exchange.NSE])
        adj = await ws.gp(sec, AS_OF, lookback_days=183)
        raw = await ws.gp(sec, AS_OF, lookback_days=183, adjust_splits=False, adjust_dividends=False)
        return adj, raw

    adj, raw = run(views())
    i = adj.dates.index(max(d for d in adj.dates if d < BONUS_EX))
    # Raw prints halve across the 1:1 bonus; the adjusted series is continuous (no -50% gap).
    assert raw.close[i + 1] / raw.close[i] < 0.6
    assert 0.9 < adj.close[i + 1] / adj.close[i] < 1.1
    assert adj.close[-1] == raw.close[-1]  # latest print is never restated
    rsi = [r for r in adj.rsi if r is not None]
    assert len(rsi) == len(adj.dates) and all(0 <= r <= 100 for r in rsi)  # warmed up before the window
    summary = {d.label: d for d in adj.summary}
    assert summary["Last close"].value == adj.close[-1]
    assert summary["Last close"].timestamp == datetime(2024, 12, 31, 15, 30, tzinfo=Exchange.NSE.timezone)


# ------------------------------------------------------------------------------------------ FA / EV
def test_fa_is_point_in_time_and_adjusts_share_data(market: tuple[FakeDatabase, Workspace]) -> None:
    _, ws = market

    async def views() -> tuple[FAView, FAView]:
        sec = await ws.resolve("RELIANCE", [Exchange.NSE])
        return await ws.fa(sec, AS_OF, periods=2), await ws.fa(sec, datetime(2024, 6, 28, 23, 59, tzinfo=UTC))

    now, then = run(views())
    rows = {r.label: r for r in now.rows}
    assert now.periods == [date(2024, 3, 31), date(2023, 3, 31)]
    revenue_now = rows[Metric.REVENUE].cells[0]
    assert revenue_now.value == float(RELIANCE_FY24[Metric.REVENUE] + 1500 * CRORE)  # restatement visible
    assert revenue_now.timestamp == RESTATED
    revenue_then = {r.label: r for r in then.rows}[Metric.REVENUE].cells[0]
    assert revenue_then.value == float(RELIANCE_FY24[Metric.REVENUE])  # original filing only
    # Shares/EPS restated for the October bonus only when the bonus is known (as_of after announcement).
    shares_now, eps_now = rows[Metric.DILUTED_SHARES].cells[0], rows[Metric.EPS_DILUTED].cells[0]
    assert shares_now.value == pytest.approx(float(RELIANCE_FY24[Metric.DILUTED_SHARES]) * 2)
    assert eps_now.value == pytest.approx(float(RELIANCE_FY24[Metric.EPS_DILUTED]) / 2)
    assert "bonus 2024-10-28" in (eps_now.note or "")
    assert {r.label: r for r in then.rows}[Metric.DILUTED_SHARES].cells[0].value == float(
        RELIANCE_FY24[Metric.DILUTED_SHARES])
    margin = {r.label: r for r in now.ratios}["EBITDA margin"].cells[0]
    assert margin.value == pytest.approx(178677 / (901064 + 1500) * 100)
    assert margin.source == DERIVED and margin.timestamp == RESTATED


def test_fa_ev_bridge_reconciles_and_flags_vendor_break(market: tuple[FakeDatabase, Workspace]) -> None:
    _, ws = market

    async def view() -> FAView:
        return await ws.fa(await ws.resolve("RELIANCE", [Exchange.NSE]), AS_OF)

    fa = run(view())
    ev = {d.label: d for d in fa.ev_table}
    price, shares = ev["Last price"].value, ev["Diluted shares"].value
    assert price is not None and shares is not None
    mcap = price * shares
    assert ev["Market cap"].value == pytest.approx(mcap)
    bridge = mcap + sum(float(RELIANCE_FY24[m]) for m in (Metric.TOTAL_DEBT, Metric.LEASE_LIABILITIES,
                                                          Metric.MINORITY_INTEREST)) \
        - float(RELIANCE_FY24[Metric.CASH_AND_EQUIVALENTS])
    assert ev["Enterprise value"].value == pytest.approx(bridge, rel=1e-9)
    assert ev["EV/EBITDA"].value == pytest.approx(bridge / float(RELIANCE_FY24[Metric.EBITDA]), rel=1e-9)
    assert ev["Preferred stock"].value == 0.0 and ev["Preferred stock"].confidence_score == ABSENT_CONFIDENCE
    # Revenue: restated XBRL vs vendor copy of the original -> break, alert, lowered confidence.
    rev = next(i for i in fa.reconciliation if i.metric == Metric.REVENUE and i.period_end_date == date(2024, 3, 31))
    assert rev.is_break and rev.confidence_score < 1.0
    assert any("revenue" in n for n in fa.notes)
    assert ev["EV/Sales"].confidence_score == min(rev.confidence_score, ev["Enterprise value"].confidence_score)


def test_bank_ev_excludes_debt_and_cash(market: tuple[FakeDatabase, Workspace]) -> None:
    _, ws = market

    async def table() -> list[Any]:
        sec = await ws.resolve("HDFCBANK", [Exchange.NSE])
        return (await ws.enterprise_value(sec, AS_OF))[0]

    ev = {d.label: d for d in run(table())}
    assert "Total debt" not in ev and "Cash & ST investments" not in ev and "EV/EBITDA" not in ev
    assert ev["Enterprise value"].note == "bank/NBFC basis: debt, leases and cash excluded"
    mcap = ev["Market cap"].value
    assert mcap is not None and ev["Enterprise value"].value == pytest.approx(mcap + 20000 * float(CRORE))
    tbv = (455636 - 7000 - 1000) * float(CRORE)
    assert ev["P/TBV"].value == pytest.approx(mcap / tbv)


# ------------------------------------------------------------------------------------------ RV / beta
def test_rv_matrix_peers_medians_and_beta(market: tuple[FakeDatabase, Workspace]) -> None:
    _, ws = market

    async def view() -> RVView:
        return await ws.rv(await ws.resolve("RELIANCE", [Exchange.NSE]), AS_OF)

    rv = run(view())
    assert [r.security.symbol for r in rv.rows] == ["RELIANCE", "ONGC"] and rv.rows[0].is_subject
    ongc = rv.rows[1].cells
    assert rv.medians["P/E"].value == ongc["P/E"].value  # one comp: median is that comp, subject excluded
    beta = rv.rows[0].cells["Beta (adj)"]
    assert beta.value is not None and 0.67 * 0.9 + 0.33 < beta.value < 0.67 * 1.3 + 0.33  # Blume-adjusted ~1.1 raw
    pe = rv.rows[0].cells["P/E"]
    assert pe.value == pytest.approx(rv.rows[0].cells["Price"].value / (float(RELIANCE_FY24[Metric.EPS_DILUTED]) / 2))


def test_rv_requires_industry(market: tuple[FakeDatabase, Workspace]) -> None:
    result = run(CommandRouter(market[1], clock=lambda: AS_OF).execute("NIFTY IN Equity RV"))
    assert result.kind == "ERROR" and "industry" in result.message


# ------------------------------------------------------------------------------------------ OVME
def test_ovme_chain_put_call_parity_and_inputs(market: tuple[FakeDatabase, Workspace]) -> None:
    _, ws = market

    async def view() -> OVMEView:
        return await ws.ovme(await ws.resolve("RELIANCE", [Exchange.NSE]), AS_OF)

    v = run(view())
    assert v.expiry == date(2025, 1, 30) and v.exercise is ExerciseStyle.EUROPEAN and len(v.chain) == 11
    spot, vol, rate, div = v.inputs
    assert rate.value == pytest.approx(math.log1p(0.066)) and rate.source == "RBI:TBILL91"
    assert vol.provenance == "quant.technicals.realized_volatility" and 0.05 < (vol.value or 0) < 0.6
    # Rs 10 paid per pre-bonus share is Rs 5 per current share after the 1:1 bonus.
    assert div.value == pytest.approx(math.log1p(5 / (spot.value or 1)))
    s, r, q = spot.value or 0, rate.value or 0, div.value or 0
    for row in v.chain:
        t = row.call.time_to_expiry_years
        parity = s * math.exp(-q * t) - row.strike * math.exp(-r * t)
        assert row.call.npv - row.put.npv == pytest.approx(parity, abs=1e-6 * s)
        assert 0 < row.call.delta < 1 and -1 < row.put.delta < 0


def test_ovme_calculator_implied_vol_round_trip(market: tuple[FakeDatabase, Workspace]) -> None:
    _, ws = market

    async def views() -> tuple[OVMEView, OVMEView]:
        sec = await ws.resolve("RELIANCE", [Exchange.NSE])
        priced = await ws.ovme(sec, AS_OF, OVMERequest(option_type=OptionType.CALL, strike=1300, volatility=0.27),
                               strikes_each_side=0)
        assert priced.calculator is not None
        implied = await ws.ovme(sec, AS_OF, OVMERequest(option_type=OptionType.CALL, strike=1300,
                                                        premium=priced.calculator.npv), strikes_each_side=0)
        return priced, implied

    priced, implied = run(views())
    assert implied.calculator is not None
    assert implied.calculator.volatility == pytest.approx(0.27, abs=1e-6)
    assert any("implied from premium" in n for n in implied.notes)


def test_ovme_bse_requires_explicit_expiry(market: tuple[FakeDatabase, Workspace]) -> None:
    _, ws = market
    run(ws.securities.upsert("SBIN", Exchange.BSE))
    result = run(CommandRouter(ws, clock=lambda: AS_OF).execute("SBIN IB Equity OVME"))
    assert result.kind == "ERROR" and "pass the expiry explicitly" in result.message


# ------------------------------------------------------------------------------------------ hygiene audit
def test_tick_hygiene_summary_reads_quarantine(market: tuple[FakeDatabase, Workspace]) -> None:
    db, ws = market
    at = datetime(2024, 12, 31, 9, 0, tzinfo=UTC)
    db.outliers.extend([
        OutlierRow("RELIANCE", "NSE", at, 1700.0, 10, "MAD_OUTLIER", 1260.0, 1.2, 366.7, None, None, None, "KITE"),
        OutlierRow("RELIANCE", "NSE", at.replace(hour=10), 12.6, 5, "MAD_OUTLIER", 1261.0, 1.1, 1135.0, None, None,
                   None, "KITE"),
        OutlierRow("RELIANCE", "NSE", at.replace(year=2023), 1.0, 1, "MAD_OUTLIER", 1.0, 1.0, 9.0, None, None, None,
                   "KITE"),
    ])

    async def summary() -> list[Any]:
        return await ws.tick_hygiene(await ws.resolve("RELIANCE", [Exchange.NSE]), AS_OF, hours=24)

    total, worst = run(summary())
    assert total.value == 2 and total.timestamp == at.replace(hour=10)
    assert worst.value == 1135.0 and format_value(worst) == "1,135.0 MAD"
