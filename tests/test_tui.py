from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from fakes import InMemoryBroker
from market_seed import AS_OF, seeded_workspace
from rich.text import Text
from textual.pilot import Pilot
from textual.widgets import ContentSwitcher, DataTable, Input, RichLog, Static

from quantterm.agent.market_agent import AgentContext, LLMReply, MarketAgent, ToolCall
from quantterm.models import Exchange
from quantterm.streaming import tick_channel
from quantterm.terminal.app import QuantTermApp
from quantterm.terminal.command_router import CommandRouter
from quantterm.workspace import FA_METRICS

SIZE = (220, 70)


async def settle(pilot: Pilot[None], *groups: str) -> None:
    app = pilot.app
    await pilot.pause()
    pending = [w for w in app.workers if w.group in groups]
    if pending:  # an empty list would make Textual wait for *all* workers, incl. the endless live feed
        await app.workers.wait_for_complete(pending)
    await pilot.pause()


async def build(**kwargs: Any) -> QuantTermApp:
    _, ws = await seeded_workspace()
    router = CommandRouter(ws, clock=lambda: AS_OF)
    agent = kwargs.pop("agent_llm", None)
    return QuantTermApp(
        router=router, workspace=ws,
        agent=MarketAgent(ws, agent, context=lambda: AgentContext(security=router.security,
                                                                  as_of=router.effective_as_of)) if agent else None,
        **kwargs,
    )


async def command(pilot: Pilot[None], text: str) -> None:
    box = pilot.app.query_one("#command", Input)
    box.focus()
    box.value = text
    await pilot.press("enter")
    await settle(pilot, "command")


def viewport(app: QuantTermApp) -> str | None:
    return app.query_one("#viewport", ContentSwitcher).current


def status(app: QuantTermApp) -> str:
    return str(app.query_one("#status", Static).content)


def test_command_bar_and_function_keys_drive_the_viewport() -> None:
    async def scenario() -> None:
        app = await build()
        async with app.run_test(size=SIZE) as pilot:
            assert viewport(app) == "welcome" and app.focused is app.query_one("#command")
            await command(pilot, "RELIANCE IN Equity GP 6M <GO>")
            assert viewport(app) == "gp" and status(app).startswith("RELIANCE IN Equity | GP | live")
            assert "Last close" in str(app.query_one("#gp-summary", Static).content)

            await pilot.press("f3")
            await settle(pilot, "command")
            assert viewport(app) == "fa"
            assert app.query_one("#fa-table", DataTable).row_count == len(FA_METRICS) + 4
            ev = app.query_one("#fa-ev", DataTable)
            assert ev.row_count > 5 and str(ev.get_row_at(0)[2]) == "raw_market_bars"
            assert "breaks" in str(app.query_one("#fa-notes", Static).content)

            await pilot.press("f4")
            await settle(pilot, "command")
            assert viewport(app) == "rv" and app.query_one("#rv-table", DataTable).row_count == 3

            await pilot.press("f5")
            await settle(pilot, "command")
            assert viewport(app) == "ovme" and app.query_one("#ovme-chain", DataTable).row_count == 11
            assert app.query_one("#ovme-inputs", DataTable).row_count == 4

            await command(pilot, "OVME C 1300 vol=25%")
            assert "Calculator CALL 1300" in str(app.query_one("#ovme-calc", Static).content)

            await pilot.press("ctrl+b")
            await settle(pilot, "command")
            assert viewport(app) == "ovme" and "Calculator" not in str(app.query_one("#ovme-calc", Static).content)

            await command(pilot, "ONGC IN Equity")  # function is kept across securities
            assert viewport(app) == "ovme" and status(app).startswith("ONGC IN Equity | OVME")

            await pilot.press("f1")
            await settle(pilot, "command")
            assert viewport(app) == "welcome"

    asyncio.run(scenario())


def test_errors_are_reported_without_leaving_the_view() -> None:
    async def scenario() -> None:
        app = await build()
        async with app.run_test(size=SIZE) as pilot:
            await command(pilot, "FA")
            assert viewport(app) == "welcome"
            await command(pilot, "RELIANCE IN Equity FA")
            await command(pilot, "TCS IN Equity")
            assert viewport(app) == "fa" and status(app).startswith("RELIANCE IN Equity | FA")
            assert any("TCS" in n.message for n in app._notifications)

    asyncio.run(scenario())


def test_watchlist_streams_validated_ticks_and_opens_security() -> None:
    async def scenario() -> None:
        broker = InMemoryBroker()
        app = await build(broker=broker, watchlist=[(Exchange.NSE, "RELIANCE"), (Exchange.NSE, "ONGC"),
                                                    (Exchange.NSE, "TCS")])
        async with app.run_test(size=SIZE) as pilot:
            await settle(pilot, "watchlist")
            table = app.query_one("#watchlist", DataTable)
            assert table.row_count == 3 and str(table.get_cell("NSE:TCS", "last")) == "err"
            prev = float(str(table.get_cell("NSE:RELIANCE", "last")).replace(",", "").split()[0])
            for _ in range(50):
                if broker._subscribers:
                    break
                await pilot.pause(0.01)
            tick = {"tick": {"symbol": "RELIANCE", "exchange": "NSE", "price": round(prev * 1.02, 2), "size": 10,
                             "timestamp": datetime(2025, 1, 1, 4, 0, 5, tzinfo=UTC).isoformat()},
                    "median": prev, "deviation_mads": 0.4}
            await broker.publish(tick_channel(Exchange.NSE, "RELIANCE"), json.dumps(tick))
            await broker.publish(tick_channel(Exchange.NSE, "INFY"), json.dumps(tick))  # not watched
            await pilot.pause(0.05)
            assert str(table.get_cell("NSE:RELIANCE", "last")) == f"{prev * 1.02:,.2f}"
            assert str(table.get_cell("NSE:RELIANCE", "chg")) == "+2.00"
            assert str(table.get_cell("NSE:RELIANCE", "time")) == "09:30:05"  # exchange-local time

            table.focus()
            await pilot.press("enter")  # first row
            await settle(pilot, "command")
            assert status(app).startswith("RELIANCE IN Equity")

    asyncio.run(scenario())


class OneShotLLM:
    async def complete(self, messages: Sequence[dict[str, Any]], tools: Sequence[dict[str, Any]]) -> LLMReply:
        tool_msgs = [m for m in messages if m["role"] == "tool"]
        if not tool_msgs:
            assert "RELIANCE IN Equity" in messages[0]["content"]  # terminal context reaches the model
            return LLMReply(tool_calls=[ToolCall(id="c1", name="get_beta",
                                                 arguments='{"ticker":"RELIANCE","exchange_code":"IN"}')])
        facts = {f["label"]: f["id"] for f in json.loads(tool_msgs[-1]["content"])["facts"]}
        return LLMReply(content=f"Adjusted beta is {{{{{facts['Adjusted beta']}}}}} against NIFTY.")


def test_agent_panel_shows_grounded_answer_with_provenance() -> None:
    async def scenario() -> None:
        app = await build(agent_llm=OneShotLLM())
        async with app.run_test(size=SIZE) as pilot:
            await command(pilot, "RELIANCE IN Equity")
            box = app.query_one("#agent-input", Input)
            await pilot.press("f6")
            assert app.focused is box
            box.value = "What is the beta?"
            await pilot.press("enter")
            await settle(pilot, "agent")
            log = app.query_one("#agent-log", RichLog)
            text = " ".join(strip.text for strip in log.lines)
            assert "> What is the beta?" in text and "Adjusted beta is" in text
            assert "raw_market_bars | conf 1.00" in text and "quant.equity_metrics.compute_beta" in text
            assert "{{" not in text and "tools: get_beta" in text

    asyncio.run(scenario())


def test_agent_disabled_message() -> None:
    async def scenario() -> None:
        app = await build()
        async with app.run_test(size=SIZE) as pilot:
            box = app.query_one("#agent-input", Input)
            box.focus()
            box.value = "hello"
            await pilot.press("enter")
            await settle(pilot, "agent")
            text = " ".join(s.text for s in app.query_one("#agent-log", RichLog).lines)
            assert "Agent disabled" in text

    asyncio.run(scenario())


def test_styled_cells_flag_low_confidence() -> None:
    from quantterm.terminal.app import styled
    from quantterm.workspace import Datum

    def d(conf: float, value: float | None = 1.5) -> Text:
        return styled(Datum(label="x", value=value, unit="x", source="S", confidence_score=conf, timestamp=None,
                            provenance="p"))

    assert d(0.95).style == "" and d(0.7).style == "yellow" and d(0.3).style == "red"
    assert d(1.0, None).plain == "n/a" and d(1.0).plain == "1.50x"
