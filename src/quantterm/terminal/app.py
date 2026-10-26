"""QuantTerm keyboard-first workspace (Textual).

Layout: command bar (top) | watchlist (left) | analytical viewport GP/FA/RV/OVME (centre) | AI agent (right).

Keys: F8/Esc command bar, F1 help, F2 GP, F3 FA, F4 RV, F5 OVME, F6 agent, F7 watchlist, Ctrl+B back,
Ctrl+Q quit. Enter on a watchlist row loads that security into the active function.

Run: ``python -m quantterm.terminal.app`` (needs the same ``QT_*`` environment as the API service).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import ContentSwitcher, DataTable, Footer, Input, Label, RichLog, Static
from textual_plotext import PlotextPlot

from quantterm.agent.market_agent import AgentContext, MarketAgent, OpenAICompatibleClient
from quantterm.db.repositories import SecurityRecord
from quantterm.models import Exchange
from quantterm.pubsub import Broker
from quantterm.streaming import tick_channel
from quantterm.terminal.command_router import (
    HELP_TEXT,
    CommandRouter,
    RouterResult,
    bloomberg_code,
    display_name,
)
from quantterm.quant.derivatives_ovme import OptionValuation
from quantterm.settings import Settings
from quantterm.workspace import Datum, FAView, GPView, OVMEView, QuotedLeg, RVView, Workspace, format_value

LOW_CONFIDENCE = 0.8
WARN_CONFIDENCE = 0.5


def styled(d: Datum) -> Text:
    """A datum as a table cell: dim when unavailable, amber/red when confidence is degraded."""
    if d.value is None:
        return Text("n/a", style="dim")
    style = "" if d.confidence_score >= LOW_CONFIDENCE else ("yellow" if d.confidence_score >= WARN_CONFIDENCE else "red")
    return Text(format_value(d), style=style, justify="right")


def stamp(ts: datetime | None) -> Text:
    return Text("-" if ts is None else ts.strftime("%Y-%m-%d %H:%M %Z"), style="dim")


def provenance_row(d: Datum) -> tuple[Text, ...]:
    return (Text(d.label), styled(d), Text(d.source), Text(f"{d.confidence_score:.2f}"), stamp(d.timestamp),
            Text(d.note or "", style="dim"))


def parse_listing(ref: str) -> tuple[Exchange, str]:
    exchange, _, ticker = ref.partition(":")
    return Exchange(exchange.upper()), ticker.upper()


class QuantTermApp(App[None]):
    TITLE = "QuantTerm"
    CSS = """
    #command { dock: top; border: tall $accent; }
    #status { dock: top; height: 1; padding: 0 1; background: $panel; color: $text-muted; }
    #body { height: 1fr; }
    #watch-pane { width: 34; border: round $primary; }
    #viewport { width: 1fr; border: round $primary; }
    #agent-pane { width: 52; border: round $secondary; }
    .pane-title { width: 1fr; text-style: bold; background: $boost; padding: 0 1; }
    #watchlist { height: 1fr; }
    #agent-log { height: 1fr; }
    #gp-price { height: 2fr; }
    #gp-volume, #gp-rsi, #gp-macd { height: 1fr; }
    #gp-summary, #fa-notes, #ovme-calc, #rv-header { height: auto; padding: 0 1; }
    #fa-table, #fa-ev, #rv-table, #ovme-inputs, #ovme-chain { height: auto; max-height: 24; }
    """
    BINDINGS = [
        Binding("f8", "focus('command')", "Command", priority=True),
        Binding("escape", "focus('command')", "Command", show=False, priority=True),
        Binding("f1", "run('HELP')", "Help", priority=True),
        Binding("f2", "run('GP')", "GP", priority=True),
        Binding("f3", "run('FA')", "FA", priority=True),
        Binding("f4", "run('RV')", "RV", priority=True),
        Binding("f5", "run('OVME')", "OVME", priority=True),
        Binding("f6", "focus('agent-input')", "Agent", priority=True),
        Binding("f7", "focus('watchlist')", "Watchlist", priority=True),
        Binding("ctrl+b", "run('BACK')", "Back", priority=True),
    ]

    def __init__(
        self,
        *,
        router: CommandRouter,
        workspace: Workspace,
        agent: MarketAgent | None = None,
        broker: Broker | None = None,
        watchlist: Sequence[tuple[Exchange, str]] = (),
    ) -> None:
        super().__init__()
        self.router = router
        self.workspace = workspace
        self.agent = agent
        self.broker = broker
        self._watch_refs = list(dict.fromkeys(watchlist))
        self._watch: dict[str, SecurityRecord] = {}
        self._prev_close: dict[str, float] = {}

    # ------------------------------------------------------------------ layout
    def compose(self) -> ComposeResult:
        yield Input(placeholder="<TICKER> <CODE> Equity <GO>  |  GP  FA  RV  OVME  |  ASOF yyyy-mm-dd  |  HELP",
                    id="command")
        yield Static(Text(self.router.context), id="status")
        with Horizontal(id="body"):
            with Vertical(id="watch-pane"):
                yield Label("WATCHLIST", classes="pane-title")
                yield DataTable(id="watchlist", cursor_type="row", zebra_stripes=True)
            with ContentSwitcher(id="viewport", initial="welcome"):
                yield VerticalScroll(Static(Text(HELP_TEXT), id="welcome-text"), id="welcome")
                with Vertical(id="gp"):
                    yield Static(id="gp-summary")
                    yield PlotextPlot(id="gp-price")
                    yield PlotextPlot(id="gp-volume")
                    yield PlotextPlot(id="gp-rsi")
                    yield PlotextPlot(id="gp-macd")
                with VerticalScroll(id="fa"):
                    yield Label("POINT-IN-TIME FUNDAMENTALS", classes="pane-title")
                    yield DataTable(id="fa-table", zebra_stripes=True)
                    yield Label("RECONCILED ENTERPRISE VALUE", classes="pane-title")
                    yield DataTable(id="fa-ev", zebra_stripes=True)
                    yield Static(id="fa-notes")
                with VerticalScroll(id="rv"):
                    yield Static(id="rv-header")
                    yield DataTable(id="rv-table", zebra_stripes=True)
                with VerticalScroll(id="ovme"):
                    yield Label("MARKET INPUTS", classes="pane-title")
                    yield DataTable(id="ovme-inputs", zebra_stripes=True)
                    yield Label("CHAIN  (calls | strike | puts)", classes="pane-title")
                    yield DataTable(id="ovme-chain", zebra_stripes=True)
                    yield Static(id="ovme-calc")
            with Vertical(id="agent-pane"):
                yield Label("AI MARKET AGENT", classes="pane-title")
                yield RichLog(id="agent-log", wrap=True, markup=False)
                yield Input(placeholder="Ask about the active security... (F6)", id="agent-input")
        yield Footer()

    async def on_mount(self) -> None:
        table = self.query_one("#watchlist", DataTable)
        for label, key in (("Security", "sec"), ("Last", "last"), ("Chg %", "chg"), ("Time", "time")):
            table.add_column(label, key=key)
        log = self.query_one("#agent-log", RichLog)
        log.write(Text("Agent ready. Every figure it cites comes from the engine with source, confidence and "
                       "knowledge time." if self.agent else "Agent disabled: configure QT_AGENT_API_KEY / "
                       "QT_AGENT_BASE_URL.", style="dim"))
        self.query_one("#command", Input).focus()
        self.run_worker(self._load_watchlist(), group="watchlist", exclusive=True)

    # ------------------------------------------------------------------ actions
    def action_run(self, command: str) -> None:
        self.run_command(command)

    def run_command(self, text: str) -> None:
        self.run_worker(self._execute(text), group="command", exclusive=True)

    @on(Input.Submitted, "#command")
    def _command_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.clear()
        if text:
            self.run_command(text)

    @on(Input.Submitted, "#agent-input")
    def _agent_submitted(self, event: Input.Submitted) -> None:
        question = event.value.strip()
        event.input.clear()
        if question:
            self.run_worker(self._ask(question), group="agent", exclusive=True)

    @on(DataTable.RowSelected, "#watchlist")
    def _watch_selected(self, event: DataTable.RowSelected) -> None:
        sec = self._watch.get(str(event.row_key.value))
        if sec is not None:
            self.run_command(display_name(sec))

    # ------------------------------------------------------------------ command execution
    def _status(self, message: str) -> None:
        self.query_one("#status", Static).update(Text(message))

    async def _execute(self, text: str) -> None:
        self._status(f"{self.router.context} | running: {text}")
        try:
            result = await self.router.execute(text)
        except Exception as exc:  # infrastructure failure (DB/network): report, keep the terminal alive
            self._status(self.router.context)
            self.notify(f"{type(exc).__name__}: {exc}", severity="error", markup=False)
            return
        await self.show(result)

    async def show(self, result: RouterResult) -> None:
        self._status(self.router.context)
        if result.kind == "ERROR":
            self.notify(result.message, severity="error", markup=False)
        elif result.kind == "HELP":
            self.query_one("#welcome-text", Static).update(Text(result.message))
            self.query_one("#viewport", ContentSwitcher).current = "welcome"
        elif result.kind == "STATUS":
            self.notify(result.message, markup=False)
        elif result.kind == "WATCH":
            assert result.security is not None
            ref = (result.security.exchange, result.security.symbol)
            if result.watch_action == "ADD" and ref not in self._watch_refs:
                self._watch_refs.append(ref)
            elif result.watch_action == "DEL" and ref in self._watch_refs:
                self._watch_refs.remove(ref)
            self.run_worker(self._load_watchlist(), group="watchlist", exclusive=True)
        elif result.view is not None:
            if isinstance(result.view, GPView):
                self.render_gp(result.view)
            elif isinstance(result.view, FAView):
                self.render_fa(result.view)
            elif isinstance(result.view, RVView):
                self.render_rv(result.view)
            elif isinstance(result.view, OVMEView):
                self.render_ovme(result.view)
            assert result.function is not None
            self.query_one("#viewport", ContentSwitcher).current = result.function.value.lower()

    # ------------------------------------------------------------------ GP
    def render_gp(self, v: GPView) -> None:
        dates = [d.isoformat() for d in v.dates]
        basis = "split+dividend adjusted" if v.adjust_dividends else ("split adjusted" if v.adjust_splits else "raw")
        summary = Text(f"{display_name(v.security)}  GP  ({basis}, as of {v.as_of:%Y-%m-%d})\n", style="bold")
        for d in v.summary:
            summary.append(f"{d.label}: ")
            summary.append_text(styled(d))
            summary.append("   ")
        self.query_one("#gp-summary", Static).update(summary)

        def draw(widget_id: str, title: str, series: dict[str, list[float | None]], kind: str = "line",
                 levels: Sequence[float] = ()) -> None:
            plot = self.query_one(f"#{widget_id}", PlotextPlot)
            plt = plot.plt
            plt.clear_figure()
            plt.date_form("Y-m-d")
            plt.title(title)
            for name, values in series.items():
                xs = [x for x, y in zip(dates, values, strict=True) if y is not None]
                ys = [y for y in values if y is not None]
                if not ys:
                    continue
                if kind == "bar":
                    plt.bar(xs, ys, label=name)
                else:
                    plt.plot(xs, ys, label=name, marker="braille")
            for level in levels:
                plt.hline(level)
            plot.refresh()

        draw("gp-price", f"{v.security.symbol} close ({v.security.currency})", {"Close": list(v.close)})
        draw("gp-volume", "Volume (split-adjusted)", {"Volume": list(v.volume)}, kind="bar")
        draw("gp-rsi", "RSI(14)", {"RSI": v.rsi}, levels=(30.0, 70.0))
        draw("gp-macd", "MACD(12,26,9)", {"MACD": v.macd, "Signal": v.signal}, levels=(0.0,))

    # ------------------------------------------------------------------ FA
    def render_fa(self, v: FAView) -> None:
        table = self.query_one("#fa-table", DataTable)
        table.clear(columns=True)
        table.add_column(Text(f"{display_name(v.security)} {v.period_type.value}"), key="metric")
        for end in v.periods:
            table.add_column(Text(end.isoformat(), justify="right"), key=end.isoformat())
        for row in v.rows:
            table.add_row(Text(row.label), *(styled(c) for c in row.cells))
        for row in v.ratios:
            table.add_row(Text(row.label, style="italic"), *(styled(c) for c in row.cells))

        ev = self.query_one("#fa-ev", DataTable)
        ev.clear(columns=True)
        for label in ("Component", "Value", "Source", "Conf", "Knowledge time", "Note"):
            ev.add_column(label)
        for d in v.ev_table:
            ev.add_row(*provenance_row(d))

        notes = Text(f"As of {v.as_of:%Y-%m-%d %H:%M %Z}: only filings public by then. ", style="dim")
        breaks = [i for i in v.reconciliation if i.is_break]
        notes.append(f"Reconciliation: {len(v.reconciliation)} items, {len(breaks)} breaks.\n",
                     style="yellow" if breaks else "green")
        for n in v.notes:
            notes.append(f"! {n}\n", style="yellow")
        self.query_one("#fa-notes", Static).update(notes)

    # ------------------------------------------------------------------ RV
    def render_rv(self, v: RVView) -> None:
        self.query_one("#rv-header", Static).update(
            Text(f"{display_name(v.security)}  RV  industry: {v.industry}  ({len(v.rows)} listings, "
                 f"as of {v.as_of:%Y-%m-%d})", style="bold"))
        table = self.query_one("#rv-table", DataTable)
        table.clear(columns=True)
        table.add_column("Security", key="sec")
        for c in v.columns:
            table.add_column(Text(c, justify="right"), key=c)
        for row in v.rows:
            name = Text(display_name(row.security), style="bold reverse" if row.is_subject else "")
            table.add_row(name, *(styled(row.cells[c]) for c in v.columns))
        table.add_row(Text("Peer median", style="bold"), Text(""), *(styled(v.medians[c]) for c in v.columns[1:]))

    # ------------------------------------------------------------------ OVME
    def render_ovme(self, v: OVMEView) -> None:
        inputs = self.query_one("#ovme-inputs", DataTable)
        inputs.clear(columns=True)
        for label in ("Input", "Value", "Source", "Conf", "Knowledge time", "Note"):
            inputs.add_column(label)
        for d in v.inputs:
            inputs.add_row(*provenance_row(d))

        chain = self.query_one("#ovme-chain", DataTable)
        chain.clear(columns=True)
        sides = ("Bid", "Ask", "Mkt IV", "Model", "Delta", "Gamma", "Theta", "Vega")
        for label in (*(f"C {s}" for s in sides), "Strike", *(f"P {s}" for s in reversed(sides))):
            chain.add_column(Text(label, justify="right"))
        spot = v.inputs[0].value

        def cell(text: str, style: str = "") -> Text:
            return Text(text, style=style, justify="right")

        def side(o: OptionValuation, q: QuotedLeg | None) -> list[Text]:
            market = ([cell("-", "dim")] * 3 if q is None else [
                cell(f"{q.bid:,.2f}"), cell(f"{q.ask:,.2f}"),
                cell("n/a", "dim") if q.implied_vol is None else cell(f"{q.implied_vol:.2%}", "bold")])
            return market + [cell(f"{o.npv:,.2f}"), cell(f"{o.delta:.3f}"), cell(f"{o.gamma:.4f}"),
                             cell(f"{o.theta_per_day:.3f}"), cell(f"{o.vega_per_vol_point:.3f}")]

        for row in v.chain:
            atm = spot is not None and abs(row.strike - spot) == min(abs(r.strike - spot) for r in v.chain)
            chain.add_row(*side(row.call, row.call_quote),
                          Text(f"{row.strike:g}", style="bold reverse" if atm else "bold", justify="center"),
                          *reversed(side(row.put, row.put_quote)))

        calc = Text(f"{display_name(v.security)}  OVME  expiry {v.expiry} ({v.exercise.value.lower()})\n", style="bold")
        if v.calculator is not None and v.calculator_contract is not None:
            c, k = v.calculator, v.calculator_contract
            calc.append(f"Calculator {k.option_type.value} {k.strike:g}: premium {c.npv:,.4f}  delta {c.delta:.4f}  "
                        f"gamma {c.gamma:.5f}  theta/day {c.theta_per_day:.4f}  vega/pt {c.vega_per_vol_point:.4f}  "
                        f"rho/1% {c.rho_per_pct:.4f}  vol {c.volatility:.2%}\n")
        for n in v.notes:
            calc.append(f"{n}\n", style="dim")
        self.query_one("#ovme-calc", Static).update(calc)

    # ------------------------------------------------------------------ agent
    async def _ask(self, question: str) -> None:
        log = self.query_one("#agent-log", RichLog)
        log.write(Text(f"> {question}", style="bold cyan"))
        if self.agent is None:
            log.write(Text("Agent disabled.", style="red"))
            return
        log.write(Text("working...", style="dim"))
        try:
            answer = await self.agent.ask(question)
        except Exception as exc:  # transport / DB failure: report in the pane, keep the session
            log.write(Text(f"agent error: {type(exc).__name__}: {exc}", style="red"))
            return
        if answer.fallback:
            log.write(Text("Model answer withheld; deterministic figures shown.", style="yellow"))
        log.write(Text(answer.text))
        log.write(Text(f"[{answer.steps} steps; tools: {', '.join(answer.tool_calls) or 'none'}; "
                       f"{len(answer.cited)} facts cited]", style="dim"))

    # ------------------------------------------------------------------ watchlist
    def _watch_key(self, exchange: Exchange, symbol: str) -> str:
        return f"{exchange.value}:{symbol}"

    async def _load_watchlist(self) -> None:
        table = self.query_one("#watchlist", DataTable)
        table.clear()
        self._watch.clear()
        as_of = self.router.effective_as_of
        for exchange, symbol in self._watch_refs:
            key = self._watch_key(exchange, symbol)
            try:
                sec = await self.workspace.resolve(symbol, (exchange,))
                last = await self.workspace.last_price(sec, as_of)
            except Exception as exc:  # one bad listing must not blank the watchlist
                table.add_row(Text(f"{symbol} {bloomberg_code(exchange)}"), Text("err", style="red"), Text(""),
                              Text(type(exc).__name__, style="dim"), key=key)
                continue
            self._watch[key] = sec
            if last.value is not None:
                self._prev_close[key] = last.value
            table.add_row(Text(f"{symbol} {bloomberg_code(exchange)}"), styled(last), Text(""),
                          stamp(last.timestamp), key=key)
        if self.broker is not None and self._watch and self.router.as_of is None:
            self.run_worker(self._stream(list(self._watch.values())), group="stream", exclusive=True)

    async def _stream(self, securities: Sequence[SecurityRecord]) -> None:
        """Live validated prints (post MAD filter) from the ingestion pipeline's Redis fan-out."""
        assert self.broker is not None
        channels = {tick_channel(s.exchange, s.symbol): self._watch_key(s.exchange, s.symbol) for s in securities}
        try:
            async with self.broker.subscribe(list(channels)) as messages:
                async for channel, data in messages:
                    key = channels.get(channel)
                    if key is not None:
                        self.apply_tick(key, json.loads(data))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # broker outage: watchlist stays on last values
            self.notify(f"live feed stopped: {type(exc).__name__}: {exc}", severity="warning", markup=False)

    def apply_tick(self, key: str, message: dict[str, Any]) -> None:
        tick = message["tick"]
        price = float(tick["price"])
        ts = datetime.fromisoformat(tick["timestamp"])
        table = self.query_one("#watchlist", DataTable)
        prev = self._prev_close.get(key)
        table.update_cell(key, "last", Text(f"{price:,.2f}", justify="right"))
        if prev:
            change = (price / prev - 1) * 100
            table.update_cell(key, "chg", Text(f"{change:+.2f}", style="green" if change >= 0 else "red",
                                               justify="right"))
        sec = self._watch.get(key)
        local = ts.astimezone(sec.exchange.timezone) if sec else ts
        table.update_cell(key, "time", Text(local.strftime("%H:%M:%S"), style="dim"))


# ---------------------------------------------------------------------------------- entrypoint
async def _run(settings: Settings | None = None) -> None:
    from redis.asyncio import Redis

    from quantterm.db.pool import create_pool
    from quantterm.live import build_pipeline, start_live_feeds
    from quantterm.pubsub import RedisBroker

    settings = settings or Settings()
    pool = await create_pool(settings)
    redis: Redis = Redis.from_url(settings.redis_url.get_secret_value(), decode_responses=True)
    llm = OpenAICompatibleClient.from_settings(settings)
    feeds: list[asyncio.Task[None]] = []
    pipeline = None
    try:
        broker = RedisBroker(redis)
        workspace = Workspace(pool, settings)
        router = CommandRouter(workspace, clock=workspace.clock)
        agent_enabled = settings.agent_api_key is not None or "api.openai.com" not in settings.agent_base_url
        agent = MarketAgent(
            workspace, llm, max_steps=settings.agent_max_steps,
            context=lambda: AgentContext(security=router.security, as_of=router.effective_as_of),
        ) if agent_enabled else None
        app = QuantTermApp(router=router, workspace=workspace, agent=agent, broker=broker,
                           watchlist=[parse_listing(r) for r in settings.watchlist])
        if settings.terminal_live_feed:
            pipeline = build_pipeline(settings, pool, broker)
            feeds = start_live_feeds(settings, pipeline,
                                     on_stop=lambda msg: app.notify(msg, severity="error", markup=False, timeout=30))
            if feeds:
                feeds.append(asyncio.create_task(pipeline.run_periodic_flush(settings.tick_flush_interval_seconds)))
        await app.run_async()
    finally:
        for task in feeds:
            task.cancel()
        await asyncio.gather(*feeds, return_exceptions=True)
        if pipeline is not None:
            await pipeline.flush()
        await llm.aclose()
        await redis.aclose()
        await pool.close()


def main(settings: Settings | None = None) -> None:
    asyncio.run(_run(settings))


if __name__ == "__main__":
    main()
