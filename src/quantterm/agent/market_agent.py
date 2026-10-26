"""AI market agent over the validated engine, hygiene and quant layers, with a numeric-grounding guard.

How hallucinated numbers are made impossible to ship:

1. The model never sees a number it could be tempted to restate as its own: every figure returned by a
   tool is registered in a ``FactLedger`` under an id (``F1``, ``F2``...) together with its source,
   ``confidence_score`` and knowledge ``timestamp`` (filing / dissemination / session-close time).
2. The model must write figures as ``{{F<n>}}`` placeholders. It may not type digits, round, convert or
   compute: a computed number has no deterministic provenance.
3. ``check_grounding`` rejects any answer containing a digit sequence or spelled-out quantity that is not a
   placeholder, unless that exact token occurs in deterministic tool text (labels such as ``RSI(14)``,
   period-end dates, the as-of date), and rejects placeholders that reference unknown facts.
4. Rejected answers are returned to the model with the violations; after ``max_repairs`` failures the agent
   answers with a deterministic digest of the facts it retrieved instead of the model's prose.
5. Placeholders are rendered from the ledger as value + unit + [source | confidence | timestamp], and a
   source table is appended, so every number shown carries its audit trail.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Awaitable, Callable, Iterable, Sequence
from datetime import date, datetime, time
from time import monotonic
from typing import Any, Literal, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from quantterm.connectors.base import Metric
from quantterm.db.repositories import SecurityRecord
from quantterm.models import PeriodType
from quantterm.quant.derivatives_ovme import OptionType
from quantterm.quant.equity_metrics import ReturnFrequency
from quantterm.settings import Settings
from quantterm.terminal.command_router import EXCHANGE_CODES, display_name
from quantterm.workspace import (
    EV_LINE_ITEMS,
    RV_COLUMNS,
    Datum,
    OVMERequest,
    Workspace,
    derive,
    format_value,
    option_datums,
    reconciled_datum,
)

PLACEHOLDER = re.compile(r"\{\{\s*(F\d+)\s*\}\}")
_NUMBER = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")
_WORD_NUMBERS = (
    r"zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|"
    r"sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|"
    r"thousand|half|a quarter|a third|double|triple"
)
_SPELLED_QUANTITY = re.compile(
    rf"\b(?:{_WORD_NUMBERS})(?:[\s-]+(?:{_WORD_NUMBERS}))*[\s-]+"
    r"(?:percent|per\s+cent|percentage\s+points?|basis\s+points?|bps|million|billion|trillion|crores?|lakhs?|"
    r"times|x|dollars?|rupees?)\b",
    re.IGNORECASE,
)
FACT_PROMPT_LIMIT = 120  # facts per tool result shown to the model

ExchangeCode = Literal["US", "UN", "UW", "UQ", "IN", "IS", "IB", "NYSE", "NASDAQ", "NSE", "BSE"]

SYSTEM_PROMPT = """You are the QuantTerm market analyst for US (NYSE/NASDAQ) and Indian (NSE/BSE) equities.
Terminal context: {context}. All data is point-in-time as of {as_of}: nothing published later exists.

Numeric grounding rules (enforced by an automatic checker; violating answers are rejected):
1. Every number about a security or market (price, return, ratio, multiple, EV, beta, Greek, volatility,
   rate, share count, amount, count) must be written as a fact placeholder like {{{{F7}}}} that refers to a
   fact id returned by a tool in this conversation. Never type the digits yourself.
2. Never round, convert, add, subtract or otherwise compute with facts; if a figure you need is not a fact,
   say it is unavailable or call another tool.
3. Do not restate units, sources or dates next to a placeholder; the renderer adds value, unit, source,
   confidence score and timestamp automatically.
4. Flag facts with confidence below 0.8 and any reconciliation break or alert you were given.
5. Securities are identified by ticker plus Bloomberg exchange code: US/UN/UW for US listings, IN for NSE,
   IB for BSE. Prefer the terminal's active security when the user says "it" or "this stock".
Write concise institutional prose. Comparisons in words ("above the peer median") are allowed."""


# ------------------------------------------------------------------------------------- facts
class Fact(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    tool: str
    datum: Datum


def _tokens(text: str) -> set[str]:
    out = set()
    for tok in _NUMBER.findall(text):
        plain = tok.replace(",", "")
        out.add(plain if "." in plain else str(int(plain)))
    return out


def _stamp(ts: datetime | None) -> str:
    return "time n/a" if ts is None else ts.strftime("%Y-%m-%d %H:%M %Z").strip()


class FactLedger:
    """Session registry of every number the tools produced; the only numbers the agent may cite."""

    def __init__(self) -> None:
        self._facts: dict[str, Fact] = {}
        self._allowed: set[str] = set()

    def __len__(self) -> int:
        return len(self._facts)

    def add(self, datums: Iterable[Datum], tool: str) -> list[Fact]:
        added = []
        for d in datums:
            fact = Fact(id=f"F{len(self._facts) + 1}", tool=tool, datum=d)
            self._facts[fact.id] = fact
            self.allow_text(d.label, d.note or "")
            added.append(fact)
        return added

    def allow_text(self, *texts: str) -> None:
        """Numbers inside deterministic descriptive text (labels, notes, dates) may be quoted verbatim."""
        for t in texts:
            self._allowed |= _tokens(t)

    @property
    def allowed_numbers(self) -> frozenset[str]:
        return frozenset(self._allowed)

    def get(self, fact_id: str) -> Fact | None:
        return self._facts.get(fact_id)

    def facts(self) -> list[Fact]:
        return list(self._facts.values())

    def render(self, fact_id: str) -> str:
        f = self._facts[fact_id]
        d = f.datum
        value = format_value(d) if d.value is not None else f"n/a ({d.note or 'unavailable'})"
        return f"{value} [{d.source} | conf {d.confidence_score:.2f} | {_stamp(d.timestamp)}]"

    def source_line(self, fact_id: str) -> str:
        d = self._facts[fact_id].datum
        note = f" | {d.note}" if d.note else ""
        return (f"[{fact_id}] {d.label} = {format_value(d)} | {d.source} via {d.provenance} | "
                f"conf {d.confidence_score:.2f} | {_stamp(d.timestamp)}{note}")


class GuardReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    cited: list[str]
    unknown_facts: list[str]
    uncited_numbers: list[str]
    spelled_quantities: list[str]

    @property
    def ok(self) -> bool:
        return not (self.unknown_facts or self.uncited_numbers or self.spelled_quantities)

    def feedback(self) -> str:
        parts = []
        if self.uncited_numbers:
            parts.append(f"numbers typed without a fact placeholder: {', '.join(self.uncited_numbers)}")
        if self.spelled_quantities:
            parts.append(f"quantities written in words: {', '.join(self.spelled_quantities)}")
        if self.unknown_facts:
            parts.append(f"placeholders for facts that do not exist: {', '.join(self.unknown_facts)}")
        return ("GROUNDING CHECK FAILED - " + "; ".join(parts) + ". Rewrite the full answer. Every number must "
                "be a {{F<n>}} placeholder for a fact a tool returned; do not round or compute. If the figure "
                "is not available from a tool, say so instead of estimating.")


def check_grounding(text: str, ledger: FactLedger) -> GuardReport:
    cited = PLACEHOLDER.findall(text)
    unknown = sorted({c for c in cited if ledger.get(c) is None})
    stripped = PLACEHOLDER.sub(" ", text)
    allowed = ledger.allowed_numbers
    uncited = []
    for tok in _NUMBER.findall(stripped):
        plain = tok.replace(",", "")
        norm = plain if "." in plain else str(int(plain))
        if norm not in allowed and tok not in uncited:
            uncited.append(tok)
    spelled = [m.group(0) for m in _SPELLED_QUANTITY.finditer(stripped)]
    return GuardReport(cited=list(dict.fromkeys(cited)), unknown_facts=unknown, uncited_numbers=uncited,
                       spelled_quantities=spelled)


def render_answer(text: str, ledger: FactLedger) -> tuple[str, list[Fact]]:
    cited = list(dict.fromkeys(PLACEHOLDER.findall(text)))
    body = PLACEHOLDER.sub(lambda m: ledger.render(m.group(1)), text)
    facts = [f for c in cited if (f := ledger.get(c)) is not None]
    if not facts:
        return body, []
    return body + "\n\nSources:\n" + "\n".join(ledger.source_line(f.id) for f in facts), facts


# ------------------------------------------------------------------------------------- tool args
class _Args(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SecurityArgs(_Args):
    ticker: str = Field(pattern=r"^[A-Za-z0-9&.\-]{1,20}$", description="Exchange ticker, e.g. AAPL, RELIANCE")
    exchange_code: ExchangeCode = Field(description="Bloomberg code: US/UN/UW (US), IN (NSE), IB (BSE)")
    as_of: date | None = Field(default=None, description="Point-in-time date; omit to use the terminal's as-of")


class PriceArgs(SecurityArgs):
    lookback_days: int = Field(default=365, ge=30, le=3660)
    adjust_dividends: bool = Field(default=True, description="False = split/bonus-adjusted price return only")


class FundamentalsArgs(SecurityArgs):
    period_type: Literal["FY", "Q", "H"] = "FY"
    periods: int = Field(default=4, ge=1, le=12)


class RevisionArgs(SecurityArgs):
    metric: Metric
    period_type: Literal["FY", "Q", "H"] = "FY"
    period_end_date: date
    source: str | None = Field(default=None, description="Source authority; default is the regulatory filing")


class ReconcileArgs(SecurityArgs):
    period_type: Literal["FY", "Q", "H"] = "FY"


class BetaArgs(SecurityArgs):
    years: int = Field(default=2, ge=1, le=5)
    frequency: ReturnFrequency = ReturnFrequency.WEEKLY


class PeerArgs(SecurityArgs):
    max_peers: int = Field(default=10, ge=1, le=25)


class ChainArgs(SecurityArgs):
    expiry: date | None = Field(default=None, description="Omit for the next standard monthly expiry")
    strikes_each_side: int = Field(default=3, ge=1, le=10)


class OptionArgs(SecurityArgs):
    option_type: OptionType
    strike: float = Field(gt=0)
    expiry: date | None = Field(default=None, description="Omit for the next standard monthly expiry")
    volatility: float | None = Field(default=None, gt=0, le=5, description="Annualised, decimal (0.25 = 25%)")
    rate: float | None = Field(default=None, ge=-0.05, le=1, description="Continuously compounded, decimal")
    dividend_yield: float | None = Field(default=None, ge=0, le=1)


class ImpliedVolArgs(OptionArgs):
    premium: float = Field(gt=0, description="Observed option price in the underlying's currency")


class TickHygieneArgs(SecurityArgs):
    hours: int = Field(default=24, ge=1, le=720)


class ToolOutput(BaseModel):
    model_config = ConfigDict(frozen=True)

    context: dict[str, str] = Field(default_factory=dict)
    facts: list[Datum] = Field(default_factory=list)


class Tool(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    name: str
    description: str
    args: type[SecurityArgs]
    run: Callable[[Any], Awaitable[ToolOutput]]

    def schema_for_llm(self) -> dict[str, Any]:
        return {"type": "function", "function": {"name": self.name, "description": self.description,
                                                 "parameters": self.args.model_json_schema()}}


# ------------------------------------------------------------------------------------- LLM transport
class ToolCall(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    arguments: str
    # Opaque provider data that must be echoed back, e.g. Gemini 3 thought signatures.
    extra_content: dict[str, Any] | None = None

    def as_message(self) -> dict[str, Any]:
        call: dict[str, Any] = {"id": self.id, "type": "function",
                                "function": {"name": self.name, "arguments": self.arguments}}
        if self.extra_content:
            call["extra_content"] = self.extra_content
        return call


class LLMReply(BaseModel):
    model_config = ConfigDict(frozen=True)

    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)


class LLMClient(Protocol):
    async def complete(self, messages: Sequence[dict[str, Any]], tools: Sequence[dict[str, Any]]) -> LLMReply: ...


class OpenAICompatibleClient:
    """Chat-completions client (OpenAI, Azure OpenAI-compatible gateways, vLLM, Ollama, LM Studio).

    ``fallback_models`` are tried in order when a model is out of quota (429), retired (404) or keeps failing;
    an exhausted model is skipped until the provider's retry delay (e.g. the daily free-tier reset) passes.
    """

    def __init__(self, base_url: str, model: str, api_key: str | None = None, *, timeout: float = 60.0,
                 client: httpx.AsyncClient | None = None, retries: int = 2, backoff: float = 2.0,
                 reasoning_effort: str | None = None, fallback_models: Sequence[str] = (),
                 clock: Callable[[], float] = monotonic) -> None:
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._models = list(dict.fromkeys([model, *fallback_models]))
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = client is None
        self._retries = retries
        self._backoff = backoff
        self._reasoning_effort = reasoning_effort
        self._clock = clock
        self._cooldown_until: dict[str, float] = {}
        self.last_model: str | None = None

    @classmethod
    def from_settings(cls, settings: Settings) -> OpenAICompatibleClient:
        key = settings.agent_api_key.get_secret_value() if settings.agent_api_key else None
        return cls(settings.agent_base_url, settings.agent_model, key, timeout=settings.agent_timeout_seconds,
                   retries=settings.agent_retries, reasoning_effort=settings.agent_reasoning_effort,
                   fallback_models=settings.agent_fallback_models)

    def _candidates(self) -> list[str]:
        now = self._clock()
        ready = [m for m in self._models if self._cooldown_until.get(m, 0) <= now]
        # Everything cooling down: still try, soonest-available first, rather than refusing outright.
        return ready or sorted(self._models, key=lambda m: self._cooldown_until.get(m, 0))

    async def _post(self, payload: dict[str, Any]) -> httpx.Response:
        for attempt in range(self._retries + 1):
            try:
                response = await self._client.post(self._url, json=payload, headers=self._headers)
            except httpx.TransportError:  # includes timeouts: hosted models stall under load
                if attempt == self._retries:
                    raise
                await asyncio.sleep(self._backoff * (2 ** attempt))
                continue
            delay = _retry_delay(response)
            # Short 429/5xx waits are load shedding; long 429 waits are spent quota -> caller switches model.
            transient = response.status_code in (500, 502, 503, 504) or (
                response.status_code == 429 and (delay is None or delay <= 30))
            if transient and attempt < self._retries:
                await asyncio.sleep(delay if delay is not None else self._backoff * (2 ** attempt))
                continue
            return response
        raise AssertionError("unreachable")

    async def complete(self, messages: Sequence[dict[str, Any]], tools: Sequence[dict[str, Any]]) -> LLMReply:
        base: dict[str, Any] = {"messages": list(messages), "temperature": 0}
        if self._reasoning_effort:
            base["reasoning_effort"] = self._reasoning_effort
        if tools:
            base["tools"] = list(tools)
            base["tool_choice"] = "auto"
        failure: Exception | None = None
        for model in self._candidates():
            try:
                response = await self._post(base | {"model": model})
            except httpx.TransportError as exc:
                self._cooldown_until[model] = self._clock() + 60
                failure = exc
                continue
            if response.status_code in (404, 429, 500, 502, 503, 504):
                delay = _retry_delay(response)
                cool = 86_400 if response.status_code == 404 else delay if delay is not None else 60
                self._cooldown_until[model] = self._clock() + cool
                failure = httpx.HTTPStatusError(f"HTTP {response.status_code} from {model}",
                                                request=response.request, response=response)
                continue
            response.raise_for_status()
            self.last_model = model
            return _parse_reply(response.json())
        assert failure is not None
        raise failure

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


# ------------------------------------------------------------------------------------- agent
def _retry_delay(response: httpx.Response) -> float | None:
    """Seconds the provider asks us to wait (Retry-After header or Google RetryInfo ``retryDelay``)."""
    header = response.headers.get("Retry-After", "")
    if header.isdigit():
        return float(header)
    try:
        body = response.json()
    except ValueError:
        return None
    errors = body if isinstance(body, list) else [body]
    for err in errors:
        details = err.get("error", {}).get("details", []) if isinstance(err, dict) else []
        for d in details:
            raw = d.get("retryDelay") if isinstance(d, dict) else None
            if isinstance(raw, str) and raw.endswith("s"):
                try:
                    return float(raw[:-1])
                except ValueError:
                    return None
    return None


def _parse_reply(body: Any) -> LLMReply:
    message = body["choices"][0]["message"]
    # Some OpenAI-compatible gateways (e.g. Gemini) may omit or blank tool-call ids.
    calls = [ToolCall(id=c.get("id") or f"call_{i}", name=c["function"]["name"],
                      arguments=c["function"].get("arguments") or "{}", extra_content=c.get("extra_content"))
             for i, c in enumerate(message.get("tool_calls") or [])]
    return LLMReply(content=message.get("content"), tool_calls=calls)


def _llm_failure(exc: Exception) -> str:
    if isinstance(exc, httpx.TimeoutException):
        return "the model provider timed out under load; please retry"
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        if code in (429, 503):
            return f"HTTP {code}: the model provider is overloaded or rate-limited; please retry shortly"
        if code in (401, 403):
            return f"HTTP {code}: the API key was rejected; check QT_AGENT_API_KEY"
        if code == 404:
            return "HTTP 404: model not found; check QT_AGENT_MODEL"
        return f"HTTP {code}"
    return type(exc).__name__


class AgentContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    security: SecurityRecord | None = None
    as_of: datetime


class AgentAnswer(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    raw: str
    cited: list[Fact]
    grounded: bool
    fallback: bool
    steps: int
    tool_calls: list[str]
    guard_failures: list[GuardReport]


class MarketAgent:
    def __init__(
        self,
        workspace: Workspace,
        llm: LLMClient,
        *,
        context: Callable[[], AgentContext] | None = None,
        max_steps: int = 8,
        max_repairs: int = 2,
        history_turns: int = 6,
    ) -> None:
        self._ws = workspace
        self._llm = llm
        self._context = context or (lambda: AgentContext(as_of=workspace.clock()))
        self._max_steps = max_steps
        self._max_repairs = max_repairs
        self._history_turns = history_turns
        self.ledger = FactLedger()
        self._history: list[tuple[str, str]] = []
        self.tools: dict[str, Tool] = {t.name: t for t in self._build_tools()}

    def reset(self) -> None:
        self.ledger = FactLedger()
        self._history.clear()

    def tool_schemas(self) -> list[dict[str, Any]]:
        return [t.schema_for_llm() for t in self.tools.values()]

    # ---------------------------------------------------------------- tool plumbing
    async def _security(self, args: SecurityArgs) -> tuple[SecurityRecord, datetime]:
        sec = await self._ws.resolve(args.ticker.upper(), EXCHANGE_CODES[args.exchange_code])
        ctx_as_of = self._context().as_of
        if args.as_of is None:
            return sec, ctx_as_of
        # End of the requested local trading day, never beyond the terminal's own as-of.
        requested = datetime.combine(args.as_of, time.max, tzinfo=sec.exchange.timezone)
        return sec, min(requested, ctx_as_of)

    @staticmethod
    def _sec_context(sec: SecurityRecord, as_of: datetime) -> dict[str, str]:
        return {"security": display_name(sec), "currency": sec.currency, "sector_class": sec.sector_class.value,
                "industry": sec.industry or "unclassified", "as_of": as_of.isoformat()}

    async def call_tool(self, name: str, arguments: str | dict[str, Any]) -> str:
        """Run one tool; returns the JSON the model sees. Errors are reported, never raised to the model loop."""
        tool = self.tools.get(name)
        if tool is None:
            return json.dumps({"error": f"unknown tool '{name}'", "available": sorted(self.tools)})
        try:
            raw = json.loads(arguments) if isinstance(arguments, str) else arguments
            args = tool.args.model_validate(raw)
            out = await tool.run(args)
        except (ValidationError, json.JSONDecodeError) as exc:
            return json.dumps({"error": f"invalid arguments: {exc}"})
        except (RuntimeError, LookupError, ValueError) as exc:  # WorkspaceError, QuantLib and domain errors
            return json.dumps({"error": str(exc)})
        facts = self.ledger.add(out.facts, name)
        self.ledger.allow_text(*out.context.values())
        shown = facts[:FACT_PROMPT_LIMIT]
        payload = {
            "context": out.context,
            "facts": [{"id": f.id, "label": f.datum.label, "value": format_value(f.datum),
                       "source": f.datum.source, "confidence_score": round(f.datum.confidence_score, 3),
                       "timestamp": f.datum.timestamp.isoformat() if f.datum.timestamp else None,
                       "note": f.datum.note} for f in shown],
            "instruction": "Cite these figures only as {{id}} placeholders, e.g. {{" + (shown[0].id if shown else "F1") + "}}.",
        }
        if len(facts) > len(shown):
            payload["truncated"] = f"{len(facts) - len(shown)} further facts omitted; narrow the request"
        return json.dumps(payload, default=str)

    # ---------------------------------------------------------------- tools
    def _build_tools(self) -> list[Tool]:
        ws = self._ws

        async def resolve_security(a: SecurityArgs) -> ToolOutput:
            sec, as_of = await self._security(a)
            ctx = self._sec_context(sec, as_of)
            if sec.isin:
                ctx["isin"] = sec.isin
            return ToolOutput(context=ctx)

        async def price_history(a: PriceArgs) -> ToolOutput:
            sec, as_of = await self._security(a)
            view = await ws.gp(sec, as_of, lookback_days=a.lookback_days, adjust_dividends=a.adjust_dividends)
            return ToolOutput(context=self._sec_context(sec, as_of) | {
                "window": f"{view.dates[0]} to {view.dates[-1]}",
                "rsi_convention": "RSI(14) above 70 is conventionally overbought, below 30 oversold"},
                facts=view.summary)

        async def fundamentals(a: FundamentalsArgs) -> ToolOutput:
            sec, as_of = await self._security(a)
            view = await ws.fa(sec, as_of, period_type=PeriodType(a.period_type), periods=a.periods)
            facts = [c.model_copy(update={"label": f"{row.label} {a.period_type} {end}"})
                     for row in view.rows + view.ratios for end, c in zip(view.periods, row.cells, strict=True)]
            ctx = self._sec_context(sec, as_of) | {"periods": ", ".join(str(p) for p in view.periods)}
            if view.notes:
                ctx["reconciliation_alerts"] = " | ".join(view.notes)
            return ToolOutput(context=ctx, facts=facts)

        async def revision_history(a: RevisionArgs) -> ToolOutput:
            sec, as_of = await self._security(a)
            facts = await ws.revisions(sec, a.metric, PeriodType(a.period_type), a.period_end_date, as_of, a.source)
            return ToolOutput(context=self._sec_context(sec, as_of), facts=facts)

        async def reconcile(a: ReconcileArgs) -> ToolOutput:
            sec, as_of = await self._security(a)
            report = await ws.reconcile(sec, as_of, PeriodType(a.period_type))
            facts = [reconciled_datum(it, f"{m} {a.period_type} {it.period_end_date}")
                     for m in EV_LINE_ITEMS if (it := report.resolved(m)) is not None]
            ctx = self._sec_context(sec, as_of) | {"primary": report.primary_source,
                                                   "secondary": report.secondary_source,
                                                   "breaks": str(report.breaks)}
            if report.alerts:
                ctx["alerts"] = " | ".join(x.message for x in report.alerts)
            return ToolOutput(context=ctx, facts=facts)

        async def enterprise_value(a: ReconcileArgs) -> ToolOutput:
            sec, as_of = await self._security(a)
            table, report = await ws.enterprise_value(sec, as_of, PeriodType(a.period_type))
            ctx = self._sec_context(sec, as_of) | {"basis": "bank/NBFC (no debt/cash)" if sec.sector_class.is_financial
                                                   else "Bloomberg EV: mkt cap + debt + leases + pref + minority - cash"}
            if report.alerts:
                ctx["alerts"] = " | ".join(x.message for x in report.alerts)
            return ToolOutput(context=ctx, facts=table)

        async def beta(a: BetaArgs) -> ToolOutput:
            sec, as_of = await self._security(a)
            return ToolOutput(context=self._sec_context(sec, as_of),
                              facts=await ws.beta(sec, as_of, years=a.years, frequency=a.frequency))

        async def relative_valuation(a: PeerArgs) -> ToolOutput:
            sec, as_of = await self._security(a)
            view = await ws.rv(sec, as_of, max_peers=a.max_peers + 1)
            facts: list[Datum] = []
            for row in view.rows:
                cols = RV_COLUMNS if row.is_subject else ("EV/EBITDA", "P/E", "P/TBV", "Beta (adj)")
                facts += [row.cells[c].model_copy(update={"label": f"{row.security.symbol} {c}"}) for c in cols]
            facts += list(view.medians.values())
            return ToolOutput(context=self._sec_context(sec, as_of) | {
                "peers": ", ".join(r.security.symbol for r in view.rows if not r.is_subject)}, facts=facts)

        async def option_chain(a: ChainArgs) -> ToolOutput:
            sec, as_of = await self._security(a)
            view = await ws.ovme(sec, as_of, OVMERequest(expiry=a.expiry), strikes_each_side=a.strikes_each_side)
            prov = "quant.derivatives_ovme.OptionValuationEngine"
            facts = list(view.inputs)
            for row in view.chain:
                for kind, v, q in (("CALL", row.call, row.call_quote), ("PUT", row.put, row.put_quote)):
                    tag = f"{sec.symbol} {row.strike:g} {kind} {view.expiry}"
                    facts.append(derive(f"{tag} premium", v.npv, sec.currency, view.inputs, prov))
                    facts.append(derive(f"{tag} delta", v.delta, "", view.inputs, prov))
                    if q is not None and q.implied_vol is not None:
                        facts.append(Datum(
                            label=f"{tag} market implied vol", value=q.implied_vol, source=q.source,
                            confidence_score=1.0, timestamp=q.timestamp,
                            provenance="quant.derivatives_ovme.OptionValuationEngine.implied_volatility",
                            note=f"from mid of bid {q.bid:g} / ask {q.ask:g}"))
            return ToolOutput(context=self._sec_context(sec, as_of) | {"expiry": str(view.expiry),
                                                                       "exercise": view.exercise.value,
                                                                       "notes": " | ".join(view.notes)}, facts=facts)

        async def _calculator(a: OptionArgs, premium: float | None) -> ToolOutput:
            sec, as_of = await self._security(a)
            request = OVMERequest(option_type=a.option_type, strike=a.strike, expiry=a.expiry, volatility=a.volatility,
                                  rate=a.rate, dividend_yield=a.dividend_yield, premium=premium)
            view = await ws.ovme(sec, as_of, request, strikes_each_side=0)
            assert view.calculator is not None and view.calculator_contract is not None
            c = view.calculator_contract
            label = f"{sec.symbol} {c.strike:g} {c.option_type.value} {view.calculator.adjusted_expiry}"
            inputs = list(view.inputs)
            if premium is not None:
                inputs = [d for d in inputs if d.label != "Volatility"] + [Datum(
                    label="Observed premium", value=premium, unit=sec.currency, source="USER_INPUT",
                    confidence_score=1.0, timestamp=ws.clock(), provenance="agent tool argument")]
            return ToolOutput(context=self._sec_context(sec, as_of) | {"exercise": view.calculator.exercise.value,
                                                                       "notes": " | ".join(view.notes)},
                              facts=inputs + option_datums(view.calculator, sec.currency, inputs, label))

        async def price_option(a: OptionArgs) -> ToolOutput:
            return await _calculator(a, None)

        async def implied_volatility(a: ImpliedVolArgs) -> ToolOutput:
            return await _calculator(a, a.premium)

        async def tick_hygiene(a: TickHygieneArgs) -> ToolOutput:
            sec, as_of = await self._security(a)
            return ToolOutput(context=self._sec_context(sec, as_of), facts=await ws.tick_hygiene(sec, as_of, a.hours))

        def tool(name: str, args: type[SecurityArgs], run: Callable[[Any], Awaitable[ToolOutput]], text: str) -> Tool:
            return Tool(name=name, description=text, args=args, run=run)

        return [
            tool("resolve_security", SecurityArgs, resolve_security,
                 "Identify a listing: exchange, currency, sector class (bank/NBFC changes EV rules), industry."),
            tool("get_price_history", PriceArgs, price_history,
                 "GP: corporate-action adjusted closes (engine.adjustments) with last close, period change, "
                 "high/low, RSI(14) and MACD(12,26,9) (quant.technicals)."),
            tool("get_fundamentals", FundamentalsArgs, fundamentals,
                 "FA: point-in-time reported fundamentals and margins/ROE/leverage per period "
                 "(engine.pit_query; only filings public at as-of)."),
            tool("get_revision_history", RevisionArgs, revision_history,
                 "Every restatement/vintage of one reported figure with its filing timestamp."),
            tool("reconcile_fundamentals", ReconcileArgs, reconcile,
                 "Regulatory XBRL vs vendor reconciliation (hygiene.reconciler): resolved values, confidence, "
                 "breaks above the 0.5% tolerance."),
            tool("get_enterprise_value", ReconcileArgs, enterprise_value,
                 "Street-convention EV bridge and EV/EBITDA, EV/Sales, P/TBV (quant.equity_metrics); "
                 "sector-aware for banks/NBFCs."),
            tool("get_beta", BetaArgs, beta,
                 "Bloomberg-style beta vs the market index: raw, Blume-adjusted, std error, R-squared."),
            tool("get_relative_valuation", PeerArgs, relative_valuation,
                 "RV: industry peer multiples (EV/EBITDA, P/E, P/TBV, beta) and peer medians."),
            tool("get_option_chain", ChainArgs, option_chain,
                 "OVME chain: QuantLib premiums and deltas around the money with the market inputs used."),
            tool("price_option", OptionArgs, price_option,
                 "OVME calculator: premium and Greeks for one option (European US index-style, American "
                 "equity, Indian stock options European) on the exchange holiday calendar."),
            tool("implied_volatility", ImpliedVolArgs, implied_volatility,
                 "Solve implied volatility from an observed premium, with the Greeks at that volatility."),
            tool("get_tick_hygiene", TickHygieneArgs, tick_hygiene,
                 "Data-quality: prints the 5xMAD bad-tick filter quarantined recently, by reason."),
        ]

    # ---------------------------------------------------------------- conversation
    def _system_prompt(self) -> str:
        ctx = self._context()
        sec = display_name(ctx.security) if ctx.security else "no active security"
        prompt = SYSTEM_PROMPT.format(context=sec, as_of=ctx.as_of.isoformat())
        self.ledger.allow_text(sec, ctx.as_of.date().isoformat())
        return prompt

    def _digest(self, facts: Sequence[Fact], reason: str) -> str:
        if not facts:
            return f"{reason}. No verified figures were retrieved; use GP, FA, RV or OVME in the terminal."
        lines = "\n".join(f"- {f.datum.label}: {{{{{f.id}}}}}" for f in facts)
        return f"{reason}. Verified figures retrieved for this question:\n{lines}"

    async def ask(self, question: str) -> AgentAnswer:
        messages: list[dict[str, Any]] = [{"role": "system", "content": self._system_prompt()}]
        for q, a in self._history[-self._history_turns:]:
            messages += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        messages.append({"role": "user", "content": question})
        first_new = len(self.ledger) + 1
        calls: list[str] = []
        failures: list[GuardReport] = []
        steps = 0
        reason = "The model did not produce a verifiable answer within the step budget"
        while steps < self._max_steps:
            steps += 1
            try:
                reply = await self._llm.complete(messages, self.tool_schemas())
            except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
                reason = f"The language model is unavailable ({_llm_failure(exc)})"
                break
            if reply.tool_calls:
                messages.append({"role": "assistant", "content": reply.content,
                                 "tool_calls": [c.as_message() for c in reply.tool_calls]})
                for c in reply.tool_calls:
                    calls.append(c.name)
                    messages.append({"role": "tool", "tool_call_id": c.id,
                                     "content": await self.call_tool(c.name, c.arguments)})
                continue
            draft = (reply.content or "").strip()
            report = check_grounding(draft, self.ledger)
            if report.ok and draft:
                text, cited = render_answer(draft, self.ledger)
                self._history.append((question, draft))
                return AgentAnswer(text=text, raw=draft, cited=cited, grounded=True, fallback=False, steps=steps,
                                   tool_calls=calls, guard_failures=failures)
            failures.append(report)
            if len(failures) > self._max_repairs:
                reason = "The model's answer failed numeric grounding and was withheld"
                break
            messages.append({"role": "assistant", "content": draft})
            messages.append({"role": "user", "content": report.feedback() if draft else
                             "Empty answer. Answer the question using tool facts as {{F<n>}} placeholders."})

        new_facts = [f for f in self.ledger.facts() if int(f.id[1:]) >= first_new]
        digest = self._digest(new_facts, reason)
        text, cited = render_answer(digest, self.ledger)
        return AgentAnswer(text=text, raw=digest, cited=cited, grounded=True, fallback=True, steps=steps,
                           tool_calls=calls, guard_failures=failures)
