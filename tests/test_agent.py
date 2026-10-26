from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Iterator, Sequence
from datetime import date
from typing import Any

import httpx
import pytest
from fakes import FakeDatabase
from market_seed import AS_OF, CRORE, RELIANCE_FY24, seeded_workspace

from quantterm.agent.market_agent import (
    AgentContext,
    FactLedger,
    LLMReply,
    MarketAgent,
    OpenAICompatibleClient,
    ToolCall,
    check_grounding,
)
from quantterm.connectors.base import Metric
from quantterm.workspace import Datum, Workspace, format_value

Step = Callable[[list[dict[str, Any]]], LLMReply]


def run(coro: Any) -> Any:
    return asyncio.run(coro)


class ScriptedLLM:
    """Plays a fixed conversation; each step sees the messages exactly as a real model would."""

    def __init__(self, steps: Sequence[Step]) -> None:
        self.steps = list(steps)
        self.transcripts: list[list[dict[str, Any]]] = []
        self.tools_seen: list[Sequence[dict[str, Any]]] = []

    async def complete(self, messages: Sequence[dict[str, Any]], tools: Sequence[dict[str, Any]]) -> LLMReply:
        self.transcripts.append(list(messages))
        self.tools_seen.append(tools)
        step = self.steps.pop(0) if len(self.steps) > 1 else self.steps[0]
        return step(list(messages))


def call(name: str, **arguments: Any) -> Step:
    return lambda _: LLMReply(tool_calls=[ToolCall(id=f"call_{name}", name=name, arguments=json.dumps(arguments,
                                                                                                    default=str))])


def say(text: str) -> Step:
    return lambda _: LLMReply(content=text)


def last_facts(messages: list[dict[str, Any]]) -> dict[str, str]:
    """label -> fact id from the most recent tool result."""
    tool_msg = next(m for m in reversed(messages) if m["role"] == "tool")
    return {f["label"]: f["id"] for f in json.loads(tool_msg["content"])["facts"]}


@pytest.fixture(scope="module")
def market() -> Iterator[tuple[FakeDatabase, Workspace]]:
    yield run(seeded_workspace())


def agent_for(ws: Workspace, llm: Any, **kwargs: Any) -> MarketAgent:
    return MarketAgent(ws, llm, context=lambda: AgentContext(as_of=AS_OF), **kwargs)


# ------------------------------------------------------------------------------------------ guard
def _ledger_with(*datums: Datum) -> FactLedger:
    ledger = FactLedger()
    ledger.add(datums, "test")
    return ledger


def _datum(label: str, value: float, note: str | None = None) -> Datum:
    return Datum(label=label, value=value, unit="", source="raw_market_bars", confidence_score=1.0,
                 timestamp=AS_OF, provenance="quant.technicals.rsi", note=note)


def test_guard_accepts_placeholders_and_label_numbers() -> None:
    ledger = _ledger_with(_datum("RSI(14)", 61.2, "period 2024-03-31"))
    report = check_grounding("RSI(14) is {{F1}} for the period ending 2024-03-31.", ledger)
    assert report.ok and report.cited == ["F1"]


@pytest.mark.parametrize(("text", "field", "expected"), [
    ("RSI is 61.2, see {{F1}}", "uncited_numbers", ["61.2"]),
    ("Revenue was 9,01,064 crore", "uncited_numbers", ["9", "01,064"]),
    ("Market cap is 1,234,567.89 USD", "uncited_numbers", ["1,234,567.89"]),
    ("Beta of {{F7}}", "unknown_facts", ["F7"]),
    ("Shares fell about twenty percent", "spelled_quantities", ["twenty percent"]),
    ("Margins expanded by two hundred basis points", "spelled_quantities", ["two hundred basis points"]),
])
def test_guard_rejects_ungrounded_figures(text: str, field: str, expected: list[str]) -> None:
    report = check_grounding(text, _ledger_with(_datum("RSI(14)", 61.2)))
    assert not report.ok and getattr(report, field) == expected
    assert "GROUNDING CHECK FAILED" in report.feedback()


def test_guard_does_not_whitelist_fact_values() -> None:
    # The value itself is never quotable: it must go through the placeholder so provenance is attached.
    report = check_grounding("RSI is 61.2", _ledger_with(_datum("RSI(14)", 61.2)))
    assert report.uncited_numbers == ["61.2"]


# ------------------------------------------------------------------------------------------ loop
def test_agent_repairs_hallucinated_numbers_then_cites_facts(market: tuple[FakeDatabase, Workspace]) -> None:
    _, ws = market

    def grounded(messages: list[dict[str, Any]]) -> LLMReply:
        ids = last_facts(messages)
        return LLMReply(content=f"RELIANCE last closed at {{{{{ids['Last close']}}}}} with RSI(14) at "
                                f"{{{{{ids['RSI(14)']}}}}}; momentum is neither overbought nor oversold.")

    llm = ScriptedLLM([
        call("get_price_history", ticker="RELIANCE", exchange_code="IN", lookback_days=183),
        say("RELIANCE last closed at 1,263.40 with RSI(14) at 55."),
        grounded,
    ])
    answer = run(agent_for(ws, llm).ask("Where is Reliance trading and how is momentum?"))

    assert answer.grounded and not answer.fallback and answer.tool_calls == ["get_price_history"]
    assert len(answer.guard_failures) == 1 and answer.guard_failures[0].uncited_numbers == ["1,263.40", "55"]
    assert "GROUNDING CHECK FAILED" in llm.transcripts[2][-1]["content"]
    last_close = next(f for f in answer.cited if f.datum.label == "Last close").datum
    assert f"{format_value(last_close)} [raw_market_bars | conf 1.00 | 2024-12-31 15:30 IST]" in answer.text
    assert "Sources:" in answer.text and "engine.adjustments.get_adjusted_series" in answer.text
    assert "quant.technicals.rsi" in answer.text and "{{" not in answer.text


def test_agent_withholds_answer_that_never_grounds(market: tuple[FakeDatabase, Workspace]) -> None:
    _, ws = market
    llm = ScriptedLLM([call("get_beta", ticker="RELIANCE", exchange_code="IN"), say("Beta is roughly 1.1.")])
    answer = run(agent_for(ws, llm, max_repairs=2).ask("What's the beta?"))
    assert answer.fallback and len(answer.guard_failures) == 3
    assert "withheld" in answer.text and "Adjusted beta" in answer.text and "1.1" not in answer.raw
    adjusted = next(f for f in answer.cited if f.datum.label == "Adjusted beta").datum
    assert format_value(adjusted) in answer.text and "quant.equity_metrics.compute_beta" in answer.text


def test_agent_falls_back_when_model_unreachable(market: tuple[FakeDatabase, Workspace]) -> None:
    class Down:
        async def complete(self, messages: Sequence[dict[str, Any]], tools: Sequence[dict[str, Any]]) -> LLMReply:
            raise httpx.ConnectError("connection refused")

    answer = run(agent_for(market[1], Down()).ask("P/E?"))
    assert answer.fallback and "unavailable (ConnectError)" in answer.text and answer.cited == []


def test_agent_tool_results_are_point_in_time(market: tuple[FakeDatabase, Workspace]) -> None:
    agent = agent_for(market[1], ScriptedLLM([say("ok")]))
    early = json.loads(run(agent.call_tool("get_fundamentals", {"ticker": "RELIANCE", "exchange_code": "IN",
                                                               "as_of": "2024-06-28", "periods": 1})))
    late = json.loads(run(agent.call_tool("get_fundamentals", {"ticker": "RELIANCE", "exchange_code": "IN",
                                                              "periods": 1})))
    label = f"{Metric.REVENUE} FY 2024-03-31"
    rev_early = next(f for f in early["facts"] if f["label"] == label)
    rev_late = next(f for f in late["facts"] if f["label"] == label)
    assert agent.ledger.get(rev_early["id"]).datum.value == float(RELIANCE_FY24[Metric.REVENUE])  # type: ignore[union-attr]
    assert agent.ledger.get(rev_late["id"]).datum.value == float(  # type: ignore[union-attr]
        RELIANCE_FY24[Metric.REVENUE] + 1500 * CRORE)
    assert early["context"]["as_of"].startswith("2024-06-28T23:59:59")


def test_option_tools_carry_input_provenance(market: tuple[FakeDatabase, Workspace]) -> None:
    agent = agent_for(market[1], ScriptedLLM([say("ok")]))
    priced = json.loads(run(agent.call_tool("price_option", {
        "ticker": "RELIANCE", "exchange_code": "IN", "option_type": "PUT", "strike": 1240, "volatility": 0.22})))
    facts = {f["label"]: agent.ledger.get(f["id"]).datum for f in priced["facts"]}  # type: ignore[union-attr]
    premium = facts["RELIANCE 1240 PUT 2025-01-30 premium"]
    inputs = [facts[k] for k in ("Last price", "Volatility", "Risk-free rate", "Dividend yield")]
    assert premium.confidence_score == min(d.confidence_score for d in inputs)
    assert premium.timestamp == max(d.timestamp for d in inputs if d.timestamp is not None)
    assert facts["Volatility"].source == "USER_INPUT" and facts["Risk-free rate"].source == "RBI:TBILL91"

    implied = json.loads(run(agent.call_tool("implied_volatility", {
        "ticker": "RELIANCE", "exchange_code": "IN", "option_type": "PUT", "strike": 1240,
        "premium": premium.value})))
    ifacts = {f["label"]: agent.ledger.get(f["id"]).datum for f in implied["facts"]}  # type: ignore[union-attr]
    assert "Volatility" not in ifacts and ifacts["Observed premium"].value == premium.value
    assert ifacts["RELIANCE 1240 PUT 2025-01-30 volatility"].value == pytest.approx(0.22, abs=1e-6)


@pytest.mark.parametrize(("name", "arguments", "error"), [
    ("get_quotes", {}, "unknown tool"),
    ("get_beta", "{not json", "invalid arguments"),
    ("get_beta", {"ticker": "RELIANCE", "exchange_code": "XX"}, "invalid arguments"),
    ("get_beta", {"ticker": "RELIANCE", "exchange_code": "IN", "years": 50}, "invalid arguments"),
    ("get_beta", {"ticker": "TCS", "exchange_code": "IN"}, "not registered"),
    ("get_relative_valuation", {"ticker": "NIFTY", "exchange_code": "IN"}, "no industry"),
])
def test_tool_errors_are_returned_to_model_without_facts(market: tuple[FakeDatabase, Workspace], name: str,
                                                        arguments: Any, error: str) -> None:
    agent = agent_for(market[1], ScriptedLLM([say("ok")]))
    out = json.loads(run(agent.call_tool(name, arguments)))
    assert error in out["error"] and len(agent.ledger) == 0


def test_tool_schemas_are_strict_json_schema(market: tuple[FakeDatabase, Workspace]) -> None:
    agent = agent_for(market[1], ScriptedLLM([say("ok")]))
    schemas = {s["function"]["name"]: s["function"]["parameters"] for s in agent.tool_schemas()}
    assert len(schemas) == 12
    beta = schemas["get_beta"]
    assert set(beta["required"]) == {"ticker", "exchange_code"} and beta["additionalProperties"] is False
    assert "IN" in beta["properties"]["exchange_code"]["enum"]


def test_tick_hygiene_tool(market: tuple[FakeDatabase, Workspace]) -> None:
    agent = agent_for(market[1], ScriptedLLM([say("ok")]))
    out = json.loads(run(agent.call_tool("get_tick_hygiene", {"ticker": "ONGC", "exchange_code": "IN"})))
    assert out["facts"][0]["label"] == "Rejected ticks (24h)" and out["facts"][0]["note"] == "none"


# ------------------------------------------------------------------------------------------ transport
def test_openai_compatible_client_wire_format() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "get_beta",
                                                          "arguments": '{"ticker":"AAPL","exchange_code":"US"}'}}]}}]})

    async def scenario() -> LLMReply:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = OpenAICompatibleClient("https://llm.internal/v1/", "gpt-4o-mini", "sk-test", client=http)
            return await client.complete([{"role": "user", "content": "beta?"}],
                                         [{"type": "function", "function": {"name": "get_beta"}}])

    reply = run(scenario())
    assert seen["url"] == "https://llm.internal/v1/chat/completions" and seen["auth"] == "Bearer sk-test"
    assert seen["body"]["temperature"] == 0 and seen["body"]["tool_choice"] == "auto"
    assert reply.tool_calls == [ToolCall(id="c1", name="get_beta", arguments='{"ticker":"AAPL","exchange_code":"US"}')]


def test_resolve_security_context_numbers_become_quotable(market: tuple[FakeDatabase, Workspace]) -> None:
    agent = agent_for(market[1], ScriptedLLM([say("ok")]))
    out = json.loads(run(agent.call_tool("resolve_security", {"ticker": "AAPL", "exchange_code": "US",
                                                              "as_of": date(2024, 12, 2).isoformat()})))
    assert out["context"]["security"] == "AAPL US Equity" and out["facts"] == []
    assert check_grounding("As of 2024-12-02 AAPL is classified as Technology Hardware.", agent.ledger).ok


def test_client_round_trips_provider_tool_call_extras_and_retries_overload() -> None:
    sig = {"google": {"thought_signature": "abc"}}
    statuses = [503]

    def handler(request: httpx.Request) -> httpx.Response:
        if statuses:
            return httpx.Response(statuses.pop())
        return httpx.Response(200, json={"choices": [{"message": {"content": None, "tool_calls": [
            {"id": "", "type": "function", "extra_content": sig,
             "function": {"name": "get_price_history", "arguments": "{}"}}]}}]})

    async def scenario() -> LLMReply:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = OpenAICompatibleClient("https://llm.test/v1", "m", "k", client=http, backoff=0)
            return await client.complete([{"role": "user", "content": "q"}], [])

    (call,) = asyncio.run(scenario()).tool_calls
    assert call.id == "call_0" and call.extra_content == sig
    assert call.as_message()["extra_content"] == sig
    assert "extra_content" not in ToolCall(id="x", name="n", arguments="{}").as_message()


def test_client_retries_timeouts_then_reports_cause() -> None:
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        assert json.loads(request.content)["reasoning_effort"] == "low"
        if attempts["n"] == 1:
            raise httpx.ReadTimeout("slow", request=request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    async def scenario() -> LLMReply:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = OpenAICompatibleClient("https://llm.test/v1", "m", "k", client=http, backoff=0,
                                            reasoning_effort="low")
            return await client.complete([{"role": "user", "content": "q"}], [])

    assert asyncio.run(scenario()).content == "ok" and attempts["n"] == 2


def test_client_falls_back_when_a_model_quota_is_exhausted_and_remembers_it() -> None:
    seen: list[str] = []
    quota = [{"error": {"code": 429, "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo",
                                                  "retryDelay": "71764s"}]}}]

    def handler(request: httpx.Request) -> httpx.Response:
        model = json.loads(request.content)["model"]
        seen.append(model)
        if model == "primary":
            return httpx.Response(429, json=quota)
        return httpx.Response(200, json={"choices": [{"message": {"content": f"from {model}"}}]})

    now = [0.0]

    async def scenario() -> list[str | None]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = OpenAICompatibleClient("https://llm.test/v1", "primary", "k", client=http, backoff=0,
                                            fallback_models=["backup"], clock=lambda: now[0])
            first = await client.complete([{"role": "user", "content": "q"}], [])
            second = await client.complete([{"role": "user", "content": "q"}], [])
            now[0] = 80_000.0  # quota reset passed: primary is tried again
            await client.complete([{"role": "user", "content": "q"}], [])
            return [first.content, second.content, client.last_model]

    assert asyncio.run(scenario()) == ["from backup", "from backup", "backup"]
    # long retryDelay = spent daily quota: no in-place retries, skipped on the 2nd call, retried after reset
    assert seen == ["primary", "backup", "backup", "primary", "backup"]
