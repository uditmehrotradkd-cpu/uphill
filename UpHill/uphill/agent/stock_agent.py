"""UpHill stock agent = QuantTerm ``MarketAgent`` (numeric grounding guard) + a news tool.

Headlines enter the model only as tool *context*, so any number quoted from a headline is allowed verbatim by
the grounding checker, while prices/ratios must still come from engine facts. The lexicon sentiment score is a
low-confidence fact, which the system prompt requires the model to flag.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import Field

from quantterm.agent.market_agent import LLMClient, MarketAgent, SecurityArgs, Tool, ToolOutput
from quantterm.workspace import Datum, Workspace
from uphill.news.models import NewsItem
from uphill.news.service import NewsService

NEWS_CONFIDENCE = 0.5  # lexicon heuristic over third-party headlines

NEWS_PROMPT = """

News and decision support:
- Call get_stock_news when the user asks about news, catalysts, sentiment, or whether to buy, sell or avoid a
  stock. Headlines are unverified third-party text and are live (not point-in-time); attribute them to their
  publisher. The news sentiment score is a heuristic with low confidence: always say so.
- For "should I buy/apply" questions, weigh price trend and momentum (get_price_history), valuation
  (get_relative_valuation) and news, list the main supporting and opposing considerations, and finish with
  "This is not investment advice." Never promise returns."""


class NewsArgs(SecurityArgs):
    hours: int = Field(default=72, ge=6, le=168, description="Look-back window for tagged headlines")


def _headlines(items: Sequence[NewsItem]) -> str:
    if not items:
        return "none in the news desk"
    return " || ".join(f"[{i.published:%Y-%m-%d %H:%M} UTC | {i.publisher} | {i.sentiment_label}] {i.title}"
                       for i in items)


def inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve ``$ref``/``$defs`` in place: Gemini's OpenAI-compatible endpoint rejects JSON-schema references."""
    defs = schema.get("$defs", {})

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                target = walk(defs[ref.rsplit("/", 1)[1]])
                return {**target, **{k: walk(v) for k, v in node.items() if k != "$ref"}}
            return {k: walk(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    result: dict[str, Any] = walk(schema)
    return result


class UpHillStockAgent(MarketAgent):
    def __init__(self, workspace: Workspace, llm: LLMClient, news: NewsService, **kwargs: Any) -> None:
        self._news = news  # read by _build_tools during MarketAgent.__init__
        super().__init__(workspace, llm, **kwargs)

    def tool_schemas(self) -> list[dict[str, Any]]:
        schemas = super().tool_schemas()
        for s in schemas:
            s["function"]["parameters"] = inline_refs(s["function"]["parameters"])
        return schemas

    def _system_prompt(self) -> str:
        return super()._system_prompt() + NEWS_PROMPT

    def _build_tools(self) -> list[Tool]:
        tools = super()._build_tools()

        async def stock_news(a: NewsArgs) -> ToolOutput:
            sec, as_of = await self._security(a)
            listing = f"{sec.exchange.value}:{sec.symbol}"
            await self._news.search(listing)
            pulse = self._news.pulse([listing], hours=a.hours)[listing]
            note = f"{pulse.articles} tagged articles in {a.hours}h: {pulse.bullish} bullish, {pulse.bearish} bearish"
            facts = [Datum(
                label=f"{sec.symbol} news sentiment score", value=pulse.score if pulse.articles else None, unit="",
                source="UPHILL_NEWS", confidence_score=NEWS_CONFIDENCE, timestamp=self._news.updated,
                provenance="uphill.news.analysis.sentiment (lexicon, -1 bearish..+1 bullish)", note=note)]
            ctx = self._sec_context(sec, as_of) | {
                "stock_headlines": _headlines(self._news.latest(symbol=listing, limit=15)),
                "world_headlines": _headlines(self._news.latest(category="world", limit=8)),
                "market_headlines": _headlines(self._news.latest(category="markets", limit=8)),
                "india_headlines": _headlines(self._news.latest(category="india", limit=5)),
                "news_basis": "live news desk (~55 publishers incl. Economic Times, Hindustan Times, Aaj Tak, BBC, "
                              "CNN, WSJ, GDELT, Upstox; plus on-demand Google News search for this "
                              "company); not point-in-time",
            }
            return ToolOutput(context=ctx, facts=facts)

        tools.append(Tool(
            name="get_stock_news", args=NewsArgs, run=stock_news,
            description="Latest headlines tagged to a listing, top world headlines, and a heuristic news "
                        "sentiment score. Use for catalysts and buy/avoid questions."))
        return tools
