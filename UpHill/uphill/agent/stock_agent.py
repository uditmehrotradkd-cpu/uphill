"""UpHill stock agent = QuantTerm ``MarketAgent`` (numeric grounding guard) + a news tool.

Headlines enter the model only as tool *context*, so any number quoted from a headline is allowed verbatim by
the grounding checker, while prices/ratios must still come from engine facts. The lexicon sentiment score is a
low-confidence fact, which the system prompt requires the model to flag.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from quantterm.agent.market_agent import (
    PLACEHOLDER,
    AgentAnswer,
    Fact,
    GuardReport,
    LLMClient,
    MarketAgent,
    SecurityArgs,
    Tool,
    ToolOutput,
    check_grounding,
    render_answer,
)
from quantterm.workspace import Datum, Workspace
from uphill.news.models import NewsItem
from uphill.news.service import NewsService

NEWS_CONFIDENCE = 0.5  # lexicon heuristic over third-party headlines
GENERAL_TAG = "[GENERAL]"

CHAT_PROMPT = """

Conversation style: you are also a warm, helpful everyday assistant (like Gemini), not only a market terminal.
- When the user greets you, thanks you or makes small talk, reply naturally and kindly, briefly introduce
  yourself as UpHill's AI analyst and offer what you can help with. Match the user's tone. The app already shows
  a welcome message, so for any other question answer directly: no greeting and no self-introduction.
- Answer general questions about the world (science, history, geography, technology, culture, health basics,
  definitions, how-to, explanations of financial concepts) from your own knowledge, and questions about how to
  use this website from the site guide below.
- Start every answer that contains no market data (greetings, small talk, general knowledge, site help, concept
  explanations) with the tag [GENERAL]; the tag is removed before display. Never tag answers that discuss a
  specific stock, index, sector, currency, commodity or investment decision.
- In [GENERAL] answers ordinary numbers are fine (dates, distances, populations, steps), but never state prices,
  returns, percentages, valuation multiples or money amounts from memory: market figures must come from a tool
  and be cited as placeholders in an untagged answer.
- For current events or "what is happening in the world/markets today" call get_news first. You have no other
  live information and your own knowledge may be out of date, so say so for anything recent.
- Questions about specific stocks, the market or investing decisions are not general: use the tools and the
  numeric grounding rules.

UpHill site guide (for questions about this website):
- Dashboard: watchlist with live NSE prices (add or remove symbols), a Bloomberg-style terminal (type e.g.
  "RELIANCE IN Equity GP"; functions GP chart, FA fundamentals, RV peer valuation, OVME options; ASOF
  yyyy-mm-dd for point-in-time views; WATCH ADD/DEL; HELP) and this AI analyst.
- Analytics: search any stock, then use the tabs Overview (live price, returns, 52-week range, RSI, volume,
  news pulse), Chart & studies (candlestick or line chart; 1m, 5m, 15m, 30m, 1h or 1D bars with a live candle;
  pick SMA, EMA, Bollinger Bands, VWAP, RSI, MACD, Stochastic or ATR in the Function menu, set the period and
  press Apply; remove a study with the x on its chip), Fundamentals, Peer valuation and Options & Greeks.
- All Stocks: the Nifty 500 universe split into large, mid and small caps with live prices, returns, 52-week
  range, RSI, volume and news; sort, filter, search and open any stock.
- News: world, markets, business and India news from about 55 publishers with sentiment tags.
- Chat: EN / हिं switches between English and Hindi answers; the speaker button reads answers aloud; the
  sliders button chooses the voice and speed.
- Live prices are NSE quotes refreshed during market hours; history is end-of-day. Nothing here is investment
  advice."""

NEWS_PROMPT = """

News and decision support:
- Call get_stock_news when the user asks about news, catalysts, sentiment, or whether to buy, sell or avoid a
  stock. Headlines are unverified third-party text and are live (not point-in-time); attribute them to their
  publisher. The news sentiment score is a heuristic with low confidence: always say so.
- For "should I buy/apply" questions, weigh price trend and momentum (get_price_history), valuation
  (get_relative_valuation) and news, list the main supporting and opposing considerations, and finish with
  "This is not investment advice." Never promise returns."""

Language = Literal["en", "hi"]

LANGUAGE_PROMPTS: dict[Language, str] = {
    "en": """

Language: answer in English. Answers may be read aloud, so prefer short, plain sentences; light markdown
(**bold** and short bullet lists) is fine, but no tables or headings.""",
    "hi": """

Language: answer in Hindi (Devanagari script), even if the question is in English or Hinglish. Answers are
read aloud by a Hindi voice, so use short, simple, natural sentences and avoid tables and markdown.
- Keep tickers, company names, exchange codes, the [GENERAL] tag and {{F<n>}} placeholders exactly as written.
- For market data the numeric grounding rules still apply in Hindi: never write such figures as numerals in
  any script (0-9 or ०-९) or in words (e.g. "दस प्रतिशत", "दो सौ करोड़"); use the placeholders. In [GENERAL]
  answers write ordinary numbers with 0-9 digits.
- Greet in Hindi (e.g. "नमस्ते!").
- End buy/sell/avoid answers with "यह निवेश सलाह नहीं है।" instead of the English disclaimer.""",
}
HINDI_DIGEST = "मॉडल का उत्तर सत्यापित नहीं हो सका ({reason})."
HINDI_FACTS = "इस प्रश्न के लिए प्राप्त सत्यापित आंकड़े:"
HINDI_NO_FACTS = "कोई सत्यापित आंकड़ा नहीं मिला; टर्मिनल में GP, FA, RV या OVME देखें।"


class NewsArgs(SecurityArgs):
    hours: int = Field(default=72, ge=6, le=168, description="Look-back window for tagged headlines")


class WorldNewsArgs(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    category: Literal["world", "markets", "business", "india", "stock", "all"] = "world"
    query: str | None = Field(default=None, max_length=60,
                              description="Optional keyword filter, e.g. 'oil', 'election', 'RBI'")
    limit: int = Field(default=12, ge=3, le=25)


_NUM = r"\d[\d,]*(?:\.\d+)?"
# Money amounts and percentages are never allowed from model memory, even in a general answer.
_MONEY = re.compile(
    rf"(?:[₹$€£]|\b(?:rs\.?|inr|usd|eur|gbp)\s?)\s*({_NUM})"
    rf"|({_NUM})\s*(?:%|x\b|per\s?cent\b|percent\b|bps\b|basis\s+points?\b|crores?\b|cr\b|lakhs?\b|rupees?\b|"
    r"dollars?\b|प्रतिशत|फ़ीसदी|फीसदी|करोड़|लाख|रुपये|रुपए|डॉलर)",
    re.IGNORECASE,
)
_MARKET_WORDS = re.compile(
    r"\b(?:rsi|macd|p/?e|eps|ebitda|beta|market\s+cap|52[- ]week|nifty|sensex|stock\s+price|share\s+price|"
    r"target\s+price|implied\s+vol)\b|सेंसेक्स|निफ्टी|शेयर\s+का\s+भाव",
    re.IGNORECASE,
)


class GeneralGuardReport(GuardReport):
    def feedback(self) -> str:
        parts = [f"figures typed from memory: {', '.join(self.uncited_numbers + self.spelled_quantities)}"]
        if self.unknown_facts:
            parts.append(f"placeholders for facts that do not exist: {', '.join(self.unknown_facts)}")
        return ("GENERAL ANSWER CHECK FAILED - " + "; ".join(parts) + ". In a [GENERAL] answer you may use "
                "ordinary numbers (dates, distances, populations) but never prices, returns, percentages, "
                "valuation multiples, indicator values or money amounts unless they appear in a tool result. "
                "Remove them or describe without numbers, or call a tool and answer without the tag using "
                "{{F<n>}} placeholders. Rewrite the full answer.")


def _norm(token: str) -> str:
    plain = token.replace(",", "")
    return plain if "." in plain else str(int(plain))


def is_general(draft: str) -> bool:
    return draft.lstrip().startswith(GENERAL_TAG)


def strip_general(draft: str) -> str:
    return draft.lstrip()[len(GENERAL_TAG):].lstrip() if is_general(draft) else draft


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
        self.language: Language = "en"
        super().__init__(workspace, llm, **kwargs)

    async def ask(self, question: str, language: Language = "en") -> AgentAnswer:
        self.language = language
        return await super().ask(question)

    def tool_schemas(self) -> list[dict[str, Any]]:
        schemas = super().tool_schemas()
        for s in schemas:
            s["function"]["parameters"] = inline_refs(s["function"]["parameters"])
        return schemas

    def _system_prompt(self) -> str:
        return super()._system_prompt() + NEWS_PROMPT + CHAT_PROMPT + LANGUAGE_PROMPTS[self.language]

    def _check(self, draft: str) -> GuardReport:
        if not is_general(draft):
            return super()._check(draft)
        body = strip_general(draft)
        report = check_grounding(body, self.ledger)
        text = unicodedata.normalize("NFC", PLACEHOLDER.sub(" ", body))
        if report.uncited_numbers and _MARKET_WORDS.search(text):  # market indicators quoted from memory
            return GeneralGuardReport(**report.model_dump())
        allowed = self.ledger.allowed_numbers
        money = [m.group(0).strip() for m in _MONEY.finditer(text) if _norm(m.group(1) or m.group(2)) not in allowed]
        return GeneralGuardReport(cited=report.cited, unknown_facts=report.unknown_facts,
                                  uncited_numbers=list(dict.fromkeys(money)),
                                  spelled_quantities=report.spelled_quantities)

    def _render(self, draft: str) -> tuple[str, list[Fact]]:
        return render_answer(strip_general(draft), self.ledger)

    def _digest(self, facts: Sequence[Fact], reason: str) -> str:
        if self.language != "hi":
            return super()._digest(facts, reason)
        head = HINDI_DIGEST.format(reason=reason)
        if not facts:
            return f"{head} {HINDI_NO_FACTS}"
        lines = "\n".join(f"- {f.datum.label}: {{{{{f.id}}}}}" for f in facts)
        return f"{head} {HINDI_FACTS}\n{lines}"

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

        async def news(a: WorldNewsArgs) -> ToolOutput:
            category = None if a.category == "all" else a.category
            items = self._news.latest(category=category, query=a.query, limit=a.limit)
            ctx: dict[str, str] = {"category": a.category}
            if a.query and not items:
                items = self._news.latest(category=category, limit=a.limit)
                ctx["note"] = f"no headline matched '{a.query}'; showing the latest {a.category} headlines instead"
            ctx |= {
                "headlines": _headlines(items),
                "news_desk_updated": self._news.updated.isoformat() if self._news.updated else "not yet refreshed",
                "news_basis": "live news desk (~55 publishers incl. Reuters via GDELT, BBC, Guardian, NYT, WSJ, "
                              "Economic Times, Hindustan Times); unverified third-party headlines",
            }
            return ToolOutput(context=ctx)

        tools.append(Tool(
            name="get_news", args=WorldNewsArgs, run=news,
            description="Latest live headlines by category (world, markets, business, india, stock or all), "
                        "optionally filtered by keyword. Use for current events and 'what is happening' "
                        "questions; it needs no security."))
        return tools
