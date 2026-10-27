from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Iterator, Sequence
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest
from fakes import InMemoryBroker
from fastapi.testclient import TestClient

from quantterm.db.repositories import SecurityRecord
from quantterm.models import Exchange, PeriodType, SectorClass
from quantterm.settings import Settings
from quantterm.streaming import tick_channel
from quantterm.terminal.command_router import CommandRouter
from quantterm.workspace import Datum, FAView, GPView, OVMERequest, OVMEView, RVView, WorkspaceError
from uphill.agent.stock_agent import inline_refs
from uphill.news.analysis import SymbolTagger, sentiment
from uphill.news.models import NewsItem, item_id
from uphill.news.service import NewsService
from uphill.news.sources import google_publisher, parse_gdelt, parse_rss, parse_upstox_news, safe_url
from uphill.server import create_app
from uphill.settings import UpHillSettings
from uphill.state import Session, SessionStore, UpHillState

NOW = datetime(2026, 10, 26, 6, 0, tzinfo=UTC)
SID = {"X-Session-Id": "0123456789abcdef-test"}


# ------------------------------------------------------------------------------------------ sources
RSS = b"""<?xml version="1.0"?>
<rss version="2.0" xmlns:media="http://search.yahoo.com/mrss/"><channel>
<item><title>Infosys shares surge after strong &lt;b&gt;Q2&lt;/b&gt; beat</title>
<link>https://example.com/infy</link><description>&lt;p&gt;IT major rallies&lt;/p&gt;</description>
<pubDate>Mon, 26 Oct 2026 04:30:00 +0000</pubDate><media:content url="https://img.example.com/a.jpg"/></item>
<item><title>Bad link</title><link>javascript:alert(1)</link></item>
</channel></rss>"""

ATOM = b"""<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">
<entry><title>Oil falls on demand fears</title><link href="https://example.com/oil"/>
<updated>2026-10-26T03:00:00Z</updated><summary>Crude slides</summary></entry></feed>"""


def test_rss_and_atom_parse_and_drop_unsafe_links() -> None:
    items = parse_rss(RSS, "ET Markets", "markets", NOW)
    assert [i.title for i in items] == ["Infosys shares surge after strong Q2 beat"]
    assert items[0].summary == "IT major rallies" and items[0].image == "https://img.example.com/a.jpg"
    assert items[0].published == datetime(2026, 10, 26, 4, 30, tzinfo=UTC)
    (oil,) = parse_rss(ATOM, "BBC World", "world", NOW)
    assert oil.url == "https://example.com/oil" and oil.published.hour == 3
    assert safe_url("javascript:alert(1)") is None and safe_url("data:x") is None


def test_rss_rejects_entity_expansion() -> None:
    bomb = b'<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "aaaa">]><rss><channel><item><title>&a;</title></item></channel></rss>'
    with pytest.raises(Exception, match="unreadable"):
        parse_rss(bomb, "x", "world", NOW)


def test_gdelt_and_upstox_news_parse() -> None:
    gdelt = parse_gdelt({"articles": [
        {"url": "https://reuters.com/x", "title": "Central bank holds rates", "seendate": "20261026T051500Z",
         "domain": "reuters.com", "socialimage": ""},
        {"url": "https://x.com/y", "title": "bad date", "seendate": "nope"},
    ]})
    assert [(i.publisher, i.category, i.published.hour) for i in gdelt] == [("reuters.com", "world", 5)]

    payload = {"status": "success", "data": {
        "NSE_EQ|INE009A01021": [{"heading": "IT stocks rally", "summary": "s", "article_link": "https://u.com/a",
                                 "published_time": 1793000000000, "thumbnail": "https://u.com/t.webp"}],
        "NSE_EQ|INE467B01029": [{"heading": "IT stocks rally", "summary": "s", "article_link": "https://u.com/a",
                                 "published_time": 1793000000000}],
    }}
    listings = {"NSE_EQ|INE009A01021": "NSE:INFY", "NSE_EQ|INE467B01029": "NSE:TCS"}
    (item,) = parse_upstox_news(payload, listings)
    assert item.symbols == ["NSE:INFY", "NSE:TCS"] and item.category == "stock"


def test_sentiment_and_tagging() -> None:
    assert sentiment("Reliance shares surge to record on strong profit")[1] == "bullish"
    assert sentiment("HDFC Bank slumps as loan losses rise")[1] == "bearish"
    assert sentiment("Markets did not fall despite war fears")[0] > -1
    assert sentiment("Board meeting scheduled")[1] == "neutral"
    tagger = SymbolTagger({"NSE:RELIANCE": ["Reliance Industries", "RIL"], "NSE:TCS": ["Tata Consultancy"],
                           "NSE:M&M": []})
    assert tagger.tag("RIL and tata consultancy services gain") == ["NSE:RELIANCE", "NSE:TCS"]
    assert tagger.tag("Trilogy of tcs results") == []  # ticker match is case-sensitive and word-bounded


def _item(title: str, url: str, published: datetime, **kw: Any) -> NewsItem:
    return NewsItem(id=item_id(url), title=title, url=url, source="T", publisher="T", category="markets",
                    published=published, **kw)


def test_news_service_isolates_failures_dedupes_and_pulses() -> None:
    async def good() -> list[NewsItem]:
        return [_item("Infosys surges on record deal", "https://a/1", NOW - timedelta(hours=1)),
                _item("Infosys surges on record deal", "https://a/1", NOW - timedelta(hours=1)),
                _item("Infosys falls on weak guidance", "https://a/2", NOW - timedelta(hours=2)),
                _item("Old Infosys gain", "https://a/3", NOW - timedelta(days=10))]

    async def broken() -> list[NewsItem]:
        raise RuntimeError("feed down")

    news = NewsService({"good": good, "broken": broken}, SymbolTagger({"NSE:INFY": ["Infosys"]}), clock=lambda: NOW)
    assert asyncio.run(news.refresh()) == 3
    assert news.status["broken"].ok is False and "feed down" in (news.status["broken"].error or "")
    assert [i.url for i in news.latest(symbol="NSE:INFY", limit=2)] == ["https://a/1", "https://a/2"]
    pulse = news.pulse(["NSE:INFY", "NSE:TCS"])
    assert (pulse["NSE:INFY"].articles, pulse["NSE:INFY"].bullish, pulse["NSE:INFY"].bearish) == (2, 1, 1)
    assert pulse["NSE:TCS"].articles == 0 and pulse["NSE:TCS"].label == "neutral"
    assert news.latest(sentiment_label="bearish")[0].url == "https://a/2"


def test_google_publisher_moves_suffix_out_of_title() -> None:
    item = google_publisher(_item("Sensex jumps 483 points - Business Standard", "https://g/1", NOW))
    assert (item.title, item.publisher) == ("Sensex jumps 483 points", "Business Standard")
    assert google_publisher(_item("No suffix here", "https://g/2", NOW)).publisher == "T"


def test_news_search_tags_listing_merges_and_respects_ttl() -> None:
    clock = [NOW]
    calls: list[str] = []

    async def searcher(listing: str) -> list[NewsItem]:
        calls.append(listing)
        return [_item("Company wins large order", f"https://s/{len(calls)}", clock[0] - timedelta(minutes=5))]

    news = NewsService({}, SymbolTagger({}), searcher=searcher, search_ttl=timedelta(minutes=5),
                       clock=lambda: clock[0])
    assert asyncio.run(news.search("NSE:TCS")) == 1
    assert asyncio.run(news.search("NSE:TCS")) == 0 and calls == ["NSE:TCS"]  # cached within TTL
    assert news.latest(symbol="NSE:TCS")[0].sentiment_label == "bullish"
    clock[0] += timedelta(minutes=6)
    assert asyncio.run(news.search("NSE:TCS")) == 1 and len(news.latest(symbol="NSE:TCS")) == 2

    async def failing(listing: str) -> list[NewsItem]:
        raise RuntimeError("google down")

    assert asyncio.run(NewsService({}, SymbolTagger({}), searcher=failing).search("NSE:TCS")) == 0


def test_inline_refs_removes_json_schema_references() -> None:
    schema = {"properties": {"t": {"$ref": "#/$defs/T", "description": "d"}}, "$defs": {"T": {"enum": ["A"]}}}
    assert inline_refs(schema) == {"properties": {"t": {"enum": ["A"], "description": "d"}}}


def test_session_store_is_bounded_lru() -> None:
    made: list[Session] = []

    def factory() -> Session:
        made.append(Session(router=CommandRouter(_Workspace()), agent=None))
        return made[-1]

    store = SessionStore(factory, max_sessions=2)
    a = store.get("a")
    store.get("b")
    assert store.get("a") is a
    store.get("c")  # evicts b (least recently used)
    assert len(store) == 2 and store.get("a") is a and len(made) == 3


# ---------------------------------------------------------------------------------------------- API
INFY = SecurityRecord(symbol="INFY", exchange=Exchange.NSE, isin="INE009A01021", currency="INR",
                      sector_class=SectorClass.GENERAL, industry="IT Services")


class _Workspace:
    clock = staticmethod(lambda: NOW)

    async def resolve(self, ticker: str, exchanges: Sequence[Exchange]) -> SecurityRecord:
        if ticker != "INFY":
            raise WorkspaceError(f"unknown security {ticker}")
        return INFY

    async def gp(self, sec: SecurityRecord, as_of: datetime, *, lookback_days: int, adjust_splits: bool,
                 adjust_dividends: bool) -> GPView:
        def d(label: str, v: float | None, unit: str = "INR") -> Datum:
            return Datum(label=label, value=v, unit=unit, source="PRICES", confidence_score=1.0, timestamp=NOW,
                         provenance="test")

        return GPView(security=sec, as_of=as_of, adjust_splits=adjust_splits, adjust_dividends=adjust_dividends,
                      dates=[date(2026, 10, 22), date(2026, 10, 23)], close=[1500.0, 1512.5], volume=[1.0, 2.0],
                      rsi=[None, 55.0], macd=[None, 1.0], signal=[None, float("nan")],
                      summary=[d("Last close", 1512.5), d("Period change", 0.83, "%"), d("Period high", 1520.0),
                               d("Period low", 1490.0), d("RSI(14)", 55.0, "")])

    async def fa(self, sec: SecurityRecord, as_of: datetime, *, period_type: PeriodType, periods: int) -> FAView:
        raise WorkspaceError("no fundamentals")

    async def rv(self, sec: SecurityRecord, as_of: datetime) -> RVView:
        raise WorkspaceError("no peers")

    async def ovme(self, sec: SecurityRecord, as_of: datetime, request: OVMERequest | None = None) -> OVMEView:
        raise WorkspaceError("no options")


@pytest.fixture
def broker() -> InMemoryBroker:
    return InMemoryBroker()


@pytest.fixture
def client(broker: InMemoryBroker) -> Iterator[TestClient]:
    async def feed() -> list[NewsItem]:
        return [_item("Infosys surges on record deal", "https://a/1", NOW - timedelta(hours=1))]

    settings = Settings(api_token="a" * 32, ingest_token="i" * 32, upstox_symbols=["NSE:INFY"], _env_file=None)
    news = NewsService({"t": feed}, SymbolTagger({"NSE:INFY": ["Infosys"]}), clock=lambda: NOW)
    asyncio.run(news.refresh())
    ws = _Workspace()
    state = UpHillState(
        settings=settings, uphill=UpHillSettings(_env_file=None), workspace=ws, broker=broker, news=news,
        sessions=SessionStore(lambda: Session(router=CommandRouter(ws, clock=lambda: NOW), agent=None), 10),
        watchlist=[(Exchange.NSE, "INFY")], agent_enabled=False, live_feeds=1, clock=lambda: NOW,
    )
    with TestClient(create_app(state=state)) as c:
        yield c


def test_terminal_command_keeps_per_session_context(client: TestClient) -> None:
    assert client.post("/api/terminal/command", json={"command": "HELP"}).status_code == 422  # no session header
    bad = client.post("/api/terminal/command", json={"command": "HELP"}, headers={"X-Session-Id": "../x"})
    assert bad.status_code == 400
    r = client.post("/api/terminal/command", json={"command": "INFY IN Equity GP 1M"}, headers=SID).json()
    assert r["kind"] == "VIEW" and r["function"] == "GP" and r["listing"] == "NSE:INFY"
    assert r["view"]["signal"] == [None, None]  # NaN serialised as null
    assert r["view"]["summary"][0]["display"] == "1,512.50 INR"
    again = client.post("/api/terminal/command", json={"command": "FA"}, headers=SID).json()
    assert again["kind"] == "ERROR" and "no fundamentals" in again["message"]
    other = client.post("/api/terminal/command", json={"command": "FA"},
                        headers={"X-Session-Id": "another-session-0001"}).json()
    assert "no active security" in other["message"]


def test_market_overview_and_news_endpoints(client: TestClient) -> None:
    (row,) = client.get("/api/market/overview").json()
    assert row["listing"] == "NSE:INFY" and row["last_close"] == 1512.5 and row["live"] is True
    assert row["news"]["articles"] == 1 and row["news"]["label"] == "bullish"
    missing = client.get("/api/market/overview", params={"symbols": ["NSE:TCS"]}).json()
    assert missing[0]["error"].startswith("WorkspaceError")
    news = client.get("/api/news", params={"symbol": "NSE:INFY"}).json()
    assert news[0]["title"] == "Infosys surges on record deal"
    assert len(client.get("/api/news", params={"tagged": ["NSE:TCS", "NSE:INFY"]}).json()) == 1
    assert client.get("/api/news", params={"tagged": ["NSE:TCS"]}).json() == []
    assert client.get("/api/news", params={"category": "evil"}).status_code == 422
    assert client.get("/api/news/sources").json()["sources"][0]["ok"] is True
    assert client.get("/api/market/status").json()["agent_enabled"] is False


def test_screener_endpoint_merges_live_flag_and_news(client: TestClient) -> None:
    from uphill.market.screener import Screener
    from uphill.market.universe import UniverseStock

    state: UpHillState = client.app.state.uphill  # type: ignore[attr-defined]
    assert client.get("/api/market/screener").json() == {"updated": None, "rows": []}
    universe = [UniverseStock(symbol="INFY", name="Infosys Ltd.", industry="Information Technology", isin="", cap="large")]
    state.screener = Screener(state.workspace, universe, clock=lambda: NOW)
    client.portal.call(state.screener.refresh)  # type: ignore[union-attr]
    body = client.get("/api/market/screener").json()
    (row,) = body["rows"]
    assert row["listing"] == "NSE:INFY" and row["cap"] == "large" and row["last_close"] == 1512.5
    assert row["live"] is True and row["news"]["articles"] == 1 and body["updated"] is not None


def test_candles_endpoint_aggregates_day_and_validates(client: TestClient) -> None:
    state: UpHillState = client.app.state.uphill  # type: ignore[attr-defined]
    assert client.get("/api/market/candles", params={"symbol": "NSE:INFY"}).status_code == 503

    class Quotes:
        calls: list[tuple[str, int]] = []

        async def intraday(self, symbol: str, minutes: int) -> list[tuple[int, float, float, float, float, float]] | None:
            self.calls.append((symbol, minutes))
            if symbol != "INFY":
                return None
            return [(100, 10.0, 12.0, 9.0, 11.0, 5.0), (1900, 11.0, 13.0, 10.5, 12.5, 7.0)]

    state.quotes = Quotes()  # type: ignore[assignment]
    five = client.get("/api/market/candles", params={"symbol": "nse:infy", "interval": "5"}).json()
    assert five["listing"] == "NSE:INFY" and len(five["candles"]) == 2
    day = client.get("/api/market/candles", params={"symbol": "NSE:INFY", "interval": "day"}).json()
    assert day["candles"] == [[100, 10.0, 13.0, 9.0, 12.5, 12.0]] and Quotes.calls[-1] == ("INFY", 30)
    assert client.get("/api/market/candles", params={"symbol": "NSE:TCS"}).status_code == 404
    assert client.get("/api/market/candles", params={"symbol": "NSE:INFY", "interval": "7"}).status_code == 422
    assert client.get("/api/market/candles", params={"symbol": "../etc"}).status_code == 422


def test_agent_disabled_is_reported(client: TestClient) -> None:
    r = client.post("/api/agent/ask", json={"question": "Should I buy INFY?"}, headers=SID)
    assert r.status_code == 503 and "QT_AGENT_API_KEY" in r.json()["detail"]
    hindi = client.post("/api/agent/ask", json={"question": "INFY?", "language": "hi"}, headers=SID)
    assert hindi.status_code == 503
    bad = client.post("/api/agent/ask", json={"question": "INFY?", "language": "fr"}, headers=SID)
    assert bad.status_code == 422


def test_stock_agent_answers_in_requested_language() -> None:
    from quantterm.agent.market_agent import LLMReply
    from uphill.agent.stock_agent import UpHillStockAgent

    prompts: list[str] = []
    replies = iter(["RELIANCE के लिए अभी कोई आंकड़ा नहीं है।", "No figures yet.",
                    *["शेयर दस प्रतिशत गिरा।"] * 3])

    class LLM:
        async def complete(self, messages: Sequence[dict[str, Any]], tools: Sequence[dict[str, Any]]) -> LLMReply:
            prompts.append(messages[0]["content"])
            return LLMReply(content=next(replies))

    news = NewsService({}, SymbolTagger({}), clock=lambda: NOW)
    agent = UpHillStockAgent(_Workspace(), LLM(), news)  # type: ignore[arg-type]
    hindi = asyncio.run(agent.ask("Is RELIANCE a buy?", "hi"))
    assert not hindi.fallback and hindi.text.startswith("RELIANCE के लिए")
    assert "answer in Hindi (Devanagari script)" in prompts[-1]
    asyncio.run(agent.ask("And now?", "en"))
    assert "answer in English" in prompts[-1] and "Hindi" not in prompts[-1]
    withheld = asyncio.run(agent.ask("कितना गिरा?", "hi"))  # spelled Hindi quantity fails grounding 3x
    assert withheld.fallback and withheld.text.startswith("मॉडल का उत्तर सत्यापित नहीं हो सका")


def test_market_websocket_forwards_validated_ticks(client: TestClient, broker: InMemoryBroker) -> None:
    with client.websocket_connect("/ws/market?symbols=NSE:INFY") as ws:
        message = json.dumps({"tick": {"symbol": "INFY", "price": 1513.0, "size": 7,
                                       "timestamp": "2026-10-26T04:00:00Z"}})

        async def publish() -> None:
            await broker.publish(tick_channel(Exchange.NSE, "INFY"), message)

        assert broker._subscribers  # subscribed before the handshake completed
        client.portal.call(publish)  # type: ignore[union-attr]
        assert ws.receive_json() == {"listing": "NSE:INFY", "price": 1513.0, "size": 7,
                                     "timestamp": "2026-10-26T04:00:00Z", "prev_close": None}
