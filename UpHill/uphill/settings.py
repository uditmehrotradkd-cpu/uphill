"""UpHill runtime settings (prefix ``UPHILL_``); market/engine settings stay in ``quantterm.settings`` (``QT_``)."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_RSS_FEEDS: dict[str, str] = {
    "ET Markets|markets": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "ET Stocks|stock": "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
    "ET Economy|business": "https://economictimes.indiatimes.com/news/economy/rssfeeds/1373380680.cms",
    "Livemint Markets|markets": "https://www.livemint.com/rss/markets",
    "Livemint Companies|stock": "https://www.livemint.com/rss/companies",
    "Livemint Economy|business": "https://www.livemint.com/rss/economy",
    "Business Standard Markets|markets": "https://www.business-standard.com/rss/markets-106.rss",
    "Business Standard Companies|stock": "https://www.business-standard.com/rss/companies-101.rss",
    "BusinessLine Markets|markets": "https://www.thehindubusinessline.com/markets/feeder/default.rss",
    "Google News India Markets|markets":
        "https://news.google.com/rss/search?q=Sensex+OR+Nifty+OR+%22stock+market%22+when:1d&hl=en-IN&gl=IN&ceid=IN:en",
    "Google News Business|business":
        "https://news.google.com/rss/headlines/section/topic/BUSINESS?hl=en-IN&gl=IN&ceid=IN:en",
    "Google News World|world": "https://news.google.com/rss/headlines/section/topic/WORLD?hl=en-IN&gl=IN&ceid=IN:en",
    "BBC Business|business": "https://feeds.bbci.co.uk/news/business/rss.xml",
    "BBC World|world": "https://feeds.bbci.co.uk/news/world/rss.xml",
    "CNBC World|world": "https://www.cnbc.com/id/100727362/device/rss/rss.html",
    "CNBC Economy|business": "https://www.cnbc.com/id/20910258/device/rss/rss.html",
    "MarketWatch|business": "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "Al Jazeera|world": "https://www.aljazeera.com/xml/rss/all.xml",
}

DEFAULT_ALIASES: dict[str, list[str]] = {
    "NSE:RELIANCE": ["Reliance Industries", "RIL", "Mukesh Ambani", "Jio"],
    "NSE:TCS": ["Tata Consultancy", "TCS"],
    "NSE:INFY": ["Infosys"],
    "NSE:HDFCBANK": ["HDFC Bank"],
    "NSE:ICICIBANK": ["ICICI Bank"],
    "NASDAQ:AAPL": ["Apple"],
    "NASDAQ:MSFT": ["Microsoft"],
    "NASDAQ:NVDA": ["Nvidia"],
    "NASDAQ:AMZN": ["Amazon"],
    "NASDAQ:GOOGL": ["Alphabet", "Google"],
    "NASDAQ:META": ["Meta Platforms", "Facebook"],
    "NASDAQ:TSLA": ["Tesla"],
    "NYSE:JPM": ["JPMorgan"],
    "NYSE:XOM": ["Exxon"],
}


class UpHillSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="UPHILL_", env_file=".env", extra="ignore")

    host: str = "127.0.0.1"
    port: int = Field(default=8800, ge=1, le=65535)
    # Run the Upstox/Kite feeds in this process; set false if the QuantTerm API or TUI already runs them.
    live_feed: bool = True
    # Stream every large/mid/small-cap stock in the universe (Upstox full mode allows up to 2000 instruments).
    live_universe: bool = True
    screener_refresh_seconds: int = Field(default=900, ge=60)
    max_sessions: int = Field(default=200, ge=1)
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173", "http://127.0.0.1:5173"])

    news_refresh_seconds: int = Field(default=120, ge=60)
    news_max_items: int = Field(default=3000, ge=50)
    # On-demand Google News search per listing (agent questions, News page symbol filter); cached this long.
    news_search_ttl_seconds: int = Field(default=300, ge=60)
    # "Name|category" -> RSS/Atom URL
    news_rss_feeds: dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_RSS_FEEDS))
    news_gdelt_query: str = (
        '(economy OR "stock market" OR "central bank" OR inflation OR "interest rates" OR tariffs OR oil '
        'OR sanctions OR war OR election) sourcelang:english'
    )
    news_gdelt_records: int = Field(default=75, ge=10, le=250)
    # EXCHANGE:SYMBOL -> company names used to tag headlines with listings
    news_aliases: dict[str, list[str]] = Field(default_factory=lambda: dict(DEFAULT_ALIASES))
