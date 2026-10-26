"""UpHill runtime settings (prefix ``UPHILL_``); market/engine settings stay in ``quantterm.settings`` (``QT_``)."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_RSS_FEEDS: dict[str, str] = {
    "ET Markets|markets": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "Livemint Markets|markets": "https://www.livemint.com/rss/markets",
    "Moneycontrol|markets": "https://www.moneycontrol.com/rss/latestnews.xml",
    "BBC Business|business": "https://feeds.bbci.co.uk/news/business/rss.xml",
    "BBC World|world": "https://feeds.bbci.co.uk/news/world/rss.xml",
    "CNBC World|world": "https://www.cnbc.com/id/100727362/device/rss/rss.html",
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

    news_refresh_seconds: int = Field(default=300, ge=60)
    news_max_items: int = Field(default=600, ge=50)
    # "Name|category" -> RSS/Atom URL
    news_rss_feeds: dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_RSS_FEEDS))
    news_gdelt_query: str = (
        '(economy OR "stock market" OR "central bank" OR inflation OR "interest rates" OR tariffs OR oil '
        'OR sanctions OR war OR election) sourcelang:english'
    )
    news_gdelt_records: int = Field(default=75, ge=10, le=250)
    # EXCHANGE:SYMBOL -> company names used to tag headlines with listings
    news_aliases: dict[str, list[str]] = Field(default_factory=lambda: dict(DEFAULT_ALIASES))
