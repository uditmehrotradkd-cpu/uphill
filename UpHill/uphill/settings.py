"""UpHill runtime settings (prefix ``UPHILL_``); market/engine settings stay in ``quantterm.settings`` (``QT_``)."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_GN = "https://news.google.com/rss/search?q="
_GN_IN = "&hl=en-IN&gl=IN&ceid=IN:en"
_GN_HI = "&hl=hi&gl=IN&ceid=IN:hi"
_GN_US = "&hl=en-US&gl=US&ceid=US:en"

# "Publisher|category" -> RSS URL. Publishers without a working RSS feed (CNN, Aaj Tak business) use a Google
# News query scoped to their site.
DEFAULT_RSS_FEEDS: dict[str, str] = {
    # Economic Times
    "ET Markets|markets": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "ET Stocks|stock": "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
    "ET IPO|stock": "https://economictimes.indiatimes.com/markets/ipos/fpos/rssfeeds/14655708.cms",
    "ET Commodities|markets": "https://economictimes.indiatimes.com/markets/commodities/rssfeeds/1808152121.cms",
    "ET Economy|business": "https://economictimes.indiatimes.com/news/economy/rssfeeds/1373380680.cms",
    "ET Industry|business": "https://economictimes.indiatimes.com/industry/rssfeeds/13352306.cms",
    "ET Tech|business": "https://economictimes.indiatimes.com/tech/rssfeeds/13357270.cms",
    "ET Top Stories|india": "https://economictimes.indiatimes.com/rssfeedstopstories.cms",
    # Hindustan Times / Livemint (HT Media)
    "Hindustan Times Business|business": "https://www.hindustantimes.com/feeds/rss/business/rssfeed.xml",
    "Hindustan Times Markets|markets": f"{_GN}site:hindustantimes.com/business+when:1d{_GN_IN}",
    "Hindustan Times India|india": "https://www.hindustantimes.com/feeds/rss/india-news/rssfeed.xml",
    "Hindustan Times World|world": "https://www.hindustantimes.com/feeds/rss/world-news/rssfeed.xml",
    "Livemint Markets|markets": "https://www.livemint.com/rss/markets",
    "Livemint Companies|stock": "https://www.livemint.com/rss/companies",
    "Livemint Economy|business": "https://www.livemint.com/rss/economy",
    # Aaj Tak (Hindi)
    "Aaj Tak Share Bazaar|markets":
        f"{_GN}site:aajtak.in+(%E0%A4%B6%E0%A5%87%E0%A4%AF%E0%A4%B0+OR+%E0%A4%B8%E0%A5%87%E0%A4%82%E0%A4%B8%E0%A5%87%E0%A4%95%E0%A5%8D%E0%A4%B8"
        f"+OR+%E0%A4%A8%E0%A4%BF%E0%A4%AB%E0%A5%8D%E0%A4%9F%E0%A5%80)+when:1d{_GN_HI}",
    "Aaj Tak Business|business": f"{_GN}site:aajtak.in/business+when:1d{_GN_HI}",
    # Other Indian outlets
    "Business Standard Markets|markets": "https://www.business-standard.com/rss/markets-106.rss",
    "Business Standard Companies|stock": "https://www.business-standard.com/rss/companies-101.rss",
    "BusinessLine Markets|markets": "https://www.thehindubusinessline.com/markets/feeder/default.rss",
    "The Hindu Business|business": "https://www.thehindu.com/business/feeder/default.rss",
    "The Hindu Markets|markets": "https://www.thehindu.com/business/markets/feeder/default.rss",
    "Times of India Business|business": "https://timesofindia.indiatimes.com/rssfeeds/1898055.cms",
    "Times of India|india": "https://timesofindia.indiatimes.com/rssfeeds/-2128936835.cms",
    "Indian Express Business|business": "https://indianexpress.com/section/business/feed/",
    "Business Today|business": "https://www.businesstoday.in/rssfeeds/?id=home",
    "CNBC-TV18 Market|markets": "https://www.cnbctv18.com/commonfeeds/v1/cne/rss/market.xml",
    "CNBC-TV18 Business|business": "https://www.cnbctv18.com/commonfeeds/v1/cne/rss/business.xml",
    "News18 Business|business": "https://www.news18.com/rss/business.xml",
    "Firstpost Business|business": "https://www.firstpost.com/commonfeeds/v1/mfp/rss/business.xml",
    "Google News India Markets|markets": f"{_GN}Sensex+OR+Nifty+OR+%22stock+market%22+when:1d{_GN_IN}",
    "Google News Business|business":
        "https://news.google.com/rss/headlines/section/topic/BUSINESS?hl=en-IN&gl=IN&ceid=IN:en",
    # BBC
    "BBC Business|business": "https://feeds.bbci.co.uk/news/business/rss.xml",
    "BBC Technology|business": "https://feeds.bbci.co.uk/news/technology/rss.xml",
    "BBC India|india": "https://feeds.bbci.co.uk/news/world/asia/india/rss.xml",
    "BBC Asia|world": "https://feeds.bbci.co.uk/news/world/asia/rss.xml",
    "BBC World|world": "https://feeds.bbci.co.uk/news/world/rss.xml",
    # CNN (official RSS feeds are frozen since 2023)
    "CNN|world": f"{_GN}site:cnn.com+(business+OR+markets+OR+economy+OR+world)+when:1d{_GN_US}",
    # Global business / markets
    "WSJ Markets|markets": "https://feeds.content.dowjones.io/public/rss/RSSMarketsMain",
    "WSJ World|world": "https://feeds.content.dowjones.io/public/rss/RSSWorldNews",
    "MarketWatch|business": "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "CNBC World|world": "https://www.cnbc.com/id/100727362/device/rss/rss.html",
    "CNBC Economy|business": "https://www.cnbc.com/id/20910258/device/rss/rss.html",
    "NYT Business|business": "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml",
    "NYT World|world": "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
    "Guardian Business|business": "https://www.theguardian.com/uk/business/rss",
    "Guardian World|world": "https://www.theguardian.com/world/rss",
    "Fox Business|markets": "https://moxie.foxbusiness.com/google-publisher/markets.xml",
    "Sky News Business|business": "https://feeds.skynews.com/feeds/rss/business.xml",
    "Investing.com|markets": "https://www.investing.com/rss/news.rss",
    "Nasdaq|markets": "https://www.nasdaq.com/feed/rssoutbound?category=Markets",
    "Al Jazeera|world": "https://www.aljazeera.com/xml/rss/all.xml",
    "Google News World|world": "https://news.google.com/rss/headlines/section/topic/WORLD?hl=en-IN&gl=IN&ceid=IN:en",
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
    # Load each trading day's NSE bhavcopy into raw_market_bars (catches up missed sessions on start).
    eod_update: bool = True
    eod_check_seconds: int = Field(default=1800, ge=300)
    # Poll Upstox last-traded prices for every universe stock during NSE hours (needs QT_UPSTOX_ANALYTICS_TOKEN).
    live_quotes: bool = True
    live_quote_seconds: int = Field(default=15, ge=5)
    max_sessions: int = Field(default=200, ge=1)
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173", "http://127.0.0.1:5173"])

    news_refresh_seconds: int = Field(default=120, ge=60)
    news_max_items: int = Field(default=6000, ge=50)
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
