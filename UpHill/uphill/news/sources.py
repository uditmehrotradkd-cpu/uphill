"""News sources. Every item's links are restricted to http(s) so feeds cannot inject ``javascript:`` URLs."""

from __future__ import annotations

import html
import json
import logging
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import urlsplit
from xml.etree.ElementTree import Element

from defusedxml import ElementTree
from pydantic import SecretStr

from quantterm.connectors.base import SourceError
from quantterm.connectors.http import HttpFetcher
from uphill.news.models import Category, NewsItem, item_id

log = logging.getLogger(__name__)

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
GOOGLE_NEWS_SEARCH = "https://news.google.com/rss/search"
BROWSER_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) UpHill/0.1"
_TAG = re.compile(r"<[^>]+>")
_SPACE = re.compile(r"\s+")
_ATOM = "{http://www.w3.org/2005/Atom}"
_MEDIA = "{http://search.yahoo.com/mrss/}"


def safe_url(url: str | None) -> str | None:
    if not url:
        return None
    url = url.strip()
    parts = urlsplit(url)
    return url if parts.scheme in ("http", "https") and parts.netloc else None


def clean_text(text: str | None, limit: int = 400) -> str:
    plain = _SPACE.sub(" ", html.unescape(_TAG.sub(" ", text or ""))).strip()
    return plain if len(plain) <= limit else plain[: limit - 1].rsplit(" ", 1)[0] + "…"


def _make(title: str, url: str | None, *, summary: str = "", image: str | None = None, source: str,
          publisher: str, category: Category, published: datetime, symbols: list[str] | None = None) -> NewsItem | None:
    link = safe_url(url)
    title = clean_text(title, 300)
    if link is None or not title:
        return None
    return NewsItem(id=item_id(link), title=title, summary=clean_text(summary), url=link, image=safe_url(image),
                    source=source, publisher=publisher, category=category, published=published,
                    symbols=symbols or [])


# ------------------------------------------------------------------------------------------- Upstox
def parse_upstox_news(payload: Mapping[str, Any], listings: Mapping[str, str]) -> list[NewsItem]:
    """``listings`` maps Upstox instrument keys (``NSE_EQ|ISIN``) to ``EXCHANGE:SYMBOL``."""
    if payload.get("status") != "success":
        raise SourceError(f"Upstox news error: {payload.get('errors') or payload.get('message') or 'unknown'}")
    items: dict[str, NewsItem] = {}
    for key, articles in (payload.get("data") or {}).items():
        listing = listings.get(key)
        for a in articles or []:
            published = datetime.fromtimestamp(int(a.get("published_time") or 0) / 1000, tz=UTC)
            item = _make(a.get("heading", ""), a.get("article_link"), summary=a.get("summary", ""),
                         image=a.get("thumbnail"), source="Upstox", publisher="Upstox News", category="stock",
                         published=published, symbols=[listing] if listing else [])
            if item is None:
                continue
            prev = items.get(item.id)
            if prev is not None:  # one article can cover several requested instruments
                item = item.model_copy(update={"symbols": sorted({*prev.symbols, *item.symbols})})
            items[item.id] = item
    return list(items.values())


async def fetch_upstox_news(http: HttpFetcher, token: SecretStr, listings: Mapping[str, str],
                            base_url: str = "https://api.upstox.com") -> list[NewsItem]:
    keys = list(listings)
    out: list[NewsItem] = []
    for i in range(0, len(keys), 30):  # API limit: 30 instrument keys per request
        payload = await http.get_json(
            f"{base_url}/v2/news",
            params={"category": "instrument_keys", "instrument_keys": ",".join(keys[i : i + 30]), "page_size": 50},
            headers={"Accept": "application/json", "Authorization": f"Bearer {token.get_secret_value()}"},
        )
        out += parse_upstox_news(payload, listings)
    return out


# -------------------------------------------------------------------------------------------- GDELT
def parse_gdelt(payload: Mapping[str, Any]) -> list[NewsItem]:
    items = []
    for a in payload.get("articles") or []:
        try:
            published = datetime.strptime(a.get("seendate", ""), "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
        except ValueError:
            continue
        domain = a.get("domain") or "web"
        item = _make(a.get("title", ""), a.get("url"), image=a.get("socialimage"), source="GDELT",
                     publisher=domain, category="world", published=published)
        if item is not None:
            items.append(item)
    return items


async def fetch_gdelt(http: HttpFetcher, query: str, records: int) -> list[NewsItem]:
    response = await http.get(GDELT_URL, params={
        "query": query, "mode": "ArtList", "format": "json", "maxrecords": records, "timespan": "24h",
        "sort": "DateDesc",
    })
    try:
        payload = json.loads(response.text)
    except ValueError as exc:  # GDELT answers plain text for malformed queries / throttling
        raise SourceError(f"GDELT: {clean_text(response.text, 120)}") from exc
    return parse_gdelt(payload)


# ---------------------------------------------------------------------------------------------- RSS
def _text(el: Element | None) -> str:
    return (el.text or "").strip() if el is not None else ""


def _rss_date(raw: str) -> datetime | None:
    if not raw:
        return None
    try:
        dt = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def _image(entry: Element) -> str | None:
    for tag in (f"{_MEDIA}content", f"{_MEDIA}thumbnail", "enclosure"):
        el = entry.find(tag)
        if el is not None and el.get("url") and (tag != "enclosure" or (el.get("type") or "").startswith("image")):
            return el.get("url")
    return None


def parse_rss(xml: bytes, publisher: str, category: Category, now: datetime) -> list[NewsItem]:
    try:
        root = ElementTree.fromstring(xml)
    except Exception as exc:  # defusedxml raises several parse/security error types
        raise SourceError(f"{publisher}: unreadable feed ({type(exc).__name__})") from exc
    items: list[NewsItem] = []
    for entry in root.iter("item"):
        item = _make(_text(entry.find("title")), _text(entry.find("link")), summary=_text(entry.find("description")),
                     image=_image(entry), source="RSS", publisher=publisher, category=category,
                     published=_rss_date(_text(entry.find("pubDate"))) or now)
        if item is not None:
            items.append(item)
    for entry in root.iter(f"{_ATOM}entry"):
        link = entry.find(f"{_ATOM}link")
        raw_date = _text(entry.find(f"{_ATOM}published")) or _text(entry.find(f"{_ATOM}updated"))
        item = _make(_text(entry.find(f"{_ATOM}title")), link.get("href") if link is not None else None,
                     summary=_text(entry.find(f"{_ATOM}summary")), image=_image(entry), source="RSS",
                     publisher=publisher, category=category, published=_rss_date(raw_date) or now)
        if item is not None:
            items.append(item)
    return items


async def fetch_rss(http: HttpFetcher, url: str, publisher: str, category: Category) -> list[NewsItem]:
    response = await http.get(url, headers={"User-Agent": BROWSER_UA, "Accept": "application/rss+xml, */*"})
    items = parse_rss(response.content, publisher, category, datetime.now(UTC))
    return [google_publisher(i) for i in items] if urlsplit(url).hostname == "news.google.com" else items


# -------------------------------------------------------------------------------------- Google News
def google_publisher(item: NewsItem) -> NewsItem:
    """Google News titles end with ' - Publisher'; move it to ``publisher``."""
    title, sep, publisher = item.title.rpartition(" - ")
    if not sep or not title or not publisher or len(publisher) > 60:
        return item
    return item.model_copy(update={"title": title, "publisher": publisher})


async def search_google_news(http: HttpFetcher, query: str, *, india: bool = True, days: int = 3,
                             category: Category = "stock") -> list[NewsItem]:
    locale = {"hl": "en-IN", "gl": "IN", "ceid": "IN:en"} if india else {"hl": "en-US", "gl": "US", "ceid": "US:en"}
    response = await http.get(GOOGLE_NEWS_SEARCH, params={"q": f"{query} when:{days}d", **locale},
                              headers={"User-Agent": BROWSER_UA, "Accept": "application/rss+xml, */*"})
    items = parse_rss(response.content, "Google News", category, datetime.now(UTC))
    return [google_publisher(i).model_copy(update={"source": "Google News"}) for i in items]
