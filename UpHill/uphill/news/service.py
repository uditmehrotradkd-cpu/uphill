"""In-memory news desk: periodic refresh from every source, dedupe, listing tags, sentiment, per-listing pulse."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import get_args

from uphill.news.analysis import SymbolTagger, label_for, sentiment
from uphill.news.models import Category, NewsItem, SourceStatus, SymbolPulse

log = logging.getLogger(__name__)

Fetcher = Callable[[], Awaitable[list[NewsItem]]]
Searcher = Callable[[str], Awaitable[list[NewsItem]]]  # listing -> fresh headlines about it
CATEGORIES: tuple[str, ...] = get_args(Category)


class NewsService:
    def __init__(self, sources: Mapping[str, Fetcher], tagger: SymbolTagger, *, max_items: int = 600,
                 searcher: Searcher | None = None, search_ttl: timedelta = timedelta(minutes=5),
                 clock: Callable[[], datetime] = lambda: datetime.now(UTC)) -> None:
        self._sources = dict(sources)
        self._tagger = tagger
        self._max = max_items
        self._searcher = searcher
        self._search_ttl = search_ttl
        self._searched: dict[str, datetime] = {}
        self._clock = clock
        self._items: dict[str, NewsItem] = {}
        self.status: dict[str, SourceStatus] = {name: SourceStatus(name=name, ok=False) for name in sources}
        self.updated: datetime | None = None
        self._lock = asyncio.Lock()

    def _enrich(self, item: NewsItem) -> NewsItem:
        text = f"{item.title}. {item.summary}"
        score, label = sentiment(text)
        symbols = sorted({*item.symbols, *self._tagger.tag(text)})
        return item.model_copy(update={"sentiment": round(score, 3), "sentiment_label": label, "symbols": symbols})

    async def _run_source(self, name: str, fetch: Fetcher) -> list[NewsItem]:
        try:
            items = await fetch()
        except Exception as exc:  # one failing feed must not blank the desk
            log.warning("news source %s failed: %s", name, exc)
            self.status[name] = SourceStatus(name=name, ok=False, error=f"{type(exc).__name__}: {exc}"[:200],
                                             checked=self._clock())
            return []
        self.status[name] = SourceStatus(name=name, ok=True, items=len(items), checked=self._clock())
        return items

    async def refresh(self) -> int:
        """Returns the number of new articles."""
        async with self._lock:
            batches = await asyncio.gather(*(self._run_source(n, f) for n, f in self._sources.items()))
            added = await self._merge([item for batch in batches for item in batch])
            self.updated = self._clock()
            return added

    async def search(self, listing: str) -> int:
        """Pull fresh headlines for one listing into the desk (at most once per TTL); returns new articles."""
        now = self._clock()
        last = self._searched.get(listing)
        if self._searcher is None or (last is not None and now - last < self._search_ttl):
            return 0
        if len(self._searched) > 5000:
            self._searched.clear()
        self._searched[listing] = now
        try:
            found = await self._searcher(listing)
        except Exception as exc:  # search is best-effort; the periodic desk still answers
            log.warning("news search for %s failed: %s", listing, exc)
            return 0
        async with self._lock:
            return await self._merge([i.model_copy(update={"symbols": sorted({*i.symbols, listing})}) for i in found])

    async def _merge(self, raw_items: list[NewsItem]) -> int:
        fresh: dict[str, NewsItem] = {}
        for raw in raw_items:
            prev = self._items.get(raw.id)
            if prev is None:
                fresh.setdefault(raw.id, raw)
            elif not set(raw.symbols) <= set(prev.symbols):  # known article: only add newly attributed listings
                self._items[raw.id] = prev.model_copy(update={"symbols": sorted({*prev.symbols, *raw.symbols})})
        # Sentiment + tagging is CPU work; keep it off the event loop so health checks stay responsive.
        enriched = await asyncio.to_thread(lambda: [self._enrich(i) for i in fresh.values()])
        self._items.update((i.id, i) for i in enriched)
        if len(self._items) > self._max:
            keep = sorted(self._items.values(), key=lambda i: i.published, reverse=True)[: self._max]
            self._items = {i.id: i for i in keep}
        return len(enriched)

    async def run(self, interval_seconds: float) -> None:
        while True:
            try:
                added = await self.refresh()
                log.info("news refreshed: %d new, %d total", added, len(self._items))
            except Exception:
                log.exception("news refresh failed")
            await asyncio.sleep(interval_seconds)

    def latest(self, *, category: str | None = None, symbol: str | None = None, query: str | None = None,
               sentiment_label: str | None = None, any_of: Sequence[str] = (), limit: int = 50) -> list[NewsItem]:
        q = query.lower() if query else None
        wanted = set(any_of)
        out = []
        for item in sorted(self._items.values(), key=lambda i: i.published, reverse=True):
            if category and item.category != category:
                continue
            if symbol and symbol not in item.symbols:
                continue
            if wanted and wanted.isdisjoint(item.symbols):
                continue
            if sentiment_label and item.sentiment_label != sentiment_label:
                continue
            if q and q not in item.title.lower() and q not in item.summary.lower():
                continue
            out.append(item)
            if len(out) >= limit:
                break
        return out

    def pulse(self, listings: Sequence[str], hours: int = 72) -> dict[str, SymbolPulse]:
        since = self._clock() - timedelta(hours=hours)
        result = {}
        for listing in listings:
            tagged = [i for i in self._items.values() if listing in i.symbols and i.published >= since]
            score = sum(i.sentiment for i in tagged) / len(tagged) if tagged else 0.0
            result[listing] = SymbolPulse(
                listing=listing, articles=len(tagged),
                bullish=sum(i.sentiment_label == "bullish" for i in tagged),
                bearish=sum(i.sentiment_label == "bearish" for i in tagged),
                score=round(score, 3), label=label_for(score) if tagged else "neutral",
                latest=max((i.published for i in tagged), default=None),
            )
        return result
