from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Category = Literal["stock", "markets", "business", "india", "world"]
SentimentLabel = Literal["bullish", "bearish", "neutral"]


class NewsItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    title: str
    summary: str = ""
    url: str
    image: str | None = None
    source: str
    publisher: str
    category: Category
    published: datetime
    symbols: list[str] = Field(default_factory=list)  # EXCHANGE:SYMBOL
    sentiment: float = 0.0  # -1 bearish .. +1 bullish (lexicon heuristic)
    sentiment_label: SentimentLabel = "neutral"


class SymbolPulse(BaseModel):
    model_config = ConfigDict(frozen=True)

    listing: str
    articles: int
    bullish: int
    bearish: int
    score: float  # mean sentiment of tagged articles
    label: SentimentLabel
    latest: datetime | None


class SourceStatus(BaseModel):
    name: str
    ok: bool
    items: int = 0
    error: str | None = None
    checked: datetime | None = None


def item_id(url: str) -> str:
    return hashlib.sha1(url.encode(), usedforsecurity=False).hexdigest()[:16]
