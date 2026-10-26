from __future__ import annotations

import re
from typing import Annotated

from fastapi import APIRouter, Query

from uphill.news.models import NewsItem, SymbolPulse
from uphill.state import State

router = APIRouter(prefix="/api/news", tags=["news"])

LISTING = r"^[A-Z]{3,6}:[A-Z0-9&.\-]{1,32}$"
_LISTING_RE = re.compile(LISTING)


@router.get("", response_model=list[NewsItem])
async def news(
    state: State,
    category: Annotated[str | None, Query(pattern=r"^(stock|markets|business|india|world)$")] = None,
    symbol: Annotated[str | None, Query(pattern=LISTING)] = None,
    q: Annotated[str | None, Query(max_length=80)] = None,
    sentiment: Annotated[str | None, Query(pattern=r"^(bullish|bearish|neutral)$")] = None,
    tagged: Annotated[list[str] | None, Query(max_length=50, description="any of these listings")] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 60,
) -> list[NewsItem]:
    if symbol:
        await state.news.search(symbol)
    return state.news.latest(category=category, symbol=symbol, query=q, sentiment_label=sentiment,
                             any_of=[s for s in tagged or [] if _LISTING_RE.fullmatch(s)], limit=limit)


@router.get("/pulse", response_model=dict[str, SymbolPulse])
async def pulse(
    state: State,
    symbols: Annotated[list[str], Query(min_length=1, max_length=50)],
    hours: Annotated[int, Query(ge=1, le=168)] = 72,
) -> dict[str, SymbolPulse]:
    return state.news.pulse([s for s in symbols if _LISTING_RE.fullmatch(s)], hours)


@router.get("/sources")
async def sources(state: State) -> dict[str, object]:
    return {"updated": state.news.updated, "sources": [s.model_dump(mode="json") for s in state.news.status.values()]}
