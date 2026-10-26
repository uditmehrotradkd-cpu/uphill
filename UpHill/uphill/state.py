"""Shared application state and per-browser sessions (each tab gets its own terminal context and agent memory)."""

from __future__ import annotations

import asyncio
import re
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import Depends, Header, HTTPException, Request, status

from quantterm.agent.market_agent import MarketAgent
from quantterm.models import Exchange
from quantterm.pubsub import Broker
from quantterm.settings import Settings
from quantterm.terminal.command_router import CommandRouter, WorkspaceLike
from uphill.market.eod import EodUpdater
from uphill.market.quotes import LiveQuotePoller
from uphill.market.screener import Screener
from uphill.news.service import NewsService
from uphill.settings import UpHillSettings

SESSION_ID = re.compile(r"^[A-Za-z0-9-]{16,64}$")


@dataclass
class Session:
    router: CommandRouter
    agent: MarketAgent | None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class SessionStore:
    """Bounded LRU so unauthenticated clients cannot grow server memory without limit."""

    def __init__(self, factory: Callable[[], Session], max_sessions: int) -> None:
        self._factory = factory
        self._max = max_sessions
        self._sessions: OrderedDict[str, Session] = OrderedDict()

    def get(self, session_id: str) -> Session:
        session = self._sessions.get(session_id)
        if session is None:
            session = self._sessions[session_id] = self._factory()
            while len(self._sessions) > self._max:
                self._sessions.popitem(last=False)
        else:
            self._sessions.move_to_end(session_id)
        return session

    def __len__(self) -> int:
        return len(self._sessions)


@dataclass
class UpHillState:
    settings: Settings
    uphill: UpHillSettings
    workspace: WorkspaceLike
    broker: Broker
    news: NewsService
    sessions: SessionStore
    watchlist: list[tuple[Exchange, str]]
    agent_enabled: bool
    live_feeds: int = 0
    clock: Callable[[], datetime] = lambda: datetime.now(UTC)
    snapshots: dict[str, tuple[float, dict[str, Any]]] = field(default_factory=dict)  # listing -> (monotonic, row)
    screener: Screener | None = None
    eod: EodUpdater | None = None
    quotes: LiveQuotePoller | None = None


def get_state(request: Request) -> UpHillState:
    state: UpHillState = request.app.state.uphill
    return state


def get_session(request: Request, x_session_id: Annotated[str, Header()]) -> Session:
    if not SESSION_ID.fullmatch(x_session_id):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "invalid X-Session-Id")
    return get_state(request).sessions.get(x_session_id)


State = Annotated[UpHillState, Depends(get_state)]
CurrentSession = Annotated[Session, Depends(get_session)]
