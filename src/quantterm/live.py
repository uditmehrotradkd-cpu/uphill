"""Live market feeds (Upstox, Kite) pumped into the tick ingestion pipeline.

Used by both the API service and the terminal, so a single ``python -m quantterm.terminal.app`` is enough.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import AsyncIterator, Callable, Coroutine, Iterable
from datetime import timedelta
from typing import Any

import httpx

from quantterm.connectors.http import HttpFetcher
from quantterm.connectors.kite import KiteTickDecoder, KiteTickerClient, load_kite_instruments
from quantterm.connectors.upstox import UpstoxFeedClient, UpstoxFeedDecoder, load_upstox_instruments
from quantterm.db.repositories import (
    Database,
    OutlierTickRepository,
    PriceRepository,
    SecurityRepository,
    TickRepository,
    UnknownSecurityError,
)
from quantterm.hygiene.tick_filter import HygieneConfig
from quantterm.models import Exchange, OrderBookSnapshot, Tick
from quantterm.pubsub import Broker
from quantterm.settings import Settings
from quantterm.streaming import TickIngestionPipeline

log = logging.getLogger(__name__)

SYMBOL_PATTERN = r"^[A-Z0-9][A-Z0-9&.\-]{0,31}$"
_SYMBOL_RE = re.compile(SYMBOL_PATTERN)


def parse_listing(ref: str) -> tuple[Exchange, str]:
    """``"NSE:INFY"`` -> (Exchange.NSE, "INFY")."""
    exchange, sep, symbol = ref.partition(":")
    if not sep or not _SYMBOL_RE.fullmatch(symbol):
        raise ValueError(f"invalid listing {ref!r}; expected EXCHANGE:SYMBOL")
    return Exchange(exchange), symbol


def build_pipeline(
    settings: Settings, db: Database, broker: Broker, securities: SecurityRepository | None = None,
    prices: PriceRepository | None = None,
) -> TickIngestionPipeline:
    securities = securities or SecurityRepository(db)
    return TickIngestionPipeline(
        securities=securities,
        ticks=TickRepository(db),
        outliers=OutlierTickRepository(db),
        hygiene_config=HygieneConfig(window=settings.hygiene_window, mad_multiplier=settings.hygiene_mad_multiplier),
        prices=prices or PriceRepository(db, securities),
        broker=broker,
        candle_interval=timedelta(seconds=settings.candle_interval_seconds),
        flush_size=settings.tick_flush_size,
    )


def _by_exchange(refs: Iterable[str]) -> dict[Exchange, list[str]]:
    grouped: dict[Exchange, list[str]] = {}
    for ref in refs:
        exchange, symbol = parse_listing(ref)
        grouped.setdefault(exchange, []).append(symbol)
    return grouped


async def _pump(events: AsyncIterator[Tick | OrderBookSnapshot], pipeline: TickIngestionPipeline, source: str) -> None:
    async for event in events:
        try:
            if isinstance(event, OrderBookSnapshot):
                await pipeline.submit_book(event)
            else:
                await pipeline.submit(event, source)
        except UnknownSecurityError:
            log.warning("%s event for unregistered %s:%s dropped", source, event.exchange.value, event.symbol)


async def run_upstox_feed(settings: Settings, pipeline: TickIngestionPipeline) -> None:
    assert settings.upstox_analytics_token is not None
    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        http = HttpFetcher(client, rate_per_second=2)
        instruments: dict[str, tuple[Exchange, str]] = {}
        for exchange, symbols in _by_exchange(settings.upstox_symbols).items():
            instruments.update(await load_upstox_instruments(http, exchange, symbols, strict=False))
        feed = UpstoxFeedClient(
            settings.upstox_analytics_token, UpstoxFeedDecoder(instruments), http, base_url=settings.upstox_base_url
        )
        await _pump(feed.events(), pipeline, "UPSTOX")


async def run_kite_feed(settings: Settings, pipeline: TickIngestionPipeline) -> None:
    assert settings.kite_api_key is not None and settings.kite_access_token is not None
    instruments: dict[int, tuple[Exchange, str]] = {}
    async with httpx.AsyncClient(timeout=30) as client:
        http = HttpFetcher(client, rate_per_second=3)
        for exchange, symbols in _by_exchange(settings.kite_symbols).items():
            instruments.update(
                await load_kite_instruments(http, settings.kite_api_key, settings.kite_access_token, exchange, symbols)
            )
    ticker = KiteTickerClient(settings.kite_api_key, settings.kite_access_token, KiteTickDecoder(instruments))
    await _pump(ticker.events(), pipeline, "KITE")


def start_live_feeds(
    settings: Settings, pipeline: TickIngestionPipeline, on_stop: Callable[[str], None] | None = None
) -> list[asyncio.Task[None]]:
    """Starts every configured feed; ``on_stop`` receives a message if a feed dies (e.g. token rejected)."""
    runners: list[tuple[str, Callable[[Settings, TickIngestionPipeline], Coroutine[Any, Any, None]]]] = []
    if settings.upstox_symbols:
        if settings.upstox_analytics_token:
            runners.append(("Upstox", run_upstox_feed))
        else:
            log.warning("QT_UPSTOX_SYMBOLS set but QT_UPSTOX_ANALYTICS_TOKEN missing; Upstox feed off")
    if settings.kite_symbols:
        if settings.kite_api_key and settings.kite_access_token:
            runners.append(("Kite", run_kite_feed))
        else:
            log.warning("QT_KITE_SYMBOLS set but QT_KITE_API_KEY / QT_KITE_ACCESS_TOKEN missing; Kite feed off")

    def reporter(name: str) -> Callable[[asyncio.Task[None]], None]:
        def done(task: asyncio.Task[None]) -> None:
            if task.cancelled() or (exc := task.exception()) is None:
                return
            message = f"{name} live feed stopped: {type(exc).__name__}: {exc}"
            log.error(message)
            if on_stop is not None:
                on_stop(message)
        return done

    tasks: list[asyncio.Task[None]] = []
    for name, runner in runners:
        task = asyncio.create_task(runner(settings, pipeline))
        task.add_done_callback(reporter(name))
        tasks.append(task)
    return tasks
