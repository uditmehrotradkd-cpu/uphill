"""UpHill web server: one process for live market, Bloomberg terminal, stock agent and news.

Run from the repository root (where ``.env`` lives):  ``python -m uphill``  ->  http://127.0.0.1:8800
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from functools import partial
from pathlib import Path
from typing import cast
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from redis.asyncio import Redis

from quantterm.agent.market_agent import AgentContext, OpenAICompatibleClient
from quantterm.connectors.http import HttpFetcher
from quantterm.connectors.upstox import load_upstox_instruments
from quantterm.db.pool import create_pool
from quantterm.live import build_pipeline, parse_listing, start_live_feeds
from quantterm.models import Exchange, Market
from quantterm.pubsub import RedisBroker
from quantterm.settings import Settings
from quantterm.terminal.command_router import CommandRouter
from quantterm.workspace import Workspace
from uphill.agent import routes as agent_routes
from uphill.agent.stock_agent import UpHillStockAgent
from uphill.market import routes as market_routes
from uphill.market.screener import Screener
from uphill.market.universe import load_universe
from uphill.market.universe import news_aliases as universe_aliases
from uphill.news import routes as news_routes
from uphill.news.analysis import SymbolTagger
from uphill.news.models import Category, NewsItem
from uphill.news.service import CATEGORIES, Fetcher, NewsService
from uphill.news.sources import fetch_gdelt, fetch_rss, fetch_upstox_news
from uphill.settings import UpHillSettings
from uphill.state import Session, SessionStore, UpHillState
from uphill.terminal import routes as terminal_routes

log = logging.getLogger(__name__)

WEB_DIST = Path(__file__).resolve().parents[1] / "web" / "dist"


def agent_enabled(settings: Settings) -> bool:
    host = urlsplit(settings.agent_base_url).hostname or ""
    return settings.agent_api_key is not None or host in ("localhost", "127.0.0.1")


def news_sources(settings: Settings, uphill: UpHillSettings, client: httpx.AsyncClient,
                 listings: list[tuple[Exchange, str]]) -> dict[str, Fetcher]:
    sources: dict[str, Fetcher] = {}
    rss_http = HttpFetcher(client, rate_per_second=4, max_retries=1)
    for key, url in uphill.news_rss_feeds.items():
        name, _, raw_category = key.partition("|")
        category = cast(Category, raw_category) if raw_category in CATEGORIES else "markets"
        sources[name] = partial(fetch_rss, rss_http, url, name, category)
    gdelt_http = HttpFetcher(client, rate_per_second=0.2, max_retries=1)  # GDELT: one request per 5 s
    sources["GDELT World"] = partial(fetch_gdelt, gdelt_http, uphill.news_gdelt_query, uphill.news_gdelt_records)

    token = settings.upstox_analytics_token
    indian = [(e, s) for e, s in listings if e.market is Market.IN]
    if token is not None and indian:
        upstox_http = HttpFetcher(client, rate_per_second=2)
        keys: dict[str, str] = {}

        async def upstox() -> list[NewsItem]:
            if not keys:
                by_exchange: dict[Exchange, list[str]] = {}
                for e, s in indian:
                    by_exchange.setdefault(e, []).append(s)
                for e, symbols in by_exchange.items():
                    found = await load_upstox_instruments(upstox_http, e, symbols, strict=False)
                    keys.update({k: f"{ex.value}:{sym}" for k, (ex, sym) in found.items()})
            return await fetch_upstox_news(upstox_http, token, keys, settings.upstox_base_url)

        sources["Upstox Stock News"] = upstox
    return sources


@contextlib.asynccontextmanager
async def _runtime(app: FastAPI, settings: Settings, uphill: UpHillSettings) -> AsyncIterator[None]:
    async with contextlib.AsyncExitStack() as stack:
        pool = await create_pool(settings)
        stack.push_async_callback(pool.close)
        redis: Redis = Redis.from_url(settings.redis_url.get_secret_value(), decode_responses=True)
        stack.push_async_callback(redis.aclose)
        client = httpx.AsyncClient(timeout=30, follow_redirects=True)
        stack.push_async_callback(client.aclose)
        llm = OpenAICompatibleClient.from_settings(settings)
        stack.push_async_callback(llm.aclose)

        broker = RedisBroker(redis)
        workspace = Workspace(pool, settings)
        try:
            universe = await load_universe(client)
        except Exception as exc:  # site still works on the configured watchlist alone
            log.warning("stock universe unavailable (%s); run `python -m uphill.market.universe`", exc)
            universe = []
        if uphill.live_universe and universe:
            live = sorted(set(settings.upstox_symbols) | {s.listing for s in universe})
            settings = settings.model_copy(update={"upstox_symbols": live})
        refs = settings.watchlist or settings.upstox_symbols or settings.kite_symbols
        watchlist = list(dict.fromkeys(parse_listing(r.upper()) for r in refs))
        aliases: dict[str, list[str]] = {s.listing: universe_aliases(s) for s in universe}
        aliases |= {f"{e.value}:{s}": aliases.get(f"{e.value}:{s}", []) for e, s in watchlist}
        for listing, names in uphill.news_aliases.items():
            aliases[listing] = sorted({*aliases.get(listing, []), *names})
        news_listings = list(dict.fromkeys([*watchlist, *((Exchange.NSE, s.symbol) for s in universe)]))
        news = NewsService(news_sources(settings, uphill, client, news_listings), SymbolTagger(aliases),
                           max_items=uphill.news_max_items)
        screener = Screener(workspace, universe, clock=workspace.clock)
        enabled = agent_enabled(settings)

        def new_session() -> Session:
            router = CommandRouter(workspace, clock=workspace.clock)
            agent = UpHillStockAgent(
                workspace, llm, news, max_steps=settings.agent_max_steps,
                context=lambda: AgentContext(security=router.security, as_of=router.effective_as_of),
            ) if enabled else None
            return Session(router=router, agent=agent)

        state = UpHillState(settings=settings, uphill=uphill, workspace=workspace, broker=broker, news=news,
                            sessions=SessionStore(new_session, uphill.max_sessions), watchlist=watchlist,
                            agent_enabled=enabled, clock=workspace.clock, screener=screener)
        tasks = [asyncio.create_task(news.run(uphill.news_refresh_seconds)),
                 asyncio.create_task(screener.run(uphill.screener_refresh_seconds))]
        pipeline = None
        if uphill.live_feed:
            pipeline = build_pipeline(settings, pool, broker)
            feeds = start_live_feeds(settings, pipeline)
            state.live_feeds = len(feeds)
            if feeds:
                tasks += [*feeds, asyncio.create_task(pipeline.run_periodic_flush(settings.tick_flush_interval_seconds))]
        app.state.uphill = state
        try:
            yield
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if pipeline is not None:
                try:
                    await pipeline.flush()
                except Exception:
                    log.exception("final tick flush failed")


def create_app(state: UpHillState | None = None, settings: Settings | None = None,
               uphill: UpHillSettings | None = None) -> FastAPI:
    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if state is not None:
            app.state.uphill = state
            yield
            return
        async with _runtime(app, settings or Settings(), uphill or UpHillSettings()):
            yield

    app = FastAPI(title="UpHill", version="0.1.0", lifespan=lifespan)
    for module in (market_routes, terminal_routes, agent_routes, news_routes):
        app.include_router(module.router)

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    if WEB_DIST.is_dir():
        app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        async def spa(path: str) -> FileResponse:
            if path.startswith(("api/", "ws/")):
                raise HTTPException(404)
            candidate = (WEB_DIST / path).resolve()
            if path and candidate.is_file() and candidate.is_relative_to(WEB_DIST):
                return FileResponse(candidate)
            if "." in path.rsplit("/", 1)[-1]:  # missing asset (e.g. optional background art): 404, not the app shell
                raise HTTPException(404)
            return FileResponse(WEB_DIST / "index.html")

    return app


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    uphill = UpHillSettings()
    if not WEB_DIST.is_dir():
        log.warning("frontend not built: run `npm install && npm run build` in UpHill/web (API still served)")
    uvicorn.run(create_app(uphill=uphill), host=uphill.host, port=uphill.port)
