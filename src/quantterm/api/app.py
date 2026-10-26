"""FastAPI service: PIT fundamentals, query-time adjusted prices, reconciliation, tick ingest + stream.

Run: ``uvicorn quantterm.api.app:app_factory --factory``.
Timestamps in query strings must be URL-encoded (``+05:30`` -> ``%2B05:30``) or use ``Z``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import secrets
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Path, Query, Request, WebSocket, WebSocketDisconnect, status
from fastapi.responses import JSONResponse, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, SecretStr, TypeAdapter, ValidationError
from redis.asyncio import Redis

from quantterm.db.pool import apply_schema, create_pool
from quantterm.db.repositories import (
    Database,
    PriceRepository,
    ReconciliationRepository,
    SecurityRepository,
    UnknownSecurityError,
)
from quantterm.engine.adjustments import get_adjusted_series
from quantterm.hygiene.reconciler import ReconciledLineItem
from quantterm.live import SYMBOL_PATTERN, build_pipeline, start_live_feeds
from quantterm.live import parse_listing as parse_listing
from quantterm.models import Exchange, FundamentalFact, OrderBookSnapshot, PeriodType, Tick
from quantterm.pit import PointInTimeFundamentalsRepository
from quantterm.pubsub import Broker, Cache, RedisBroker, RedisCache, cache_key
from quantterm.services import ReconciliationService
from quantterm.settings import Settings
from quantterm.streaming import TickIngestionPipeline, candle_channel, tick_channel

log = logging.getLogger(__name__)

SOURCE_PATTERN = r"^[A-Z0-9_:]{1,32}$"
MAX_INGEST_MESSAGE_BYTES = 1_000_000
MAX_INGEST_BATCH = 5_000
MAX_PRICE_RANGE = timedelta(days=366 * 30)

Symbol = Annotated[str, Path(pattern=SYMBOL_PATTERN)]
_FACTS = TypeAdapter(list[FundamentalFact])
_EVENTS = TypeAdapter(list[Tick | OrderBookSnapshot])


@dataclass
class TerminalServices:
    db: Database
    securities: SecurityRepository
    fundamentals: PointInTimeFundamentalsRepository
    prices: PriceRepository
    reconciliation: ReconciliationService
    pipeline: TickIngestionPipeline
    broker: Broker
    cache: Cache
    health_checks: Sequence[Callable[[], Awaitable[object]]] = ()


def build_services(
    settings: Settings,
    db: Database,
    broker: Broker,
    cache: Cache,
    health_checks: Sequence[Callable[[], Awaitable[object]]] = (),
) -> TerminalServices:
    securities = SecurityRepository(db)
    prices = PriceRepository(db, securities)
    pit = PointInTimeFundamentalsRepository(db)
    return TerminalServices(
        db=db,
        securities=securities,
        fundamentals=pit,
        prices=prices,
        reconciliation=ReconciliationService(
            pit, securities, ReconciliationRepository(db), settings.reconciliation_tolerance
        ),
        pipeline=build_pipeline(settings, db, broker, securities, prices),
        broker=broker,
        cache=cache,
        health_checks=health_checks,
    )


def _token_ok(presented: str | None, expected: SecretStr) -> bool:
    return presented is not None and secrets.compare_digest(
        presented.encode(), expected.get_secret_value().encode()
    )


def _ws_bearer(ws: WebSocket) -> str | None:
    scheme, _, token = ws.headers.get("authorization", "").partition(" ")
    return token if scheme.lower() == "bearer" and token else None


_bearer = HTTPBearer(auto_error=False)


async def require_api_token(
    request: Request, creds: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)]
) -> None:
    settings: Settings = request.app.state.settings
    if creds is None or not _token_ok(creds.credentials, settings.api_token):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, "invalid or missing token", {"WWW-Authenticate": "Bearer"}
        )


def get_services(request: Request) -> TerminalServices:
    svc: TerminalServices = request.app.state.services
    return svc


Services = Annotated[TerminalServices, Depends(get_services)]


class AdjustedBar(BaseModel):
    timestamp: datetime
    session_date: date
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float | None
    price_factor: float
    volume_factor: float


class ReconciliationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metrics: list[str] = Field(min_length=1, max_length=50)
    period_type: PeriodType
    primary_source: str = Field(pattern=SOURCE_PATTERN)
    secondary_source: str = Field(pattern=SOURCE_PATTERN)
    as_of: AwareDatetime


class ReconciliationResponse(BaseModel):
    as_of: datetime
    breaks: int
    rows: list[ReconciledLineItem]
    alerts: list[str]


def create_app(settings: Settings, services: TerminalServices | None = None) -> FastAPI:
    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        closers: list[Callable[[], Awaitable[Any]]] = []
        svc = services
        if svc is None:
            pool = await create_pool(settings)
            closers.append(pool.close)
            if settings.auto_migrate:
                await apply_schema(pool)
            redis: Redis = Redis.from_url(settings.redis_url.get_secret_value(), decode_responses=True)
            closers.append(redis.aclose)

            async def db_ok() -> object:
                return await pool.fetchval("SELECT 1")

            async def redis_ok() -> object:
                return await redis.ping()

            svc = build_services(settings, pool, RedisBroker(redis), RedisCache(redis), (db_ok, redis_ok))
        app.state.services = svc

        tasks = [asyncio.create_task(svc.pipeline.run_periodic_flush(settings.tick_flush_interval_seconds))]
        tasks.extend(start_live_feeds(settings, svc.pipeline))
        try:
            yield
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            try:
                await svc.pipeline.flush()
            except Exception:
                log.exception("final tick flush failed; %d rows lost", svc.pipeline.pending)
            for close in reversed(closers):
                await close()

    app = FastAPI(title="QuantTerm", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings

    @app.exception_handler(UnknownSecurityError)
    async def _unknown_security(_: Request, exc: UnknownSecurityError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=status.HTTP_404_NOT_FOUND)

    def _cacheable(as_of: datetime | None) -> bool:
        # Past answers only change on late backfill (bounded by TTL); "now" answers change continuously.
        return as_of is not None and as_of <= datetime.now(UTC)

    @app.get("/health")
    async def health(svc: Services) -> JSONResponse:
        for check in svc.health_checks:
            try:
                await check()
            except Exception:
                log.exception("health check failed")
                return JSONResponse({"status": "degraded"}, status_code=status.HTTP_503_SERVICE_UNAVAILABLE)
        return JSONResponse({"status": "ok", "pending_ticks": svc.pipeline.pending})

    @app.get(
        "/v1/fundamentals/{exchange}/{symbol}",
        response_model=list[FundamentalFact],
        dependencies=[Depends(require_api_token)],
    )
    async def fundamentals(
        exchange: Exchange,
        symbol: Symbol,
        svc: Services,
        metrics: Annotated[list[str], Query(min_length=1, max_length=50)],
        period_type: PeriodType,
        source: Annotated[str, Query(pattern=SOURCE_PATTERN)],
        as_of: AwareDatetime,
    ) -> Response:
        key = cache_key(
            "fundamentals", exchange=exchange.value, symbol=symbol, metrics=sorted(set(metrics)),
            period_type=period_type.value, source=source, as_of=as_of.isoformat(),
        )
        if _cacheable(as_of) and (hit := await svc.cache.get(key)) is not None:
            return Response(hit, media_type="application/json")
        await svc.securities.get(symbol, exchange)
        facts = await svc.fundamentals.fetch_as_of([symbol], exchange, metrics, period_type, source, as_of)
        body = _FACTS.dump_json(facts).decode()
        if _cacheable(as_of):
            await svc.cache.set(key, body, settings.cache_ttl_seconds)
        return Response(body, media_type="application/json")

    @app.get(
        "/v1/prices/{exchange}/{symbol}/daily",
        response_model=list[AdjustedBar],
        dependencies=[Depends(require_api_token)],
    )
    async def daily_prices(
        exchange: Exchange,
        symbol: Symbol,
        svc: Services,
        start: date,
        end: date,
        adjust_splits: bool = True,
        adjust_dividends: bool = True,
        as_of: AwareDatetime | None = None,
    ) -> Response:
        """Bloomberg GP semantics: capital-change and cash-dividend adjustment toggled independently."""
        if start > end or end - start > MAX_PRICE_RANGE:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "require start <= end within 30 years")
        key = cache_key(
            "prices", exchange=exchange.value, symbol=symbol, start=start.isoformat(), end=end.isoformat(),
            adjust_splits=adjust_splits, adjust_dividends=adjust_dividends,
            as_of=as_of.isoformat() if as_of else None,
        )
        if _cacheable(as_of) and (hit := await svc.cache.get(key)) is not None:
            return Response(hit, media_type="application/json")
        await svc.securities.get(symbol, exchange)
        frame = await get_adjusted_series(
            symbol, start, end, as_of or datetime.now(UTC), adjust_splits, adjust_dividends,
            exchange=exchange, db=svc.db,
        )
        body = frame.write_json()
        if _cacheable(as_of):
            await svc.cache.set(key, body, settings.cache_ttl_seconds)
        return Response(body, media_type="application/json")

    @app.post(
        "/v1/reconciliation/{exchange}/{symbol}",
        dependencies=[Depends(require_api_token)],
    )
    async def run_reconciliation(
        exchange: Exchange, symbol: Symbol, request: ReconciliationRequest, svc: Services
    ) -> ReconciliationResponse:
        try:
            report = await svc.reconciliation.run(
                symbol, exchange, request.metrics, request.period_type,
                request.primary_source, request.secondary_source, request.as_of,
            )
        except ValueError as exc:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
        return ReconciliationResponse(
            as_of=request.as_of, breaks=report.breaks, rows=list(report.items),
            alerts=[a.message for a in report.alerts],
        )

    @app.websocket("/v1/ws/ingest")
    async def ingest(ws: WebSocket, source: Annotated[str, Query(pattern=SOURCE_PATTERN)] = "WS") -> None:
        if not _token_ok(_ws_bearer(ws), settings.ingest_token):
            await ws.close(code=status.WS_1008_POLICY_VIOLATION)
            return
        svc: TerminalServices = ws.app.state.services
        await ws.accept()
        try:
            while True:
                raw = await ws.receive_text()
                if len(raw) > MAX_INGEST_MESSAGE_BYTES:
                    await ws.send_json({"error": "message too large"})
                    continue
                try:
                    payload = json.loads(raw)
                    events = _EVENTS.validate_python(payload if isinstance(payload, list) else [payload])
                except ValueError as exc:  # JSONDecodeError and ValidationError
                    await ws.send_json({"error": "invalid payload", "detail": str(exc)[:500]})
                    continue
                if len(events) > MAX_INGEST_BATCH:
                    await ws.send_json({"error": f"batch exceeds {MAX_INGEST_BATCH} events"})
                    continue
                accepted = quarantined = books = 0
                unknown: set[str] = set()
                for event in events:
                    try:
                        if isinstance(event, OrderBookSnapshot):
                            await svc.pipeline.submit_book(event)
                            books += 1
                            continue
                        verdict = await svc.pipeline.submit(event, source)
                    except UnknownSecurityError:
                        unknown.add(f"{event.exchange.value}:{event.symbol}")
                        continue
                    if verdict.accepted:
                        accepted += 1
                    else:
                        quarantined += 1
                await ws.send_json({"accepted": accepted, "quarantined": quarantined, "books": books,
                                    "unknown": sorted(unknown)})
        except WebSocketDisconnect:
            return

    @app.websocket("/v1/ws/stream")
    async def stream(ws: WebSocket, symbols: Annotated[list[str], Query(min_length=1, max_length=200)]) -> None:
        if not _token_ok(_ws_bearer(ws), settings.api_token):
            await ws.close(code=status.WS_1008_POLICY_VIOLATION)
            return
        try:
            listings = [parse_listing(s) for s in symbols]
        except ValueError:
            await ws.close(code=status.WS_1008_POLICY_VIOLATION)
            return
        channels = [ch for e, s in listings for ch in (tick_channel(e, s), candle_channel(e, s))]
        svc: TerminalServices = ws.app.state.services
        # Subscribe before accepting so no message published after the handshake can be missed.
        async with svc.broker.subscribe(channels) as messages:
            await ws.accept()

            async def forward() -> None:
                async for channel, data in messages:
                    await ws.send_text(f'{{"channel":{json.dumps(channel)},"data":{data}}}')

            # TaskGroup guarantees neither side outlives the connection, including on cancellation.
            try:
                async with asyncio.TaskGroup() as tg:
                    forwarder = tg.create_task(forward())
                    while (await ws.receive())["type"] != "websocket.disconnect":
                        pass
                    forwarder.cancel()
            except* (WebSocketDisconnect, RuntimeError, OSError) as group:
                log.warning("stream closed with %s", [type(e).__name__ for e in group.exceptions])

    return app


def app_factory() -> FastAPI:
    return create_app(Settings())  # required tokens come from QT_* environment variables
