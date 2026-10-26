"""Redis-backed pub/sub fan-out and response cache."""

from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any, Protocol

from redis.asyncio import Redis
from redis.asyncio.client import PubSub


class Broker(Protocol):
    async def publish(self, channel: str, message: str) -> None: ...

    def subscribe(
        self, channels: Sequence[str]
    ) -> AbstractAsyncContextManager[AsyncIterator[tuple[str, str]]]:
        """Subscription is active once the context is entered; iterate for ``(channel, message)``."""
        ...


class Cache(Protocol):
    async def get(self, key: str) -> str | None: ...

    async def set(self, key: str, value: str, ttl_seconds: int) -> None: ...


def cache_key(namespace: str, **params: Any) -> str:
    """Deterministic key; every parameter that changes the answer (incl. ``as_of``) must be passed."""
    payload = json.dumps(params, sort_keys=True, default=str, separators=(",", ":"))
    return f"qt:{namespace}:{hashlib.sha256(payload.encode()).hexdigest()}"


class RedisBroker:
    def __init__(self, client: Redis) -> None:
        self._client = client

    async def publish(self, channel: str, message: str) -> None:
        await self._client.publish(channel, message)

    @asynccontextmanager
    async def subscribe(self, channels: Sequence[str]) -> AsyncIterator[AsyncIterator[tuple[str, str]]]:
        pubsub = self._client.pubsub(ignore_subscribe_messages=True)
        await pubsub.subscribe(*channels)
        try:
            yield _messages(pubsub)
        finally:
            await pubsub.unsubscribe()
            await pubsub.aclose()  # type: ignore[no-untyped-call]


async def _messages(pubsub: PubSub) -> AsyncIterator[tuple[str, str]]:
    async for msg in pubsub.listen():
        if msg["type"] == "message":
            yield _text(msg["channel"]), _text(msg["data"])


def _text(v: str | bytes) -> str:
    return v.decode() if isinstance(v, bytes) else v


class RedisCache:
    def __init__(self, client: Redis) -> None:
        self._client = client

    async def get(self, key: str) -> str | None:
        value = await self._client.get(key)
        return None if value is None else _text(value)

    async def set(self, key: str, value: str, ttl_seconds: int) -> None:
        await self._client.set(key, value, ex=ttl_seconds)
