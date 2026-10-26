"""Rate-limited HTTP with bounded retries. Error messages never include query strings (API keys)."""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

import httpx

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class SourceHTTPError(RuntimeError):
    def __init__(self, status: int, url: httpx.URL) -> None:
        super().__init__(f"HTTP {status} from {url.copy_with(query=None)}")
        self.status = status


class RateLimiter:
    """Spaces requests at least ``1 / rate_per_second`` apart across all concurrent callers."""

    def __init__(self, rate_per_second: float) -> None:
        if rate_per_second <= 0:
            raise ValueError("rate_per_second must be positive")
        self._interval = 1.0 / rate_per_second
        self._next = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self._lock:
            now = time.monotonic()
            delay = self._next - now
            self._next = max(now, self._next) + self._interval
        if delay > 0:
            await asyncio.sleep(delay)


class HttpFetcher:
    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        rate_per_second: float,
        max_retries: int = 4,
        backoff_base: float = 0.5,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._client = client
        self._limiter = RateLimiter(rate_per_second)
        self._max_retries = max_retries
        self._backoff = backoff_base
        self._sleep = sleep

    def _delay(self, attempt: int, response: httpx.Response | None) -> float:
        if response is not None:
            retry_after = response.headers.get("Retry-After")
            if retry_after is not None and retry_after.isdigit():
                return float(retry_after)
        return float(self._backoff * (2**attempt) * (1 + random.random() * 0.25))

    async def get(
        self,
        url: str,
        *,
        params: Mapping[str, str | int] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> httpx.Response:
        # httpx replaces a URL's query with ``params``; merge so pagination cursors in next_url survive.
        target = httpx.URL(url).copy_merge_params(params) if params else httpx.URL(url)
        for attempt in range(self._max_retries + 1):
            await self._limiter.wait()
            try:
                response = await self._client.get(target, headers=headers)
            except httpx.TransportError:
                if attempt == self._max_retries:
                    raise
                await self._sleep(self._delay(attempt, None))
                continue
            if response.status_code in RETRYABLE_STATUS and attempt < self._max_retries:
                await self._sleep(self._delay(attempt, response))
                continue
            if response.status_code >= 400:
                raise SourceHTTPError(response.status_code, response.request.url)
            return response
        raise AssertionError("unreachable")

    async def get_json(
        self,
        url: str,
        *,
        params: Mapping[str, str | int] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Any:
        return (await self.get(url, params=params, headers=headers)).json()
