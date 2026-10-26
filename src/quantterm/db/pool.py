"""Connection pool and idempotent schema migration."""

from __future__ import annotations

from importlib.resources import files
from typing import Any

import asyncpg

from quantterm.settings import Settings

# Arbitrary constant shared by all app instances so concurrent deploys serialise migrations.
_MIGRATION_LOCK_ID = 7_301_995_517


def schema_sql() -> str:
    return files("quantterm.db").joinpath("schema.sql").read_text(encoding="utf-8")


async def create_pool(settings: Settings) -> Any:
    # UTC session time zone keeps the ``filing_timestamp::date`` CHECK deterministic.
    return await asyncpg.create_pool(
        dsn=settings.database_url.get_secret_value(),
        min_size=settings.db_pool_min,
        max_size=settings.db_pool_max,
        server_settings={"timezone": "UTC", "application_name": "quantterm"},
    )


async def apply_schema(pool: Any) -> None:
    async with pool.acquire() as conn:
        await conn.execute("SELECT pg_advisory_lock($1)", _MIGRATION_LOCK_ID)
        try:
            await conn.execute(schema_sql())
        finally:
            await conn.execute("SELECT pg_advisory_unlock($1)", _MIGRATION_LOCK_ID)
