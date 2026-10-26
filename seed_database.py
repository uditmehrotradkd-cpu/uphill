"""Create the demo database, apply src/quantterm/db/schema.sql and load the synthetic demo market.

    python seed_database.py                      # -> database "quantterm_demo" on the QT_DATABASE_URL server
    python seed_database.py --database my_demo --as-of 2026-10-23T16:30:00-04:00

The demo data (RELIANCE, HDFCBANK, TCS, AAPL + beta benchmarks) is synthetic and labelled DEMO_*. It is
written to its own database and the script refuses to touch a database that already holds real data.
If PostgreSQL is not reachable and Docker is installed, the docker-compose stack is started first.
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from quantterm.db.pool import apply_schema, create_pool
from quantterm.demo import (
    DEFAULT_DATABASE,
    DatabaseUnavailable,
    DemoSeedRefused,
    SeedReport,
    assert_demo_only,
    demo_settings,
    ensure_database,
    seed_demo,
)
from quantterm.settings import Settings

ROOT = Path(__file__).resolve().parent
DOCKER_WAIT_SECONDS = 90


def _start_docker_stack() -> bool:
    if shutil.which("docker") is None:
        return False
    print("PostgreSQL unreachable; starting the docker compose stack (timescaledb, redis)...", flush=True)
    result = subprocess.run(["docker", "compose", "up", "-d", "timescaledb", "redis"], cwd=ROOT, check=False)
    return result.returncode == 0


async def _prepare(settings: Settings, database: str) -> str:
    server = settings.database_url.get_secret_value()
    try:
        return await ensure_database(server, database)
    except DatabaseUnavailable:
        if not _start_docker_stack():
            raise
    deadline = time.monotonic() + DOCKER_WAIT_SECONDS
    while True:
        try:
            return await ensure_database(server, database)
        except DatabaseUnavailable:
            if time.monotonic() > deadline:
                raise
            await asyncio.sleep(3)


async def _seed(settings: Settings, database: str, as_of: datetime | None) -> tuple[Settings, SeedReport]:
    dsn = await _prepare(settings, database)
    demo = demo_settings(settings, dsn)
    pool = await create_pool(demo)
    try:
        await apply_schema(pool)
        await assert_demo_only(pool)
        report = await seed_demo(pool, demo, as_of)
    finally:
        await pool.close()
    return demo, report


def seed(argv: list[str] | None = None) -> Settings:
    """Parse arguments, seed, print a summary; returns Settings pointing at the seeded demo database."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--database", default=DEFAULT_DATABASE, help=f"target database (default {DEFAULT_DATABASE})")
    parser.add_argument("--as-of", type=datetime.fromisoformat, help="ISO-8601 with offset; default now")
    args = parser.parse_args(argv)
    if args.as_of is not None and args.as_of.tzinfo is None:
        parser.error("--as-of needs a UTC offset, e.g. 2026-10-23T16:30:00-04:00")
    try:
        demo, report = asyncio.run(_seed(Settings(), args.database, args.as_of))
    except DatabaseUnavailable as exc:
        sys.exit(f"{exc}\nStart PostgreSQL + TimescaleDB (docker compose up -d, or your local server) and re-run.")
    except DemoSeedRefused as exc:
        sys.exit(str(exc))
    print(f"seeded database {args.database!r} as of {report.as_of:%Y-%m-%d %H:%M %Z}: "
          f"{report.securities} listings, {report.bars} new bars, {report.splits} split, {report.dividends} dividends, "
          f"{report.fundamentals} new fundamental vintages, {report.macro} rate prints, "
          f"{report.option_quotes} new option quotes")
    return demo


if __name__ == "__main__":
    seed()
    print("Launch the terminal on it with:  python seed_and_launch.py")
