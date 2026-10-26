from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest
from fakes import DictCache, FakeDatabase, InMemoryBroker
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from quantterm.api.app import build_services, create_app, parse_listing
from quantterm.db.repositories import DAILY, FundamentalsRepository, SecurityRepository
from quantterm.models import Exchange, FundamentalFact, PeriodType, session_bar_ts
from quantterm.settings import Settings

API = "a" * 32
INGEST = "i" * 32
AUTH = {"Authorization": f"Bearer {API}"}


@pytest.fixture
def db() -> FakeDatabase:
    db = FakeDatabase()
    securities = SecurityRepository(db)
    facts = FundamentalsRepository(db, securities)

    async def seed() -> None:
        await securities.upsert("AAPL", Exchange.NASDAQ)
        filed = datetime(2024, 11, 1, 10, 1, 36, tzinfo=UTC)
        await facts.append(Exchange.NASDAQ, [
            FundamentalFact(symbol="AAPL", metric="revenue", period_type=PeriodType.QUARTER,
                            period_end_date=date(2024, 9, 28), filing_timestamp=filed,
                            value=Decimal("94930000000"), currency="USD", source="SEC_EDGAR",
                            source_document_id="0000320193-24-000123:Q4_DERIVED"),
            FundamentalFact(symbol="AAPL", metric="revenue", period_type=PeriodType.QUARTER,
                            period_end_date=date(2024, 9, 28), filing_timestamp=datetime(2025, 1, 5, tzinfo=UTC),
                            value=Decimal("95100000000"), currency="USD", source="FMP"),
        ])

    asyncio.run(seed())
    return db


@pytest.fixture
def client(db: FakeDatabase) -> Iterator[TestClient]:
    settings = Settings(api_token=API, ingest_token=INGEST, _env_file=None)
    services = build_services(settings, db, InMemoryBroker(), DictCache())
    with TestClient(create_app(settings, services)) as c:
        yield c


def test_requires_bearer_token(client: TestClient) -> None:
    r = client.get("/v1/fundamentals/NASDAQ/AAPL", params={"metrics": "revenue", "period_type": "Q",
                                                            "source": "SEC_EDGAR", "as_of": "2025-01-01T00:00:00Z"})
    assert r.status_code == 401
    bad = client.get("/v1/fundamentals/NASDAQ/AAPL", headers={"Authorization": "Bearer wrong"})
    assert bad.status_code == 401


def test_fundamentals_point_in_time_and_naive_as_of_rejected(client: TestClient) -> None:
    params = {"metrics": "revenue", "period_type": "Q", "source": "SEC_EDGAR"}
    before = client.get("/v1/fundamentals/NASDAQ/AAPL", headers=AUTH,
                        params={**params, "as_of": "2024-11-01T10:00:00Z"})
    after = client.get("/v1/fundamentals/NASDAQ/AAPL", headers=AUTH,
                       params={**params, "as_of": "2024-11-01T10:05:00Z"})
    naive = client.get("/v1/fundamentals/NASDAQ/AAPL", headers=AUTH, params={**params, "as_of": "2024-11-02T00:00:00"})
    assert before.status_code == 200 and before.json() == []
    assert after.json()[0]["value"] == "94930000000"
    assert after.json()[0]["source_document_id"].endswith("Q4_DERIVED")
    assert naive.status_code == 422


def test_fundamentals_cached_for_past_as_of(client: TestClient, db: FakeDatabase) -> None:
    params = {"metrics": "revenue", "period_type": "Q", "source": "SEC_EDGAR", "as_of": "2024-12-01T00:00:00Z"}
    first = client.get("/v1/fundamentals/NASDAQ/AAPL", headers=AUTH, params=params).json()
    queries = len(db.queries)
    second = client.get("/v1/fundamentals/NASDAQ/AAPL", headers=AUTH, params=params).json()
    assert first == second and len(db.queries) == queries


def test_unknown_security_is_404(client: TestClient) -> None:
    r = client.get("/v1/fundamentals/NYSE/AAPL", headers=AUTH,
                   params={"metrics": "revenue", "period_type": "Q", "source": "SEC_EDGAR",
                           "as_of": "2024-12-01T00:00:00Z"})
    assert r.status_code == 404


def test_reconciliation_flags_vendor_variance(client: TestClient, db: FakeDatabase) -> None:
    body = {"metrics": ["revenue"], "period_type": "Q", "primary_source": "SEC_EDGAR",
            "secondary_source": "FMP", "as_of": "2025-02-01T00:00:00+00:00"}
    r = client.post("/v1/reconciliation/NASDAQ/AAPL", headers=AUTH, json=body)
    assert r.status_code == 200
    payload = r.json()
    assert payload["breaks"] == 0  # 0.18% < 0.5%
    row = payload["rows"][0]
    assert (row["status"], row["resolution"], row["confidence_score"]) == ("MATCH", "CONSENSUS", 1.0)
    assert row["resolved_value"] == "94930000000" and payload["alerts"] == []

    early = client.post("/v1/reconciliation/NASDAQ/AAPL", headers=AUTH,
                        json={**body, "as_of": "2024-12-01T00:00:00Z"}).json()
    assert early["rows"][0]["status"] == "MISSING_SECONDARY"  # vendor value not yet observed
    assert early["rows"][0]["confidence_score"] == 0.9
    assert db.breaks[-1]["status"] == "MISSING_SECONDARY"

    same = client.post("/v1/reconciliation/NASDAQ/AAPL", headers=AUTH, json={**body, "secondary_source": "SEC_EDGAR"})
    assert same.status_code == 422


def test_daily_prices_endpoint_returns_query_time_adjusted_rows(client: TestClient, db: FakeDatabase) -> None:
    ts = session_bar_ts(Exchange.NASDAQ, date(2024, 1, 2))
    db.bars[("AAPL", "NASDAQ", DAILY, ts)] = dict(timestamp=ts, open=187.15, high=188.44, low=183.885,
                                                close=185.64, volume=82488674, vwap=185.9, source="POLYGON")
    r = client.get("/v1/prices/NASDAQ/AAPL/daily", headers=AUTH,
                   params={"start": "2024-01-01", "end": "2024-01-05", "as_of": "2024-02-01T00:00:00Z",
                           "adjust_dividends": "false"})
    assert r.status_code == 200
    (row,) = r.json()
    assert row["session_date"] == "2024-01-02" and row["close"] == 185.64 and row["price_factor"] == 1.0
    assert row["vwap"] == 185.9
    bad = client.get("/v1/prices/NASDAQ/AAPL/daily", headers=AUTH, params={"start": "2024-02-01", "end": "2024-01-01"})
    assert bad.status_code == 422


def test_ingest_requires_ingest_token(client: TestClient) -> None:
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect("/v1/ws/ingest", headers=AUTH) as ws:
            ws.receive_text()
    assert exc.value.code == 1008


def test_ingest_to_stream_end_to_end(client: TestClient, db: FakeDatabase) -> None:
    t0 = datetime(2025, 1, 16, 14, 30, tzinfo=UTC)
    ticks = [
        {"symbol": "AAPL", "exchange": "NASDAQ", "timestamp": (t0 + timedelta(seconds=i)).isoformat(),
         "price": 230.0 + (i % 3) * 0.01, "size": 100}
        for i in range(12)
    ]
    ticks.append({"symbol": "AAPL", "exchange": "NASDAQ", "timestamp": (t0 + timedelta(seconds=12)).isoformat(),
                  "price": 2.30, "size": 100})
    ticks.append({"symbol": "MSFT", "exchange": "NASDAQ", "timestamp": t0.isoformat(), "price": 400.0, "size": 1})
    ticks.insert(5, {"symbol": "AAPL", "exchange": "NASDAQ", "timestamp": (t0 + timedelta(seconds=5)).isoformat(),
                     "bids": [{"price": 229.99, "quantity": 300}], "asks": [{"price": 230.02, "quantity": 200}]})

    with client.websocket_connect("/v1/ws/stream?symbols=NASDAQ:AAPL", headers=AUTH) as stream:
        with client.websocket_connect("/v1/ws/ingest?source=TEST_FEED",
                                      headers={"Authorization": f"Bearer {INGEST}"}) as ingest:
            ingest.send_text(json.dumps(ticks))
            ack = ingest.receive_json()
            ingest.send_text("{not json")
            error = ingest.receive_json()
        received = [json.loads(stream.receive_text()) for _ in range(12)]

    assert ack == {"accepted": 12, "quarantined": 1, "books": 1, "unknown": ["NASDAQ:MSFT"]}
    assert error["error"] == "invalid payload"
    assert {m["channel"] for m in received} == {"qt:ticks:NASDAQ:AAPL"}
    assert all(m["data"]["tick"]["price"] > 200 for m in received)


def test_stream_rejects_malformed_listing(client: TestClient) -> None:
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/v1/ws/stream?symbols=LSE:VOD", headers=AUTH) as ws:
            ws.receive_text()
    assert parse_listing("NSE:M&M") == (Exchange.NSE, "M&M")
    with pytest.raises(ValueError):
        parse_listing("NSE:infy")
