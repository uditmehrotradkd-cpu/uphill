from __future__ import annotations

import asyncio
from datetime import date, datetime, time
from decimal import Decimal

import httpx
from pydantic import SecretStr

from quantterm.connectors.http import HttpFetcher
from quantterm.connectors.prices import (
    PolygonCorporateActionsSource,
    parse_polygon_aggs,
    parse_polygon_dividends,
    parse_polygon_splits,
)
from quantterm.corporate_actions import CorporateActionType
from quantterm.models import Exchange

ET = Exchange.NASDAQ.timezone

# Shapes as returned by Polygon /v3/reference/splits and /v3/reference/dividends.
NVDA_SPLITS = [
    {"execution_date": "2024-06-10", "id": "E36416cce", "split_from": 1, "split_to": 10, "ticker": "NVDA"},
    {"execution_date": "2021-07-20", "id": "E9f7b2a1", "split_from": 1, "split_to": 4, "ticker": "NVDA"},
]
AAPL_DIVIDENDS = [
    {"cash_amount": 0.26, "currency": "USD", "declaration_date": "2025-07-31", "dividend_type": "CD",
     "ex_dividend_date": "2025-08-11", "frequency": 4, "pay_date": "2025-08-14", "record_date": "2025-08-11",
     "ticker": "AAPL"},
    {"cash_amount": 0.25, "currency": "USD", "ex_dividend_date": "2024-11-08", "ticker": "AAPL"},
    {"cash_amount": 1.0, "currency": "CAD", "ex_dividend_date": "2024-05-10", "ticker": "AAPL"},
]


def test_splits_become_capital_change_factors_known_from_ex_date() -> None:
    actions = parse_polygon_splits(NVDA_SPLITS, "NVDA", Exchange.NASDAQ)
    assert [(a.ex_date, a.adjustment_factor) for a in actions] == [
        (date(2024, 6, 10), Decimal("0.1")), (date(2021, 7, 20), Decimal("0.25"))]
    assert all(a.action_type is CorporateActionType.SPLIT for a in actions)
    assert actions[0].announced_at == datetime(2024, 6, 10, tzinfo=ET)


def test_dividends_use_declaration_date_and_skip_foreign_currency() -> None:
    declared, undeclared = parse_polygon_dividends(AAPL_DIVIDENDS, "AAPL", Exchange.NASDAQ, "USD")
    assert declared.raw_dividend_amount == Decimal("0.26") and declared.ex_date == date(2025, 8, 11)
    assert declared.announced_at == datetime.combine(date(2025, 7, 31), time.max, tzinfo=ET)
    assert undeclared.announced_at == datetime(2024, 11, 8, tzinfo=ET)  # no declaration date -> ex-date


def test_earnings_day_bar_kept_when_vwap_includes_after_hours() -> None:
    # MSFT 2025-01-29 shape: post-earnings after-hours trading pulls vw below the regular-session low.
    rows = [{"t": 1738126800000, "o": 446.2, "h": 446.88, "l": 440.4, "c": 442.33, "v": 23581370, "vw": 439.3597},
            {"t": 1738213200000, "o": 418.77, "h": 422.86, "l": 413.16, "c": 414.99, "v": 54586300, "vw": 416.73}]
    early, late = parse_polygon_aggs(rows, Exchange.NASDAQ)
    assert early.close == 442.33 and early.vwap is None
    assert late.vwap == 416.73


def test_special_and_regular_dividend_on_same_ex_date_are_summed() -> None:
    rows = [{"cash_amount": 0.08, "currency": "USD", "declaration_date": "2004-07-20", "ex_dividend_date": "2004-11-15"},
            {"cash_amount": 3.0, "currency": "USD", "declaration_date": "2004-07-20", "ex_dividend_date": "2004-11-15",
             "dividend_type": "SC"}]
    (combined,) = parse_polygon_dividends(rows, "MSFT", Exchange.NASDAQ, "USD")
    assert combined.raw_dividend_amount == Decimal("3.08")


def test_malformed_rows_are_dropped_not_fatal() -> None:
    assert parse_polygon_splits([{"execution_date": "2024-01-02", "split_from": 0, "split_to": 2}], "X",
                                Exchange.NYSE) == []
    assert parse_polygon_dividends([{"cash_amount": -1, "ex_dividend_date": "2024-01-02"}], "X", Exchange.NYSE,
                                   "USD") == []


def test_source_pages_both_endpoints_and_keeps_key_out_of_errors() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path == "/v3/reference/splits":
            if "cursor" in request.url.params:
                return httpx.Response(200, json={"results": NVDA_SPLITS[1:]})
            return httpx.Response(200, json={"results": NVDA_SPLITS[:1],
                                             "next_url": "https://api.polygon.io/v3/reference/splits?cursor=abc"})
        return httpx.Response(200, json={"results": []})

    async def scenario() -> int:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            source = PolygonCorporateActionsSource(HttpFetcher(client, rate_per_second=1000), SecretStr("k"))
            return len(await source.fetch("NVDA", Exchange.NASDAQ))

    assert asyncio.run(scenario()) == 2
    assert seen == ["/v3/reference/splits", "/v3/reference/splits", "/v3/reference/dividends"]
