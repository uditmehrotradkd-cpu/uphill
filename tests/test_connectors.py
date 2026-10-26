from __future__ import annotations

import asyncio
import hashlib
import io
import struct
import zipfile
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any
from zoneinfo import ZoneInfo

import httpx
import pytest
from pydantic import SecretStr

from quantterm.connectors.base import SourceError
from quantterm.connectors.edgar import parse_acceptance_times, parse_companyfacts
from quantterm.connectors.http import HttpFetcher, SourceHTTPError
from quantterm.connectors.india_xbrl import XbrlFilingRef, parse_ind_as_results, parse_nse_results_index
from quantterm.connectors.kite import (
    KiteTickDecoder,
    create_kite_session,
    kite_session_checksum,
    parse_kite_instruments,
    parse_request_token,
    split_packets,
)
from quantterm.connectors.prices import NseBhavcopySource, parse_nse_bhavcopy, parse_polygon_aggs
from quantterm.connectors.vendors import (
    FMP_INCOME_FIELDS,
    fmp_symbol,
    parse_fmp_statements,
    parse_polygon_financials,
)
from quantterm.models import Exchange, OrderBookSnapshot, PeriodType, Tick

IST = ZoneInfo("Asia/Kolkata")
ET = ZoneInfo("America/New_York")


async def _no_sleep(_: float) -> None:
    return None


def _fetcher(handler: Any) -> tuple[HttpFetcher, httpx.AsyncClient]:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return HttpFetcher(client, rate_per_second=1000, sleep=_no_sleep), client


# ------------------------------------------------------------------------------------------- EDGAR
def _e(start: str | None, end: str, val: int, accn: str, form: str, filed: str) -> dict[str, Any]:
    entry: dict[str, Any] = {"end": end, "val": val, "accn": accn, "form": form, "filed": filed, "fy": 2024, "fp": "Q"}
    if start is not None:
        entry["start"] = start
    return entry


FY24_10K = "0000320193-24-000123"
COMPANYFACTS: dict[str, Any] = {
    "facts": {
        "us-gaap": {
            "RevenueFromContractWithCustomerExcludingAssessedTax": {
                "units": {
                    "USD": [
                        _e("2023-10-01", "2023-12-30", 119_575_000_000, "0000320193-24-000006", "10-Q", "2024-02-02"),
                        _e("2023-12-31", "2024-03-30", 90_753_000_000, "0000320193-24-000069", "10-Q", "2024-05-03"),
                        _e("2023-10-01", "2024-03-30", 210_328_000_000, "0000320193-24-000069", "10-Q", "2024-05-03"),
                        _e("2024-03-31", "2024-06-29", 85_777_000_000, "0000320193-24-000081", "10-Q", "2024-08-02"),
                        _e("2023-10-01", "2024-06-29", 296_105_000_000, "0000320193-24-000081", "10-Q", "2024-08-02"),
                        _e("2023-10-01", "2024-09-28", 391_035_000_000, FY24_10K, "10-K", "2024-11-01"),
                        _e("2023-10-01", "2023-12-30", 119_575_000_000, "0000320193-24-000006", "8-K", "2024-02-01"),
                    ]
                }
            },
            "Assets": {
                "units": {"USD": [_e(None, "2024-09-28", 364_980_000_000, FY24_10K, "10-K", "2024-11-01")]}
            },
            "EarningsPerShareDiluted": {
                "units": {"USD/shares": [_e("2023-10-01", "2024-09-28", 6.08, FY24_10K, "10-K", "2024-11-01")]}
            },
        }
    }
}
ACCEPTANCE = {
    "0000320193-24-000006": datetime(2024, 2, 2, 11, 0, 32, tzinfo=UTC),
    "0000320193-24-000069": datetime(2024, 5, 3, 10, 2, 11, tzinfo=UTC),
    "0000320193-24-000081": datetime(2024, 8, 2, 10, 1, 49, tzinfo=UTC),
    FY24_10K: datetime(2024, 11, 1, 10, 1, 36, tzinfo=UTC),
}


def test_edgar_period_typing_and_acceptance_times() -> None:
    facts = parse_companyfacts(COMPANYFACTS, "AAPL", ACCEPTANCE)
    revenue_q = [f for f in facts if f.metric == "revenue" and f.period_type is PeriodType.QUARTER]
    assert [f.period_end_date for f in revenue_q] == [
        date(2023, 12, 30), date(2024, 3, 30), date(2024, 6, 29), date(2024, 9, 28)
    ]
    # 6-month YTD is typed as a half; 9-month YTD and the 8-K are dropped.
    assert {f.period_type for f in facts if f.metric == "revenue"} == {
        PeriodType.QUARTER, PeriodType.HALF_YEAR, PeriodType.FISCAL_YEAR
    }
    assert revenue_q[0].filing_timestamp == ACCEPTANCE["0000320193-24-000006"]
    eps = next(f for f in facts if f.metric == "eps_diluted")
    assert eps.value == Decimal("6.08") and eps.currency == "USD"
    assets = next(f for f in facts if f.metric == "total_assets")
    assert assets.period_type is PeriodType.FISCAL_YEAR


def test_edgar_derives_fiscal_q4_from_10k_minus_known_quarters() -> None:
    facts = parse_companyfacts(COMPANYFACTS, "AAPL", ACCEPTANCE)
    q4 = next(f for f in facts if f.metric == "revenue" and f.period_type is PeriodType.QUARTER
              and f.period_end_date == date(2024, 9, 28))
    assert q4.value == Decimal(391_035_000_000 - 119_575_000_000 - 90_753_000_000 - 85_777_000_000)
    assert q4.filing_timestamp == ACCEPTANCE[FY24_10K]  # not knowable before the 10-K
    assert q4.source_document_id == f"{FY24_10K}:Q4_DERIVED"
    assert not any(f.metric == "eps_diluted" and f.period_type is PeriodType.QUARTER for f in facts)


def test_edgar_missing_acceptance_falls_back_to_end_of_filing_day() -> None:
    facts = parse_companyfacts(COMPANYFACTS, "AAPL", {})
    first = next(f for f in facts if f.period_end_date == date(2023, 12, 30) and f.metric == "revenue")
    assert first.filing_timestamp == datetime(2024, 2, 2, 23, 59, 59, tzinfo=ET)


def test_edgar_rejects_payload_without_us_gaap() -> None:
    with pytest.raises(SourceError):
        parse_companyfacts({"facts": {"ifrs-full": {}}}, "SAP", {})


def test_acceptance_times_parse_utc() -> None:
    out = parse_acceptance_times(
        {"accessionNumber": ["a", "b"], "acceptanceDateTime": ["2026-07-31T10:01:02.000Z", ""]}
    )
    assert out == {"a": datetime(2026, 7, 31, 10, 1, 2, tzinfo=UTC)}


# -------------------------------------------------------------------------------------- Ind-AS XBRL
def _ctx(cid: str, period: str, scenario: str = "") -> str:
    return (
        f'<xbrli:context id="{cid}"><xbrli:entity><xbrli:identifier scheme="http://www.nseindia.com/NSESymbol">'
        f"INFY</xbrli:identifier></xbrli:entity><xbrli:period>{period}</xbrli:period>{scenario}</xbrli:context>"
    )


def _xbrl(nature: str = "Consolidated", extra: str = "") -> bytes:
    q = "<xbrli:startDate>2024-07-01</xbrli:startDate><xbrli:endDate>2024-09-30</xbrli:endDate>"
    dims = (
        '<xbrli:scenario><xbrldi:explicitMember dimension="in-bse-fin:SegmentsAxis">'
        "in-bse-fin:FinancialServicesMember</xbrldi:explicitMember></xbrli:scenario>"
    )
    m = 'unitRef="INR" decimals="-7"'
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<xbrli:xbrl xmlns:xbrli="http://www.xbrl.org/2003/instance" xmlns:xbrldi="http://xbrl.org/2006/xbrldi"
  xmlns:in-bse-fin="http://www.bseindia.com/xbrl/fin/2020-03-31/in-bse-fin"
  xmlns:iso4217="http://www.xbrl.org/2003/iso4217">
{_ctx("OneD", q)}
{_ctx("FourD", q)}
{_ctx("OneI", "<xbrli:instant>2024-09-30</xbrli:instant>")}
{_ctx("SegD", q, dims)}
<xbrli:unit id="INR"><xbrli:measure>iso4217:INR</xbrli:measure></xbrli:unit>
<xbrli:unit id="INRPerShare"><xbrli:divide><xbrli:unitNumerator><xbrli:measure>iso4217:INR</xbrli:measure>
</xbrli:unitNumerator><xbrli:unitDenominator><xbrli:measure>xbrli:shares</xbrli:measure></xbrli:unitDenominator>
</xbrli:divide></xbrli:unit>
<in-bse-fin:DateOfStartOfFinancialYear contextRef="OneD">2024-04-01</in-bse-fin:DateOfStartOfFinancialYear>
<in-bse-fin:DateOfEndOfFinancialYear contextRef="OneD">2025-03-31</in-bse-fin:DateOfEndOfFinancialYear>
<in-bse-fin:NatureOfReportStandaloneConsolidated contextRef="OneD">{nature}</in-bse-fin:NatureOfReportStandaloneConsolidated>
<in-bse-fin:DateOfStartOfReportingPeriod contextRef="OneD">2024-07-01</in-bse-fin:DateOfStartOfReportingPeriod>
<in-bse-fin:DateOfEndOfReportingPeriod contextRef="OneD">2024-09-30</in-bse-fin:DateOfEndOfReportingPeriod>
<in-bse-fin:DateOfStartOfReportingPeriod contextRef="FourD">2024-04-01</in-bse-fin:DateOfStartOfReportingPeriod>
<in-bse-fin:DateOfEndOfReportingPeriod contextRef="FourD">2024-09-30</in-bse-fin:DateOfEndOfReportingPeriod>
<in-bse-fin:RevenueFromOperations contextRef="OneD" {m}>409860000000.00</in-bse-fin:RevenueFromOperations>
<in-bse-fin:RevenueFromOperations contextRef="FourD" {m}>803000000000.00</in-bse-fin:RevenueFromOperations>
<in-bse-fin:RevenueFromOperations contextRef="SegD" {m}>1.00</in-bse-fin:RevenueFromOperations>
<in-bse-fin:ProfitLossForPeriod contextRef="OneD" {m}>65160000000.00</in-bse-fin:ProfitLossForPeriod>
<in-bse-fin:ProfitOrLossAttributableToOwnersOfParent contextRef="OneD" {m}>65060000000.00</in-bse-fin:ProfitOrLossAttributableToOwnersOfParent>
<in-bse-fin:DilutedEarningsLossPerShareFromContinuingAndDiscontinuedOperations contextRef="OneD" unitRef="INRPerShare" decimals="INF">15.68</in-bse-fin:DilutedEarningsLossPerShareFromContinuingAndDiscontinuedOperations>
<in-bse-fin:Assets contextRef="OneI" {m}>1418700000000.00</in-bse-fin:Assets>
<in-bse-fin:EquityAttributableToOwnersOfParent contextRef="OneI" {m}>904630000000.00</in-bse-fin:EquityAttributableToOwnersOfParent>
<in-bse-fin:Equity contextRef="OneI" {m}>908300000000.00</in-bse-fin:Equity>
{extra}
</xbrli:xbrl>""".encode()


REF = XbrlFilingRef(
    url="https://nsearchives.nseindia.com/corporate/xbrl/INDAS_112851_1276022_17102024074550.xml",
    symbol="INFY", exchange=Exchange.NSE, disseminated_at=datetime(2024, 10, 17, 19, 46, 12, tzinfo=IST),
    consolidated=True, document_id="NSE:1181591",
)


def test_ind_as_uses_reporting_period_facts_not_context_dates() -> None:
    facts = {(f.metric, f.period_type): f for f in parse_ind_as_results(_xbrl(), REF)}
    # FourD carries the quarter's dates in its context but is the H1 YTD per DateOfStartOfReportingPeriod.
    assert facts[("revenue", PeriodType.QUARTER)].value == Decimal("409860000000.00")
    assert facts[("revenue", PeriodType.HALF_YEAR)].value == Decimal("803000000000.00")
    assert facts[("net_income", PeriodType.QUARTER)].value == Decimal("65060000000.00")  # owners of parent
    assert facts[("eps_diluted", PeriodType.QUARTER)].currency == "INR"
    # Half-year balance sheet; parent equity preferred over total equity incl. NCI.
    assert facts[("total_equity", PeriodType.HALF_YEAR)].value == Decimal("904630000000.00")
    assert all(f.filing_timestamp == REF.disseminated_at and f.source == "NSE_XBRL" for f in facts.values())
    assert Decimal("1.00") not in {f.value for f in facts.values()}  # dimensional segment fact excluded


def test_ind_as_rejects_standalone_when_consolidated_expected() -> None:
    with pytest.raises(SourceError, match="Standalone"):
        parse_ind_as_results(_xbrl(nature="Standalone"), REF)


def test_ind_as_blocks_entity_expansion() -> None:
    bomb = b'<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "aaaa"><!ENTITY b "&a;&a;">]><x>&b;</x>'
    with pytest.raises(SourceError, match="rejected"):
        parse_ind_as_results(bomb, REF)


NSE_INDEX_ROWS = [
    {"consolidated": "Consolidated", "cumulative": "Non-cumulative", "exchdisstime": "16-Jan-2025 19:40:31",
     "broadCastDate": "16-Jan-2025 19:40:13", "seqNumber": "1189811", "toDate": "31-Dec-2024",
     "xbrl": "https://nsearchives.nseindia.com/corporate/xbrl/INDAS_117292_1348213_16012025074012.xml"},
    {"consolidated": "Non-Consolidated", "cumulative": "Non-cumulative", "exchdisstime": "16-Jan-2025 19:42:44",
     "broadCastDate": "16-Jan-2025 19:42:10", "seqNumber": "1189815", "toDate": "31-Dec-2024",
     "xbrl": "https://nsearchives.nseindia.com/corporate/xbrl/INDAS_117294_134819_16012025074210.xml"},
    {"consolidated": "Consolidated", "cumulative": "Non-cumulative", "exchdisstime": None,
     "broadCastDate": "08-Jun-2018 15:02:15", "seqNumber": "1038789", "toDate": "31-Mar-2018",
     "xbrl": "https://nsearchives.nseindia.com/corporate/xbrl/-"},
]


def test_nse_index_uses_exchange_dissemination_time_in_ist() -> None:
    refs = parse_nse_results_index(NSE_INDEX_ROWS, "INFY", consolidated=True)
    assert [r.document_id for r in refs] == ["NSE:1189811"]
    assert refs[0].disseminated_at == datetime(2025, 1, 16, 19, 40, 31, tzinfo=IST)
    assert refs[0].disseminated_at.astimezone(UTC).hour == 14


# ------------------------------------------------------------------------------------------ vendors
OBSERVED = datetime(2025, 2, 1, 6, 0, tzinfo=UTC)


def test_fmp_values_are_stamped_with_observation_time() -> None:
    rows = [
        {"date": "2024-12-31", "period": "Q3", "reportedCurrency": "INR", "acceptedDate": "2025-01-16 19:40:00",
         "revenue": 417640000000, "netIncome": 68060000000, "epsdiluted": 16.39},
        {"date": "2024-12-31", "period": "9M", "reportedCurrency": "INR", "revenue": 1},
    ]
    facts = parse_fmp_statements(rows, "INFY", FMP_INCOME_FIELDS, OBSERVED)
    assert {f.metric for f in facts} == {"revenue", "net_income", "eps_diluted"}
    assert all(f.filing_timestamp == OBSERVED and f.source == "FMP" for f in facts)
    assert fmp_symbol("INFY", Exchange.NSE) == "INFY.NS" and fmp_symbol("BRK.B", Exchange.NYSE) == "BRK-B"


def test_fmp_missing_currency_is_a_source_error() -> None:
    with pytest.raises(SourceError):
        parse_fmp_statements([{"date": "2024-12-31", "period": "Q3", "revenue": 1}], "INFY",
                             FMP_INCOME_FIELDS, OBSERVED)


def test_polygon_financials_units_and_timeframes() -> None:
    results = [{
        "end_date": "2024-09-28", "timeframe": "quarterly",
        "financials": {
            "income_statement": {
                "revenues": {"value": 94930000000, "unit": "USD"},
                "diluted_earnings_per_share": {"value": 0.97, "unit": "USD / shares"},
            },
            "balance_sheet": {"equity_attributable_to_parent": {"value": 56950000000, "unit": "USD"}},
        },
    }, {"end_date": "2024-09-28", "timeframe": "ttm", "financials": {}}]
    facts = {f.metric: f for f in parse_polygon_financials(results, "AAPL", OBSERVED)}
    assert facts["eps_diluted"].value == Decimal("0.97") and facts["eps_diluted"].currency == "USD"
    assert facts["total_equity"].period_type is PeriodType.QUARTER
    assert len(facts) == 3


# ------------------------------------------------------------------------------------------- prices
BHAV_HEADER = (
    "TradDt,BizDt,Sgmt,Src,FinInstrmTp,FinInstrmId,ISIN,TckrSymb,SctySrs,XpryDt,FininstrmActlXpryDt,StrkPric,"
    "OptnTp,FinInstrmNm,OpnPric,HghPric,LwPric,ClsPric,LastPric,PrvsClsgPric,UndrlygPric,SttlmPric,OpnIntrst,"
    "ChngInOpnIntrst,TtlTradgVol,TtlTrfVal,TtlNbOfTxsExctd,SsnId,NewBrdLotQty,Rmks,Rsvd1,Rsvd2,Rsvd3,Rsvd4"
)
BHAV_ROWS = [
    "2025-01-16,2025-01-16,CM,NSE,STK,1594,INE009A01021,INFY,EQ,,,,,INFOSYS LIMITED,1965.95,1966.95,1916.85,"
    "1928.45,1920.05,1949.65,,1928.40,,,7533579,14610485616.25,233848,F1,1,,,,,",
    "2025-01-16,2025-01-16,CM,NSE,STK,9999,INE000000000,BADROW,EQ,,,,,BROKEN,10,9,8,9.5,9.5,9.5,,9.5,,,1,1,1,F1,1,,,,,",
    "2025-01-16,2025-01-16,CM,NSE,STK,1595,INE009A01021,INFYBOND,N1,,,,,BOND,100,100,100,100,100,100,,100,,,1,1,1,F1,1,,,,,",
]


def test_bhavcopy_parses_raw_bar_and_drops_inconsistent_rows() -> None:
    bars = parse_nse_bhavcopy("\n".join([BHAV_HEADER, *BHAV_ROWS]))
    assert set(bars) == {"INFY"}
    infy = bars["INFY"]
    assert (infy.open, infy.high, infy.low, infy.close, infy.volume) == (1965.95, 1966.95, 1916.85, 1928.45, 7533579)
    assert infy.ts == datetime(2025, 1, 16, tzinfo=IST)


def test_bhavcopy_holiday_returns_empty_and_zip_is_read() -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("BhavCopy_NSE_CM_0_0_0_20250116_F_0000.csv", "\n".join([BHAV_HEADER, BHAV_ROWS[0]]))

    def handler(request: httpx.Request) -> httpx.Response:
        if "20250126" in request.url.path:
            return httpx.Response(404)
        return httpx.Response(200, content=buf.getvalue())

    async def scenario() -> tuple[dict[str, Any], dict[str, Any]]:
        http, client = _fetcher(handler)
        async with client:
            source = NseBhavcopySource(http)
            return await source.fetch_session(date(2025, 1, 26)), await source.fetch_session(date(2025, 1, 16))

    holiday, session = asyncio.run(scenario())
    assert holiday == {} and set(session) == {"INFY"}


def test_polygon_aggs_session_date_from_eastern_midnight() -> None:
    t = int(datetime(2024, 1, 2, tzinfo=ET).timestamp() * 1000)
    (bar,) = parse_polygon_aggs([{"t": t, "o": 187.15, "h": 188.44, "l": 183.885, "c": 185.64, "v": 82488674.0}],
                                Exchange.NASDAQ)
    assert bar.ts == datetime(2024, 1, 2, tzinfo=ET) and bar.volume == 82488674


# --------------------------------------------------------------------------------------------- Kite
INFY_TOKEN = 408065  # low byte 1 -> NSE segment


def _full_packet(token: int, ltp_paise: int, ltq: int, volume: int, ltt: int,
                 depth: list[tuple[int, int, int]] | None = None) -> bytes:
    head = struct.pack(">IIIIIIIIIII", token, ltp_paise, ltq, ltp_paise, volume, 0, 0, 0, 0, 0, 0)
    tail = struct.pack(">IIIII", ltt, 0, 0, 0, ltt)
    levels = (depth or []) + [(0, 0, 0)] * (10 - len(depth or []))
    book = b"".join(struct.pack(">IIH2x", qty, price, orders) for qty, price, orders in levels)
    return head + tail + book


def test_kite_full_packet_yields_depth_before_trade() -> None:
    decoder = KiteTickDecoder({INFY_TOKEN: (Exchange.NSE, "INFY")})
    ltt = int(datetime(2025, 1, 16, 9, 30, tzinfo=IST).timestamp())
    bids = [(1200, 192840, 7), (800, 192835, 4)] + [(0, 0, 0)] * 3
    asks = [(500, 192850, 3)]
    events = decoder.decode(_frame(_full_packet(INFY_TOKEN, 192845, 25, 1_000_000, ltt, bids + asks)))
    book, tick = events
    assert isinstance(book, OrderBookSnapshot) and isinstance(tick, Tick)
    assert [(lvl.price, lvl.quantity, lvl.orders) for lvl in book.bids] == [(1928.4, 1200, 7), (1928.35, 800, 4)]
    assert book.best_ask == 1928.5 and not book.crossed
    assert book.timestamp == tick.timestamp


def _frame(*packets: bytes) -> bytes:
    return struct.pack(">H", len(packets)) + b"".join(struct.pack(">H", len(p)) + p for p in packets)


def test_kite_decoder_emits_only_on_new_volume_with_delta_size() -> None:
    decoder = KiteTickDecoder({INFY_TOKEN: (Exchange.NSE, "INFY")})
    ltt = int(datetime(2025, 1, 16, 9, 30, tzinfo=IST).timestamp())
    first = decoder.decode(_frame(_full_packet(INFY_TOKEN, 192845, 25, 1_000_000, ltt)))
    same = decoder.decode(_frame(_full_packet(INFY_TOKEN, 192845, 25, 1_000_000, ltt)))
    nxt = decoder.decode(_frame(_full_packet(INFY_TOKEN, 192850, 10, 1_000_340, ltt + 1)))
    assert [(t.price, t.size) for t in first] == [(1928.45, 25)]
    assert same == []
    assert [(t.price, t.size) for t in nxt] == [(1928.5, 340)]
    assert nxt[0].timestamp == datetime(2025, 1, 16, 4, 0, 1, tzinfo=UTC)


def test_kite_decoder_resets_baseline_on_new_session_and_skips_non_full_packets() -> None:
    decoder = KiteTickDecoder({INFY_TOKEN: (Exchange.NSE, "INFY")})
    day1 = int(datetime(2025, 1, 16, 15, 29, tzinfo=IST).timestamp())
    day2 = int((datetime(2025, 1, 17, 9, 15, tzinfo=IST)).timestamp())
    decoder.decode(_frame(_full_packet(INFY_TOKEN, 192845, 5, 7_533_579, day1)))
    ticks = decoder.decode(_frame(
        struct.pack(">II", INFY_TOKEN, 192900),  # LTP-mode packet: no trade time
        _full_packet(INFY_TOKEN, 193000, 12, 12, day2),
    ))
    assert [(t.size, t.timestamp.date()) for t in ticks] == [(12, date(2025, 1, 17))]
    assert decoder.decode(b"\x00") == []  # heartbeat


def test_kite_truncated_frame_raises() -> None:
    with pytest.raises(SourceError):
        split_packets(struct.pack(">HH", 1, 184) + b"\x00" * 10)


def test_kite_instrument_master_requires_every_symbol() -> None:
    csv_text = (
        "instrument_token,exchange_token,tradingsymbol,name,last_price,expiry,strike,tick_size,lot_size,"
        "instrument_type,segment,exchange\n"
        "408065,1594,INFY,INFOSYS,0,,0,0.05,1,EQ,NSE,NSE\n"
    )
    assert parse_kite_instruments(csv_text, Exchange.NSE, ["INFY"]) == {408065: (Exchange.NSE, "INFY")}
    with pytest.raises(SourceError, match="TCS"):
        parse_kite_instruments(csv_text, Exchange.NSE, ["INFY", "TCS"])


def test_kite_request_token_from_redirect_url() -> None:
    url = "http://127.0.0.1/?action=login&type=login&status=success&request_token=Ab12Cd34"
    assert parse_request_token(url) == "Ab12Cd34"
    assert parse_request_token(" Ab12Cd34 ") == "Ab12Cd34"
    with pytest.raises(ValueError):
        parse_request_token("http://127.0.0.1/?status=cancelled")
    with pytest.raises(ValueError):
        parse_request_token("abc&api_key=x")


def test_kite_session_exchange_posts_checksum_and_returns_token() -> None:
    key, secret = SecretStr("k"), SecretStr("s")
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(dict(httpx.QueryParams(request.content.decode())))
        if seen["request_token"] == "bad":
            return httpx.Response(403, json={"status": "error", "message": "Token is invalid or has expired."})
        return httpx.Response(200, json={"status": "success", "data": {"access_token": "tok"}})

    async def scenario(request_token: str) -> SecretStr:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await create_kite_session(client, key, secret, request_token)

    assert asyncio.run(scenario("rt")).get_secret_value() == "tok"
    assert seen["checksum"] == kite_session_checksum(key, "rt", secret) == hashlib.sha256(b"krts").hexdigest()
    with pytest.raises(SourceError, match="expired"):
        asyncio.run(scenario("bad"))


# --------------------------------------------------------------------------------------------- HTTP
def test_http_retries_throttling_and_hides_api_key() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "1"})
        if "missing" in request.url.path:
            return httpx.Response(404)
        return httpx.Response(200, json={"ok": True})

    async def scenario() -> tuple[Any, str]:
        http, client = _fetcher(handler)
        async with client:
            ok = await http.get_json("https://api.example.test/v1/x", params={"apikey": "SECRET"})
            try:
                await http.get("https://api.example.test/missing", params={"apikey": "SECRET"})
            except SourceHTTPError as exc:
                return ok, str(exc)
        raise AssertionError("expected 404")

    ok, message = asyncio.run(scenario())
    assert ok == {"ok": True} and len(calls) == 3
    assert "SECRET" not in message and "404" in message


def test_http_gives_up_after_max_retries() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    async def scenario() -> None:
        http, client = _fetcher(handler)
        async with client:
            await http.get("https://api.example.test/down")

    with pytest.raises(SourceHTTPError, match="503"):
        asyncio.run(scenario())
