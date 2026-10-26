from __future__ import annotations

import logging
from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from quantterm.hygiene.reconciler import (
    RECONCILED_LINE_ITEMS,
    ConfidencePolicy,
    DiscrepancyAlert,
    FundamentalReconciler,
    Resolution,
)
from quantterm.models import FundamentalFact, PeriodType
from quantterm.reconciliation import ReconStatus

FY24 = date(2024, 9, 28)
AS_OF = datetime(2024, 11, 15, tzinfo=UTC)

# AAPL FY2024. The 10-K XBRL carries no EBITDA or total-debt concept; the vendor computes both.
XBRL = {
    "revenue": Decimal("391035000000"),
    "cash_and_equivalents": Decimal("29943000000"),
    "minority_interest": Decimal("0"),
    "preferred_stock": Decimal("0"),
    "diluted_shares": Decimal("15408095000"),
}
VENDOR = {
    "revenue": Decimal("391035000000"),
    "ebitda": Decimal("134661000000"),
    "total_debt": Decimal("106629000000"),
    "cash_and_equivalents": Decimal("30242430000"),  # +1.0 %
    "minority_interest": Decimal("0"),
    "preferred_stock": Decimal("0"),
    "diluted_shares": Decimal("15408095"),  # reported in thousands
}


def _report(primary: dict[str, Decimal], secondary: dict[str, Decimal], **kw: object):  # type: ignore[no-untyped-def]
    reconciler = FundamentalReconciler(**kw)  # type: ignore[arg-type]
    return reconciler.reconcile_line_items(
        "AAPL", PeriodType.FISCAL_YEAR, FY24, primary, secondary, currency="USD", as_of=AS_OF,
        primary_source="SEC_EDGAR", secondary_source="FMP",
    )


def test_every_line_item_is_classified_scored_and_resolved() -> None:
    report = _report(XBRL, VENDOR)
    items = {i.metric: i for i in report.items}
    assert set(items) == set(RECONCILED_LINE_ITEMS)

    revenue = items["revenue"]
    assert (revenue.status, revenue.resolution, revenue.confidence_score) == (
        ReconStatus.MATCH, Resolution.CONSENSUS, 1.0)

    cash = items["cash_and_equivalents"]
    assert cash.status is ReconStatus.VARIANCE_BREACH and cash.resolution is Resolution.PRIMARY_OVERRIDE
    assert cash.resolved_value == XBRL["cash_and_equivalents"]  # XBRL wins
    assert cash.relative_variance == pytest.approx(0.01)
    assert cash.confidence_score == pytest.approx(1 - 0.25 * 0.30103, abs=1e-4)

    shares = items["diluted_shares"]
    assert shares.status is ReconStatus.SCALE_MISMATCH and shares.confidence_score == 0.9
    assert shares.resolved_value == XBRL["diluted_shares"] and shares.currency == "XXX"

    ebitda = items["ebitda"]
    assert ebitda.resolution is Resolution.SECONDARY_FALLBACK and ebitda.confidence_score == 0.6
    assert ebitda.resolved_value == VENDOR["ebitda"] and ebitda.primary_value is None

    for zero_item in ("minority_interest", "preferred_stock"):
        assert items[zero_item].status is ReconStatus.MATCH and items[zero_item].confidence_score == 1.0

    assert {a.item.metric for a in report.alerts} == {"cash_and_equivalents", "diluted_shares", "ebitda", "total_debt"}
    assert report.breaks == 4


def test_half_percent_is_the_boundary() -> None:
    at = _report({"revenue": Decimal(1000)}, {"revenue": Decimal(1005)})
    over = _report({"revenue": Decimal(1000)}, {"revenue": Decimal("1005.1")})
    assert at.items[0].confidence_score == 1.0 and not at.alerts
    assert over.items[0].status is ReconStatus.VARIANCE_BREACH
    assert over.items[0].confidence_score == 0.95  # capped below 1.0 for any breach
    assert over.items[0].resolved_value == Decimal(1000)


@pytest.mark.parametrize(
    ("secondary", "expected"),
    [("1010", 0.925), ("1050", 0.75), ("1500", 0.5), ("100000", 0.5)],
)
def test_confidence_decays_with_variance(secondary: str, expected: float) -> None:
    item = _report({"revenue": Decimal(1000)}, {"revenue": Decimal(secondary)}).items[0]
    assert item.confidence_score == pytest.approx(expected, abs=1e-3)


def test_sign_flip_and_unverified_primary() -> None:
    report = _report({"ebitda": Decimal(-500), "revenue": Decimal(9000)}, {"ebitda": Decimal(500)})
    items = {i.metric: i for i in report.items}
    assert items["ebitda"].status is ReconStatus.SIGN_MISMATCH and items["ebitda"].confidence_score == 0.5
    assert items["ebitda"].resolved_value == Decimal(-500)
    unverified = items["revenue"]
    assert unverified.resolution is Resolution.PRIMARY_UNVERIFIED and unverified.confidence_score == 0.9
    assert [a.item.metric for a in report.alerts] == ["ebitda"]  # missing vendor value is not a discrepancy


def test_alerts_are_logged_and_delivered(caplog: pytest.LogCaptureFixture) -> None:
    received: list[DiscrepancyAlert] = []
    with caplog.at_level(logging.WARNING, logger="quantterm.hygiene.reconciler"):
        report = _report(XBRL, VENDOR, on_alert=received.append)
    assert received == list(report.alerts)
    assert sum("reconciliation discrepancy" in r.message for r in caplog.records) == 4
    cash_alert = next(a for a in received if a.item.metric == "cash_and_equivalents")
    assert "VARIANCE_BREACH" in cash_alert.message and "PRIMARY_OVERRIDE" in cash_alert.message


def test_line_item_filter_and_custom_policy() -> None:
    report = _report({"revenue": Decimal(1000), "net_income": Decimal(10)},
                     {"revenue": Decimal(1020), "net_income": Decimal(99)},
                     policy=ConfidencePolicy(tolerance=0.05))
    assert [i.metric for i in report.items] == ["revenue"]  # net_income is not an EV line item
    assert report.items[0].status is ReconStatus.MATCH  # 2 % inside a 5 % tolerance


def _fact(metric: str, end: date, value: str, source: str) -> FundamentalFact:
    return FundamentalFact(symbol="AAPL", metric=metric, period_type=PeriodType.QUARTER, period_end_date=end,
                           filing_timestamp=AS_OF, value=Decimal(value), currency="USD", source=source)


def test_multi_period_facts_and_frame_export() -> None:
    q3, q4 = date(2024, 6, 29), date(2024, 9, 28)
    primary = [_fact("revenue", q3, "85777000000", "SEC_EDGAR"), _fact("revenue", q4, "94930000000", "SEC_EDGAR")]
    secondary = [_fact("revenue", q3, "85777000000", "FMP"), _fact("revenue", q4, "95930000000", "FMP")]
    report = FundamentalReconciler().reconcile(primary, secondary)
    latest = report.resolved("revenue")
    assert latest is not None and latest.period_end_date == q4 and latest.resolved_value == Decimal("94930000000")
    frame = report.to_frame()
    assert {"metric", "period_type", "period_end_date", "value_primary", "value_secondary",
            "relative_variance", "status", "is_break", "confidence_score"} <= set(frame.columns)
    assert frame.filter(frame["is_break"])["period_end_date"].to_list() == [q4]


def test_sides_must_be_single_source_and_point_in_time_resolved() -> None:
    q = date(2024, 9, 28)
    with pytest.raises(ValueError, match="mixes sources"):
        FundamentalReconciler().reconcile(
            [_fact("revenue", q, "1", "SEC_EDGAR"), _fact("ebitda", q, "1", "NSE_XBRL")], [])
    with pytest.raises(ValueError, match="two values"):
        FundamentalReconciler().reconcile([_fact("revenue", q, "1", "SEC_EDGAR")] * 2, [])
