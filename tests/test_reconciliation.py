from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from quantterm.reconciliation import ReconStatus, reconcile

PE = date(2024, 9, 30)


def _frame(rows: list[tuple[str, str, float, str]], source: str) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": [r[0] for r in rows],
            "metric": [r[1] for r in rows],
            "period_type": ["Q"] * len(rows),
            "period_end_date": [PE] * len(rows),
            "value": [r[2] for r in rows],
            "currency": [r[3] for r in rows],
            "source": [source] * len(rows),
        }
    )


@pytest.fixture
def result() -> pl.DataFrame:
    primary = _frame(
        [
            ("AAPL", "revenue", 94_930_000_000.0, "USD"),
            ("AAPL", "net_income", 14_736_000_000.0, "USD"),
            ("TCS", "revenue", 642_590_000_000.0, "INR"),  # rupees in XBRL
            ("AAPL", "other_income", -19_000_000.0, "USD"),
            ("AAPL", "eps_diluted", 0.97, "USD"),
            ("HDFCBANK", "nii", 301_140_000_000.0, "INR"),
            ("AAPL", "minority_interest", 0.0, "USD"),
            ("AAPL", "capex", 2_908_000_000.0, "USD"),
        ],
        "SEC_EDGAR",
    )
    secondary = _frame(
        [
            ("AAPL", "revenue", 94_930_000_000.0 * 1.004, "USD"),  # 0.4 % -> inside tolerance
            ("AAPL", "net_income", 14_736_000_000.0 * 1.006, "USD"),  # 0.6 % -> breach
            ("TCS", "revenue", 64_259.0, "INR"),  # vendor reports in crores
            ("AAPL", "other_income", 19_000_000.0, "USD"),
            ("AAPL", "eps_diluted", 0.97, "INR"),
            ("AAPL", "minority_interest", 0.2, "USD"),
            ("AAPL", "gross_margin", 0.462, "USD"),
            ("AAPL", "capex", 2_908_000_000.0 * 0.9951, "USD"),  # 0.49 % -> just inside
        ],
        "FMP",
    )
    return reconcile(primary, secondary)


def _status(df: pl.DataFrame, symbol: str, metric: str) -> str:
    return str(df.filter((pl.col("symbol") == symbol) & (pl.col("metric") == metric))["status"].item())


@pytest.mark.parametrize(
    ("symbol", "metric", "expected"),
    [
        ("AAPL", "revenue", ReconStatus.MATCH),
        ("AAPL", "net_income", ReconStatus.VARIANCE_BREACH),
        ("TCS", "revenue", ReconStatus.SCALE_MISMATCH),
        ("AAPL", "other_income", ReconStatus.SIGN_MISMATCH),
        ("AAPL", "eps_diluted", ReconStatus.CURRENCY_MISMATCH),
        ("HDFCBANK", "nii", ReconStatus.MISSING_SECONDARY),
        ("AAPL", "gross_margin", ReconStatus.MISSING_PRIMARY),
        ("AAPL", "minority_interest", ReconStatus.MATCH),
        ("AAPL", "capex", ReconStatus.MATCH),
    ],
)
def test_statuses(result: pl.DataFrame, symbol: str, metric: str, expected: ReconStatus) -> None:
    assert _status(result, symbol, metric) == expected.value


def test_relative_variance_and_break_flag(result: pl.DataFrame) -> None:
    row = result.filter((pl.col("symbol") == "AAPL") & (pl.col("metric") == "net_income"))
    assert row["relative_variance"].item() == pytest.approx(0.006)
    assert row["is_break"].item() is True
    assert result.filter(pl.col("status") == "MATCH")["is_break"].to_list() == [False, False, False]


def test_duplicate_keys_rejected() -> None:
    dup = _frame([("AAPL", "revenue", 1.0, "USD"), ("AAPL", "revenue", 2.0, "USD")], "FMP")
    with pytest.raises(ValueError, match="duplicate"):
        reconcile(dup, dup)
