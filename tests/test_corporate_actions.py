from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from quantterm.corporate_actions import (
    CorporateAction,
    CorporateActionType,
    bonus_factor,
    rights_factor,
    split_factor,
)
from quantterm.models import Exchange

ANNOUNCED = datetime(2024, 5, 22, tzinfo=UTC)


def test_split_factors_forward_and_reverse() -> None:
    assert split_factor(Decimal(10), Decimal(1)) == Decimal("0.100000000000")  # NVDA 10-for-1, 2024
    assert split_factor(Decimal(1), Decimal(10)) == Decimal(10)  # 1-for-10 reverse split
    assert split_factor(Decimal(3), Decimal(1)) == Decimal("0.333333333333")  # quantised to NUMERIC(24,12)


def test_bonus_factor_indian_convention() -> None:
    assert bonus_factor(Decimal(1), Decimal(1)) == Decimal("0.5")  # RELIANCE 1:1, Oct 2024
    assert bonus_factor(Decimal(3), Decimal(2)) == Decimal("0.4")  # 3 bonus for every 2 held


def test_rights_factor_uses_terp_over_cum_close() -> None:
    # 1 right per 15 held at 1257 with a 1445 cum-rights close: TERP = (15*1445 + 1257) / 16 = 1433.25.
    factor = rights_factor(Decimal(1), Decimal(15), Decimal(1257), Decimal(1445))
    assert float(factor) == pytest.approx(1433.25 / 1445, rel=1e-12)
    assert factor < 1


def test_out_of_the_money_rights_need_no_adjustment() -> None:
    assert rights_factor(Decimal(1), Decimal(4), Decimal(120), Decimal(100)) == Decimal(1)


@pytest.mark.parametrize("bad", [Decimal(0), Decimal(-1)])
def test_factor_inputs_must_be_positive(bad: Decimal) -> None:
    with pytest.raises(ValueError):
        split_factor(bad, Decimal(1))
    with pytest.raises(ValueError):
        rights_factor(Decimal(1), Decimal(1), Decimal(1), bad)


def _action(**kw: object) -> CorporateAction:
    base: dict[str, object] = dict(symbol="NVDA", exchange=Exchange.NASDAQ, ex_date=date(2024, 6, 10),
                                   announced_at=ANNOUNCED, source="NASDAQ_CA")
    return CorporateAction.model_validate({**base, **kw})


def test_action_shape_validation() -> None:
    assert _action(action_type=CorporateActionType.SPLIT, adjustment_factor=Decimal("0.1")).adjustment_factor
    with pytest.raises(ValueError, match="adjustment_factor only"):
        _action(action_type=CorporateActionType.SPLIT, raw_dividend_amount=Decimal(1))
    with pytest.raises(ValueError, match="raw_dividend_amount only"):
        _action(action_type=CorporateActionType.CASH_DIVIDEND, adjustment_factor=Decimal("0.99"))
    with pytest.raises(ValueError, match="dilutes"):
        _action(action_type=CorporateActionType.BONUS, adjustment_factor=Decimal(2))
    with pytest.raises(ValueError, match="timezone-aware"):
        _action(action_type=CorporateActionType.SPLIT, adjustment_factor=Decimal("0.1"),
                announced_at=datetime(2024, 5, 22))
