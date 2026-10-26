"""Corporate-action model and Bloomberg-convention adjustment factors.

``adjustment_factor`` multiplies every price strictly before ``ex_date``; volume is divided by it.
  * SPLIT n-for-m: m / n                         (2-for-1 -> 0.5, 1-for-10 reverse split -> 10)
  * BONUS a:b (a bonus shares per b held):  b / (a + b)
  * RIGHTS a:b at subscription S, cum-rights close P: TERP / P with TERP = (b*P + a*S) / (a + b)
  * CASH_DIVIDEND D: (P - D) / P on the prior raw close, derived at query time by ``engine.adjustments``.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from quantterm.models import Exchange

_FACTOR_QUANTUM = Decimal("1e-12")  # NUMERIC(24, 12) in corporate_actions.adjustment_factor


class CorporateActionType(StrEnum):
    SPLIT = "SPLIT"
    BONUS = "BONUS"
    RIGHTS = "RIGHTS"
    CASH_DIVIDEND = "CASH_DIVIDEND"


CAPITAL_CHANGE_TYPES: frozenset[CorporateActionType] = frozenset(
    {CorporateActionType.SPLIT, CorporateActionType.BONUS, CorporateActionType.RIGHTS}
)


def _positive(**values: Decimal) -> None:
    for name, v in values.items():
        if v <= 0:
            raise ValueError(f"{name} must be positive, got {v}")


def split_factor(new_shares: Decimal, old_shares: Decimal) -> Decimal:
    _positive(new_shares=new_shares, old_shares=old_shares)
    return (old_shares / new_shares).quantize(_FACTOR_QUANTUM)


def bonus_factor(bonus_shares: Decimal, held_shares: Decimal) -> Decimal:
    _positive(bonus_shares=bonus_shares, held_shares=held_shares)
    return (held_shares / (bonus_shares + held_shares)).quantize(_FACTOR_QUANTUM)


def rights_factor(
    rights_shares: Decimal, held_shares: Decimal, subscription_price: Decimal, cum_rights_close: Decimal
) -> Decimal:
    """TERP / cum-rights close. Rights priced at or above market carry no value, hence no adjustment."""
    _positive(rights_shares=rights_shares, held_shares=held_shares, cum_rights_close=cum_rights_close)
    if subscription_price < 0:
        raise ValueError("subscription_price cannot be negative")
    if subscription_price >= cum_rights_close:
        return Decimal(1)
    terp = (held_shares * cum_rights_close + rights_shares * subscription_price) / (held_shares + rights_shares)
    return (terp / cum_rights_close).quantize(_FACTOR_QUANTUM)


class CorporateAction(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str = Field(min_length=1)
    exchange: Exchange
    action_type: CorporateActionType
    ex_date: date
    announced_at: datetime
    adjustment_factor: Decimal | None = Field(default=None, gt=0)
    raw_dividend_amount: Decimal | None = Field(default=None, gt=0)
    source: str = Field(min_length=1)

    @model_validator(mode="after")
    def _shape(self) -> CorporateAction:
        if self.announced_at.tzinfo is None or self.announced_at.utcoffset() is None:
            raise ValueError("announced_at must be timezone-aware")
        if self.action_type is CorporateActionType.CASH_DIVIDEND:
            if self.raw_dividend_amount is None or self.adjustment_factor is not None:
                raise ValueError("CASH_DIVIDEND requires raw_dividend_amount only")
        else:
            if self.adjustment_factor is None or self.raw_dividend_amount is not None:
                raise ValueError(f"{self.action_type} requires adjustment_factor only")
            if self.action_type is not CorporateActionType.SPLIT and self.adjustment_factor > 1:
                raise ValueError(f"{self.action_type} dilutes: adjustment_factor must be <= 1")
        return self
