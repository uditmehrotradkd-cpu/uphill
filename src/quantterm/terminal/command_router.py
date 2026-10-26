"""Bloomberg-style command line: ``<TICKER> <CODE> Equity [FUNCTION [args]] [<GO>]``.

Examples::

    RELIANCE IN Equity            AAPL US Equity GP 2Y
    GP 6M NODIV                   FA Q 8
    RV                            OVME P 250 2025-01-17 vol=24% rate=4.3%
    ASOF 2024-06-28               ASOF NOW
    WATCH ADD INFY IN Equity      BACK      HELP

Loading a security keeps the active function, so ``MSFT US Equity`` while in FA re-renders FA for MSFT.
``ASOF`` pins every view to a past instant: all reads are point-in-time, never revised hindsight.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime, time
from enum import StrEnum
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from quantterm.db.repositories import SecurityRecord
from quantterm.models import Exchange, Market, PeriodType
from quantterm.quant.derivatives_ovme import OptionType
from quantterm.workspace import FAView, GPView, OVMERequest, OVMEView, RVView, WorkspaceError

# Bloomberg exchange codes -> candidate venues (composite codes span several).
EXCHANGE_CODES: dict[str, tuple[Exchange, ...]] = {
    "US": (Exchange.NASDAQ, Exchange.NYSE),
    "UN": (Exchange.NYSE,),
    "UW": (Exchange.NASDAQ,),
    "UQ": (Exchange.NASDAQ,),
    "IN": (Exchange.NSE,),
    "IS": (Exchange.NSE,),
    "IB": (Exchange.BSE,),
    **{e.value: (e,) for e in Exchange},
}
YELLOW_KEYS = {"EQUITY", "EQ"}
GP_PERIODS: dict[str, int] = {"1M": 31, "3M": 92, "6M": 183, "1Y": 366, "2Y": 731, "3Y": 1096, "5Y": 1827, "10Y": 3653}


def bloomberg_code(exchange: Exchange) -> str:
    return "US" if exchange.market is Market.US else ("IB" if exchange is Exchange.BSE else "IN")


def display_name(sec: SecurityRecord) -> str:
    return f"{sec.symbol} {bloomberg_code(sec.exchange)} Equity"


class Function(StrEnum):
    GP = "GP"
    FA = "FA"
    RV = "RV"
    OVME = "OVME"


class CommandError(ValueError):
    pass


class SecurityRef(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    code: str
    exchanges: tuple[Exchange, ...]


class GPArgs(BaseModel):
    model_config = ConfigDict(frozen=True)

    lookback_days: int = 366
    adjust_splits: bool = True
    adjust_dividends: bool = True


class FAArgs(BaseModel):
    model_config = ConfigDict(frozen=True)

    period_type: PeriodType = PeriodType.FISCAL_YEAR
    periods: int = Field(default=5, ge=1, le=20)


class RVArgs(BaseModel):
    model_config = ConfigDict(frozen=True)


FunctionArgs = GPArgs | FAArgs | RVArgs | OVMERequest


class Command(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["RUN", "ASOF", "BACK", "HELP", "WATCH"]
    security: SecurityRef | None = None
    function: Function | None = None
    args: FunctionArgs | None = None
    as_of: datetime | None = None
    watch_action: Literal["ADD", "DEL"] | None = None


_PCT = re.compile(r"^(-?\d+(?:\.\d+)?)%$")


def _number(token: str, field: str) -> float:
    m = _PCT.match(token)
    try:
        return float(m.group(1)) / 100 if m else float(token)
    except ValueError as exc:
        raise CommandError(f"{field}: '{token}' is not a number") from exc


def _parse_security(tokens: list[str]) -> tuple[SecurityRef | None, list[str]]:
    """Consume ``TICKER CODE EQUITY`` from the front of ``tokens`` if present."""
    if len(tokens) >= 3 and tokens[2].upper() in YELLOW_KEYS:
        code = tokens[1].upper()
        if code not in EXCHANGE_CODES:
            raise CommandError(f"unknown exchange code '{tokens[1]}' (try US, UN, UW, IN, IB)")
        return SecurityRef(ticker=tokens[0].upper(), code=code, exchanges=EXCHANGE_CODES[code]), tokens[3:]
    return None, tokens


def _parse_gp(args: list[str]) -> GPArgs:
    lookback, splits, dividends = 366, True, True
    for a in (x.upper() for x in args):
        if a in GP_PERIODS:
            lookback = GP_PERIODS[a]
        elif a == "YTD":
            today = datetime.now(UTC).date()
            lookback = (today - date(today.year, 1, 1)).days + 1
        elif a == "NODIV":
            dividends = False
        elif a == "RAW":
            splits = dividends = False
        else:
            raise CommandError(f"GP: unknown argument '{a}' (periods {', '.join(GP_PERIODS)}, YTD; NODIV, RAW)")
    return GPArgs(lookback_days=lookback, adjust_splits=splits, adjust_dividends=dividends)


def _parse_fa(args: list[str]) -> FAArgs:
    period_type, periods = PeriodType.FISCAL_YEAR, 5
    for a in (x.upper() for x in args):
        if a in ("A", "FY", "ANNUAL"):
            period_type = PeriodType.FISCAL_YEAR
        elif a in ("Q", "QTR", "QUARTERLY"):
            period_type = PeriodType.QUARTER
        elif a in ("H", "SEMI"):
            period_type = PeriodType.HALF_YEAR
        elif a.isdigit():
            periods = int(a)
        else:
            raise CommandError(f"FA: unknown argument '{a}' (A, Q, H, number of periods)")
    return FAArgs(period_type=period_type, periods=periods)


_OVME_KEYS = {"VOL": "volatility", "IV": "volatility", "RATE": "rate", "R": "rate", "DIV": "dividend_yield",
              "Q": "dividend_yield", "PX": "premium", "PREMIUM": "premium"}


def _parse_ovme(args: list[str]) -> OVMERequest:
    fields: dict[str, object] = {}
    for raw in args:
        a = raw.upper()
        if "=" in a:
            key, _, value = a.partition("=")
            if key not in _OVME_KEYS:
                raise CommandError(f"OVME: unknown input '{key}' (vol=, rate=, div=, px=)")
            name = _OVME_KEYS[key]
            fields[name] = _number(value, name) if name != "premium" else float(value)
        elif a in ("C", "CALL"):
            fields["option_type"] = OptionType.CALL
        elif a in ("P", "PUT"):
            fields["option_type"] = OptionType.PUT
        elif re.fullmatch(r"\d{4}-\d{2}-\d{2}", a):
            fields["expiry"] = date.fromisoformat(a)
        else:
            fields["strike"] = _number(a, "strike")
    try:
        return OVMERequest.model_validate(fields)
    except ValueError as exc:
        raise CommandError(f"OVME: {exc}") from exc


def parse(text: str) -> Command:
    tokens = [t for t in text.strip().split() if t.upper() not in ("<GO>", "GO")]
    if not tokens:
        raise CommandError("empty command")
    head = tokens[0].upper()
    if head == "HELP":
        return Command(kind="HELP")
    if head == "BACK":
        return Command(kind="BACK")
    if head == "ASOF":
        if len(tokens) != 2:
            raise CommandError("usage: ASOF YYYY-MM-DD | ASOF NOW")
        if tokens[1].upper() == "NOW":
            return Command(kind="ASOF", as_of=None)
        try:
            d = date.fromisoformat(tokens[1])
        except ValueError as exc:
            raise CommandError(f"ASOF: '{tokens[1]}' is not YYYY-MM-DD") from exc
        # End of that UTC day: everything published on the date is visible, nothing after it.
        return Command(kind="ASOF", as_of=datetime.combine(d, time.max, tzinfo=UTC))
    if head == "WATCH":
        if len(tokens) < 2 or tokens[1].upper() not in ("ADD", "DEL"):
            raise CommandError("usage: WATCH ADD|DEL <TICKER> <CODE> Equity")
        ref, rest = _parse_security(tokens[2:])
        if ref is None or rest:
            raise CommandError("usage: WATCH ADD|DEL <TICKER> <CODE> Equity")
        action: Literal["ADD", "DEL"] = "ADD" if tokens[1].upper() == "ADD" else "DEL"
        return Command(kind="WATCH", security=ref, watch_action=action)

    ref, rest = _parse_security(tokens)
    if not rest:
        if ref is None:
            raise CommandError(f"unrecognised command '{text.strip()}' (type HELP)")
        return Command(kind="RUN", security=ref)
    name = rest[0].upper()
    if name not in Function.__members__:
        raise CommandError(f"unknown function '{rest[0]}' (GP, FA, RV, OVME)")
    function = Function(name)
    parsers: dict[Function, Callable[[list[str]], FunctionArgs]] = {
        Function.GP: _parse_gp, Function.FA: _parse_fa, Function.RV: _no_args, Function.OVME: _parse_ovme,
    }
    return Command(kind="RUN", security=ref, function=function, args=parsers[function](rest[1:]))


def _no_args(args: list[str]) -> RVArgs:
    if args:
        raise CommandError(f"RV takes no arguments, got {' '.join(args)}")
    return RVArgs()


HELP_TEXT = __doc__ or ""


# ------------------------------------------------------------------------------------------ router
class WorkspaceLike(Protocol):
    async def resolve(self, ticker: str, exchanges: Sequence[Exchange]) -> SecurityRecord: ...

    async def gp(self, sec: SecurityRecord, as_of: datetime, *, lookback_days: int, adjust_splits: bool,
                 adjust_dividends: bool) -> GPView: ...

    async def fa(self, sec: SecurityRecord, as_of: datetime, *, period_type: PeriodType, periods: int) -> FAView: ...

    async def rv(self, sec: SecurityRecord, as_of: datetime) -> RVView: ...

    async def ovme(self, sec: SecurityRecord, as_of: datetime, request: OVMERequest | None = None) -> OVMEView: ...


View = GPView | FAView | RVView | OVMEView


class RouterResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["VIEW", "STATUS", "HELP", "WATCH", "ERROR"]
    message: str
    function: Function | None = None
    view: View | None = None
    security: SecurityRecord | None = None
    watch_action: Literal["ADD", "DEL"] | None = None


class _Frame(BaseModel):
    model_config = ConfigDict(frozen=True)

    security: SecurityRecord
    function: Function
    args: FunctionArgs


class CommandRouter:
    """Holds the active security, function, arguments and as-of; turns command text into loaded views."""

    def __init__(self, workspace: WorkspaceLike, clock: Callable[[], datetime] = lambda: datetime.now(UTC),
                 history_limit: int = 50) -> None:
        self._ws = workspace
        self._clock = clock
        self.security: SecurityRecord | None = None
        self.function: Function | None = None
        self.args: FunctionArgs | None = None
        self.as_of: datetime | None = None
        self._history: list[_Frame] = []
        self._history_limit = history_limit

    @property
    def effective_as_of(self) -> datetime:
        now = self._clock()
        return self.as_of if self.as_of is not None and self.as_of < now else now

    @property
    def context(self) -> str:
        sec = display_name(self.security) if self.security else "no security"
        when = f"as of {self.as_of:%Y-%m-%d}" if self.as_of else "live"
        return f"{sec} | {self.function.value if self.function else '-'} | {when}"

    async def execute(self, text: str) -> RouterResult:
        try:
            command = parse(text)
            return await self._dispatch(command)
        except (CommandError, WorkspaceError, LookupError, ValueError) as exc:
            return RouterResult(kind="ERROR", message=str(exc))

    async def _dispatch(self, command: Command) -> RouterResult:
        if command.kind == "HELP":
            return RouterResult(kind="HELP", message=HELP_TEXT)
        if command.kind == "ASOF":
            self.as_of = command.as_of
            if self.security and self.function and self.args is not None:
                return await self._run(self.security, self.function, self.args, record=False)
            return RouterResult(kind="STATUS", message=f"context: {self.context}")
        if command.kind == "BACK":
            if len(self._history) < 2:
                return RouterResult(kind="STATUS", message="no previous screen")
            self._history.pop()
            frame = self._history.pop()
            return await self._run(frame.security, frame.function, frame.args)
        if command.kind == "WATCH":
            assert command.security is not None and command.watch_action is not None
            sec = await self._ws.resolve(command.security.ticker, command.security.exchanges)
            return RouterResult(kind="WATCH", message=f"watchlist {command.watch_action} {display_name(sec)}",
                                security=sec, watch_action=command.watch_action)

        if command.security is not None:
            self.security = await self._ws.resolve(command.security.ticker, command.security.exchanges)
        if command.function is None:
            assert self.security is not None
            if self.function is not None and self.args is not None:
                return await self._run(self.security, self.function, self.args)
            return RouterResult(kind="STATUS", message=f"loaded {display_name(self.security)}", security=self.security)
        if self.security is None:
            raise CommandError("no active security: type e.g. 'AAPL US Equity' first")
        assert command.args is not None
        return await self._run(self.security, command.function, command.args)

    async def _run(self, sec: SecurityRecord, function: Function, args: FunctionArgs, record: bool = True) -> RouterResult:
        as_of = self.effective_as_of
        view: View
        if function is Function.GP:
            assert isinstance(args, GPArgs)
            view = await self._ws.gp(sec, as_of, lookback_days=args.lookback_days, adjust_splits=args.adjust_splits,
                                     adjust_dividends=args.adjust_dividends)
        elif function is Function.FA:
            assert isinstance(args, FAArgs)
            view = await self._ws.fa(sec, as_of, period_type=args.period_type, periods=args.periods)
        elif function is Function.RV:
            view = await self._ws.rv(sec, as_of)
        else:
            assert isinstance(args, OVMERequest)
            view = await self._ws.ovme(sec, as_of, args)
        self.security, self.function, self.args = sec, function, args
        if record:
            self._history.append(_Frame(security=sec, function=function, args=args))
            del self._history[:-self._history_limit]
        return RouterResult(kind="VIEW", message=self.context, function=function, view=view, security=sec)
