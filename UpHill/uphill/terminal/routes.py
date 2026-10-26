from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from quantterm.terminal.command_router import HELP_TEXT, RouterResult, display_name
from quantterm.workspace import Datum, format_value
from uphill.state import CurrentSession

router = APIRouter(prefix="/api/terminal", tags=["terminal"])

_DATUM_KEYS = {"label", "value", "unit", "source", "confidence_score", "timestamp", "provenance"}


class CommandRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command: str = Field(min_length=1, max_length=200)


def with_display(node: Any) -> Any:
    """Adds ``display`` (the TUI's exact formatting) to every serialized ``Datum`` in a view."""
    if isinstance(node, list):
        return [with_display(v) for v in node]
    if not isinstance(node, dict):
        return node
    out = {k: with_display(v) for k, v in node.items()}
    if _DATUM_KEYS <= node.keys():
        try:
            out["display"] = format_value(Datum.model_validate(node))
        except ValidationError:
            pass
    return out


def serialize(result: RouterResult, context: str) -> dict[str, Any]:
    # model_dump_json writes NaN/inf as null, which plain json.dumps would not
    body: dict[str, Any] = with_display(json.loads(result.model_dump_json()))
    body["context"] = context
    if result.security is not None:
        body["display_name"] = display_name(result.security)
        body["listing"] = f"{result.security.exchange.value}:{result.security.symbol}"
    return body


@router.post("/command")
async def command(body: CommandRequest, session: CurrentSession) -> JSONResponse:
    async with session.lock:
        try:
            result = await session.router.execute(body.command)
        except Exception as exc:  # DB/network failure: report to the terminal instead of a bare 500
            result = RouterResult(kind="ERROR", message=f"{type(exc).__name__}: {exc}"[:300])
        context = session.router.context
    return JSONResponse(serialize(result, context))


@router.get("/help")
async def help_text() -> dict[str, str]:
    return {"text": HELP_TEXT}
