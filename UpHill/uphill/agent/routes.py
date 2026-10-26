from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from quantterm.workspace import format_value
from uphill.state import CurrentSession, State

router = APIRouter(prefix="/api/agent", tags=["agent"])


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=1000)


class CitedFact(BaseModel):
    id: str
    label: str
    display: str
    source: str
    confidence: float
    timestamp: datetime | None
    note: str | None


class AskResponse(BaseModel):
    text: str
    fallback: bool
    steps: int
    tool_calls: list[str]
    cited: list[CitedFact]
    context: str


@router.get("/status")
async def agent_status(state: State) -> dict[str, object]:
    return {"enabled": state.agent_enabled, "model": state.settings.agent_model}


@router.post("/ask", response_model=AskResponse)
async def ask(body: AskRequest, state: State, session: CurrentSession) -> AskResponse:
    if session.agent is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                            "agent disabled: set QT_AGENT_API_KEY (Gemini key) in .env and restart UpHill")
    async with session.lock:
        answer = await session.agent.ask(body.question.strip())
        context = session.router.context
    return AskResponse(
        text=answer.text, fallback=answer.fallback, steps=answer.steps, tool_calls=answer.tool_calls, context=context,
        cited=[CitedFact(id=f.id, label=f.datum.label, display=format_value(f.datum), source=f.datum.source,
                         confidence=f.datum.confidence_score, timestamp=f.datum.timestamp, note=f.datum.note)
               for f in answer.cited],
    )


@router.post("/reset", status_code=status.HTTP_204_NO_CONTENT)
async def reset(session: CurrentSession) -> None:
    if session.agent is not None:
        async with session.lock:
            session.agent.reset()
