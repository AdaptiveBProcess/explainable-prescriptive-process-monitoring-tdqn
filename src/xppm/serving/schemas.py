from __future__ import annotations

from pydantic import BaseModel


class Event(BaseModel):
    activity: str
    timestamp: str
    amount: float | None = None


class CaseRequest(BaseModel):
    case_id: str
    events: list[Event]


class ActionRecommendation(BaseModel):
    case_id: str
    action: str
    score: float


