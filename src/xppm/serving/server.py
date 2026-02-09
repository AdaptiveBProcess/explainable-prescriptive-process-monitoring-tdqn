from __future__ import annotations

from fastapi import FastAPI

from xppm.serving.guard import guard_action
from xppm.serving.schemas import ActionRecommendation, CaseRequest


app = FastAPI(title="xPPM TDQN Policy Server")


@app.post("/recommend", response_model=ActionRecommendation)
def recommend_action(req: CaseRequest) -> ActionRecommendation:
    # TODO: load trained policy, compute state, choose action
    dummy_action = guard_action("start_standard")
    return ActionRecommendation(case_id=req.case_id, action=dummy_action, score=0.0)



