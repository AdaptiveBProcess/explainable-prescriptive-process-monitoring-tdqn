"""Pydantic schemas for Decision Support System API."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Literal, Optional

import numpy as np
from pydantic import BaseModel, Field, validator


class CaseFeatures(BaseModel):
    """Features tabulares del caso (compatible con surrogate)."""

    amount: float = Field(..., description="Loan amount", ge=0)
    est_quality: float = Field(..., description="Estimated quality", ge=0, le=1)
    unc_quality: float = Field(..., description="Quality uncertainty", ge=0, le=1)
    cum_cost: float = Field(..., description="Cumulative cost", ge=0)
    elapsed_time: float = Field(..., description="Elapsed time (days)", ge=0)
    prefix_len: int = Field(..., description="Prefix length", ge=1)

    # Activity counts
    count_validate_application: int = Field(0, ge=0)
    count_skip_contact: int = Field(0, ge=0)
    count_contact_headquarters: int = Field(0, ge=0)

    @validator("*", pre=True)
    def check_finite(cls, v):
        """Verifica que no haya NaN/Inf."""
        if isinstance(v, float) and not np.isfinite(v):
            raise ValueError(f"Feature must be finite, got {v}")
        return v


class DecisionRequest(BaseModel):
    """Request para decision endpoint."""

    request_id: str = Field(..., description="Unique request identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    case_id: str = Field(..., description="Case identifier")
    t: int = Field(..., description="Step/position in case", ge=0)

    features: CaseFeatures

    # Opcionales
    valid_actions: Optional[List[int]] = Field(None, description="Valid action IDs")
    override: Optional[Dict] = Field(None, description="Human override {action_id, reason}")
    context: Optional[Dict] = Field(None, description="Additional metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "request_id": "req_12345",
                "case_id": "case_789",
                "t": 3,
                "features": {
                    "amount": 15000.0,
                    "est_quality": 0.65,
                    "unc_quality": 0.15,
                    "cum_cost": 250.0,
                    "elapsed_time": 4.5,
                    "prefix_len": 3,
                    "count_validate_application": 2,
                    "count_skip_contact": 1,
                    "count_contact_headquarters": 0,
                },
                "valid_actions": [0, 1, 2],
            }
        }


class PolicyVersions(BaseModel):
    """Versioning information."""

    model_version: str = Field(..., description="Model checkpoint hash (SHA256)")
    surrogate_version: str = Field(..., description="Surrogate tree hash")
    data_version: str = Field(..., description="Training data hash")
    config_version: str = Field(..., description="Config file hash")
    git_commit: str = Field(..., description="Git commit SHA")
    deployed_at: datetime = Field(..., description="Deployment timestamp")


class Explanation(BaseModel):
    """XAI explanation snippet."""

    top_drivers: List[Dict[str, float]] = Field(
        ..., description="Top feature drivers {feature: importance}"
    )
    delta_q: Optional[float] = Field(None, description="ΔQ vs alternative")
    risk_score: Optional[float] = Field(None, description="Risk/value score")


class FidelityBadge(BaseModel):
    """Fidelity badge from offline tests."""

    q_drop_pass: bool = Field(..., description="Q-drop test passed")
    action_flip_pass: bool = Field(..., description="Action-flip test passed")
    rank_consistency: Optional[float] = Field(None, description="Spearman ρ")
    overall_status: Literal["PASS", "CAUTION", "FAIL"]


class DecisionResponse(BaseModel):
    """Response from decision endpoint."""

    request_id: str
    case_id: str
    t: int

    # Decision
    action_id: int = Field(..., description="Recommended action ID")
    action_name: str = Field(..., description="Action name")
    source: Literal["surrogate", "teacher", "baseline", "override"]

    # Confidence & uncertainty
    confidence: float = Field(..., ge=0, le=1, description="Decision confidence")
    uncertainty: float = Field(..., ge=0, le=1, description="Decision uncertainty")
    ood: bool = Field(..., description="Out-of-distribution flag")

    # Explanations
    explanation: Optional[Explanation] = None

    # Fidelity
    fidelity_badge: Optional[FidelityBadge] = None

    # Versions
    versions: PolicyVersions

    # Shadow (opcional)
    shadow: Optional[Dict] = Field(None, description="Teacher decision for comparison")

    # Metadata
    latency_ms: float = Field(..., description="Processing time (ms)")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "request_id": "req_12345",
                "case_id": "case_789",
                "t": 3,
                "action_id": 1,
                "action_name": "contact_headquarters",
                "source": "surrogate",
                "confidence": 0.89,
                "uncertainty": 0.11,
                "ood": False,
                "explanation": {
                    "top_drivers": [
                        {"elapsed_time": 0.45},
                        {"count_validate_application": 0.32},
                    ],
                    "delta_q": 125.3,
                },
                "fidelity_badge": {
                    "q_drop_pass": True,
                    "action_flip_pass": True,
                    "overall_status": "PASS",
                },
                "versions": {
                    "model_version": "abc123...",
                    "surrogate_version": "def456...",
                    "data_version": "ghi789...",
                    "config_version": "jkl012...",
                    "git_commit": "a1b2c3d",
                    "deployed_at": "2026-02-14T10:30:00Z",
                },
                "latency_ms": 1.2,
            }
        }
