"""
xppm-serve — FastAPI serving layer for xppm policies.

Install with:
    pip install xppm[serve]

Usage:
    uvicorn xppm_serve.server:app --host 0.0.0.0 --port 8000
"""

from xppm_serve.guard import PolicyGuard
from xppm_serve.logger import DecisionLogger, FeedbackLogger
from xppm_serve.schemas import (
    CaseFeatures,
    DecisionRequest,
    DecisionResponse,
    Explanation,
    FidelityBadge,
    PolicyVersions,
)

__all__ = [
    "PolicyGuard",
    "DecisionLogger",
    "FeedbackLogger",
    "CaseFeatures",
    "DecisionRequest",
    "DecisionResponse",
    "Explanation",
    "FidelityBadge",
    "PolicyVersions",
]
