"""Decision logging for auditing."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict


class DecisionLogger:
    """Logs decisions to JSONL for auditing."""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_decision(self, request: Dict, response: Dict):
        """Log a single decision."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request["request_id"],
            "case_id": request["case_id"],
            "t": request["t"],
            "action_id": response["action_id"],
            "action_name": response["action_name"],
            "source": response["source"],
            "confidence": response["confidence"],
            "ood": response["ood"],
            "latency_ms": response["latency_ms"],
        }

        with open(self.log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")


class FeedbackLogger:
    """
    Logs decisions with full state/action/reward for RL reconstruction.
    """

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_transition(
        self,
        case_id: str,
        t: int,
        state_features: Dict,
        action_recommended: int,
        action_executed: int,
        source: str,
        reward: float = None,
        next_state_features: Dict = None,
        done: bool = False,
    ):
        """Log a single RL transition."""
        transition = {
            "case_id": case_id,
            "t": t,
            "state_features": state_features,
            "action_recommended": action_recommended,
            "action_executed": action_executed,
            "source": source,
            "reward": reward,
            "next_state_features": next_state_features,
            "done": done,
            "timestamp": datetime.utcnow().isoformat(),
        }

        with open(self.log_path, "a") as f:
            f.write(json.dumps(transition) + "\n")
