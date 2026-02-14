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
