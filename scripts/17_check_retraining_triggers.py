#!/usr/bin/env python3
"""
Check retraining triggers:
1. Drift detected (OOD spike, coverage drop)
2. Expected gain drop (if available)
3. Override rate spike
"""

import json
from datetime import datetime
from pathlib import Path


def check_retraining_triggers(metrics_dir: Path):
    """
    Check retraining triggers:
    1. Drift detected (OOD spike, coverage drop)
    2. Expected gain drop (if available)
    3. Override rate spike
    """
    # Load drift report
    drift_path = metrics_dir / "drift_report.json"
    if not drift_path.exists():
        print("❌ No drift report found")
        return False

    with open(drift_path) as f:
        drift = json.load(f)

    # Triggers
    triggers = []

    # Trigger 1: Drift detected
    if drift["drift_detected"]:
        triggers.append(
            {
                "type": "drift",
                "reason": "OOD/coverage/override spike detected",
                "details": drift["signals"],
            }
        )

    # Trigger 2: Expected gain drop (TODO: implement if available)
    # if expected_gain_drop > 0.3:
    #     triggers.append({'type': 'gain_drop', ...})

    # Decision
    should_retrain = len(triggers) > 0

    if should_retrain:
        # Create retraining ticket
        ticket = {
            "timestamp": datetime.now().isoformat(),
            "triggers": triggers,
            "status": "pending",
            "action_required": "Review monitoring and initiate retraining pipeline",
        }

        ticket_path = metrics_dir / "retraining_ticket.json"
        with open(ticket_path, "w") as f:
            json.dump(ticket, f, indent=2)

        print("🔔 Retraining triggered")
        print(f"   Ticket: {ticket_path}")
        for trigger in triggers:
            print(f"   - {trigger['type']}: {trigger['reason']}")
    else:
        print("✅ No retraining triggers")

    return should_retrain


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-dir", default="artifacts/monitoring")
    args = parser.parse_args()

    check_retraining_triggers(Path(args.metrics_dir))
