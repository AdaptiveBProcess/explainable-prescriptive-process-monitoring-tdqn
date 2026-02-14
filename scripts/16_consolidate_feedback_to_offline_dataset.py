#!/usr/bin/env python3
"""
Consolidate feedback logs → D_offline.npz

Process:
1. Load feedback JSONL
2. Group by case_id
3. Construct (s, a, r, s', done) transitions
4. Export to npz format compatible with training pipeline
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def consolidate_feedback_logs(feedback_log_path: Path, output_path: Path, min_cases: int = 100):
    """
    Consolida feedback logs → D_offline.npz

    Process:
    1. Load feedback JSONL
    2. Group by case_id
    3. Construct (s, a, r, s', done) transitions
    4. Export to npz format compatible with training pipeline
    """
    # Load logs
    transitions = []
    with open(feedback_log_path) as f:
        for line in f:
            if line.strip():
                transitions.append(json.loads(line))

    if not transitions:
        print("❌ No feedback logs found")
        return

    df = pd.DataFrame(transitions)

    # Group by case
    cases = defaultdict(list)
    for _, row in df.iterrows():
        cases[row["case_id"]].append(row.to_dict())

    # Filter cases con suficientes transitions
    valid_cases = {k: v for k, v in cases.items() if len(v) >= 2}

    if len(valid_cases) < min_cases:
        print(f"⚠️ Only {len(valid_cases)} valid cases, need {min_cases}")
        return

    # Build MDP dataset
    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []
    case_ids = []
    ts = []

    for case_id, case_transitions in valid_cases.items():
        # Sort by t
        case_transitions = sorted(case_transitions, key=lambda x: x["t"])

        for i in range(len(case_transitions) - 1):
            curr = case_transitions[i]
            next_trans = case_transitions[i + 1]

            # State (convert dict to array - adapt to your feature order)
            state_feat = curr["state_features"]
            next_state_feat = next_trans["state_features"]

            # Extract features in order (adapt to your schema)
            # For now, assume numeric features only
            state_array = []
            next_state_array = []

            # Common features (adapt to your actual feature names)
            feature_order = [
                "amount",
                "est_quality",
                "unc_quality",
                "cum_cost",
                "elapsed_time",
                "prefix_len",
                "count_validate_application",
                "count_skip_contact",
                "count_contact_headquarters",
            ]

            for feat in feature_order:
                state_array.append(state_feat.get(feat, 0.0))
                next_state_array.append(next_state_feat.get(feat, 0.0))

            # Action (executed, not recommended)
            action = curr["action_executed"]

            # Reward (terminal si done, 0 si no)
            if curr.get("done"):
                reward = curr.get("reward", 0.0)
            else:
                reward = 0.0  # Sparse reward

            states.append(state_array)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state_array)
            dones.append(curr.get("done", False))
            case_ids.append(case_id)
            ts.append(curr["t"])

    # Convert to npz (simplified, adapt to your format)
    dataset = {
        "states": np.array(states, dtype=np.float32),
        "actions": np.array(actions, dtype=np.int32),
        "rewards": np.array(rewards, dtype=np.float32),
        "next_states": np.array(next_states, dtype=np.float32),
        "dones": np.array(dones, dtype=bool),
        "case_ids": np.array(case_ids),
        "ts": np.array(ts, dtype=np.int32),
    }

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **dataset)

    print(f"✅ Consolidated {len(states)} transitions from {len(valid_cases)} cases")
    print(f"   Saved to: {output_path}")

    return dataset


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--feedback-log", default="artifacts/deploy/v1/feedback.jsonl")
    parser.add_argument("--output", default="data/processed/D_offline_incremental.npz")
    parser.add_argument("--min-cases", type=int, default=100)
    args = parser.parse_args()

    consolidate_feedback_logs(Path(args.feedback_log), Path(args.output), args.min_cases)
