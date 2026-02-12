from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from xppm.utils.io import load_npz, load_parquet, save_npz
from xppm.utils.logging import ensure_dir, get_logger

logger = get_logger(__name__)


@dataclass
class Prefixes:
    """Container for prefix data."""

    X: np.ndarray  # [N, max_len] int32
    mask: np.ndarray  # [N, max_len] uint8
    case_ptr: np.ndarray  # [N] int32
    t_ptr: np.ndarray  # [N] int32
    ts_last: np.ndarray  # [N] int64


def load_prefixes(path: str | Path) -> Prefixes:
    """Load prefixes from NPZ file.

    Args:
        path: Path to prefixes.npz

    Returns:
        Prefixes dataclass
    """
    data = load_npz(path)
    return Prefixes(
        X=data["X"],
        mask=data["mask"],
        case_ptr=data["case_ptr"],
        t_ptr=data["t_ptr"],
        ts_last=data.get("ts_last", np.zeros(len(data["X"]), dtype=np.int64)),
    )


def group_by_case(prefixes: Prefixes) -> dict[int, list[int]]:
    """Group prefix indices by case_id.

    Args:
        prefixes: Prefixes container

    Returns:
        Dictionary mapping case_id to list of prefix indices
    """
    case_groups: dict[int, list[int]] = {}
    for idx, case_id in enumerate(prefixes.case_ptr):
        if case_id not in case_groups:
            case_groups[case_id] = []
        case_groups[case_id].append(idx)
    return case_groups


def get_last_activity(prefix_X: np.ndarray, prefix_mask: np.ndarray, id2token: list[str]) -> str:
    """Get the last activity name from a prefix.

    Args:
        prefix_X: Prefix token IDs [max_len]
        prefix_mask: Prefix mask [max_len]
        id2token: Vocabulary mapping ID to token

    Returns:
        Activity name (or UNK if not found)
    """
    # Get non-padding tokens
    real_tokens = prefix_X[prefix_mask == 1]
    if len(real_tokens) == 0:
        return "<UNK>"
    last_token_id = int(real_tokens[-1])
    if last_token_id < len(id2token):
        return id2token[last_token_id]
    return "<UNK>"


def is_decision_point(last_activity: str, config: dict[str, Any]) -> bool:
    """Check if a prefix represents a decision point.

    Args:
        last_activity: Last activity name in the prefix
        config: MDP configuration

    Returns:
        True if this is a decision point
    """
    decision_cfg = config.get("decision_points", {})
    mode = decision_cfg.get("mode", "by_last_activity")

    if mode == "by_last_activity":
        activities = decision_cfg.get("activities", [])
        return last_activity in activities
    elif mode == "all":
        return True
    else:
        return False


def build_action_mask(
    last_activity: str, action_names: list[str], config: dict[str, Any]
) -> np.ndarray:
    """Build action mask for a given state.

    Args:
        last_activity: Last activity name
        action_names: List of action names
        config: MDP configuration

    Returns:
        Binary mask [A] where 1 = valid action
    """
    mask_cfg = config.get("action_mask", {})
    n_actions = len(action_names)
    mask = np.zeros(n_actions, dtype=np.uint8)

    # Get default valid actions
    default_valid = mask_cfg.get("default_valid", ["NOOP"])
    default_ids = [i for i, name in enumerate(action_names) if name in default_valid]

    # Get activity-specific valid actions
    by_activity = mask_cfg.get("by_last_activity", {})
    valid_actions = by_activity.get(last_activity, default_valid)

    # Set mask
    for action_name in valid_actions:
        if action_name in action_names:
            action_id = action_names.index(action_name)
            mask[action_id] = 1

    # If no valid actions found, use default
    if mask.sum() == 0:
        mask[default_ids] = 1

    return mask


def extract_behavior_action_optimized(
    case_id: int,
    current_t: int,
    next_t: int,
    case_activity_map: dict[tuple[int, int], bool],
    action_names: list[str],
) -> int:
    """Extract behavior action from log data (optimized version).

    For time_contact_HQ:
    - If event at position next_t-1 has activity 'contact_headquarters',
      action = 'contact_headquarters'
    - Otherwise, action = 'do_nothing'

    Args:
        case_id: Case ID
        current_t: Current prefix length (1-indexed)
        next_t: Next prefix length (1-indexed)
        case_activity_map: Pre-computed map of (case_id, position) -> has_contact_headquarters
        action_names: List of action names

    Returns:
        Action ID
    """
    # Check if the event at position next_t-1 (0-indexed) has contact_headquarters
    # next_t is 1-indexed (prefix length), so event position is next_t-1
    event_pos = next_t - 1
    if case_activity_map.get((case_id, event_pos), False):
        if "contact_headquarters" in action_names:
            return action_names.index("contact_headquarters")

    # Default: do_nothing
    if "do_nothing" in action_names:
        return action_names.index("do_nothing")
    elif "NOOP" in action_names:
        return action_names.index("NOOP")
    else:
        return 0  # First action as fallback


def extract_behavior_action(
    case_id: int,
    prefix_idx: int,
    next_prefix_idx: int | None,
    prefixes: Prefixes,
    clean_df: pd.DataFrame,
    action_names: list[str],
    id2token: list[str],
    config: dict[str, Any],
) -> int:
    """Extract behavior action from log data (legacy, slower version).

    For time_contact_HQ:
    - If next event has activity 'contact_headquarters', action = 'contact_headquarters'
    - Otherwise, action = 'do_nothing'

    Args:
        case_id: Case ID
        prefix_idx: Current prefix index
        next_prefix_idx: Next prefix index (if exists)
        prefixes: Prefixes container
        clean_df: Clean event log DataFrame
        action_names: List of action names
        id2token: Vocabulary mapping
        config: MDP configuration

    Returns:
        Action ID
    """
    # Get case events
    case_events = clean_df[clean_df["case_id"] == case_id].sort_values("timestamp")

    # Get current prefix info
    current_t = prefixes.t_ptr[prefix_idx]

    # Check if next event has contact_headquarters activity
    if next_prefix_idx is not None:
        next_t = prefixes.t_ptr[next_prefix_idx]
        # Get events between current_t and next_t
        events_in_range = case_events.iloc[current_t - 1 : next_t]
        if len(events_in_range) > 0:
            # Check if contact_headquarters occurred
            if "contact_headquarters" in events_in_range["activity"].values:
                if "contact_headquarters" in action_names:
                    return action_names.index("contact_headquarters")

    # Default: do_nothing
    if "do_nothing" in action_names:
        return action_names.index("do_nothing")
    elif "NOOP" in action_names:
        return action_names.index("NOOP")
    else:
        return 0  # First action as fallback


def compute_reward(
    case_id: int,
    done: bool,
    case_terminal_profit_map: dict[int, float],
    config: dict[str, Any],
) -> float:
    """Compute reward for a transition.

    Args:
        case_id: Case ID
        done: Whether this is a terminal state
        case_terminal_profit_map: Mapping from case_id to terminal profit
        config: MDP configuration

    Returns:
        Reward value
    """
    reward_cfg = config.get("reward", {})
    reward_type = reward_cfg.get("type", "terminal_profit_delayed")

    if reward_type == "terminal_profit_delayed":
        if done:
            return float(case_terminal_profit_map.get(case_id, 0.0))
        else:
            return float(reward_cfg.get("intermediate_reward", 0.0))
    else:
        return 0.0


def build_transitions(
    prefixes: Prefixes,
    clean_df: pd.DataFrame,
    vocab_path: str | Path,
    config: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Build MDP transitions from prefixes.

    Args:
        prefixes: Prefixes container
        clean_df: Clean event log DataFrame
        vocab_path: Path to vocabulary JSON
        config: MDP configuration

    Returns:
        Dictionary with transition arrays
    """
    # Load vocabulary
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    id2token = vocab["id2token"]

    # Get action names
    actions_cfg = config.get("actions", {})
    action_names = actions_cfg.get("id2name", ["NOOP"])

    # Build case terminal profit map (from last event outcome)
    case_terminal_profit_map = clean_df.groupby("case_id")["outcome"].last().to_dict()

    # Pre-compute case events map for efficiency
    case_events_map: dict[int, pd.DataFrame] = {}
    for case_id in clean_df["case_id"].unique():
        case_events_map[case_id] = (
            clean_df[clean_df["case_id"] == case_id].sort_values("timestamp").reset_index(drop=True)
        )

    # Pre-compute case activity positions (for faster action extraction)
    # Map: (case_id, event_position) -> has_contact_headquarters
    # event_position is 0-indexed (0 = first event, 1 = second event, etc.)
    case_activity_map: dict[tuple[int, int], bool] = {}
    for case_id, case_events in case_events_map.items():
        for pos in range(len(case_events)):
            case_activity_map[(case_id, pos)] = (
                case_events.iloc[pos]["activity"] == "contact_headquarters"
            )

    # Group prefixes by case
    case_groups = group_by_case(prefixes)

    # Build transitions
    all_s = []
    all_s_mask = []
    all_a = []
    all_r = []
    all_s_next = []
    all_s_next_mask = []
    all_done = []
    all_case_ptr = []
    all_t_ptr = []
    all_valid_actions = []
    all_behavior_action = []
    all_propensity = []

    n_invalid_actions = 0
    n_skipped_non_decision = 0

    for case_id, prefix_indices in case_groups.items():
        # Sort by t_ptr
        prefix_indices_sorted = sorted(prefix_indices, key=lambda idx: prefixes.t_ptr[idx])

        # Build transitions for this case
        for i, prefix_idx in enumerate(prefix_indices_sorted[:-1]):  # Skip last (no next)
            next_prefix_idx = prefix_indices_sorted[i + 1]

            # Get prefix info
            prefix_X = prefixes.X[prefix_idx]
            prefix_mask = prefixes.mask[prefix_idx]
            prefix_t = prefixes.t_ptr[prefix_idx]

            # Get next prefix
            next_prefix_X = prefixes.X[next_prefix_idx]
            next_prefix_mask = prefixes.mask[next_prefix_idx]
            next_prefix_t = prefixes.t_ptr[next_prefix_idx]

            # Check if same case
            if prefixes.case_ptr[prefix_idx] != prefixes.case_ptr[next_prefix_idx]:
                continue

            # Get last activity
            last_activity = get_last_activity(prefix_X, prefix_mask, id2token)

            # Check if decision point
            # For time_contact_HQ, we allow decision points after
            # start_standard or validate_application
            # But we also need to check if we're in a valid window
            is_dp = is_decision_point(last_activity, config)

            # Additional check: for time_contact_HQ, decision points should be after start_standard
            # We'll be more permissive: allow all prefixes as potential decision points
            # but filter by whether contact_headquarters can occur
            if not is_dp:
                # Check if this is after start_standard (more permissive)
                # For now, allow all prefixes as decision points if mode is not strict
                decision_mode = config.get("decision_points", {}).get("mode", "by_last_activity")
                if decision_mode == "all":
                    is_dp = True
                else:
                    n_skipped_non_decision += 1
                    continue

            if not is_dp:
                n_skipped_non_decision += 1
                continue

            # Build action mask
            valid_actions = build_action_mask(last_activity, action_names, config)

            # Extract behavior action (optimized with pre-computed maps)
            behavior_action = extract_behavior_action_optimized(
                case_id,
                prefix_t,
                next_prefix_t,
                case_activity_map,
                action_names,
            )

            # Validate action is valid
            if valid_actions[behavior_action] == 0:
                n_invalid_actions += 1
                logger.warning(
                    "Invalid action %d (%s) for case %d at t=%d (last_activity=%s)",
                    behavior_action,
                    action_names[behavior_action],
                    case_id,
                    prefix_t,
                    last_activity,
                )
                continue

            # Check if terminal
            is_last_prefix = i + 1 == len(prefix_indices_sorted) - 1
            done = 1 if is_last_prefix else 0

            # Compute reward
            reward = compute_reward(case_id, bool(done), case_terminal_profit_map, config)

            # Store transition
            all_s.append(prefix_X)
            all_s_mask.append(prefix_mask)
            all_a.append(behavior_action)
            all_r.append(reward)
            all_s_next.append(next_prefix_X)
            all_s_next_mask.append(next_prefix_mask)
            all_done.append(done)
            all_case_ptr.append(case_id)
            all_t_ptr.append(prefix_t)
            all_valid_actions.append(valid_actions)
            all_behavior_action.append(behavior_action)
            all_propensity.append(-1.0)  # Unknown, will be estimated later

    # Convert to arrays
    s = np.array(all_s, dtype=np.int32)
    s_mask = np.array(all_s_mask, dtype=np.uint8)
    a = np.array(all_a, dtype=np.int32)
    r = np.array(all_r, dtype=np.float32)
    s_next = np.array(all_s_next, dtype=np.int32)
    s_next_mask = np.array(all_s_next_mask, dtype=np.uint8)
    done_array = np.array(all_done, dtype=np.uint8)
    case_ptr = np.array(all_case_ptr, dtype=np.int32)
    t_ptr = np.array(all_t_ptr, dtype=np.int32)
    valid_actions_array = np.array(all_valid_actions, dtype=np.uint8)
    behavior_action_array = np.array(all_behavior_action, dtype=np.int32)
    propensity = np.array(all_propensity, dtype=np.float32)

    logger.info(
        "Built %d transitions (skipped %d non-decision points, %d invalid actions)",
        len(s),
        n_skipped_non_decision,
        n_invalid_actions,
    )

    return {
        "s": s,
        "s_mask": s_mask,
        "a": a,
        "r": r,
        "s_next": s_next,
        "s_next_mask": s_next_mask,
        "done": done_array,
        "case_ptr": case_ptr,
        "t_ptr": t_ptr,
        "valid_actions": valid_actions_array,
        "behavior_action": behavior_action_array,
        "propensity": propensity,
    }


def validate_transitions(transitions: dict[str, np.ndarray], n_actions: int) -> None:
    """Validate transition arrays.

    Args:
        transitions: Dictionary with transition arrays
        n_actions: Number of actions

    Raises:
        ValueError: If validation fails
    """
    n = len(transitions["a"])

    if n == 0:
        logger.warning("No transitions to validate")
        return

    # Check shapes
    assert transitions["s"].shape == (n, transitions["s"].shape[1]), "s shape mismatch"
    assert transitions["s_mask"].shape == transitions["s"].shape, "s_mask shape mismatch"
    assert transitions["a"].shape == (n,), "a shape mismatch"
    assert transitions["r"].shape == (n,), "r shape mismatch"
    assert transitions["s_next"].shape == transitions["s"].shape, "s_next shape mismatch"
    assert transitions["s_next_mask"].shape == transitions["s"].shape, "s_next_mask shape mismatch"
    assert transitions["done"].shape == (n,), "done shape mismatch"
    assert transitions["valid_actions"].shape == (n, n_actions), "valid_actions shape mismatch"

    # Check valid_actions[a] == 1 for all transitions
    invalid_mask = transitions["valid_actions"][np.arange(n), transitions["a"]] == 0
    if invalid_mask.any():
        n_invalid = invalid_mask.sum()
        raise ValueError(
            f"Found {n_invalid} transitions with invalid actions (valid_actions[a]==0)"
        )

    # Check done is binary
    assert np.all((transitions["done"] == 0) | (transitions["done"] == 1)), "done must be 0 or 1"

    logger.info("Transition validation passed: %d transitions", n)


def build_mdp_dataset(
    prefixes_path: str | Path,
    clean_log_path: str | Path,
    vocab_path: str | Path,
    output_path: str | Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Build MDP dataset from prefixes.

    Args:
        prefixes_path: Path to prefixes.npz
        clean_log_path: Path to clean.parquet
        vocab_path: Path to vocab_activity.json
        output_path: Path to output D_offline.npz
        config: MDP configuration

    Returns:
        Dictionary with build statistics
    """
    # Load data
    logger.info("Loading prefixes from %s", prefixes_path)
    prefixes = load_prefixes(prefixes_path)

    logger.info("Loading clean log from %s", clean_log_path)
    clean_df = load_parquet(clean_log_path)

    # Build transitions
    logger.info("Building transitions...")
    transitions = build_transitions(prefixes, clean_df, vocab_path, config)

    # Validate (only if we have transitions)
    # Get n_actions from config (needed for stats even if no transitions)
    actions_cfg = config.get("actions", {})
    n_actions = len(actions_cfg.get("id2name", ["NOOP"]))

    # Validate transitions
    if len(transitions["a"]) > 0:
        validate_transitions(transitions, n_actions)
    else:
        logger.warning("No transitions built! Check decision point configuration.")

    # Save dataset
    ensure_dir(Path(output_path).parent)
    save_npz(output_path, **transitions)
    logger.info("Saved MDP dataset to %s", output_path)

    # Compute statistics
    n_transitions = len(transitions["a"])
    if n_transitions > 0:
        n_cases_used = len(np.unique(transitions["case_ptr"]))
        done = transitions["done"]
        r = transitions["r"]
        terminal_rewards = r[done == 1]
        max_len = transitions["s"].shape[1]
    else:
        n_cases_used = 0
        done = np.array([], dtype=np.uint8)
        r = np.array([], dtype=np.float32)
        terminal_rewards = np.array([], dtype=np.float32)
        max_len = config.get("encoding", {}).get("max_len", 50)

    stats = {
        "n_transitions": n_transitions,
        "n_cases_used": n_cases_used,
        "n_actions": n_actions,
        "max_len": max_len,
        "pct_done": float(done.mean() * 100),
        "reward_mean": float(r.mean()),
        "reward_std": float(r.std()),
        "reward_p50": (
            float(np.percentile(terminal_rewards, 50)) if len(terminal_rewards) > 0 else 0.0
        ),
        "reward_p95": (
            float(np.percentile(terminal_rewards, 95)) if len(terminal_rewards) > 0 else 0.0
        ),
        "pct_invalid_action": 0.0,  # Should be 0 after validation
        # Additional sanity checks
        "non_terminal_reward_zero_pct": (
            float((r[done == 0] == 0).mean() * 100) if (done == 0).any() else 100.0
        ),
        "terminal_reward_nonzero_pct": (
            float((terminal_rewards != 0).mean() * 100) if len(terminal_rewards) > 0 else 0.0
        ),
    }

    return stats
