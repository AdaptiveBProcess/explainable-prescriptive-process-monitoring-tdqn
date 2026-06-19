from __future__ import annotations

from typing import Any

import numpy as np


def compute_noop_policy_probs(
    valid_actions: np.ndarray,
    noop_action_id: int,
    n_actions: int,
) -> np.ndarray:
    """Compute π_noop(a|s): uniform over valid actions, preferring noop if available.

    Args:
        valid_actions: (N, n_actions) binary mask
        noop_action_id: ID of the no-op action
        n_actions: Total number of actions

    Returns:
        (N, n_actions) probability array
    """
    valid = valid_actions > 0  # (N, n_actions) bool
    n_valid = valid.sum(axis=1, keepdims=True).astype(np.float32)  # (N, 1)

    # Base: uniform over valid actions; fall back to uniform over all if no valid action
    n_valid_safe = np.where(n_valid > 0, n_valid, n_actions)
    probs = valid.astype(np.float32) / n_valid_safe
    probs = np.where(n_valid == 0, 1.0 / n_actions, probs)

    # Override: if noop is valid, concentrate all probability on noop
    noop_valid = valid_actions[:, noop_action_id] > 0  # (N,) bool
    probs[noop_valid] = 0.0
    probs[noop_valid, noop_action_id] = 1.0

    return probs.astype(np.float32)


def compute_heuristic_policy_probs(
    valid_actions: np.ndarray,
    prefix_lengths: np.ndarray,
    n_actions: int,
    config: dict[str, Any],
) -> np.ndarray:
    """Compute π_heuristic(a|s) based on simple rules.

    Current heuristic: choose action based on prefix length.
    - Short prefixes (< threshold): prefer action 0 (or first valid)
    - Long prefixes (>= threshold): prefer action 1 (or last valid)

    Args:
        valid_actions: (N, n_actions) binary mask
        prefix_lengths: (N,) prefix lengths (from s_mask.sum(axis=1))
        n_actions: Total number of actions
        config: Configuration dict with heuristic rules

    Returns:
        (N, n_actions) probability array
    """
    n_transitions = valid_actions.shape[0]
    heuristic_cfg = config.get("ope", {}).get("heuristic", {})
    length_threshold = int(heuristic_cfg.get("length_threshold", 5))
    short_action_id = int(heuristic_cfg.get("short_action_id", 0))
    long_action_id = int(heuristic_cfg.get("long_action_id", 1))

    valid = valid_actions > 0  # (N, n_actions) bool
    n_valid = valid.sum(axis=1, keepdims=True).astype(np.float32)  # (N, 1)

    # Base: uniform over valid, fall back to uniform over all
    n_valid_safe = np.where(n_valid > 0, n_valid, n_actions)
    probs = valid.astype(np.float32) / n_valid_safe
    probs = np.where(n_valid == 0, 1.0 / n_actions, probs)

    # Choose preferred action per row based on prefix length
    preferred = np.where(prefix_lengths < length_threshold, short_action_id, long_action_id)  # (N,)
    rows = np.arange(n_transitions)

    # Rows where preferred action is valid → put 1.0 on it
    preferred_valid = valid_actions[rows, preferred] > 0
    probs[preferred_valid] = 0.0
    probs[rows[preferred_valid], preferred[preferred_valid]] = 1.0

    # Rows where preferred is NOT valid but there are valid actions → use first valid
    fallback_rows = ~preferred_valid & (n_valid.squeeze(1) > 0)
    if fallback_rows.any():
        first_valid = np.argmax(valid[fallback_rows], axis=1)
        probs[fallback_rows] = 0.0
        probs[rows[fallback_rows], first_valid] = 1.0

    return probs.astype(np.float32)


def evaluate_baseline_policy(
    pi_baseline: np.ndarray,
    pi_b: np.ndarray,
    a_logged: np.ndarray,
    r: np.ndarray,
    q_sa: np.ndarray,
    v_s: np.ndarray,
    valid_actions: np.ndarray,
    gamma: float,
    rho_cap: float,
) -> dict[str, float]:
    """Evaluate a baseline policy using DR and WIS estimators.

    Args:
        pi_baseline: (N, n_actions) baseline policy probabilities
        pi_b: (N, n_actions) behavior policy probabilities
        a_logged: (N,) logged actions
        r: (N,) rewards
        q_sa: (N,) Q(s, a_logged) values
        v_s: (N,) V(s) values
        valid_actions: (N, n_actions) action mask
        gamma: Discount factor
        rho_cap: Truncation cap for IS weights

    Returns:
        Dictionary with dr_mean, wis_mean, and diagnostics
    """
    n = len(a_logged)

    # Extract probabilities at logged actions
    pi_baseline_logged = pi_baseline[np.arange(n), a_logged]
    pi_b_logged = pi_b[np.arange(n), a_logged]

    # Propensity clipping and IS weights
    pi_b_clipped = np.clip(pi_b_logged, 1e-6, None)
    rho = pi_baseline_logged / pi_b_clipped
    rho_trunc = np.clip(rho, 0.0, rho_cap)

    # Step-wise DR estimator
    dr_step = rho_trunc * (r - q_sa) + v_s
    dr_mean = float(dr_step.mean())

    # Weighted IS (self-normalized)
    wis_num = float(np.sum(rho_trunc * r))
    wis_den = float(np.sum(rho_trunc)) if float(np.sum(rho_trunc)) > 0 else 1.0
    wis_mean = wis_num / wis_den

    # Diagnostics
    rho_percentiles = np.percentile(rho_trunc, [50, 75, 90, 95, 99]).tolist()
    ess = (np.sum(rho_trunc) ** 2) / np.sum(rho_trunc**2) if np.sum(rho_trunc**2) > 0 else 0.0
    ess_fraction = ess / float(n)

    return {
        "dr_mean": dr_mean,
        "wis_mean": wis_mean,
        "rho_percentiles": [float(x) for x in rho_percentiles],
        "ess_fraction": float(ess_fraction),
    }
