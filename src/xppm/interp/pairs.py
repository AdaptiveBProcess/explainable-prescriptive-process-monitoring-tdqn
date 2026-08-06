"""Counterfactual twin pairs for activation patching (Paper 2, Phase 0).

A *twin pair* is (risk case, healthy case): same prefix length, minimal edit
distance between unpadded activity sequences, and V(s) on opposite tails of
the value distribution. Both members are real logged prefixes, so patching
healthy activations into the risk run is an in-distribution intervention —
the property that masking-based fidelity tests lack (paper 1, Threats to
Validity).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def unpad(tokens: np.ndarray, pad_id: int = 0) -> np.ndarray:
    """Strip PAD tokens from a single (L,) token-id sequence."""
    return tokens[tokens != pad_id]


def edit_distance(a: np.ndarray, b: np.ndarray, cap: int | None = None) -> int:
    """Levenshtein distance between two token sequences with optional early-exit *cap*."""
    if len(a) < len(b):
        a, b = b, a
    if cap is not None and len(a) - len(b) > cap:
        return cap + 1
    previous = np.arange(len(b) + 1)
    for i, tok_a in enumerate(a, start=1):
        current = np.empty(len(b) + 1, dtype=np.int64)
        current[0] = i
        for j, tok_b in enumerate(b, start=1):
            current[j] = min(
                previous[j] + 1,  # deletion
                current[j - 1] + 1,  # insertion
                previous[j - 1] + (tok_a != tok_b),  # substitution
            )
        if cap is not None and current.min() > cap:
            return cap + 1
        previous = current
    return int(previous[-1])


def build_twin_pairs(
    states: np.ndarray,
    scores: np.ndarray,
    *,
    pad_id: int = 0,
    risk_percentile: float = 10.0,
    healthy_percentile: float = 60.0,
    max_edit_distance: int = 2,
    max_pairs: int = 1000,
    max_candidates_per_length: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """Match risk states to healthy twins of the same prefix length.

    Args:
        states: (N, L) padded token ids.
        scores: (N,) per-state value, e.g. V(s) from
            :func:`xppm.interp.hooked_model.compute_state_values`.
        pad_id: PAD token id (also the IG baseline token).
        risk_percentile: states with score <= this percentile form the risk set.
        healthy_percentile: states with score >= this percentile form the healthy set.
        max_edit_distance: maximum Levenshtein distance between unpadded twins.
        max_pairs: stop after this many accepted pairs.
        max_candidates_per_length: cap on healthy candidates scanned per risk state.
        seed: RNG seed for candidate subsampling and risk-state order.

    Returns:
        DataFrame with one row per pair: ``risk_idx``, ``healthy_idx``,
        ``prefix_len``, ``edit_distance``, ``diff_positions`` (';'-joined
        padded positions where tokens differ), ``risk_score``,
        ``healthy_score``.
    """
    if len(states) != len(scores):
        raise ValueError(f"states ({len(states)}) and scores ({len(scores)}) length mismatch")
    rng = np.random.default_rng(seed)

    risk_thr = np.percentile(scores, risk_percentile)
    healthy_thr = np.percentile(scores, healthy_percentile)
    risk_indices = np.flatnonzero(scores <= risk_thr)
    healthy_indices = np.flatnonzero(scores >= healthy_thr)

    lengths = (states != pad_id).sum(axis=1)
    healthy_by_length: dict[int, np.ndarray] = {}
    for length in np.unique(lengths[healthy_indices]):
        healthy_by_length[int(length)] = healthy_indices[lengths[healthy_indices] == length]

    rows: list[dict] = []
    for risk_idx in rng.permutation(risk_indices):
        if len(rows) >= max_pairs:
            break
        length = int(lengths[risk_idx])
        candidates = healthy_by_length.get(length)
        if candidates is None or len(candidates) == 0:
            continue
        if len(candidates) > max_candidates_per_length:
            candidates = rng.choice(candidates, size=max_candidates_per_length, replace=False)

        risk_seq = unpad(states[risk_idx], pad_id)
        best_idx, best_dist = -1, max_edit_distance + 1
        for healthy_idx in candidates:
            dist = edit_distance(risk_seq, unpad(states[healthy_idx], pad_id), cap=best_dist - 1)
            if dist < best_dist:
                best_idx, best_dist = int(healthy_idx), dist
                if best_dist == 1:  # same length => cannot differ in 0 positions
                    break
        if best_idx < 0 or best_dist > max_edit_distance:
            continue

        diff = np.flatnonzero(states[risk_idx] != states[best_idx])
        rows.append(
            {
                "risk_idx": int(risk_idx),
                "healthy_idx": best_idx,
                "prefix_len": length,
                "edit_distance": best_dist,
                "diff_positions": ";".join(map(str, diff.tolist())),
                "risk_score": float(scores[risk_idx]),
                "healthy_score": float(scores[best_idx]),
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "risk_idx",
            "healthy_idx",
            "prefix_len",
            "edit_distance",
            "diff_positions",
            "risk_score",
            "healthy_score",
        ],
    )
