from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from xppm.utils.io import load_npz


@dataclass
class BehaviorPolicy:
    """Very simple tabular behavior policy estimator (stub: empirical frequencies)."""

    probs: np.ndarray  # shape (n_states, n_actions) in toy version


def fit_behavior_policy(dataset_path: str | Path) -> BehaviorPolicy:
    data = load_npz(dataset_path)
    actions = data["actions"].astype(int).flatten()
    n_actions = actions.max() + 1 if actions.size > 0 else 1
    counts = np.bincount(actions, minlength=n_actions)
    probs = counts / counts.sum() if counts.sum() > 0 else np.ones_like(counts) / len(counts)
    return BehaviorPolicy(probs=probs[None, :])


