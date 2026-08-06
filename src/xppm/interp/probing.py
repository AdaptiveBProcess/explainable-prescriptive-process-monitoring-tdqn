"""Linear probing of internal representations (Paper 2, Phase 1).

Hypotheses (plan §Fase 1), refined by the architecture: without positional
encoding the encoder is permutation-equivariant, so the internal state can
encode at most (activity multiset, last activity). Probes therefore target:

- H1: per-activity counts are linearly decodable from the residual stream.
- H2: the ~5 discrete V(s) levels correspond to separable activation clusters.
- H3: process concepts (contacted yes/no, phase, recoverable) have linear
  directions.

Every probe is paired with a shuffled-label control: a probe is only evidence
if it beats both the majority/mean baseline and its control.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Label builders (from raw token states)
# ---------------------------------------------------------------------------


def activity_counts(states: np.ndarray, vocab_size: int, pad_id: int = 0) -> np.ndarray:
    """Per-activity token counts, shape (N, vocab_size); PAD column zeroed."""
    counts = np.zeros((len(states), vocab_size), dtype=np.int64)
    for token in range(vocab_size):
        counts[:, token] = (states == token).sum(axis=1)
    counts[:, pad_id] = 0
    return counts


def last_activity(states: np.ndarray, pad_id: int = 0) -> np.ndarray:
    """Token id at the pooling position (last non-PAD), shape (N,)."""
    mask = states != pad_id
    lengths = np.clip(mask.sum(axis=1) - 1, 0, states.shape[1] - 1)
    return states[np.arange(len(states)), lengths]


def discretize_values(values: np.ndarray, n_levels: int = 5, seed: int = 0) -> np.ndarray:
    """Cluster scalar V(s) into *n_levels* discrete labels (H2: the V collapse)."""
    km = KMeans(n_clusters=n_levels, n_init=10, random_state=seed)
    return km.fit_predict(values.reshape(-1, 1))


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------


def fit_probe(
    activations: np.ndarray,
    labels: np.ndarray,
    *,
    task: str = "classification",
    test_size: float = 0.2,
    seed: int = 0,
) -> dict[str, float]:
    """Fit a linear probe with a shuffled-label control.

    Args:
        activations: (N, D) feature matrix (residual-stream slice).
        labels: (N,) targets — int classes or float regression targets.
        task: ``"classification"`` (logistic) or ``"regression"`` (ridge).
        test_size: held-out fraction for the reported score.
        seed: split / shuffle seed.

    Returns:
        Dict with ``score`` (accuracy or R^2 on the held-out split),
        ``baseline`` (majority class or mean predictor) and ``control``
        (same probe on shuffled labels).
    """
    if task not in ("classification", "regression"):
        raise ValueError(f"Unknown task '{task}'")

    stratify = labels if task == "classification" else None
    x_tr, x_te, y_tr, y_te = train_test_split(
        activations, labels, test_size=test_size, random_state=seed, stratify=stratify
    )
    scaler = StandardScaler().fit(x_tr)
    x_tr, x_te = scaler.transform(x_tr), scaler.transform(x_te)

    def _fit_score(y_train: np.ndarray) -> float:
        if task == "classification":
            model = LogisticRegression(max_iter=1000, random_state=seed)
            model.fit(x_tr, y_train)
            return float(accuracy_score(y_te, model.predict(x_te)))
        model = Ridge(random_state=seed)
        model.fit(x_tr, y_train)
        return float(r2_score(y_te, model.predict(x_te)))

    if task == "classification":
        majority = np.bincount(y_tr.astype(np.int64)).argmax()
        baseline = float(np.mean(y_te == majority))
    else:
        baseline = 0.0  # R^2 of the mean predictor

    rng = np.random.default_rng(seed)
    return {
        "score": _fit_score(y_tr),
        "baseline": baseline,
        "control": _fit_score(rng.permutation(y_tr)),
    }


def probe_sweep(
    activations_by_hook: dict[str, np.ndarray],
    labels_by_task: dict[str, tuple[np.ndarray, str]],
    *,
    test_size: float = 0.2,
    seed: int = 0,
) -> pd.DataFrame:
    """Fit every (hook point x label task) probe and tabulate results.

    Args:
        activations_by_hook: hook name -> (N, D) activations.
        labels_by_task: task name -> (labels, task_type) where task_type is
            ``"classification"`` or ``"regression"``.

    Returns:
        Long DataFrame with columns ``hook``, ``task``, ``task_type``,
        ``score``, ``baseline``, ``control``, ``margin`` (score - max(baseline,
        control)).
    """
    rows = []
    for hook, acts in activations_by_hook.items():
        for task_name, (labels, task_type) in labels_by_task.items():
            metrics = fit_probe(acts, labels, task=task_type, test_size=test_size, seed=seed)
            rows.append(
                {
                    "hook": hook,
                    "task": task_name,
                    "task_type": task_type,
                    **metrics,
                    "margin": metrics["score"] - max(metrics["baseline"], metrics["control"]),
                }
            )
    return pd.DataFrame(rows).sort_values(["task", "hook"]).reset_index(drop=True)
