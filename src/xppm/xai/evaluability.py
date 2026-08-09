"""Test resolution and explanandum (paper Def. 4), computed from artifacts.

Until now this criterion existed only in the paper's prose: nothing in the
repository decided which configurations were evaluable, so the claim that the
conditions were "fixed in advance rather than read off the outcome" had no
artifact behind it.  This module computes them.

A configuration is **evaluable** for a test iff

  (i)   granularity        mean k(0.2) > mean k(0.1), with k(p) = ceil(p * len)
  (ii)  sample             at least MIN_CASES evaluated cases
  (iii) target above null  E[|dQ|] >= NULL_MARGIN_FACTOR * m_random(0.1)
                           (margin test only)

and an **at-risk band is resolvable** iff

        IQR(V) >= BAND_FACTOR * E[|delta_random(0.1)|]

Every quantity is derived from the state encoding and the *random* null only --
never from the guided arm -- so a verdict cannot be decided by the outcome it
gates.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from xppm.xai.fidelity_tests import _compute_q_values, _perturb_states_mask_topk

# --- criterion constants, declared here and used by every configuration -----
HEADLINE_P = (0.1, 0.2)
MIN_CASES = 30
NULL_MARGIN_FACTOR = 3.0
BAND_FACTOR = 3.0
DEFAULT_SEED = 123
DEFAULT_N_RANDOM = 20

__all__ = [
    "HEADLINE_P",
    "MIN_CASES",
    "NULL_MARGIN_FACTOR",
    "BAND_FACTOR",
    "k_of_p",
    "granularity_evidence",
    "random_null",
    "evaluate_configuration",
]


def k_of_p(prefix_lengths: np.ndarray, p: float) -> np.ndarray:
    """k(p) = ceil(p * len), at least one event."""
    return np.maximum(1, np.ceil(p * np.asarray(prefix_lengths)).astype(int))


def granularity_evidence(prefix_lengths: np.ndarray) -> dict[str, float]:
    """Evidence for criterion (i).

    The paper states (i) as the strict inequality ``k_bar(0.2) > k_bar(0.1)``.
    That is a knife-edge test: a configuration where the two fractions coincide
    on 90% of cases still passes it. ``frac_cases_differ`` is reported alongside
    so the verdict can be read against how much of the population actually sees
    two different perturbations. It is computed from prefix lengths alone.
    """
    k1 = k_of_p(prefix_lengths, 0.1)
    k2 = k_of_p(prefix_lengths, 0.2)
    return {
        "k_bar_0.1": float(k1.mean()),
        "k_bar_0.2": float(k2.mean()),
        "frac_cases_differ": float((k2 > k1).mean()),
    }


def _random_positions(state_masks: np.ndarray, k_list: np.ndarray, seed: int, rep: int):
    out = []
    for i in range(state_masks.shape[0]):
        np.random.seed(seed + i * 1000 + rep)
        nonpad = np.nonzero(state_masks[i] > 0)[0]
        k = int(min(k_list[i], len(nonpad)))
        out.append(list(np.random.choice(nonpad, size=k, replace=False)) if k else [])
    return out


def random_null(
    q_net: torch.nn.Module,
    states: np.ndarray,
    state_masks: np.ndarray,
    valid_actions: np.ndarray,
    p: float,
    device: torch.device,
    action_pairs: np.ndarray | None = None,
    seed: int = DEFAULT_SEED,
    n_random: int = DEFAULT_N_RANDOM,
) -> dict[str, float]:
    """Random-deletion null at fraction ``p``.

    Returns ``delta_random`` (mean |change in V|) and, when ``action_pairs`` is
    given, ``m_random`` (the margin displacement of Def. 3 under random masking).
    """
    lengths = state_masks.sum(axis=1).astype(int)
    k_list = k_of_p(lengths, p)

    q0, v0, _ = _compute_q_values(q_net, states, state_masks, valid_actions, device)
    rows = np.arange(states.shape[0])
    dq0 = (
        q0[rows, action_pairs[:, 0]] - q0[rows, action_pairs[:, 1]]
        if action_pairs is not None
        else None
    )

    abs_dv, m_terms = [], []
    for rep in range(n_random):
        pos = _random_positions(state_masks, k_list, seed, rep)
        s_r, m_r = _perturb_states_mask_topk(states, state_masks, pos)
        q_r, v_r, _ = _compute_q_values(q_net, s_r, m_r, valid_actions, device)
        abs_dv.append(np.abs(v0 - v_r))
        if action_pairs is not None and dq0 is not None:
            dq_r = q_r[rows, action_pairs[:, 0]] - q_r[rows, action_pairs[:, 1]]
            m_terms.append(np.abs(dq0) - np.abs(dq_r))

    out = {"delta_random": float(np.stack(abs_dv).mean(0).mean())}
    if m_terms:
        # m(p) = |E[ |dQ| - |dQ_masked| ]|, averaged over repetitions
        out["m_random"] = float(np.abs(np.stack(m_terms).mean(1)).mean())
        out["mean_abs_dq"] = float(np.abs(dq0).mean())
    return out


def evaluate_configuration(
    q_net: torch.nn.Module,
    states: np.ndarray,
    state_masks: np.ndarray,
    valid_actions: np.ndarray,
    device: torch.device,
    dq_states: np.ndarray | None = None,
    dq_state_masks: np.ndarray | None = None,
    dq_valid_actions: np.ndarray | None = None,
    dq_action_pairs: np.ndarray | None = None,
    case_level_v: np.ndarray | None = None,
    seed: int = DEFAULT_SEED,
    n_random: int = DEFAULT_N_RANDOM,
) -> dict[str, Any]:
    """Apply Def. 4 to one configuration and return the verdicts with evidence."""
    lengths = state_masks.sum(axis=1).astype(int)
    kbar = {p: float(k_of_p(lengths, p).mean()) for p in HEADLINE_P}
    gran_risk_evidence = granularity_evidence(lengths)
    granular_risk = kbar[0.2] > kbar[0.1]
    n_risk = int(states.shape[0])

    null_risk = random_null(
        q_net, states, state_masks, valid_actions, 0.1, device, seed=seed, n_random=n_random
    )

    risk = {
        "n_cases": n_risk,
        "k_bar": kbar,
        "granularity": bool(granular_risk),
        "granularity_evidence": gran_risk_evidence,
        "sample": bool(n_risk >= MIN_CASES),
        "delta_random_0.1": null_risk["delta_random"],
    }
    risk["evaluable"] = bool(risk["granularity"] and risk["sample"])
    risk["failing"] = [
        c for c, ok in (("i", risk["granularity"]), ("ii", risk["sample"])) if not ok
    ]

    margin: dict[str, Any] = {"evaluable": False, "failing": ["ii"], "n_cases": 0}
    if dq_states is not None and dq_action_pairs is not None and dq_states.shape[0] > 0:
        dq_masks = dq_state_masks if dq_state_masks is not None else state_masks
        dq_valid = dq_valid_actions if dq_valid_actions is not None else valid_actions
        dq_lengths = dq_masks.sum(axis=1).astype(int)
        kbar_dq = {p: float(k_of_p(dq_lengths, p).mean()) for p in HEADLINE_P}
        gran_dq_evidence = granularity_evidence(dq_lengths)
        n_dq = int(dq_states.shape[0])
        null_dq = random_null(
            q_net,
            dq_states,
            dq_masks,
            dq_valid,
            0.1,
            device,
            action_pairs=dq_action_pairs,
            seed=seed,
            n_random=n_random,
        )
        cond_i = kbar_dq[0.2] > kbar_dq[0.1]
        cond_ii = n_dq >= MIN_CASES
        cond_iii = null_dq["mean_abs_dq"] >= NULL_MARGIN_FACTOR * null_dq["m_random"]
        margin = {
            "n_cases": n_dq,
            "k_bar": kbar_dq,
            "granularity": bool(cond_i),
            "granularity_evidence": gran_dq_evidence,
            "sample": bool(cond_ii),
            "target_above_null": bool(cond_iii),
            "mean_abs_dq": null_dq["mean_abs_dq"],
            "m_random_0.1": null_dq["m_random"],
            "evaluable": bool(cond_i and cond_ii and cond_iii),
            "failing": [
                c for c, ok in (("i", cond_i), ("ii", cond_ii), ("iii", cond_iii)) if not ok
            ],
        }

    band: dict[str, Any] = {"resolvable": False}
    if case_level_v is not None and case_level_v.size > 0:
        iqr = float(np.percentile(case_level_v, 75) - np.percentile(case_level_v, 25))
        ratio = iqr / null_risk["delta_random"] if null_risk["delta_random"] > 0 else float("inf")
        band = {
            "iqr_v": iqr,
            "delta_random_0.1": null_risk["delta_random"],
            "ratio": float(ratio),
            "threshold": BAND_FACTOR,
            "resolvable": bool(ratio >= BAND_FACTOR),
        }

    return {
        "constants": {
            "headline_p": list(HEADLINE_P),
            "min_cases": MIN_CASES,
            "null_margin_factor": NULL_MARGIN_FACTOR,
            "band_factor": BAND_FACTOR,
            "seed": seed,
            "n_random": n_random,
        },
        "risk_test": risk,
        "margin_test": margin,
        "at_risk_band": band,
    }
