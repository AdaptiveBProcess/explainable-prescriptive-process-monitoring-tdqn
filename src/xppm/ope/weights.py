"""Importance-sampling weights for sequential off-policy evaluation.

The first version of this pipeline weighted every transition by a *single-step*
ratio ``rho_t = pi_e(a_t|s_t)/pi_b(a_t|s_t)``.  That is the contextual-bandit
correction: it is unbiased only when the two policies induce the same state
distribution, which is exactly the assumption a sequential intervention policy
violates.  With a terminal-only reward it also makes the estimate scale with
trajectory length, so the per-case number had to be recovered by multiplying by
the mean number of decisions per case.

This module implements the trajectory-aware correction instead: the cumulative
ratio ``rho_{1:t} = prod_{t'<=t} rho_{t'}`` restarted at every case boundary
(Precup et al.'s per-decision importance sampling).  Both the single-step and
the cumulative weights are exposed so the two can be reported side by side.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "step_weights",
    "cumulative_weights",
    "trajectory_weights",
    "effective_sample_size",
    "wis_trajectory",
    "dr_sequential",
]


def step_weights(
    pi_e_logged: np.ndarray,
    pi_b_logged: np.ndarray,
    rho_cap: float,
    propensity_floor: float = 1e-6,
) -> np.ndarray:
    """Single-step ratios ``rho_t``, clipped to ``[0, rho_cap]``."""
    return np.clip(pi_e_logged / np.clip(pi_b_logged, propensity_floor, None), 0.0, rho_cap)


def _case_order(case_ids: np.ndarray, t_ptr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (order, starts): ``order`` sorts transitions by (case, time);
    ``starts`` are the positions in that sorted array where a new case begins."""
    order = np.lexsort((t_ptr, case_ids))
    cs = case_ids[order]
    new_case = np.empty(cs.shape[0], dtype=bool)
    new_case[0] = True
    new_case[1:] = cs[1:] != cs[:-1]
    return order, np.nonzero(new_case)[0]


def cumulative_weights(
    rho_step: np.ndarray,
    case_ids: np.ndarray,
    t_ptr: np.ndarray,
    cum_cap: float | None = None,
    include_current: bool = True,
) -> np.ndarray:
    """Cumulative ratio ``rho_{1:t}`` within each case.

    The per-step ratios are expected to be clipped already (see
    :func:`step_weights`).  Clipping the *product* as well would flatten the
    upper tail and make distinct policies collide at the cap, so ``cum_cap``
    defaults to no truncation and the estimators self-normalise instead.

    Args:
        rho_step: (N,) single-step ratios, already clipped per step.
        case_ids: (N,) case identifier per transition.
        t_ptr: (N,) time index within the case.
        cum_cap: optional truncation of the cumulative weight (default: none).
        include_current: if False, returns ``rho_{1:t-1}`` (the weight the
            sequential DR estimator applies to the ``V(s_t)`` control variate).

    Returns:
        (N,) cumulative weights in the original transition order.
    """
    order, starts = _case_order(case_ids, t_ptr)
    log_rho = np.log(np.clip(rho_step[order], 1e-12, None))

    if not include_current:
        # shift within each case: position t gets the product up to t-1
        shifted = np.empty_like(log_rho)
        shifted[1:] = log_rho[:-1]
        shifted[0] = 0.0
        shifted[starts] = 0.0  # first step of each case has an empty product
        log_rho = shifted

    cum = np.cumsum(log_rho)
    prev = np.concatenate(([0.0], cum[:-1]))
    lengths = np.diff(np.append(starts, log_rho.shape[0]))
    base = np.repeat(prev[starts], lengths)
    log_cum = cum - base
    hi = np.log(cum_cap) if cum_cap is not None and cum_cap > 0 else np.inf
    log_cum = np.clip(log_cum, -700.0, hi)

    out = np.empty_like(rho_step, dtype=np.float64)
    out[order] = np.exp(log_cum)
    return out


def trajectory_weights(
    rho_step: np.ndarray,
    case_ids: np.ndarray,
    t_ptr: np.ndarray,
    cum_cap: float | None = None,
    r: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Per-case weight ``prod_t rho_t`` and, optionally, the per-case return.

    Returns:
        ``(unique_cases, weights, returns)`` with one entry per case, cases in
        ascending id order (matching ``np.unique(case_ids)``).  ``returns`` is
        ``None`` when ``r`` is not supplied.
    """
    order, starts = _case_order(case_ids, t_ptr)
    cs = case_ids[order]
    ends = np.append(starts[1:], cs.shape[0]) - 1  # last position of each case (sorted)
    cum_full = cumulative_weights(rho_step, case_ids, t_ptr, cum_cap, include_current=True)
    w_case = cum_full[order][ends]

    returns = None
    if r is not None:
        # reward is delayed to completion, so the per-case return is the sum
        returns = np.add.reduceat(np.asarray(r, dtype=np.float64)[order], starts)
    return cs[starts], w_case, returns


def effective_sample_size(w: np.ndarray) -> tuple[float, float]:
    """(ESS, ESS fraction) of a weight vector."""
    denom = float(np.sum(w**2))
    if denom <= 0.0:
        return 0.0, 0.0
    ess = float(np.sum(w) ** 2) / denom
    return ess, ess / float(w.shape[0])


def wis_trajectory(
    rho_step: np.ndarray,
    r: np.ndarray,
    case_ids: np.ndarray,
    t_ptr: np.ndarray,
    cum_cap: float | None = None,
) -> dict[str, float]:
    """Self-normalised trajectory IS on the terminal return, reported *per case*.

    With reward only at completion this is the textbook WIS estimator, and its
    units are already per case -- no rescaling by trajectory length is needed.
    """
    _, w_case, returns = trajectory_weights(rho_step, case_ids, t_ptr, cum_cap, r=r)
    assert returns is not None
    den = float(np.sum(w_case))
    value = float(np.sum(w_case * returns) / den) if den > 0 else 0.0
    ess, ess_frac = effective_sample_size(w_case)
    return {
        "value": value,
        "ess": ess,
        "ess_fraction": ess_frac,
        "n_cases": int(w_case.shape[0]),
    }


def dr_sequential(
    rho_step: np.ndarray,
    r: np.ndarray,
    q_sa: np.ndarray,
    v_s: np.ndarray,
    case_ids: np.ndarray,
    t_ptr: np.ndarray,
    cum_cap: float | None = None,
    gamma: float = 1.0,
) -> dict[str, float]:
    """Sequential doubly-robust estimator, per case.

    ``V_DR = (1/n_cases) * sum_i sum_t gamma^t [ rho_{1:t} (r_t - Q(s_t,a_t))
                                                 + rho_{1:t-1} V(s_t) ]``

    which reduces to the step-wise form only when every ``rho_{1:t-1} = 1``.
    """
    w_incl = cumulative_weights(rho_step, case_ids, t_ptr, cum_cap, include_current=True)
    w_prev = cumulative_weights(rho_step, case_ids, t_ptr, cum_cap, include_current=False)
    disc = np.power(gamma, np.asarray(t_ptr, dtype=np.float64))
    per_step = disc * (w_incl * (r - q_sa) + w_prev * v_s)

    n_cases = int(np.unique(case_ids).shape[0])
    total = float(np.sum(per_step))
    return {"value": total / n_cases if n_cases else 0.0, "n_cases": n_cases}
