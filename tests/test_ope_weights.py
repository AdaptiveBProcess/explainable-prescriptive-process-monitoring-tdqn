"""Correctness tests for the trajectory-aware IS weights (xppm.ope.weights)."""

from __future__ import annotations

import numpy as np
import pytest

from xppm.ope.weights import (
    cumulative_weights,
    dr_sequential,
    effective_sample_size,
    step_weights,
    trajectory_weights,
    wis_trajectory,
)

# two cases: A with 3 steps, B with 2 steps, deliberately interleaved in storage
CASE = np.array([0, 0, 1, 0, 1])
T = np.array([0, 1, 0, 2, 1])
RHO = np.array([2.0, 0.5, 3.0, 4.0, 0.25])
R = np.array([0.0, 0.0, 0.0, 10.0, 7.0])  # terminal reward only


def test_cumulative_restarts_at_case_boundary():
    w = cumulative_weights(RHO, CASE, T, cum_cap=None)
    # case 0: t=0 -> 2, t=1 -> 2*0.5 = 1, t=2 -> 2*0.5*4 = 4
    assert w[0] == pytest.approx(2.0)
    assert w[1] == pytest.approx(1.0)
    assert w[3] == pytest.approx(4.0)
    # case 1: t=0 -> 3, t=1 -> 3*0.25 = 0.75
    assert w[2] == pytest.approx(3.0)
    assert w[4] == pytest.approx(0.75)


def test_shifted_weights_are_the_lagged_product():
    w_prev = cumulative_weights(RHO, CASE, T, cum_cap=None, include_current=False)
    assert w_prev[0] == pytest.approx(1.0)  # empty product at the first step
    assert w_prev[1] == pytest.approx(2.0)
    assert w_prev[3] == pytest.approx(1.0)  # 2 * 0.5
    assert w_prev[2] == pytest.approx(1.0)
    assert w_prev[4] == pytest.approx(3.0)


def test_cumulative_weight_is_capped():
    w = cumulative_weights(RHO, CASE, T, cum_cap=2.5)
    assert w.max() <= 2.5 + 1e-9


def test_trajectory_weight_and_return_per_case():
    cases, w, ret = trajectory_weights(RHO, CASE, T, cum_cap=None, r=R)
    assert list(cases) == [0, 1]
    assert w[0] == pytest.approx(4.0)  # 2 * 0.5 * 4
    assert w[1] == pytest.approx(0.75)  # 3 * 0.25
    assert ret[0] == pytest.approx(10.0)
    assert ret[1] == pytest.approx(7.0)


def test_wis_matches_hand_computed_value():
    out = wis_trajectory(RHO, R, CASE, T, cum_cap=None)
    expected = (4.0 * 10.0 + 0.75 * 7.0) / (4.0 + 0.75)
    assert out["value"] == pytest.approx(expected)
    assert out["n_cases"] == 2


def test_wis_reduces_to_logged_average_under_on_policy_weights():
    """rho == 1 everywhere -> the estimate is the empirical per-case return."""
    ones = np.ones_like(RHO)
    out = wis_trajectory(ones, R, CASE, T, cum_cap=None)
    assert out["value"] == pytest.approx((10.0 + 7.0) / 2)
    assert out["ess_fraction"] == pytest.approx(1.0)


def test_dr_is_unbiased_when_q_is_perfect_and_policy_is_behavior():
    """With rho == 1 and Q == V == true return, DR collapses to the return."""
    ones = np.ones_like(RHO)
    true_return = np.array([10.0, 10.0, 7.0, 10.0, 7.0], dtype=float)
    out = dr_sequential(
        ones, R, q_sa=true_return, v_s=true_return, case_ids=CASE, t_ptr=T, cum_cap=None, gamma=1.0
    )
    # sum_t [ (r_t - Q) + V ] telescopes to sum_t r_t over each case
    # case 0: (0-10+10)+(0-10+10)+(10-10+10) = 10 ; case 1: (0-7+7)+(7-7+7) = 7
    assert out["value"] == pytest.approx((10.0 + 7.0) / 2)


def test_dr_differs_from_stepwise_form_when_ratios_deviate():
    """The old estimator used rho_t for both terms; that is a different number."""
    q = np.zeros(5)
    v = np.ones(5)
    seq = dr_sequential(RHO, R, q, v, CASE, T, cum_cap=None, gamma=1.0)["value"]
    stepwise = float(np.sum(RHO * (R - q) + v)) / 2
    assert seq != pytest.approx(stepwise)


def test_step_weights_clip():
    w = step_weights(np.array([1.0, 1.0]), np.array([1e-9, 0.5]), rho_cap=20.0)
    assert w[0] == pytest.approx(20.0)
    assert w[1] == pytest.approx(2.0)


def test_ess_bounds():
    ess, frac = effective_sample_size(np.ones(10))
    assert ess == pytest.approx(10.0) and frac == pytest.approx(1.0)
    ess, frac = effective_sample_size(np.array([1.0, 0.0, 0.0, 0.0]))
    assert frac == pytest.approx(0.25)


def test_order_invariance():
    """Storage order must not change any estimate."""
    perm = np.array([4, 0, 3, 2, 1])
    a = wis_trajectory(RHO, R, CASE, T, cum_cap=None)["value"]
    b = wis_trajectory(RHO[perm], R[perm], CASE[perm], T[perm], cum_cap=None)["value"]
    assert a == pytest.approx(b)
