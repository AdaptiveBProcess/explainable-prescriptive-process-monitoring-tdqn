"""Smoke tests for dual_level on an untrained SB3 PPO.

Run with the sb3 environment (see requirements.txt), not the repo's .venv:
    pytest experiments/generality_ppo/test_dual_level.py -v
This directory is outside the repo's default pytest testpaths on purpose --
the main suite must stay runnable without stable_baselines3.
"""

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium import spaces

sb3 = pytest.importorskip("stable_baselines3")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dual_level import (  # noqa: E402
    CriticHead,
    MarginHead,
    deletion_test,
    dual_level_study,
    integrated_gradients,
)


class _ToyEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(4, dtype=np.float32), 0.0, True, False, {}


@pytest.fixture(scope="module")
def model():
    return sb3.PPO("MlpPolicy", _ToyEnv(), n_steps=32, batch_size=16, seed=0, device="cpu")


@pytest.fixture(scope="module")
def states():
    return np.random.default_rng(0).normal(size=(64, 4)).astype(np.float32)


def test_ig_completeness(model, states):
    """sum_j phi_j must equal f(x) - f(reference) (IG completeness)."""
    head = CriticHead(model.policy)
    ref = states.mean(axis=0)
    phi = integrated_gradients(head, states, ref, n_steps=256)
    with torch.no_grad():
        fx = head(torch.from_numpy(states)).numpy()
        fr = head(torch.from_numpy(ref[None, :])).numpy()[0]
    np.testing.assert_allclose(phi.sum(axis=1), fx - fr, atol=1e-3)


def test_margin_head_matches_action_choice(model, states):
    head = MarginHead(model.policy)
    with torch.no_grad():
        m = head(torch.from_numpy(states)).numpy()
        obs = model.policy.obs_to_tensor(states)[0]
        probs = model.policy.get_distribution(obs).distribution.probs.numpy()
    np.testing.assert_array_equal(m > 0, probs[:, 1] > probs[:, 0])


def test_deletion_test_shapes(model, states):
    head = CriticHead(model.policy)
    ref = states.mean(axis=0)
    phi = integrated_gradients(head, states, ref)
    res = deletion_test(head, states, phi, ref, k=1, n_random=5)
    assert res.abs_guided >= 0 and res.abs_random >= 0 and res.abs_anti >= 0
    assert np.isfinite(res.gap_se)


def test_dual_level_study_runs(model, states):
    out = dual_level_study(model, states, ["a", "b", "c", "d"], ks=(1,), n_margin_min=5, seed=0)
    assert out["n_states"] == 64
    assert "value_test" in out and "phi_v" in out["value_test"]
    if out["margin_evaluable_sample"]:
        r = out["margin_test"]["phi_dq"]["1"]
        assert "flip_guided" in r
