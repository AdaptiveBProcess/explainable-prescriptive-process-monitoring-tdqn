"""Phase-0 gate: the instrumented model must be numerically equivalent to Q_theta.

Unit tests use a tiny random ``TransformerQNetwork``. The integration test
against the real SimBank checkpoint (case 552) is marked ``slow`` and skips
itself when the artifacts are not present locally.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from xppm.interp.cache import gather_last_position
from xppm.interp.hooked_model import HookedTDQN, compute_state_values

VOCAB, MAX_LEN, D_MODEL, N_HEADS, N_LAYERS, N_ACTIONS = 8, 12, 16, 2, 2, 2

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = REPO_ROOT / "artifacts" / "models" / "tdqn" / "20260209_191903"
NPZ_PATH = REPO_ROOT / "data" / "processed" / "D_offline.npz"

# Case 552 (paper 1, Section 3 running example): 10-event prefix, left-padded to 50.
CASE_552_TOKENS = [0] * 40 + [4, 5, 2, 3, 2, 3, 2, 3, 7, 2]
CASE_552_Q_CONTACT = 532.0  # action 1 = contact_headquarters
CASE_552_Q_DO_NOTHING = -264.0  # action 0 = do_nothing


def _make_q_net():
    from xppm.rl.train_tdqn import TransformerQNetwork

    torch.manual_seed(0)
    net = TransformerQNetwork(
        vocab_size=VOCAB,
        max_len=MAX_LEN,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        dropout=0.0,
        n_actions=N_ACTIONS,
    )
    net.eval()
    return net


def _make_batch(batch: int = 4, seed: int = 1):
    rng = np.random.default_rng(seed)
    lengths = rng.integers(1, MAX_LEN + 1, size=batch)
    states = np.zeros((batch, MAX_LEN), dtype=np.int64)
    for i, length in enumerate(lengths):
        states[i, MAX_LEN - length :] = rng.integers(1, VOCAB, size=length)
    s = torch.as_tensor(states)
    m = (s != 0).float()
    return s, m


def test_forward_equivalence():
    """Instrumentation (hooks + fastpath off + forced weights) must not change Q."""
    q_net = _make_q_net()
    s, m = _make_batch()
    hooked = HookedTDQN(q_net)
    with torch.no_grad():
        q_ref = q_net(s, m)
        q_hooked = hooked(s, m)
        q_cached, _ = hooked.run_with_cache(s, m)
    assert torch.allclose(q_ref, q_hooked, atol=1e-5)
    assert torch.allclose(q_ref, q_cached, atol=1e-5)


def test_cache_names_and_shapes():
    hooked = HookedTDQN(_make_q_net())
    s, m = _make_batch()
    with torch.no_grad():
        q, cache = hooked.run_with_cache(s, m)

    assert set(cache) == set(hooked.hook_names())
    batch = s.size(0)
    assert q.shape == (batch, N_ACTIONS)
    assert cache["embed"].shape == (batch, MAX_LEN, D_MODEL)
    assert cache["pooled"].shape == (batch, D_MODEL)
    for i in range(N_LAYERS):
        assert cache[f"resid_{i}"].shape == (batch, MAX_LEN, D_MODEL)
        assert cache[f"pattern_{i}"].shape == (batch, N_HEADS, MAX_LEN, MAX_LEN)
        # Attention rows are probability distributions.
        row_sums = cache[f"pattern_{i}"].sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)


def test_pooled_matches_gather_rule():
    """'pooled' must equal the residual stream gathered at the pooling position."""
    hooked = HookedTDQN(_make_q_net())
    s, m = _make_batch()
    with torch.no_grad():
        _, cache = hooked.run_with_cache(s, m)
    last_layer = hooked.n_layers - 1
    expected = gather_last_position(cache[f"resid_{last_layer}"], m)
    assert torch.allclose(cache["pooled"], expected, atol=1e-6)


def test_identity_patch_is_noop():
    hooked = HookedTDQN(_make_q_net())
    s, m = _make_batch()
    with torch.no_grad():
        q_ref, cache = hooked.run_with_cache(s, m)
        q_patched = hooked.run_with_hooks(s, m, edits={"resid_0": lambda _t: cache["resid_0"]})
    assert torch.allclose(q_ref, q_patched, atol=1e-6)


def test_destructive_patch_changes_output():
    hooked = HookedTDQN(_make_q_net())
    s, m = _make_batch()
    with torch.no_grad():
        q_ref = hooked(s, m)
        q_zeroed = hooked.run_with_hooks(s, m, edits={"resid_0": torch.zeros_like})
    assert not torch.allclose(q_ref, q_zeroed, atol=1e-3)


def test_unknown_and_read_only_hooks_rejected():
    hooked = HookedTDQN(_make_q_net())
    s, m = _make_batch()
    with pytest.raises(ValueError):
        hooked.run_with_hooks(s, m, edits={"nope": lambda t: t})
    with pytest.raises(ValueError):
        hooked.run_with_hooks(s, m, edits={"pattern_0": lambda t: t})


def test_compute_state_values_respects_action_mask():
    q_net = _make_q_net()
    s, m = _make_batch()
    with torch.no_grad():
        q = q_net(s, m).numpy()
    only_a0 = np.zeros((s.size(0), N_ACTIONS), dtype=np.int64)
    only_a0[:, 0] = 1
    values = compute_state_values(q_net, s.numpy(), m.numpy(), valid_actions=only_a0)
    assert np.allclose(values, q[:, 0], atol=1e-5)


@pytest.mark.slow
@pytest.mark.skipif(
    not (RUN_DIR / "Q_theta.ckpt").exists() or not NPZ_PATH.exists(),
    reason="SimBank checkpoint or D_offline.npz not available locally",
)
def test_case_552_against_real_checkpoint():
    """Reproduce the paper-1 running example: Q(contact_HQ)=532, Q(do_nothing)=-264."""
    from xppm.utils.io import load_yaml

    config = load_yaml(RUN_DIR / "config.yaml")
    hooked = HookedTDQN.from_checkpoint(
        RUN_DIR / "Q_theta.ckpt",
        NPZ_PATH,
        RUN_DIR / "vocab_activity.json",
        config,
        torch.device("cpu"),
    )
    s = torch.as_tensor([CASE_552_TOKENS], dtype=torch.long)
    m = (s != 0).float()
    with torch.no_grad():
        q = hooked(s, m)
    assert q[0, 1].item() == pytest.approx(CASE_552_Q_CONTACT, abs=1.0)
    assert q[0, 0].item() == pytest.approx(CASE_552_Q_DO_NOTHING, abs=1.0)
