"""Unit tests for Integrated Gradients attribution module."""

from __future__ import annotations

import numpy as np
import torch

from xppm.xai.attributions import (
    aggregate_token_importance,
    integrated_gradients_embedding,
)


def _make_q_net(vocab_size: int = 8, d_model: int = 16, max_len: int = 6, n_actions: int = 2):
    from xppm.rl.train_tdqn import TransformerQNetwork

    net = TransformerQNetwork(
        vocab_size=vocab_size,
        max_len=max_len,
        d_model=d_model,
        n_heads=2,
        n_layers=1,
        dropout=0.0,
        n_actions=n_actions,
    )
    net.eval()
    return net


def test_ig_output_shape():
    """IG attributions must match embedding shape."""
    q_net = _make_q_net()
    batch, seq_len, d_model = 2, 6, 16
    states = torch.zeros(batch, seq_len, dtype=torch.long)
    mask = torch.ones(batch, seq_len)
    baseline = torch.zeros(batch, seq_len, d_model)

    def target_fn(q):
        return q[:, 0]

    attr, stats = integrated_gradients_embedding(
        q_net, states, mask, target_fn, baseline, n_steps=4
    )

    assert attr.shape == (batch, seq_len, d_model)
    assert "abs_err_mean" in stats
    assert "rel_err_mean" in stats


def test_ig_completeness_near_zero_for_identical_input_and_baseline():
    """When input == baseline, IG attributions should be zero and completeness error near zero."""
    q_net = _make_q_net()
    batch, seq_len = 1, 6
    states = torch.zeros(batch, seq_len, dtype=torch.long)
    mask = torch.ones(batch, seq_len)
    # baseline is the embedding of the same all-zero tokens
    with torch.no_grad():
        baseline = q_net.embedding(states)

    def target_fn(q):
        return q[:, 0]

    attr, stats = integrated_gradients_embedding(
        q_net, states, mask, target_fn, baseline, n_steps=4
    )

    assert stats["abs_err_mean"] < 1e-3


def test_aggregate_token_importance_masks_padding():
    """PAD positions (mask=0) must have zero importance."""
    rng = np.random.default_rng(0)
    attr_emb = rng.standard_normal((3, 5, 8)).astype(np.float32)
    masks = np.array([[1, 1, 1, 0, 0], [1, 0, 0, 0, 0], [1, 1, 1, 1, 1]])

    token_imp = aggregate_token_importance(attr_emb, masks)

    assert token_imp.shape == (3, 5)
    # PAD positions must be exactly zero
    for i in range(3):
        for j in range(5):
            if masks[i, j] == 0:
                assert token_imp[i, j] == 0.0


def test_aggregate_token_importance_nonnegative():
    """Token importance is L1 norm of embedding attributions — always >= 0."""
    rng = np.random.default_rng(1)
    attr_emb = rng.standard_normal((4, 6, 16)).astype(np.float32)
    masks = np.ones((4, 6))

    token_imp = aggregate_token_importance(attr_emb, masks)

    assert (token_imp >= 0).all()


def test_ig_with_padding_mask():
    """IG should work correctly when part of the sequence is padding."""
    q_net = _make_q_net()
    batch, seq_len, d_model = 2, 6, 16
    states = torch.randint(0, 8, (batch, seq_len))
    # First 2 tokens are padding (mask=0)
    mask = torch.tensor([[0, 0, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1]], dtype=torch.float)
    baseline = torch.zeros(batch, seq_len, d_model)

    def target_fn(q):
        return q[:, 0]

    attr, stats = integrated_gradients_embedding(
        q_net, states, mask, target_fn, baseline, n_steps=4
    )

    assert attr.shape == (batch, seq_len, d_model)
    assert np.isfinite(stats["abs_err_mean"])
