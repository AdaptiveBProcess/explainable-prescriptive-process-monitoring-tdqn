"""Contract tests for the positional encoder and the pooling index."""

from __future__ import annotations

import pytest
import torch

from xppm.rl.factory import infer_encoder_version
from xppm.rl.train_tdqn import TransformerQNetwork
from xppm.xai.attributions import _forward_from_embeddings

MAX_LEN = 8


def build(use_positional: bool = True) -> TransformerQNetwork:
    torch.manual_seed(0)
    return TransformerQNetwork(
        vocab_size=12,
        max_len=MAX_LEN,
        d_model=16,
        n_heads=2,
        n_layers=1,
        dropout=0.0,
        n_actions=2,
        encoder_version=2 if use_positional else 1,
    ).eval()


# left padding: real tokens occupy the right-hand end of the row
PREFIX = torch.tensor([[0, 0, 0, 0, 0, 3, 4, 5]])
MASK = torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1.0]])


def test_pooling_points_at_the_last_real_event_not_into_the_pad_region():
    net = build()
    assert net.last_real_index(MASK).tolist() == [7]
    # the old rule was mask.sum()-1 == 2, which is a PAD position here
    assert int(MASK.sum().item()) - 1 == 2


def test_pooling_handles_interior_holes_left_by_deletion():
    """Fidelity tests zero out a mask entry; the readout must follow the survivors."""
    net = build()
    holed = MASK.clone()
    holed[0, 7] = 0.0  # delete the most recent event
    assert net.last_real_index(holed).tolist() == [6]


def test_pooling_handles_right_padding_too():
    net = build()
    right = torch.tensor([[1, 1, 1, 0, 0, 0, 0, 0.0]])
    assert net.last_real_index(right).tolist() == [2]


def test_order_changes_the_q_values():
    net = build(use_positional=True)
    permuted = torch.tensor([[0, 0, 0, 0, 0, 5, 4, 3]])
    assert not torch.allclose(net(PREFIX, MASK), net(permuted, MASK), atol=1e-6)


def test_v1_reproduces_the_original_permutation_equivariant_encoder():
    """Every pre-existing checkpoint must keep behaving as it was trained."""
    net = build(use_positional=False)
    assert net.encoder_version == 1
    permuted = torch.tensor([[0, 0, 0, 0, 0, 5, 4, 3]])
    assert torch.allclose(net(PREFIX, MASK), net(permuted, MASK), atol=1e-5)
    # v1 pools at mask.sum()-1, which is inside the PAD region under left padding
    assert net._pool_index(MASK).tolist() == [2]
    assert net.last_real_index(MASK).tolist() == [7]


def test_repeated_activities_get_distinguishable_representations():
    """Two occurrences of one activity are no longer interchangeable."""
    net = build(use_positional=True)
    a = torch.tensor([[0, 0, 0, 0, 0, 7, 7, 4]])
    b = torch.tensor([[0, 0, 0, 0, 0, 4, 7, 7]])
    assert not torch.allclose(net(a, MASK), net(b, MASK), atol=1e-6)


def test_ig_path_matches_the_deployed_forward():
    net = build()
    direct = net(PREFIX, MASK)
    from_emb = _forward_from_embeddings(net, net.embedding(PREFIX), MASK)
    assert torch.allclose(direct, from_emb, atol=1e-6)


def test_padding_positions_do_not_change_the_output():
    """Attention ignores PAD, so junk in the padded region must be inert."""
    net = build()
    noisy = PREFIX.clone()
    noisy[0, :5] = torch.tensor([9, 8, 7, 6, 2])  # non-zero ids, still masked out
    assert torch.allclose(net(PREFIX, MASK), net(noisy, MASK), atol=1e-6)


def test_all_padded_row_does_not_produce_nan():
    net = build()
    empty = torch.zeros(1, MAX_LEN, dtype=torch.long)
    assert torch.isfinite(net(empty, torch.zeros(1, MAX_LEN))).all()


@pytest.mark.parametrize(
    "ckpt,keys,expected",
    [
        ({"encoder_version": 2}, [], 2),
        ({"encoder_version": 1}, [], 1),
        ({}, ["pos_embedding.weight", "q_head.bias"], 2),
        ({}, ["embedding.weight", "q_head.bias"], 1),
    ],
)
def test_encoder_version_is_inferred_from_the_checkpoint(ckpt, keys, expected):
    assert infer_encoder_version(ckpt, dict.fromkeys(keys, None)) == expected
