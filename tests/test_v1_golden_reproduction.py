"""Golden-value regression: v1 checkpoints must keep reproducing their exact Q.

Pins the output of the paper's SimBank v1 checkpoint (run 20260209_191903,
pre-positional encoder) on a fixed, deterministic slice of the test split.
If the v1 compatibility path in ``TransformerQNetwork`` (count-based pooling,
no positional embedding, no attention mask) ever drifts, these assertions fail.

The interactive verification during the v2 refactor quoted a mean Q of
3644.0017; that figure was the training-history ``q/mean`` metric, which
depends on the RNG batch sampling at a given step and is not a fixed dataset
statistic. The values pinned here are recomputed deterministically instead:
first 2048 test-split transitions in index order, CPU, canonical loader.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
CKPT = REPO / "artifacts/models/tdqn/20260209_191903/Q_theta.ckpt"
NPZ = REPO / "data/simbank/processed/D_offline.npz"
VOCAB = REPO / "data/simbank/interim/vocab_activity.json"
SPLITS = REPO / "data/simbank/processed/splits.json"

pytestmark = pytest.mark.skipif(
    not (CKPT.exists() and NPZ.exists() and VOCAB.exists() and SPLITS.exists()),
    reason="SimBank v1 checkpoint or dataset not available",
)

# Golden values: ckpt 20260209_191903, first 2048 test transitions, CPU.
GOLDEN = {
    "q_mean": 3646.845215,
    "v_mean": 4396.604980,
    "q0_mean": 3915.320801,
    "q1_mean": 3378.369141,
    "q_first": [5922.15283203125, 4815.57080078125],
}
ATOL = 0.5  # absolute, on values of order 4e3: guards real drift, not FP noise


@pytest.fixture(scope="module")
def golden_q() -> np.ndarray:
    import yaml

    from xppm.rl.factory import load_q_network
    from xppm.utils.io import load_json, load_npz

    config = yaml.safe_load(open(REPO / "configs/config.yaml"))
    data = load_npz(NPZ)
    splits = load_json(SPLITS)
    q_net = load_q_network(CKPT, NPZ, VOCAB, config, torch.device("cpu"))
    assert q_net.encoder_version == 1, "the paper checkpoint must rebuild as v1"
    assert q_net.pos_embedding is None

    test_cases = np.asarray(splits["cases"]["test"])
    idx = np.where(np.isin(data["case_ptr"], test_cases))[0][:2048]
    with torch.no_grad():
        q = q_net(
            torch.as_tensor(data["s"][idx], dtype=torch.long),
            torch.as_tensor(data["s_mask"][idx], dtype=torch.float32),
        ).numpy()
    return q


def test_v1_checkpoint_rebuilds_and_reproduces_q_mean(golden_q: np.ndarray) -> None:
    assert abs(float(golden_q.mean()) - GOLDEN["q_mean"]) < ATOL
    assert abs(float(golden_q.max(axis=1).mean()) - GOLDEN["v_mean"]) < ATOL


def test_v1_checkpoint_reproduces_per_action_means(golden_q: np.ndarray) -> None:
    assert abs(float(golden_q[:, 0].mean()) - GOLDEN["q0_mean"]) < ATOL
    assert abs(float(golden_q[:, 1].mean()) - GOLDEN["q1_mean"]) < ATOL


def test_v1_checkpoint_reproduces_first_transition_exactly(golden_q: np.ndarray) -> None:
    np.testing.assert_allclose(golden_q[0], GOLDEN["q_first"], atol=ATOL)
