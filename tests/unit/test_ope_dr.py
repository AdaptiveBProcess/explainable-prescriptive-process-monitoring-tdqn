from pathlib import Path

from xppm.ope.doubly_robust import doubly_robust_estimate
from xppm.ope.report import save_ope_report


def test_ope_dr_smoke(tmp_path: Path) -> None:
    import numpy as np
    import torch
    from xppm.rl.models.q_network import QNetwork

    # Create tiny dataset
    states = np.zeros((1, 1), dtype="float32")
    actions = np.zeros((1, 1), dtype="int64")
    rewards = np.zeros((1, 1), dtype="float32")
    next_states = np.zeros((1, 1), dtype="float32")
    dones = np.zeros((1, 1), dtype="bool")
    data_path = tmp_path / "D_offline.npz"
    np.savez_compressed(
        data_path,
        states=states,
        actions=actions,
        rewards=rewards,
        next_states=next_states,
        dones=dones,
    )

    # Dummy Q-network checkpoint
    q_net = QNetwork(state_dim=1, n_actions=1, hidden_dim=4)
    ckpt = tmp_path / "Q_theta.ckpt"
    torch.save(q_net.state_dict(), ckpt)

    metrics = doubly_robust_estimate(ckpt, data_path)
    out = tmp_path / "ope_dr.json"
    save_ope_report(metrics, out)

    assert "dr_mean" in metrics
    assert out.exists()


