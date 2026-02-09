from pathlib import Path

from xppm.rl.train_tdqn import TDQNConfig, train_tdqn


def test_tdqn_smoke(tmp_path: Path) -> None:
    # Minimal 1-sample dataset to ensure the training loop runs.
    import numpy as np

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

    cfg = TDQNConfig(state_dim=1, n_actions=1, max_epochs=1, batch_size=1)
    ckpt = tmp_path / "Q_theta.ckpt"
    metrics = train_tdqn(cfg, data_path, ckpt, seed=0)
    assert "final_loss" in metrics
    assert ckpt.exists()


