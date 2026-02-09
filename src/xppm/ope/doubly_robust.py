from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from xppm.ope.behavior_model import BehaviorPolicy, fit_behavior_policy
from xppm.rl.models.q_network import QNetwork
from xppm.utils.io import load_npz


def doubly_robust_estimate(
    checkpoint_path: str | Path,
    dataset_path: str | Path,
    behavior: BehaviorPolicy | None = None,
) -> dict[str, Any]:
    """Very small DR estimator stub on top of offline trajectories."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_npz(dataset_path)
    states = torch.from_numpy(data["states"]).float().to(device)
    actions = data["actions"].astype(int).flatten()
    rewards = data["rewards"].astype(float).flatten()

    state_dim = states.shape[-1]
    n_actions = actions.max() + 1 if actions.size > 0 else 1

    q_net = QNetwork(state_dim, n_actions, hidden_dim=128).to(device)
    q_net.load_state_dict(torch.load(checkpoint_path, map_location=device))
    q_net.eval()

    if behavior is None:
        behavior = fit_behavior_policy(dataset_path)

    with torch.no_grad():
        q_all = q_net(states).cpu().numpy()

    # Importance weights (very naive, assuming uniform target policy for now)
    pi_e = np.ones_like(actions, dtype=float) / float(n_actions)
    pi_b = behavior.probs[0, actions]
    w = pi_e / np.clip(pi_b, 1e-8, None)

    # DR estimator: IS-corrected returns + baseline from Q
    dr = rewards * w + (1.0 - w) * q_all[np.arange(len(actions)), actions]
    return {
        "dr_mean": float(dr.mean()),
        "dr_std": float(dr.std()),
        "n_samples": int(len(dr)),
    }


