from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from xppm.rl.models.q_network import QNetwork
from xppm.utils.io import load_npz


def evaluate_q_statistics(checkpoint_path: str | Path, dataset_path: str | Path) -> dict[str, Any]:
    """Compute simple Q-value statistics on the offline dataset (stub)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_npz(dataset_path)
    states = torch.from_numpy(data["states"]).float().to(device)

    state_dim = states.shape[-1]
    # In a real setup, n_actions should come from config; here we infer from actions if available.
    n_actions = int(data.get("actions", [[0]]).max() + 1) if "actions" in data else 3

    q_net = QNetwork(state_dim, n_actions, hidden_dim=128).to(device)
    q_net.load_state_dict(torch.load(checkpoint_path, map_location=device))
    q_net.eval()

    with torch.no_grad():
        q_vals = q_net(states)

    return {
        "q_mean": float(q_vals.mean().cpu().item()),
        "q_std": float(q_vals.std().cpu().item()),
        "q_min": float(q_vals.min().cpu().item()),
        "q_max": float(q_vals.max().cpu().item()),
    }


