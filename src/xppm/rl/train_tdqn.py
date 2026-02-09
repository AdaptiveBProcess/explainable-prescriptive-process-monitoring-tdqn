from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn, optim

from xppm.rl.models.q_network import QNetwork
from xppm.rl.replay import ReplayBuffer
from xppm.utils.io import load_npz
from xppm.utils.logging import get_logger, ensure_dir
from xppm.utils.seed import set_seed


logger = get_logger(__name__)


@dataclass
class TDQNConfig:
    state_dim: int
    n_actions: int
    hidden_dim: int = 128
    gamma: float = 0.99
    learning_rate: float = 3e-4
    batch_size: int = 128
    max_epochs: int = 1  # keep very small as a smoke default


def load_replay(dataset_path: str | Path) -> ReplayBuffer:
    data = load_npz(dataset_path)
    return ReplayBuffer(
        states=data["states"],
        actions=data["actions"],
        rewards=data["rewards"],
        next_states=data["next_states"],
        dones=data["dones"],
    )


def train_tdqn(config: TDQNConfig, dataset_path: str | Path, checkpoint_path: str | Path, seed: int = 42) -> dict[str, Any]:
    """Minimal offline TDQN training loop (stub: single-network DQN-style)."""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    replay = load_replay(dataset_path)
    q_net = QNetwork(config.state_dim, config.n_actions, config.hidden_dim).to(device)
    target_q_net = QNetwork(config.state_dim, config.n_actions, config.hidden_dim).to(device)
    target_q_net.load_state_dict(q_net.state_dict())

    optimizer = optim.Adam(q_net.parameters(), lr=config.learning_rate)
    loss_fn = nn.MSELoss()

    n_samples = len(replay.states)
    logger.info("Starting TDQN training on %d samples (stub loop).", n_samples)

    for epoch in range(config.max_epochs):
        batch = replay.sample(config.batch_size)
        states = torch.from_numpy(batch["states"]).float().to(device)
        actions = torch.from_numpy(batch["actions"]).long().to(device)
        rewards = torch.from_numpy(batch["rewards"]).float().to(device)
        next_states = torch.from_numpy(batch["next_states"]).float().to(device)
        dones = torch.from_numpy(batch["dones"]).float().to(device)

        q_values = q_net(states).gather(1, actions)
        with torch.no_grad():
            next_q = target_q_net(next_states).max(1, keepdim=True).values
            target = rewards + config.gamma * (1.0 - dones) * next_q

        loss = loss_fn(q_values, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logger.info("Epoch %d - loss: %.6f", epoch + 1, float(loss.item()))

    ensure_dir(Path(checkpoint_path).parent)
    torch.save(q_net.state_dict(), checkpoint_path)
    logger.info("Saved checkpoint to %s", checkpoint_path)
    return {"final_loss": float(loss.item())}


