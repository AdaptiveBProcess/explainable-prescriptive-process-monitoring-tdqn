from __future__ import annotations

import torch
from torch import nn


class QNetwork(nn.Module):
    """Simple MLP Q_theta(s, a) over state embeddings."""

    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.net(states)


