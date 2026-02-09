from __future__ import annotations

import torch
from torch import nn


class SimpleTransformerEncoder(nn.Module):
    """Minimal transformer-style encoder as a placeholder for sequence encoding."""

    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int, dropout: float) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim * 4, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        h = self.input_proj(x)
        return self.encoder(h)  # (batch, seq_len, hidden_dim)


