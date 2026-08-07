from __future__ import annotations

import torch
from torch import nn


class SimpleTransformerEncoder(nn.Module):
    """Minimal transformer-style encoder as a placeholder for sequence encoding."""

    def __init__(
        self, input_dim: int, hidden_dim: int, n_heads: int, n_layers: int, dropout: float
    ) -> None:
        super().__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by n_heads ({n_heads})")
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Encode a batch of sequences.

        Args:
            x: (batch, seq_len, input_dim)
            key_padding_mask: (batch, seq_len) bool, True at positions to ignore.
                Rows that are entirely padding are passed through unmasked, since
                masking every key makes attention undefined.
        """
        h = self.input_proj(x)
        if key_padding_mask is not None:
            all_padded = key_padding_mask.all(dim=1, keepdim=True)
            key_padding_mask = key_padding_mask & ~all_padded
        return self.encoder(h, src_key_padding_mask=key_padding_mask)
