"""Sparse autoencoder over TDQN activations (Paper 2, Phase 4).

Standard ReLU SAE with L1 sparsity penalty and unit-norm decoder columns
(Anthropic-style dictionary learning). With d_model = 128 the dictionaries
are tiny; the 200k offline transitions provide the training activations
(cached by ``scripts/28_cache_activations.py``).

Feature labeling against the SimBank simulator (max-activating examples,
correlation with known process variables) happens in the analysis notebooks,
not here.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class SparseAutoencoder(nn.Module):
    """ReLU sparse autoencoder: latents = ReLU(W_enc (x - b_dec) + b_enc)."""

    def __init__(self, d_in: int, d_hidden: int) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.w_enc = nn.Parameter(torch.empty(d_hidden, d_in))
        self.b_enc = nn.Parameter(torch.zeros(d_hidden))
        self.w_dec = nn.Parameter(torch.empty(d_in, d_hidden))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        nn.init.kaiming_uniform_(self.w_enc, a=5**0.5)
        with torch.no_grad():
            self.w_dec.copy_(self.w_enc.t())
        self.normalize_decoder()

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        """Constrain decoder columns (features) to unit norm."""
        norms = self.w_dec.norm(dim=0, keepdim=True).clamp_min(1e-8)
        self.w_dec.div_(norms)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu((x - self.b_dec) @ self.w_enc.t() + self.b_enc)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return latents @ self.w_dec.t() + self.b_dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latents = self.encode(x)
        return self.decode(latents), latents


def train_sae(
    activations: np.ndarray,
    *,
    expansion: int = 8,
    l1_coeff: float = 1e-3,
    lr: float = 1e-3,
    batch_size: int = 1024,
    n_epochs: int = 10,
    seed: int = 0,
    device: torch.device | str = "cpu",
) -> tuple[SparseAutoencoder, list[dict[str, float]]]:
    """Train a sparse autoencoder on (N, D) cached activations.

    Args:
        activations: (N, D) float activations from one hook point.
        expansion: dictionary size multiplier (d_hidden = expansion * D).
        l1_coeff: sparsity penalty weight.
        lr: Adam learning rate.
        batch_size: minibatch size.
        n_epochs: passes over the activation set.
        seed: shuffling / init seed.
        device: torch device.

    Returns:
        (trained SAE, per-epoch history with ``recon_mse``, ``l1``, ``l0``).
    """
    torch.manual_seed(seed)
    data = torch.as_tensor(np.asarray(activations), dtype=torch.float32, device=device)
    sae = SparseAutoencoder(d_in=data.size(1), d_hidden=expansion * data.size(1)).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    history: list[dict[str, float]] = []
    for epoch in range(n_epochs):
        order = torch.randperm(len(data), generator=generator)
        epoch_mse, epoch_l1, epoch_l0, n_batches = 0.0, 0.0, 0.0, 0
        for start in range(0, len(data), batch_size):
            batch = data[order[start : start + batch_size]]
            recon, latents = sae(batch)
            mse = ((recon - batch) ** 2).mean()
            l1 = latents.abs().sum(dim=1).mean()
            loss = mse + l1_coeff * l1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sae.normalize_decoder()

            epoch_mse += float(mse.item())
            epoch_l1 += float(l1.item())
            epoch_l0 += float((latents > 0).float().sum(dim=1).mean().item())
            n_batches += 1

        history.append(
            {
                "epoch": epoch,
                "recon_mse": epoch_mse / n_batches,
                "l1": epoch_l1 / n_batches,
                "l0": epoch_l0 / n_batches,
            }
        )
    return sae, history
