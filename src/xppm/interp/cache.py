"""Activation cache and hook-point naming for the instrumented TDQN.

Hook points (batch B, sequence L = max_len, model dim D = d_model, heads H):

======================  ==============  ====================================================
Name                    Shape           Meaning
======================  ==============  ====================================================
``embed``               (B, L, D)       output of ``q_net.embedding``
``proj``                (B, L, D)       output of ``encoder.input_proj`` (residual entry)
``attn_{i}``            (B, L, D)       self-attention output of layer *i* (post out_proj)
``pattern_{i}``         (B, H, L, L)    per-head attention weights of layer *i*
``mlp_{i}``             (B, L, D)       feed-forward output of layer *i* (pre residual add)
``resid_{i}``           (B, L, D)       residual stream after encoder layer *i* (post-LN)
``pooled``              (B, D)          pooled representation at last non-PAD position
``state_repr``          (B, D)          output of ``state_proj`` (pre-ReLU)
``q``                   (B, A)          Q-values
======================  ==============  ====================================================

``pattern_{i}`` and ``q`` are read-only; the rest are editable via
:meth:`xppm.interp.hooked_model.HookedTDQN.run_with_hooks`.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping

import torch


def hook_names(n_layers: int) -> list[str]:
    """Return every hook-point name for a model with *n_layers* encoder layers."""
    names = ["embed", "proj"]
    for i in range(n_layers):
        names += [f"attn_{i}", f"pattern_{i}", f"mlp_{i}", f"resid_{i}"]
    names += ["pooled", "state_repr", "q"]
    return names


def residual_hook_names(n_layers: int) -> list[str]:
    """Hook points that live on the (B, L, D) residual stream, in forward order."""
    return ["proj"] + [f"resid_{i}" for i in range(n_layers)]


def gather_last_position(activations: torch.Tensor, state_mask: torch.Tensor) -> torch.Tensor:
    """Gather a (B, L, D) activation tensor at the pooling position of each sequence.

    Mirrors the pooling rule of ``TransformerQNetwork.forward``: the last
    non-padded position (with left-padding, the most recent event).

    Args:
        activations: (B, L, D) tensor.
        state_mask: (B, L) tensor, 1 for real tokens and 0 for PAD.

    Returns:
        (B, D) tensor.
    """
    lengths = state_mask.sum(dim=1).long() - 1
    lengths = torch.clamp(lengths, min=0, max=activations.size(1) - 1)
    batch_indices = torch.arange(activations.size(0), device=activations.device)
    return activations[batch_indices, lengths]


class ActivationCache(Mapping[str, torch.Tensor]):
    """Read-only mapping of hook-point name to captured (detached) activation."""

    def __init__(self, store: dict[str, torch.Tensor]) -> None:
        self._store = dict(store)

    def __getitem__(self, name: str) -> torch.Tensor:
        try:
            return self._store[name]
        except KeyError:
            raise KeyError(
                f"Hook point '{name}' not in cache. Cached: {sorted(self._store)}"
            ) from None

    def __iter__(self) -> Iterator[str]:
        return iter(self._store)

    def __len__(self) -> int:
        return len(self._store)

    def residual_stream(self) -> torch.Tensor:
        """Stack the residual stream as (n_layers + 1, B, L, D): entry + each layer."""
        names = [n for n in self._store if n == "proj" or n.startswith("resid_")]
        ordered = sorted(names, key=lambda n: -1 if n == "proj" else int(n.split("_")[1]))
        return torch.stack([self._store[n] for n in ordered], dim=0)
