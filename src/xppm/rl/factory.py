"""AgentFactory: load trained RL agents from checkpoints via algorithm registry.

Usage
-----
::

    from xppm.rl.factory import AgentFactory

    q_net = AgentFactory.load(
        ckpt_path="artifacts/models/tdqn/.../Q_theta.ckpt",
        npz_path="data/simbank/processed/D_offline.npz",
        vocab_path="data/simbank/interim/vocab_activity.json",
        config=cfg,        # full config dict with config["algorithm"] = "tdqn"
        device=device,
    )

Adding a new algorithm
----------------------
Implement a loader function with signature::

    def _load_my_algo(ckpt_path, npz_path, vocab_path, config, device) -> nn.Module: ...

Then register it::

    _REGISTRY["my_algo"] = _load_my_algo

The returned ``nn.Module`` must be in ``eval()`` mode and respond to
``model(states, state_mask)`` returning ``(batch, n_actions)`` tensors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn

from xppm.utils.io import load_json, load_npz

# ---------------------------------------------------------------------------
# Internal registry
# ---------------------------------------------------------------------------

# Maps algorithm name -> loader(ckpt_path, npz_path, vocab_path, config, device) -> nn.Module
_REGISTRY: dict[str, Callable[..., nn.Module]] = {}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class AgentFactory:
    """Registry-based factory for loading RL agents from checkpoints.

    Supports multiple algorithms via a simple string-keyed registry.
    Algorithm is resolved (in priority order):

    1. ``config["algorithm"]`` key in the config dict.
    2. ``"algorithm"`` field stored inside the checkpoint file.
    3. Hard-coded fallback ``"tdqn"`` (backwards compatible with old runs).
    """

    @staticmethod
    def load(
        ckpt_path: str | Path,
        npz_path: str | Path,
        vocab_path: str | Path,
        config: dict[str, Any],
        device: torch.device,
    ) -> nn.Module:
        """Load an RL agent from *ckpt_path*, dispatching by algorithm name.

        Args:
            ckpt_path: Path to checkpoint file (e.g. ``Q_theta.ckpt``).
            npz_path: Path to ``D_offline.npz`` — used to infer architecture
                dimensions (``max_len``, ``n_actions``, …).
            vocab_path: Path to ``vocab_activity.json`` — used to infer
                ``vocab_size``.
            config: Full config dict.  ``config["algorithm"]`` is read first.
            device: Target torch device.

        Returns:
            Loaded agent as an ``nn.Module`` in ``eval()`` mode.

        Raises:
            ValueError: If the resolved algorithm name is not in the registry.
        """
        # Resolve algorithm name
        algo: str | None = config.get("algorithm", None)
        if algo is None:
            try:
                raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                algo = raw.get("algorithm", "tdqn")
            except Exception:
                algo = "tdqn"

        if algo not in _REGISTRY:
            raise ValueError(
                f"Unknown algorithm '{algo}'. " f"Registered algorithms: {sorted(_REGISTRY.keys())}"
            )

        return _REGISTRY[algo](ckpt_path, npz_path, vocab_path, config, device)

    @staticmethod
    def register(name: str, loader: Callable[..., nn.Module]) -> None:
        """Register *loader* under *name* at runtime (for plugins / tests)."""
        _REGISTRY[name] = loader

    @staticmethod
    def registered() -> list[str]:
        """Return sorted list of registered algorithm names."""
        return sorted(_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Built-in loaders
# ---------------------------------------------------------------------------


def _load_tdqn(
    ckpt_path: str | Path,
    npz_path: str | Path,
    vocab_path: str | Path,
    config: dict[str, Any],
    device: torch.device,
) -> nn.Module:
    """Load a ``TransformerQNetwork`` from a TDQN checkpoint."""
    from xppm.rl.train_tdqn import TransformerQNetwork

    data = load_npz(npz_path)
    training_cfg = config.get("training", {})
    transformer_cfg = training_cfg.get("transformer", {})

    max_len = int(transformer_cfg.get("max_len", data["s"].shape[1]))
    d_model = int(transformer_cfg.get("d_model", 128))
    n_heads = int(transformer_cfg.get("n_heads", 4))
    n_layers = int(transformer_cfg.get("n_layers", 3))
    dropout = float(transformer_cfg.get("dropout", 0.1))

    vocab = load_json(vocab_path)
    token2id = vocab.get("token2id", {})
    vocab_size = int(len(token2id)) if token2id else int(data["s"].max() + 1)
    n_actions = int(data["valid_actions"].shape[1])

    q_net = TransformerQNetwork(
        vocab_size=vocab_size,
        max_len=max_len,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
        n_actions=n_actions,
    ).to(device)

    raw_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = raw_ckpt.get("model_state_dict", raw_ckpt)
    q_net.load_state_dict(state_dict, strict=False)
    q_net.eval()
    return q_net


# Register built-in algorithms
_REGISTRY["tdqn"] = _load_tdqn
