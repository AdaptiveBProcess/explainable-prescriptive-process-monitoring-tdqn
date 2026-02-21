"""Abstract base classes for offline RL trainers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class RLTrainer(ABC):
    """Abstract interface for offline RL trainers.

    Subclass this to add a new algorithm (CQL, BCQ, SAC, …).  Only two
    methods are required:

    - ``train()`` — runs the offline training loop and returns a result dict.
    - ``save_checkpoint()`` — persists model weights and training metadata.

    The internal training loop, loss function, and model architecture are
    entirely up to the subclass; this ABC enforces only the call-site contract
    used by ``scripts/04_train_tdqn_offline.py``.
    """

    @abstractmethod
    def train(self, config: Any, **kwargs: Any) -> dict[str, Any]:
        """Run the offline training loop.

        Args:
            config: Algorithm-specific training configuration (e.g.
                ``TDQNConfig`` for TDQN).
            **kwargs: Extra keyword arguments forwarded by the caller
                (``checkpoint_dir``, ``tracker``, hash strings, …).

        Returns:
            Dict containing at minimum:

            - ``"history"`` — list of per-step metric dicts.
            - ``"final_step"`` — total gradient steps taken.
            - Any trained model objects needed by ``save_checkpoint``.
        """

    @abstractmethod
    def save_checkpoint(self, path: Path, **kwargs: Any) -> dict[str, Path]:
        """Persist model weights and training metadata to *path*.

        Args:
            path: Target directory (created if absent).
            **kwargs: Algorithm-specific keyword arguments (model, optimizer,
                step, config, hash strings, …).

        Returns:
            Dict mapping artifact role to saved ``Path``,
            e.g. ``{"q_theta": Path(...), "target_q": Path(...)}``.
        """
