"""Q-variance critical state identification (offline, no extra training required).

A state s_t in a trajectory is "critical" when the agent's decision matters most,
i.e. when the Q-value spread across actions is largest:

    importance(t) = max_a Q(s_t, a) - min_a Q(s_t, a)

Large spread → the agent strongly prefers one action over others → the step is
consequential.  This is a simple baseline that requires only the trained Q-network.

Alternative criterion: advantage magnitude |A(s_t, a_t)| = |Q(s_t,a_t) - V(s_t)|.

References:
    Amir & Shabtai (2018) HIGHLIGHTS — Summarizing Agent Behavior. AAMAS 2018.
    Survey: arXiv:2502.06869 §2.2
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from xppm.utils.logging import get_logger
from xppm.xai.base import BaseStateImportance, StateImportanceResult

logger = get_logger(__name__)


class QVarianceCriticalStates(BaseStateImportance):
    """Identify critical states by Q-value spread across actions.

    This is the simplest offline state-importance method: it runs a single
    forward pass over the trajectory states and ranks them by action-value spread.

    Config keys (config.yaml → xai_extensions.critical_states, when method="q_variance"):
        n_critical:    int  — number of top steps to return as critical  (default: 10)
        criterion:     "spread" | "advantage"  (default: "spread")
            "spread"    → max_a Q - min_a Q
            "advantage" → |Q(s,a_logged) - V(s)|
    """

    @property
    def method_name(self) -> str:
        return "q_variance"

    def score_trajectory(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        q_net: Any,
        config: dict[str, Any] | None = None,
    ) -> StateImportanceResult:
        """Score each step by Q-value spread and return the critical ones.

        Args:
            states:   (T, L) token ID sequences.
            actions:  (T,) logged action at each step.
            rewards:  (T,) reward at each step (unused here, kept for API parity).
            q_net:    TransformerQNetwork in eval mode.
            config:   dict with keys: n_critical, criterion.

        Returns:
            StateImportanceResult with importance_scores (T,) and critical_indices.
        """
        cfg = config or {}
        n_critical = int(cfg.get("n_critical", 10))
        criterion = cfg.get("criterion", "spread")

        T, L = states.shape
        device = next(q_net.parameters()).device

        states_t = torch.from_numpy(states).long().to(device)
        masks_t = (states_t != 0).float()

        with torch.no_grad():
            q_vals = q_net(states_t, masks_t).cpu().numpy()  # (T, n_actions)

        if criterion == "spread":
            scores = q_vals.max(axis=1) - q_vals.min(axis=1)
        elif criterion == "advantage":
            v = q_vals.max(axis=1)
            q_logged = q_vals[np.arange(T), actions]
            scores = np.abs(q_logged - v)
        else:
            raise ValueError(f"Unknown criterion '{criterion}'. Use 'spread' or 'advantage'.")

        n_critical = min(n_critical, T)
        top_indices = np.argsort(scores)[::-1][:n_critical].tolist()

        logger.info(
            "QVarianceCriticalStates: %d/%d steps marked critical (criterion=%s)",
            n_critical,
            T,
            criterion,
        )

        return StateImportanceResult(
            importance_scores=scores,
            critical_indices=top_indices,
            method=self.method_name,
            metadata={"criterion": criterion},
        )
