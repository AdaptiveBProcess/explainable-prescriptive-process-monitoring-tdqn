"""StateMask and LazyMDP state-importance methods.

StateMask (Shi et al. 2022):
    Trains an auxiliary "mask network" M(s) → {0,1}^L in parallel with the agent.
    M selects which tokens to pass to the Q-network; states where M can "blind"
    many tokens without degrading Q are non-critical.  Requires online training.

LazyMDP (Topin et al. 2021):
    Extends the action space with a "lazy action" a_lazy that copies the previous
    action.  States where the optimal policy deviates from a_lazy are critical.
    Requires access to the environment (online rollouts).

Both require modifications to the training loop and are therefore NOT applicable
in the current offline-only setting without an environment simulator.

References:
    Shi et al. (2022) StateMask: Explaining Deep Reinforcement Learning through
      State Mask.  NeurIPS 2022.  arXiv:2209.05444
    Topin et al. (2021) Iterative Bounding MDPs: Learning Interpretable Policies
      via Non-Interpretable Methods.  AAAI 2021.
    Survey: arXiv:2502.06869 §2.2
"""

from __future__ import annotations

from typing import Any

import numpy as np

from xppm.utils.logging import get_logger
from xppm.xai.base import BaseStateImportance, StateImportanceResult

logger = get_logger(__name__)


class StateMaskExplainer(BaseStateImportance):
    """State-importance via an auxiliary token-masking network.

    Config keys (config.yaml → xai_extensions.critical_states, when method="state_mask"):
        mask_lr:          float — learning rate for the mask network  (default: 1e-3)
        mask_sparsity:    float — target fraction of tokens to mask   (default: 0.5)
        mask_hidden_dim:  int   — hidden dim of the mask MLP          (default: 64)
        n_epochs:         int   — training epochs for mask network    (default: 20)

    PREREQUISITE: This method requires online training of a mask network
    alongside (or after) the Q-network.  It is not applicable offline without
    an environment simulator.
    """

    @property
    def method_name(self) -> str:
        return "state_mask"

    def score_trajectory(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        q_net: Any,
        config: dict[str, Any] | None = None,
    ) -> StateImportanceResult:
        """Load a pre-trained mask network and compute per-step masking entropy.

        TODO: implement. Steps:
            1. Load a pre-trained MaskNetwork from config["mask_checkpoint"].
            2. Run states through MaskNetwork → token mask probabilities (T, L).
            3. Importance(t) = average masking probability over unpadded tokens at t.
               (High masking prob → token is non-critical → step may be non-critical)
               Invert: importance(t) = 1 - mean_mask(t).
            4. Return StateImportanceResult.

        Note: The MaskNetwork must be trained first with scripts/06c_critical_states.py.

        Args:
            states:   (T, L) token ID sequences.
            actions:  (T,) logged actions.
            rewards:  (T,) rewards.
            q_net:    TransformerQNetwork in eval mode.
            config:   dict with keys: mask_checkpoint, n_critical.

        Returns:
            StateImportanceResult.
        """
        raise NotImplementedError(
            "StateMaskExplainer.score_trajectory() is not yet implemented. "
            "Requires a pre-trained mask network. "
            "See docstring for the algorithm. "
            "Reference: Shi et al. (2022) arXiv:2209.05444"
        )


class LazyMDPExplainer(BaseStateImportance):
    """State-importance via deviation from a 'lazy' (no-op) baseline policy.

    In the process-monitoring context: do_nothing (action 0) is the natural
    lazy action.  Steps where the Q-network strongly prefers contact_HQ (action 1)
    over do_nothing are "critical" — those are the moments where intervention matters.

    This is a lightweight approximation of LazyMDP that works offline:
        importance(t) = Q(s_t, a=1) - Q(s_t, a=0)   (ΔQ at each step)

    A full LazyMDP implementation would require environment interaction to verify
    that the lazy action actually leads to comparable outcomes.
    """

    @property
    def method_name(self) -> str:
        return "lazy_mdp"

    def score_trajectory(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        q_net: Any,
        config: dict[str, Any] | None = None,
    ) -> StateImportanceResult:
        """Score steps by Q-deviation from the lazy (NOOP) action.

        Args:
            states:   (T, L) token ID sequences.
            actions:  (T,) logged actions.
            rewards:  (T,) rewards.
            q_net:    TransformerQNetwork in eval mode.
            config:   dict with keys: lazy_action_id (int, default 0), n_critical (int).

        Returns:
            StateImportanceResult.
        """
        import torch

        cfg = config or {}
        lazy_action_id = int(cfg.get("lazy_action_id", 0))
        n_critical = int(cfg.get("n_critical", 10))

        T, L = states.shape
        device = next(q_net.parameters()).device

        states_t = torch.from_numpy(states).long().to(device)
        masks_t = (states_t != 0).float()

        with torch.no_grad():
            q_vals = q_net(states_t, masks_t).cpu().numpy()  # (T, n_actions)

        best_action = q_vals.argmax(axis=1)
        scores = q_vals[np.arange(T), best_action] - q_vals[:, lazy_action_id]
        scores = np.maximum(scores, 0.0)  # only count when agent prefers non-lazy

        n_critical = min(n_critical, T)
        top_indices = np.argsort(scores)[::-1][:n_critical].tolist()

        return StateImportanceResult(
            importance_scores=scores,
            critical_indices=top_indices,
            method=self.method_name,
            metadata={"lazy_action_id": lazy_action_id},
        )
