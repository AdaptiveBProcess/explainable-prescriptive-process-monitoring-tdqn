"""Sequence-level counterfactual explanations (RACCER).

"What if the agent had acted differently at step t? Where would the process end up?"

RACCER (Samoilescu et al. 2022) finds a sequence of actions that leads from
the current state to a desired target state (e.g. a state where a different
recommended action would be taken), while being *reachable* according to the
learned transition model.

For offline process mining the key question is:
    "If we had contacted HQ at step 3 instead of step 7, would the outcome improve?"

This requires either:
    a) A learned transition model (world model) — p(s'|s,a).
    b) Nearest-neighbour rollout from the offline dataset.

References:
    Samoilescu et al. (2022) Model-based Counterfactual Explanations for Deep
      Reinforcement Learning.  arXiv:2209.00820  (RACCER)
    Survey: arXiv:2502.06869 §2.5 (sequence-level, least explored area)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from xppm.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SequenceCounterfactualResult:
    """Output of a sequence-level counterfactual method.

    Attributes:
        original_trajectory:      (T, L) original state sequence.
        counterfactual_trajectory: (T', L) counterfactual state sequence.
        intervention_step:         step index where the divergence was introduced.
        original_actions:          (T,) original action sequence.
        counterfactual_actions:    (T',) counterfactual action sequence.
        original_outcome:          estimated outcome (Q-value or OPE estimate).
        counterfactual_outcome:    estimated outcome of the CF trajectory.
        method:                    name of the method.
        metadata:                  method-specific extras.
    """

    original_trajectory: np.ndarray
    counterfactual_trajectory: np.ndarray
    intervention_step: int
    original_actions: np.ndarray
    counterfactual_actions: np.ndarray
    original_outcome: float
    counterfactual_outcome: float
    method: str
    metadata: dict[str, Any] = field(default_factory=dict)


class RaccerExplainer:
    """Reachable counterfactual sequence explanations (RACCER).

    Config keys (config.yaml → xai_extensions.counterfactual_sequences):
        method:              "nearest_neighbor" | "world_model"  (default: "nearest_neighbor")
        max_horizon:         int   — maximum trajectory length to search   (default: 20)
        n_candidates:        int   — rollout candidates per step            (default: 50)
        target_action:       int   — action we want the agent to choose     (default: 1)
        world_model_path:    str   — path to a pre-trained transition model
                                     (if method="world_model")
    """

    def generate_counterfactual_sequence(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        intervention_step: int,
        target_action: int,
        q_net: Any,
        dataset: dict[str, np.ndarray] | None = None,
        config: dict[str, Any] | None = None,
    ) -> SequenceCounterfactualResult:
        """Generate a counterfactual trajectory starting from intervention_step.

        TODO: implement. Steps (nearest-neighbour approach):
            1. Take states[0:intervention_step] as the fixed prefix.
            2. At step intervention_step, force action = target_action.
            3. Search dataset for a transition (s ≈ states[intervention_step], a=target_action).
            4. Follow the greedy Q-policy from the matched next-state using dataset rollout.
            5. Compare V(s_intervention) under original vs CF policy using Q-values.
            6. Return SequenceCounterfactualResult.

        Args:
            states:            (T, L) original trajectory.
            actions:           (T,) original actions.
            rewards:           (T,) original rewards.
            intervention_step: step at which to introduce the counterfactual action.
            target_action:     action to force at intervention_step.
            q_net:             TransformerQNetwork.
            dataset:           offline D (for nearest-neighbour rollout).
            config:            method-specific config dict.

        Returns:
            SequenceCounterfactualResult.
        """
        raise NotImplementedError(
            "RaccerExplainer.generate_counterfactual_sequence() is not yet implemented. "
            "See docstring for the algorithm. "
            "Reference: Samoilescu et al. (2022) arXiv:2209.00820"
        )
