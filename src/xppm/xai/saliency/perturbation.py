"""Perturbation-based and attention-based saliency for TDQN.

PerturbationSaliency: masks/replaces individual tokens and measures the
resulting drop in Q(s, a*).  This is the simplest, model-agnostic approach
and requires no gradient computation.

AttentionSaliency: reads the multi-head attention weights from the Transformer
encoder and aggregates them into a per-token importance score.  Fast, but
attention ≠ explanation (Jain & Wallace 2019); treat as a heuristic.

References:
    Greydanus et al. (2017) Visualizing and Understanding Atari Agents.
      arXiv:1711.00138
    Jain & Wallace (2019) Attention Is Not Explanation.
      arXiv:1902.10186
    Survey: arXiv:2502.06869 §2.1
"""

from __future__ import annotations

from typing import Any

import torch

from xppm.utils.logging import get_logger
from xppm.xai.base import BaseFeatureAttribution, FeatureAttributionResult

logger = get_logger(__name__)


class PerturbationSaliency(BaseFeatureAttribution):
    """Token-level perturbation saliency via Q-drop.

    Algorithm:
        1. Compute Q_base = Q(s, a*) on the original state.
        2. For each position i, replace token[i] with pad_id (or a random token).
        3. Compute Q_perturb_i = Q(s_i_masked, a*).
        4. Saliency[i] = Q_base - Q_perturb_i.

    Implementation notes:
        - Run all N perturbations in a single batched forward pass for efficiency.
        - For tabular features, use the mean feature value as the "null" substitution
          instead of pad_id.
    """

    @property
    def method_name(self) -> str:
        return "perturbation"

    def attribute(
        self,
        states: torch.Tensor,
        masks: torch.Tensor,
        q_net: Any,
        action_ids: torch.Tensor | None = None,
        config: dict[str, Any] | None = None,
    ) -> FeatureAttributionResult:
        """Compute perturbation saliency.

        TODO: implement. Steps:
            1. Forward pass on original states → Q_base (B, n_actions).
            2. Determine explained action: action_ids if given, else argmax Q_base.
            3. Build (B*L, L) batch where each row is state_b with token l masked.
            4. Batch forward pass → Q_perturb (B, L, n_actions).
            5. saliency[b, l] = Q_base[b, a] - Q_perturb[b, l, a].
            6. Return FeatureAttributionResult.

        Args:
            states:     (B, L) token IDs.
            masks:      (B, L) attention mask.
            q_net:      TransformerQNetwork.
            action_ids: (B,) optional explicit action to explain.
            config:     dict with keys: pad_id (int), substitution ("pad" | "mean").

        Returns:
            FeatureAttributionResult with attributions shape (B, L).
        """
        raise NotImplementedError(
            "PerturbationSaliency.attribute() is not yet implemented. "
            "See docstring for the algorithm. "
            "Reference: Greydanus et al. (2017) arXiv:1711.00138"
        )


class AttentionSaliency(BaseFeatureAttribution):
    """Attention-weight saliency from Transformer encoder.

    Aggregates multi-head attention weights over all layers with a chosen
    rollout strategy (mean | max | rollout).

    WARNING: Attention weights are not a reliable proxy for feature importance
    (Jain & Wallace 2019). Use alongside gradient-based methods for validation.
    """

    @property
    def method_name(self) -> str:
        return "attention"

    def attribute(
        self,
        states: torch.Tensor,
        masks: torch.Tensor,
        q_net: Any,
        action_ids: torch.Tensor | None = None,
        config: dict[str, Any] | None = None,
    ) -> FeatureAttributionResult:
        """Extract and aggregate attention weights.

        TODO: implement. Steps:
            1. Register forward hooks on each TransformerEncoderLayer to capture
               self-attention weights (attn_output_weights from F.multi_head_attention_forward).
            2. Run a forward pass.
            3. Aggregate heads: mean across heads → (B, L, L) attention matrix per layer.
            4. Aggregate layers: use attention rollout
               (Abnar & Zuidema 2020, arXiv:2005.00928) or simple mean.
            5. Take the row corresponding to the last non-padded token as saliency.
            6. Return FeatureAttributionResult with method="attention".

        Args:
            states:     (B, L) token IDs.
            masks:      (B, L) attention mask.
            q_net:      TransformerQNetwork.
            action_ids: ignored (attention is action-independent).
            config:     dict with keys: aggregation ("mean" | "rollout").

        Returns:
            FeatureAttributionResult with attributions shape (B, L).
        """
        raise NotImplementedError(
            "AttentionSaliency.attribute() is not yet implemented. "
            "See docstring for the algorithm. "
            "Reference: Abnar & Zuidema (2020) arXiv:2005.00928"
        )
