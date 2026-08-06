"""SHAP-based feature attribution for TDQN.

Uses the `shap` library (Lundberg & Lee 2017) to compute Shapley values for
each token/feature in the state representation.

Installation::

    pip install shap        # already in pyproject.toml [xai] extra

Two modes:
    - KernelSHAP: model-agnostic, works on any black-box Q-function.
      Expensive: O(2^F) model calls in the worst case; use a small background.
    - GradientSHAP: gradient-based, faster, requires PyTorch autograd.
      A hybrid between IG and SHAP (expected value over noise samples).

References:
    Lundberg & Lee (2017) A Unified Approach to Interpreting Model Predictions.
      NeurIPS 2017.  https://github.com/shap/shap
    Survey: arXiv:2507.12599 §3.1
"""

from __future__ import annotations

from typing import Any

import torch

from xppm.utils.logging import get_logger
from xppm.xai.base import BaseFeatureAttribution, FeatureAttributionResult

logger = get_logger(__name__)

_SHAP_AVAILABLE = False
try:
    import shap  # noqa: F401

    _SHAP_AVAILABLE = True
except ImportError:
    pass


class ShapExplainer(BaseFeatureAttribution):
    """SHAP-based attribution. Supports KernelSHAP and GradientSHAP.

    Config keys (config.yaml → xai_extensions.shap):
        mode:              "kernel" | "gradient"  (default: "gradient")
        background_n:      int  — number of background samples for KernelSHAP
                                  or noise baselines for GradientSHAP  (default: 50)
        n_samples:         int  — KernelSHAP sampling budget  (default: 200)
        output_type:       "Q_star" | "V"  — which Q-network output to explain
    """

    @property
    def method_name(self) -> str:
        return "shap"

    def attribute(
        self,
        states: torch.Tensor,
        masks: torch.Tensor,
        q_net: Any,
        action_ids: torch.Tensor | None = None,
        config: dict[str, Any] | None = None,
    ) -> FeatureAttributionResult:
        """Compute SHAP values.

        TODO: implement. Steps:
            1. Check _SHAP_AVAILABLE; raise ImportError with install hint if False.
            2. Build a wrapper function f(x: np.ndarray) → np.ndarray that:
               a. Embeds token IDs via q_net.embedding.
               b. Runs the Transformer + Q-head.
               c. Returns Q(s, action_id) for each row.
            3. Sample background states from the offline dataset.
            4. For "gradient" mode: shap.GradientExplainer(model, background).
               For "kernel"  mode: shap.KernelExplainer(f, background_np).
            5. Compute SHAP values → (B, L) array.
            6. Return FeatureAttributionResult(attributions=shap_vals, method="shap").

        Notes:
            - KernelSHAP works on discrete token IDs; pass states as (B, L) int array.
            - GradientSHAP works in embedding space; build a module that
              starts from embeddings so gradients flow through.

        Args:
            states:     (B, L) token IDs.
            masks:      (B, L) attention mask.
            q_net:      TransformerQNetwork.
            action_ids: (B,) action to explain; None → argmax Q.
            config:     dict with keys: mode, background_n, n_samples, output_type.

        Returns:
            FeatureAttributionResult with attributions shape (B, L).
        """
        if not _SHAP_AVAILABLE:
            raise ImportError(
                "shap is not installed. Install it with: pip install shap\n"
                "Or: pip install 'xppm-tdqn[xai]'"
            )
        raise NotImplementedError(
            "ShapExplainer.attribute() is not yet implemented. "
            "See docstring for the algorithm. "
            "Reference: Lundberg & Lee (2017) NeurIPS."
        )
