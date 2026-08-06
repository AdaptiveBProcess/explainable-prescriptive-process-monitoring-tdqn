"""LIME-based local attribution for TDQN.

Fits a locally-faithful linear model around a single state decision.
Useful when the feature space is tabular (distilled) rather than token-based;
for token sequences use the tabular representation from distill_policy.py.

Installation::

    pip install lime        # already in pyproject.toml [xai] extra

References:
    Ribeiro et al. (2016) "Why Should I Trust You?" Explaining the Predictions
      of Any Classifier.  KDD 2016.
    Survey: arXiv:2507.12599 §3.1
"""

from __future__ import annotations

from typing import Any

from xppm.utils.logging import get_logger
from xppm.xai.base import BaseFeatureAttribution, FeatureAttributionResult

logger = get_logger(__name__)

_LIME_AVAILABLE = False
try:
    import lime  # noqa: F401
    import lime.lime_tabular  # noqa: F401

    _LIME_AVAILABLE = True
except ImportError:
    pass


class LimeExplainer(BaseFeatureAttribution):
    """LIME local explanation for tabular state representations.

    Works best on the tabular features extracted by
    xppm.distill.distill_policy.extract_tabular_features(), not on raw token IDs.

    Config keys (config.yaml → xai_extensions.lime):
        num_samples:   int   — perturbation samples for LIME  (default: 500)
        kernel_width:  float — exponential kernel width       (default: 0.25)
        feature_names: list  — optional feature name list for human-readable output
        discretize:    bool  — whether to discretize continuous features (default: True)
    """

    @property
    def method_name(self) -> str:
        return "lime"

    def attribute(
        self,
        states: Any,  # (B, F) float array when using tabular features
        masks: Any,
        q_net: Any,
        action_ids: Any = None,
        config: dict[str, Any] | None = None,
    ) -> FeatureAttributionResult:
        """Compute LIME local linear coefficients.

        TODO: implement. Steps:
            1. Check _LIME_AVAILABLE; raise ImportError with install hint if False.
            2. Convert states to tabular features (call extract_tabular_features if needed).
            3. Build a predict_fn(x: np.ndarray) → np.ndarray[B, n_actions]:
               a. Feed x through q_net.
               b. Apply softmax if config["output_type"] == "prob".
            4. Build lime.lime_tabular.LimeTabularExplainer with training data
               as background, feature_names from config.
            5. For each state in the batch: explainer.explain_instance(state, predict_fn).
            6. Extract coefficients → (B, F) attribution array.
            7. Return FeatureAttributionResult(attributions=coefs, method="lime").

        Notes:
            - LIME is inherently single-instance; batch it with a loop.
            - For very large batches consider parallelizing with joblib.

        Args:
            states:     (B, F) float array of tabular features.
            masks:      ignored (no padding in tabular representation).
            q_net:      TransformerQNetwork or a sklearn-compatible predictor.
            action_ids: (B,) action to explain; None → argmax.
            config:     dict with keys: num_samples, kernel_width, feature_names.

        Returns:
            FeatureAttributionResult with attributions shape (B, F).
        """
        if not _LIME_AVAILABLE:
            raise ImportError(
                "lime is not installed. Install it with: pip install lime\n"
                "Or: pip install 'xppm-tdqn[xai]'"
            )
        raise NotImplementedError(
            "LimeExplainer.attribute() is not yet implemented. "
            "See docstring for the algorithm. "
            "Reference: Ribeiro et al. (2016) KDD."
        )
