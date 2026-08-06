"""Feature-level saliency methods.

Available implementations:
    perturbation   → PerturbationSaliency   (mask/replace tokens, measure Q-drop)
    shap           → ShapExplainer          (requires: pip install shap)
    lime           → LimeExplainer          (requires: pip install lime)
    attention      → AttentionSaliency      (reads transformer attention weights)

All classes implement xppm.xai.base.BaseFeatureAttribution.

Usage (via registry)::

    from xppm.xai.base import get_feature_attribution
    cls = get_feature_attribution("perturbation")
    result = cls().attribute(states, masks, q_net, config=cfg)

Or directly::

    from xppm.xai.saliency import PerturbationSaliency
    result = PerturbationSaliency().attribute(states, masks, q_net)
"""

from xppm.xai.saliency.lime_explainer import LimeExplainer
from xppm.xai.saliency.perturbation import AttentionSaliency, PerturbationSaliency
from xppm.xai.saliency.shap_explainer import ShapExplainer

__all__ = ["PerturbationSaliency", "AttentionSaliency", "ShapExplainer", "LimeExplainer"]
