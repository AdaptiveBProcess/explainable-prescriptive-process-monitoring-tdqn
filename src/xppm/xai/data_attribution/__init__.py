"""Dataset-level attribution: which training transitions most influenced a decision?

Available implementations:
    influence_functions → InfluenceFunctionAttribution  (stub; requires Hessian inverse)
    data_shapley        → DataShapleyAttribution        (stub; requires model retraining)
    tracin              → TracInAttribution             (stub; gradient checkpointing approach)

All classes implement xppm.xai.base.BaseDataAttribution.

Usage::

    from xppm.xai.data_attribution import InfluenceFunctionAttribution
    result = InfluenceFunctionAttribution().compute_influence(
        query_state, query_action, dataset, q_net, top_k=10
    )
"""

from xppm.xai.data_attribution.data_shapley import DataShapleyAttribution
from xppm.xai.data_attribution.influence import InfluenceFunctionAttribution, TracInAttribution

__all__ = ["InfluenceFunctionAttribution", "TracInAttribution", "DataShapleyAttribution"]
