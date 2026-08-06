"""Counterfactual and action-level explanation methods.

Available implementations:
    nearest_neighbor → NearestNeighbourCounterfactual  (stub; offline, fast)
    wachter          → WachterCounterfactual           (stub; gradient-based optimization)
    raccer           → RaccerExplainer                 (stub; reachable sequences)
    recourse         → AlgorithmicRecourse             (stub; actionable recommendations)
    nl_explanation   → NLExplainer                     (stub; LLM-based NL generation)

All state-level methods implement xppm.xai.base.BaseCounterfactualExplainer.

Usage::

    from xppm.xai.counterfactual import NearestNeighbourCounterfactual
    result = NearestNeighbourCounterfactual().generate(state, target_action=1, q_net=q_net)
"""

from xppm.xai.counterfactual.nl_explanation import NLExplainer
from xppm.xai.counterfactual.recourse import AlgorithmicRecourse
from xppm.xai.counterfactual.sequences import RaccerExplainer
from xppm.xai.counterfactual.states import NearestNeighbourCounterfactual, WachterCounterfactual

__all__ = [
    "NearestNeighbourCounterfactual",
    "WachterCounterfactual",
    "RaccerExplainer",
    "AlgorithmicRecourse",
    "NLExplainer",
]
