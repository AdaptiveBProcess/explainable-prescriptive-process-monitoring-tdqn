"""State-level importance methods: which time steps in a trajectory are most critical?

Available implementations:
    q_variance      → QVarianceCriticalStates  (implemented; fast, no extra training)
    edge            → EdgeExplainer            (stub; Gaussian Process on trajectories)
    state_mask      → StateMaskExplainer       (stub; requires auxiliary masking network)
    lazy_mdp        → LazyMDPExplainer         (stub; requires environment interaction)

All classes implement xppm.xai.base.BaseStateImportance.

Usage::

    from xppm.xai.state_importance import QVarianceCriticalStates
    result = QVarianceCriticalStates().score_trajectory(states, actions, rewards, q_net)
"""

from xppm.xai.state_importance.critical_states import QVarianceCriticalStates
from xppm.xai.state_importance.edge import EdgeExplainer
from xppm.xai.state_importance.state_mask import LazyMDPExplainer, StateMaskExplainer

__all__ = [
    "QVarianceCriticalStates",
    "EdgeExplainer",
    "StateMaskExplainer",
    "LazyMDPExplainer",
]
