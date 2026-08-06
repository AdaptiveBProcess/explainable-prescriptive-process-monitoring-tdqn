"""Abstract base classes for all XRL technique categories.

Taxonomy (from surveys arXiv:2502.06869 and arXiv:2507.12599):

  Feature-level   → BaseFeatureAttribution   (SHAP, LIME, IG, perturbation, attention)
  State-level     → BaseStateImportance      (EDGE, AIRS, StateMask, LazyMDP)
  Dataset-level   → BaseDataAttribution      (Influence Functions, Data Shapley)
  Action-level    → BaseCounterfactual       (counterfactual states/sequences, RACCER)
  Policy-level    → BasePolicySummarizer     (HIGHLIGHTS, clustering, visual toolkits)
  Model-level     → handled by src/xppm/distill/ (DT, rules, symbolic programs)

All existing implementations in this package satisfy these interfaces. New
implementations must subclass the appropriate base class and implement all
abstract methods; the registry at the bottom of this file wires method names
from config.yaml to concrete classes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class FeatureAttributionResult:
    """Output of a feature-level attribution method.

    Attributes:
        attributions:  (batch, seq_len) or (batch, n_features) — signed attribution
                       scores; positive = increases Q(s,a*), negative = decreases it.
        token_ids:     (batch, seq_len) original input token IDs (for traceability).
        action_ids:    (batch,) action explained for each item.
        method:        name of the method that produced this result.
        metadata:      method-specific extras (convergence delta, n_steps, etc.).
    """

    attributions: np.ndarray
    token_ids: np.ndarray
    action_ids: np.ndarray
    method: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StateImportanceResult:
    """Output of a state-level (critical-state) identification method.

    Attributes:
        importance_scores: (T,) score per time step in the trajectory.
        critical_indices:  sorted list of the most important step indices.
        method:            name of the method.
        metadata:          method-specific extras.
    """

    importance_scores: np.ndarray
    critical_indices: list[int]
    method: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataAttributionResult:
    """Output of a dataset-level attribution method.

    Attributes:
        influence_scores: (N_train,) influence of each training transition on the
                          test query; positive = the transition helped the model
                          make the current decision.
        top_k_indices:    indices of the most influential training transitions.
        method:           name of the method.
        metadata:         method-specific extras.
    """

    influence_scores: np.ndarray
    top_k_indices: list[int]
    method: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CounterfactualResult:
    """Output of a counterfactual explanation method.

    Attributes:
        original_state:      original state (token IDs or feature vector).
        counterfactual_state: minimally-modified state that causes target_action.
        original_action:     action chosen in the original state.
        target_action:       action the agent would choose in counterfactual_state.
        distance:            L1/L2/semantic distance between original and counterfactual.
        changed_features:    names/indices of features that were changed.
        method:              name of the method.
        metadata:            method-specific extras.
    """

    original_state: np.ndarray
    counterfactual_state: np.ndarray
    original_action: int
    target_action: int
    distance: float
    changed_features: list[int | str]
    method: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicySummaryResult:
    """Output of a policy-level summarization method.

    Attributes:
        highlights:   list of (state_idx, importance_score) for key states/moments.
        clusters:     per-state cluster assignment.
        prototypes:   representative states for each cluster.
        strategies:   human-readable description per cluster (if NL is available).
        method:       name of the method.
        metadata:     method-specific extras.
    """

    highlights: list[tuple[int, float]]
    clusters: np.ndarray
    prototypes: list[np.ndarray]
    strategies: list[str]
    method: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract base classes
# ---------------------------------------------------------------------------


class BaseFeatureAttribution(ABC):
    """Feature-level: which input features drive Q(s, a*)?

    Methods in this category:
        - Integrated Gradients (implemented: xppm.xai.attributions)
        - Perturbation saliency (stub: xppm.xai.saliency.perturbation)
        - SHAP (stub: xppm.xai.saliency.shap_explainer)
        - LIME (stub: xppm.xai.saliency.lime_explainer)
        - Attention-based (stub: xppm.xai.saliency.perturbation.AttentionSaliency)

    References:
        - Greydanus et al. (2017) Visualizing and Understanding Atari Agents
        - Lundberg & Lee (2017) SHAP
        - Ribeiro et al. (2016) LIME
        - Survey: arXiv:2502.06869 §2.1, arXiv:2507.12599 §3.1
    """

    @abstractmethod
    def attribute(
        self,
        states: torch.Tensor,
        masks: torch.Tensor,
        q_net: Any,
        action_ids: torch.Tensor | None = None,
        config: dict[str, Any] | None = None,
    ) -> FeatureAttributionResult:
        """Compute per-token/per-feature attributions.

        Args:
            states:     (B, L) token IDs or (B, F) tabular features.
            masks:      (B, L) attention mask (1=real, 0=pad).
            q_net:      the trained Q-network.
            action_ids: (B,) action to explain; if None, uses argmax Q.
            config:     method-specific config dict.

        Returns:
            FeatureAttributionResult
        """

    @property
    @abstractmethod
    def method_name(self) -> str:
        """Unique slug for this method (matches config.yaml value)."""


class BaseStateImportance(ABC):
    """State-level: which time steps in a trajectory are most critical?

    Methods in this category:
        - Q-value variance / critical state detection
          (stub: xppm.xai.state_importance.critical_states)
        - EDGE — Gaussian Process on trajectories (stub: xppm.xai.state_importance.edge)
        - StateMask — masking-based (stub: xppm.xai.state_importance.state_mask)
        - LazyMDP — lazy action extension (stub: xppm.xai.state_importance.state_mask.LazyMDP)

    References:
        - Guo et al. (2021) EDGE
        - Shi et al. (2022) StateMask
        - Survey: arXiv:2502.06869 §2.2
    """

    @abstractmethod
    def score_trajectory(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        q_net: Any,
        config: dict[str, Any] | None = None,
    ) -> StateImportanceResult:
        """Assign importance scores to each step of a trajectory.

        Args:
            states:   (T, L) or (T, F) state sequence.
            actions:  (T,) action taken at each step.
            rewards:  (T,) reward received at each step.
            q_net:    the trained Q-network.
            config:   method-specific config dict.

        Returns:
            StateImportanceResult
        """

    @property
    @abstractmethod
    def method_name(self) -> str:
        """Unique slug for this method."""


class BaseDataAttribution(ABC):
    """Dataset-level: which training transitions most influenced a test decision?

    Methods in this category:
        - Influence Functions (stub: xppm.xai.data_attribution.influence)
        - Data Shapley (stub: xppm.xai.data_attribution.data_shapley)
        - Data Masking / TracIn

    References:
        - Koh & Liang (2017) Understanding Black-box Predictions via Influence Functions
        - Ghorbani & Zou (2019) Data Shapley
        - Survey: arXiv:2502.06869 §2.3
    """

    @abstractmethod
    def compute_influence(
        self,
        query_state: np.ndarray,
        query_action: int,
        dataset: dict[str, np.ndarray],
        q_net: Any,
        top_k: int = 10,
        config: dict[str, Any] | None = None,
    ) -> DataAttributionResult:
        """Compute influence of each training transition on a test query.

        Args:
            query_state:  (L,) or (F,) test state to explain.
            query_action: action chosen at query_state.
            dataset:      full offline dataset dict (s, a, r, s_next, ...).
            q_net:        the trained Q-network.
            top_k:        number of most influential transitions to return.
            config:       method-specific config dict.

        Returns:
            DataAttributionResult
        """

    @property
    @abstractmethod
    def method_name(self) -> str:
        """Unique slug for this method."""


class BaseCounterfactualExplainer(ABC):
    """Action-level: what minimal change to the state would alter the agent's action?

    Methods in this category:
        - Nearest-Neighbour counterfactual (stub: xppm.xai.counterfactual.states)
        - GAN-based: GANterfactual-RL, SAFE-RL (stub: xppm.xai.counterfactual.states)
        - RACCER — reachable counterfactual sequences (stub: xppm.xai.counterfactual.sequences)
        - Algorithmic Recourse (stub: xppm.xai.counterfactual.recourse)
        - Natural Language Explanations (stub: xppm.xai.counterfactual.nl_explanation)

    References:
        - Wachter et al. (2017) Counterfactual Explanations
        - Huber & Böhm (2022) GANterfactual-RL
        - Samoilescu et al. (2022) RACCER
        - Survey: arXiv:2507.12599 §3.3
    """

    @abstractmethod
    def generate(
        self,
        state: np.ndarray,
        target_action: int,
        q_net: Any,
        valid_actions: np.ndarray | None = None,
        config: dict[str, Any] | None = None,
    ) -> CounterfactualResult:
        """Generate a counterfactual state that causes target_action.

        Args:
            state:         (L,) or (F,) original state (token IDs or features).
            target_action: desired action in the counterfactual.
            q_net:         the trained Q-network.
            valid_actions: (n_actions,) binary mask of valid actions.
            config:        method-specific config dict.

        Returns:
            CounterfactualResult
        """

    @property
    @abstractmethod
    def method_name(self) -> str:
        """Unique slug for this method."""


class BasePolicySummarizer(ABC):
    """Policy-level: how does the agent behave globally across many states?

    Methods in this category:
        - HIGHLIGHTS — select key state clips (stub: xppm.xai.highlights)
        - Clustering / SAMDPs (existing: xppm.xai.policy_summary)
        - Visual Toolkits / MDPVis (external)
        - Human-readable MDPs (external)

    References:
        - Amir & Shabtai (2018) HIGHLIGHTS
        - Survey: arXiv:2502.06869 §2.4
    """

    @abstractmethod
    def summarize(
        self,
        states: np.ndarray,
        masks: np.ndarray,
        q_net: Any,
        config: dict[str, Any] | None = None,
    ) -> PolicySummaryResult:
        """Produce a structured summary of the policy over a state collection.

        Args:
            states:  (N, L) or (N, F) collection of states.
            masks:   (N, L) attention masks.
            q_net:   the trained Q-network.
            config:  method-specific config dict.

        Returns:
            PolicySummaryResult
        """

    @property
    @abstractmethod
    def method_name(self) -> str:
        """Unique slug for this method."""


# ---------------------------------------------------------------------------
# Registry: config name → concrete class
# ---------------------------------------------------------------------------

_FEATURE_ATTRIBUTION_REGISTRY: dict[str, type[BaseFeatureAttribution]] = {}
_STATE_IMPORTANCE_REGISTRY: dict[str, type[BaseStateImportance]] = {}
_DATA_ATTRIBUTION_REGISTRY: dict[str, type[BaseDataAttribution]] = {}
_COUNTERFACTUAL_REGISTRY: dict[str, type[BaseCounterfactualExplainer]] = {}
_POLICY_SUMMARIZER_REGISTRY: dict[str, type[BasePolicySummarizer]] = {}


def register_feature_attribution(cls: type[BaseFeatureAttribution]) -> type[BaseFeatureAttribution]:
    _FEATURE_ATTRIBUTION_REGISTRY[cls.method_name.fget(cls)] = cls  # type: ignore[attr-defined]
    return cls


def register_state_importance(cls: type[BaseStateImportance]) -> type[BaseStateImportance]:
    _STATE_IMPORTANCE_REGISTRY[cls.method_name.fget(cls)] = cls  # type: ignore[attr-defined]
    return cls


def register_data_attribution(cls: type[BaseDataAttribution]) -> type[BaseDataAttribution]:
    _DATA_ATTRIBUTION_REGISTRY[cls.method_name.fget(cls)] = cls  # type: ignore[attr-defined]
    return cls


def register_counterfactual(
    cls: type[BaseCounterfactualExplainer],
) -> type[BaseCounterfactualExplainer]:
    _COUNTERFACTUAL_REGISTRY[cls.method_name.fget(cls)] = cls  # type: ignore[attr-defined]
    return cls


def register_policy_summarizer(cls: type[BasePolicySummarizer]) -> type[BasePolicySummarizer]:
    _POLICY_SUMMARIZER_REGISTRY[cls.method_name.fget(cls)] = cls  # type: ignore[attr-defined]
    return cls


def get_feature_attribution(method: str) -> type[BaseFeatureAttribution]:
    if method not in _FEATURE_ATTRIBUTION_REGISTRY:
        available = list(_FEATURE_ATTRIBUTION_REGISTRY)
        raise KeyError(f"Unknown feature attribution method '{method}'. Available: {available}")
    return _FEATURE_ATTRIBUTION_REGISTRY[method]


def get_state_importance(method: str) -> type[BaseStateImportance]:
    if method not in _STATE_IMPORTANCE_REGISTRY:
        available = list(_STATE_IMPORTANCE_REGISTRY)
        raise KeyError(f"Unknown state importance method '{method}'. Available: {available}")
    return _STATE_IMPORTANCE_REGISTRY[method]


def get_counterfactual(method: str) -> type[BaseCounterfactualExplainer]:
    if method not in _COUNTERFACTUAL_REGISTRY:
        available = list(_COUNTERFACTUAL_REGISTRY)
        raise KeyError(f"Unknown counterfactual method '{method}'. Available: {available}")
    return _COUNTERFACTUAL_REGISTRY[method]


def get_data_attribution(method: str) -> type[BaseDataAttribution]:
    if method not in _DATA_ATTRIBUTION_REGISTRY:
        available = list(_DATA_ATTRIBUTION_REGISTRY)
        raise KeyError(f"Unknown data attribution method '{method}'. Available: {available}")
    return _DATA_ATTRIBUTION_REGISTRY[method]
