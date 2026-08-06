"""EDGE: Explaining Deep Reinforcement Learning Policies (Guo et al. 2021).

EDGE models the relationship between states and episode rewards using a
Gaussian Process (GP) to identify which states most influence the final outcome.

Key idea:
    Train a GP: f(s_t) → future_reward_t on offline trajectory data.
    States with high GP posterior variance are ambiguous; states where the GP
    mean changes rapidly along the trajectory are the critical ones.

Implementation requirements:
    pip install gpytorch scikit-learn   (pyproject.toml [xai] extra)

References:
    Guo et al. (2021) EDGE: Explaining Deep Reinforcement Learning Policies.
      arXiv:2212.09707
    Survey: arXiv:2502.06869 §2.2
"""

from __future__ import annotations

from typing import Any

import numpy as np

from xppm.utils.logging import get_logger
from xppm.xai.base import BaseStateImportance, StateImportanceResult

logger = get_logger(__name__)

_GPYTORCH_AVAILABLE = False
try:
    import gpytorch  # noqa: F401

    _GPYTORCH_AVAILABLE = True
except ImportError:
    pass


class EdgeExplainer(BaseStateImportance):
    """EDGE state-importance via Gaussian Process reward modelling.

    Config keys (config.yaml → xai_extensions.critical_states, when method="edge"):
        n_gp_train:     int   — number of trajectory steps to train GP on  (default: 500)
        kernel:         str   — GP kernel: "rbf" | "matern"                (default: "rbf")
        n_critical:     int   — top-k critical states to return            (default: 10)
        feature_method: str   — state encoding for GP input:
                                "q_values" | "embeddings" | "tabular"      (default: "q_values")
    """

    @property
    def method_name(self) -> str:
        return "edge"

    def score_trajectory(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        q_net: Any,
        config: dict[str, Any] | None = None,
    ) -> StateImportanceResult:
        """Score trajectory steps using GP posterior mean sensitivity.

        TODO: implement. Steps:
            1. Check _GPYTORCH_AVAILABLE; raise ImportError if False.
            2. Encode states → feature matrix X (T, D):
               - "q_values": run q_net → (T, n_actions) Q-value vectors.
               - "embeddings": run q_net encoder → (T, d_model) state embeddings.
               - "tabular": call extract_tabular_features().
            3. Build future reward targets y_t = sum_{k>=t} gamma^(k-t) * r_k.
            4. Train a GP: gpytorch.models.ExactGP with chosen kernel on (X, y).
            5. Compute GP posterior mean μ(x_t) for all steps t.
            6. Importance(t) = |μ(x_t) - μ(x_{t-1})| (mean gradient along trajectory).
            7. Return StateImportanceResult.

        Args:
            states:   (T, L) token ID sequences.
            actions:  (T,) logged actions (used for "tabular" encoding).
            rewards:  (T,) rewards for computing discounted future returns.
            q_net:    TransformerQNetwork in eval mode.
            config:   dict with keys: n_gp_train, kernel, n_critical, feature_method.

        Returns:
            StateImportanceResult with importance_scores (T,) and critical_indices.
        """
        if not _GPYTORCH_AVAILABLE:
            raise ImportError(
                "gpytorch is not installed. Install it with: pip install gpytorch\n"
                "Or: pip install 'xppm-tdqn[xai]'"
            )
        raise NotImplementedError(
            "EdgeExplainer.score_trajectory() is not yet implemented. "
            "See docstring for the algorithm. "
            "Reference: Guo et al. (2021) arXiv:2212.09707"
        )
