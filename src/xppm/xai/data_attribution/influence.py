"""Influence Functions and TracIn for offline RL dataset attribution.

Influence Functions (Koh & Liang 2017):
    Estimate the effect of upweighting a training point z_i on the test loss L(z_test):
        I(z_i, z_test) = -∇_θ L(z_test)^T · H^{-1} · ∇_θ L(z_i)
    where H = ∑_i ∇²_θ L(z_i) is the Hessian of the training loss.

    In the DQN context:
        L(z_i)    = Huber( Q_θ(s_i, a_i) - y_i )     (TD error for transition z_i)
        L(z_test) = -Q_θ(s_test, a_target)             (we want to explain a_target)

    Computing H^{-1} exactly is O(P²) where P = # parameters.  Use approximations:
        - LiSSA (Agarwal et al. 2017): stochastic iterative approximation.
        - CG (conjugate gradient): moderate P (<10M params), single solve.
        - EK-FAC (George et al. 2018): Kronecker-factored approximation.

TracIn (Pruthi et al. 2020):
    Simpler alternative: approximate influence by summing gradient dot-products
    over training checkpoints:
        TracIn(z_i, z_test) = ∑_t η_t · ∇_θ_t L(z_test) · ∇_θ_t L(z_i)
    Requires saving multiple checkpoints during training.

References:
    Koh & Liang (2017) Understanding Black-box Predictions via Influence Functions.
      ICML 2017.  arXiv:1703.04730
    Pruthi et al. (2020) Estimating Training Data Influence by Tracing Gradient
      Descent.  NeurIPS 2020.  arXiv:2002.08484
    Survey: arXiv:2502.06869 §2.3
"""

from __future__ import annotations

from typing import Any

import numpy as np

from xppm.utils.logging import get_logger
from xppm.xai.base import BaseDataAttribution, DataAttributionResult

logger = get_logger(__name__)


class InfluenceFunctionAttribution(BaseDataAttribution):
    """Dataset attribution via Influence Functions (Koh & Liang 2017).

    Config keys (config.yaml → xai_extensions.data_attribution.influence):
        method:            "lissa" | "cg"           (default: "lissa")
        n_samples:         int  — LiSSA recursion depth          (default: 500)
        recursion_depth:   int  — LiSSA inner loop depth         (default: 10)
        damping:           float — regularisation for H⁻¹        (default: 0.01)
        scale:             float — LiSSA scaling factor          (default: 25.0)
        batch_size:        int  — mini-batch for H⁻¹v estimate   (default: 32)
        top_k:             int  — top influential transitions     (default: 10)
    """

    @property
    def method_name(self) -> str:
        return "influence_functions"

    def compute_influence(
        self,
        query_state: np.ndarray,
        query_action: int,
        dataset: dict[str, np.ndarray],
        q_net: Any,
        top_k: int = 10,
        config: dict[str, Any] | None = None,
    ) -> DataAttributionResult:
        """Compute per-transition influence scores via Influence Functions.

        TODO: implement. Steps:
            1. Compute test gradient: g_test = ∇_θ L(query_state, query_action).
            2. Approximate H^{-1} · g_test using LiSSA:
               - Sample mini-batches from dataset.
               - Iteratively refine the estimate.
            3. For each training transition z_i:
               g_i = ∇_θ L(z_i)
               I(z_i) = -g_test^T · (H^{-1} g_test) inner product with g_i.
               (Or: use only the H^{-1} g_test once, then dot with g_i for each z_i.)
            4. Sort by |I(z_i)| and return top_k.

        Args:
            query_state:   (L,) or (F,) test state.
            query_action:  action to explain.
            dataset:       offline D dict (s, a, r, s_next, ...).
            q_net:         TransformerQNetwork (must have parameters accessible).
            top_k:         number of most influential transitions.
            config:        dict with keys: method, n_samples, damping, top_k.

        Returns:
            DataAttributionResult.
        """
        raise NotImplementedError(
            "InfluenceFunctionAttribution.compute_influence() is not yet implemented. "
            "See docstring for the algorithm. "
            "Reference: Koh & Liang (2017) arXiv:1703.04730. "
            "Recommended: start with TracInAttribution (simpler, no Hessian)."
        )


class TracInAttribution(BaseDataAttribution):
    """Dataset attribution via TracIn (Pruthi et al. 2020).

    Simpler than Influence Functions: sums gradient dot-products across saved
    checkpoints.  Requires that checkpoints were saved during training
    (the training script already saves Q_theta.ckpt; extend it to save
    intermediate checkpoints at regular intervals).

    Config keys (config.yaml → xai_extensions.data_attribution.tracin):
        checkpoint_paths:  list[str]  — paths to intermediate checkpoints
        top_k:             int        — top influential transitions  (default: 10)
    """

    @property
    def method_name(self) -> str:
        return "tracin"

    def compute_influence(
        self,
        query_state: np.ndarray,
        query_action: int,
        dataset: dict[str, np.ndarray],
        q_net: Any,
        top_k: int = 10,
        config: dict[str, Any] | None = None,
    ) -> DataAttributionResult:
        """Compute TracIn influence scores.

        TODO: implement. Steps:
            1. Load checkpoint list from config["checkpoint_paths"].
            2. For each checkpoint c:
               a. Load q_net weights from checkpoint c.
               b. Compute g_test_c = ∇_θ_c L(query_state, query_action).
               c. For each training batch, compute g_i_c = ∇_θ_c L(z_i).
               d. tracin_c[i] = η_c · g_test_c · g_i_c.
            3. Sum across checkpoints: tracin[i] = ∑_c tracin_c[i].
            4. Return top_k most influential transitions.

        Args:
            query_state:   (L,) or (F,) test state.
            query_action:  action to explain.
            dataset:       offline D dict.
            q_net:         TransformerQNetwork.
            top_k:         number of most influential transitions.
            config:        dict with keys: checkpoint_paths, top_k.

        Returns:
            DataAttributionResult.
        """
        raise NotImplementedError(
            "TracInAttribution.compute_influence() is not yet implemented. "
            "See docstring for the algorithm. "
            "Reference: Pruthi et al. (2020) arXiv:2002.08484"
        )
