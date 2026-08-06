"""Counterfactual state explanations: "what would the state need to look like
for the agent to choose a different action?"

Two approaches:

NearestNeighbourCounterfactual (offline, fast):
    Search the offline dataset D for a state s' closest to s such that
    argmax_a Q(s', a) == target_action.  No gradient required.
    Best when the dataset is large and covers the relevant region.

WachterCounterfactual (gradient-based, Wachter et al. 2017):
    Solve: min_{s'} λ·L_pred(s', target_action) + L_dist(s, s')
    where L_pred encourages the target action and L_dist penalises deviation.
    Operates in embedding space; requires gradient backpropagation.

Note: Both methods output *token-level* counterfactuals, which must be decoded
back to human-readable process events via the vocabulary.

References:
    Wachter et al. (2017) Counterfactual Explanations without Opening the Black Box.
      Harvard Journal of Law & Technology.
    Huber & Böhm (2022) GANterfactual-RL.  arXiv:2204.01840
    Survey: arXiv:2507.12599 §3.3
"""

from __future__ import annotations

from typing import Any

import numpy as np

from xppm.utils.logging import get_logger
from xppm.xai.base import BaseCounterfactualExplainer, CounterfactualResult

logger = get_logger(__name__)


class NearestNeighbourCounterfactual(BaseCounterfactualExplainer):
    """Find the closest dataset state that causes target_action.

    Config keys (config.yaml → xai_extensions.counterfactual, when method="nearest_neighbor"):
        distance:      "cosine" | "l2"   — embedding-space distance  (default: "cosine")
        max_candidates: int              — states to search           (default: 10000)
        dataset_path:   str              — path to D_offline.npz
    """

    @property
    def method_name(self) -> str:
        return "nearest_neighbor"

    def generate(
        self,
        state: np.ndarray,
        target_action: int,
        q_net: Any,
        valid_actions: np.ndarray | None = None,
        config: dict[str, Any] | None = None,
    ) -> CounterfactualResult:
        """Search the offline dataset for the nearest state with target_action.

        TODO: implement. Steps:
            1. Load dataset from config["dataset_path"] (D_offline.npz).
            2. Filter dataset to rows where argmax Q = target_action.
            3. Encode all candidates and the query through q_net.encoder → embeddings.
            4. Compute distances (cosine or L2) between query embedding and candidates.
            5. Return the closest candidate as the counterfactual state.
            6. Compute changed_features by comparing token IDs between state and CF.

        Args:
            state:         (L,) original token ID sequence.
            target_action: desired action in the counterfactual state.
            q_net:         TransformerQNetwork.
            valid_actions: (n_actions,) binary mask.
            config:        dict with keys: distance, max_candidates, dataset_path.

        Returns:
            CounterfactualResult.
        """
        raise NotImplementedError(
            "NearestNeighbourCounterfactual.generate() is not yet implemented. "
            "See docstring for the algorithm. "
            "This is the recommended starting point — no gradients required."
        )


class WachterCounterfactual(BaseCounterfactualExplainer):
    """Gradient-based counterfactual via embedding-space optimisation.

    Config keys (config.yaml → xai_extensions.counterfactual, when method="wachter"):
        lambda_:       float — trade-off between prediction loss and distance  (default: 0.5)
        lr:            float — Adam learning rate for optimisation             (default: 0.01)
        n_steps:       int   — gradient descent steps                         (default: 200)
        distance:      "l1" | "l2"                                            (default: "l1")
        project_to_vocab: bool — project continuous CF embedding back to
                                 nearest vocabulary token after each step     (default: True)
    """

    @property
    def method_name(self) -> str:
        return "wachter"

    def generate(
        self,
        state: np.ndarray,
        target_action: int,
        q_net: Any,
        valid_actions: np.ndarray | None = None,
        config: dict[str, Any] | None = None,
    ) -> CounterfactualResult:
        """Optimise a counterfactual in embedding space.

        TODO: implement. Steps:
            1. Embed state via q_net.embedding → x_orig (1, L, d_model).
            2. Initialise x_cf = x_orig.clone().requires_grad_(True).
            3. Optimisation loop (n_steps):
               a. Compute Q(x_cf) via _forward_from_embeddings().
               b. L_pred = -Q(x_cf)[target_action] + max_{a≠target} Q(x_cf)[a].
               c. L_dist = λ * ||x_cf - x_orig||_p.
               d. loss = L_pred + L_dist; loss.backward(); Adam step.
            4. Project x_cf to nearest vocabulary token for each position:
               argmin_tok ||emb(tok) - x_cf[pos]||_2.
            5. Build CounterfactualResult: compare CF tokens vs original tokens.

        Args:
            state:         (L,) original token ID sequence.
            target_action: desired action.
            q_net:         TransformerQNetwork (must be in eval mode with grad enabled).
            valid_actions: (n_actions,) binary mask.
            config:        dict with keys: lambda_, lr, n_steps, distance.

        Returns:
            CounterfactualResult.
        """
        raise NotImplementedError(
            "WachterCounterfactual.generate() is not yet implemented. "
            "See docstring for the algorithm. "
            "Reference: Wachter et al. (2017)"
        )
