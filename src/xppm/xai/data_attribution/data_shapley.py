"""Data Shapley Values for offline RL dataset attribution.

Data Shapley (Ghorbani & Zou 2019) assigns a fair attribution value to each
training data point by treating model performance as a cooperative game:

    φ_i = ∑_{S ⊆ D\{i}} [|S|!(|D|-|S|-1)!/|D|!] · [V(S ∪ {i}) - V(S)]

where V(S) is the model performance trained on subset S.

Direct computation requires O(2^N) model retrainings.  Efficient approximations:
    - Truncated Monte Carlo: sample subsets, stop when marginal contribution < ε.
    - GroupTesting-Shapley: use group testing to identify high-influence points.
    - KNN-Shapley (Jia et al. 2019): closed-form for KNN classifiers.
      Adapted here by treating the Q-network as a soft KNN.

For offline RL the "value function" V(S) can be:
    - Policy performance (OPE estimate) when trained only on S.
    - Q-value accuracy on held-out test transitions.

References:
    Ghorbani & Zou (2019) Data Shapley: Equitable Valuation of Data for Machine
      Learning.  ICML 2019.  arXiv:1904.02868
    Jia et al. (2019) Efficient Task-Specific Data Valuation for Nearest Neighbor
      Algorithms.  VLDB 2019.  arXiv:1908.08619
    Survey: arXiv:2502.06869 §2.3
"""

from __future__ import annotations

from typing import Any

import numpy as np

from xppm.utils.logging import get_logger
from xppm.xai.base import BaseDataAttribution, DataAttributionResult

logger = get_logger(__name__)


class DataShapleyAttribution(BaseDataAttribution):
    """Data Shapley attribution for offline RL transitions.

    Config keys (config.yaml → xai_extensions.data_attribution.data_shapley):
        method:          "monte_carlo" | "knn"          (default: "monte_carlo")
        n_permutations:  int  — Monte Carlo samples     (default: 100)
        max_fraction:    float — max subset fraction    (default: 0.5)
        value_fn:        "ope" | "q_accuracy"           (default: "q_accuracy")
        top_k:           int  — top transitions         (default: 10)
        truncation_eps:  float — stop when |ΔV| < eps  (default: 0.01)
    """

    @property
    def method_name(self) -> str:
        return "data_shapley"

    def compute_influence(
        self,
        query_state: np.ndarray,
        query_action: int,
        dataset: dict[str, np.ndarray],
        q_net: Any,
        top_k: int = 10,
        config: dict[str, Any] | None = None,
    ) -> DataAttributionResult:
        """Compute Data Shapley values via truncated Monte Carlo.

        TODO: implement. Steps:
            1. Define V(S): train q_net on subset S, evaluate on held-out test.
               OR (cheaper): use Q-value difference on the query point as a proxy.
            2. Sample n_permutations random orderings of the training dataset.
            3. For each permutation, compute marginal contributions V(S ∪ {i}) - V(S).
            4. Average marginals across permutations → φ_i for each transition.
            5. Return top_k by |φ_i|.

        Note: Full Shapley requires retraining the model for each subset — extremely
        expensive.  Consider KNN-Shapley as a faster approximation:
            - Encode all transitions via q_net encoder → embeddings.
            - Use KNN in embedding space with closed-form Shapley formula.

        Args:
            query_state:  (L,) test state.
            query_action: action to explain.
            dataset:      offline D dict.
            q_net:        TransformerQNetwork.
            top_k:        number of top transitions.
            config:       dict with keys: method, n_permutations, value_fn, top_k.

        Returns:
            DataAttributionResult.
        """
        raise NotImplementedError(
            "DataShapleyAttribution.compute_influence() is not yet implemented. "
            "See docstring for the algorithm and efficiency considerations. "
            "Recommended starting point: KNN-Shapley in embedding space. "
            "Reference: Ghorbani & Zou (2019) arXiv:1904.02868"
        )
