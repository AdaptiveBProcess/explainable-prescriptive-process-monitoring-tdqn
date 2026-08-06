"""Algorithmic Recourse for process monitoring recommendations.

While counterfactual states answer "what would need to change in the PAST?",
algorithmic recourse answers "what can the PATIENT/PROCESS do NOW to improve
their expected outcome?"

In the loan process context:
    Recourse = actionable advice to the applicant or process manager:
    "If the applicant provides additional income documentation (action=contact_HQ
     at this stage), the expected profit increases by €X."

The key constraint is FEASIBILITY: only suggest changes that are actually possible
(e.g. cannot decrease elapsed_time; can only suggest future interventions).

References:
    Karimi et al. (2021) Algorithmic Recourse: from Counterfactual Explanations
      to Interventions.  FAccT 2021.
    Ustun et al. (2019) Actionable Recourse in Linear Classification.  ACM FAT* 2019.
    Survey: arXiv:2507.12599 §3.3
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from xppm.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RecourseResult:
    """Output of an algorithmic recourse method.

    Attributes:
        recommended_action:  the action to take now (from the Q-policy).
        recommended_timing:  suggested step to act (if not immediate).
        expected_gain:       ΔQ between taking recourse action vs NOOP.
        feasibility_score:   0–1 score of how feasible the recourse is.
        explanation_text:    human-readable explanation string.
        metadata:            method-specific extras.
    """

    recommended_action: int
    recommended_timing: int | None
    expected_gain: float
    feasibility_score: float
    explanation_text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class AlgorithmicRecourse:
    """Generate actionable recourse recommendations from the Q-policy.

    Lightweight version: uses the existing ΔQ explanations to generate
    structured recourse advice.  No extra training required.

    Full version (TODO): integrates causal feasibility constraints using
    a structural causal model (SCM) of the loan process.

    Config keys (config.yaml → xai_extensions.recourse):
        min_gain_threshold: float — minimum ΔQ to recommend action  (default: 0.1)
        feasibility_check:  bool  — check action against action_mask (default: True)
        timing_strategy:    "immediate" | "optimal"                  (default: "immediate")
            "immediate" → recommend action now
            "optimal"   → search for the best future step to act (requires rollout)
    """

    def generate_recourse(
        self,
        state: np.ndarray,
        q_net: Any,
        valid_actions: np.ndarray | None = None,
        action_names: list[str] | None = None,
        config: dict[str, Any] | None = None,
    ) -> RecourseResult:
        """Generate recourse recommendation for a single state.

        Lightweight implementation (no gradient required):
            1. Compute Q(s, a) for all valid actions.
            2. Identify best_action = argmax_a Q(s, a).
            3. Compute ΔQ = Q(s, best_action) - Q(s, noop_action).
            4. If ΔQ > min_gain_threshold and best_action ≠ noop_action → recommend.
            5. Generate explanation_text from action name and ΔQ.

        TODO for full implementation:
            - Check action feasibility via SCM (requires causal model of process).
            - Search for optimal timing via lookahead rollout.

        Args:
            state:         (L,) token ID sequence.
            q_net:         TransformerQNetwork.
            valid_actions: (n_actions,) binary mask.
            action_names:  list of action name strings.
            config:        dict with keys: min_gain_threshold, feasibility_check.

        Returns:
            RecourseResult.
        """
        import torch

        cfg = config or {}
        min_gain = float(cfg.get("min_gain_threshold", 0.1))
        noop_id = int(cfg.get("noop_action_id", 0))

        device = next(q_net.parameters()).device
        s = torch.from_numpy(state).long().unsqueeze(0).to(device)
        mask = (s != 0).float()

        with torch.no_grad():
            q_vals = q_net(s, mask).cpu().numpy()[0]  # (n_actions,)

        if valid_actions is not None:
            q_vals[valid_actions == 0] = -np.inf

        best_action = int(np.argmax(q_vals))
        delta_q = float(q_vals[best_action] - q_vals[noop_id])

        names = action_names or [f"action_{i}" for i in range(len(q_vals))]

        if delta_q > min_gain and best_action != noop_id:
            text = (
                f"Recommended intervention: '{names[best_action]}' "
                f"(expected gain ΔQ = {delta_q:.3f} over do-nothing)."
            )
            feasibility = 1.0 if valid_actions is None else float(valid_actions[best_action])
        else:
            best_action = noop_id
            delta_q = 0.0
            feasibility = 1.0
            text = "No beneficial intervention found. Recommend do-nothing."

        return RecourseResult(
            recommended_action=best_action,
            recommended_timing=None,
            expected_gain=delta_q,
            feasibility_score=feasibility,
            explanation_text=text,
        )
