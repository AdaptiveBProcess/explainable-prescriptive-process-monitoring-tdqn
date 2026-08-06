"""Natural Language Explanation generation for TDQN decisions.

Converts structured Q-network outputs (ΔQ, attributions, recourse) into
human-readable text suitable for process practitioners.

Two approaches:

TemplateNLExplainer (implemented):
    Fill pre-written templates with numeric values.  Fast, predictable, offline.
    Example: "At step 5, contacting HQ is recommended because the loan amount
    (€45,000) combined with uncertain quality (0.7) creates a high-risk scenario.
    The expected profit gain from this action is €1,230."

LLMNLExplainer (stub):
    Pass the structured explanation dict to an LLM (GPT-4, Claude, Llama) to
    generate fluent, context-aware explanations.  Requires API access or local
    model.  RLHF-based alignment (Ouyang et al. 2022) is the key technique
    enabling this category of XRL explanation.

References:
    Ehsan et al. (2019) Automated Rationale Generation.  IUI 2019.
    Ouyang et al. (2022) Training Language Models to Follow Instructions with
      Human Feedback (InstructGPT).  NeurIPS 2022.
    Survey: arXiv:2507.12599 §3.3
"""

from __future__ import annotations

from typing import Any

from xppm.utils.logging import get_logger

logger = get_logger(__name__)


class TemplateNLExplainer:
    """Fill template strings from structured explanation dicts.

    Inputs expected (as produced by scripts/06_explain_policy.py):
        - risk_explanation: {"V": float, "top_drivers": [{feature, attribution, token}]}
        - delta_q: {"delta_q": float, "a_star": str, "a_baseline": str}
        - recourse: RecourseResult
    """

    RISK_TEMPLATE = (
        "At this point in the process, the estimated risk level is {v:.2f}. "
        "The main risk drivers are: {drivers}. "
        "Immediate attention is {attention_level}."
    )

    INTERVENTION_TEMPLATE = (
        "Recommended action: {a_star}. "
        "Compared to doing nothing, this is expected to yield an additional "
        "profit of {delta_q:.2f} (ΔQ). "
        "The decision is primarily driven by {top_driver}."
    )

    RECOURSE_TEMPLATE = (
        "{recourse_text} "
        "This recommendation is valid at the current process stage "
        "({feasibility_pct:.0f}% feasible given current constraints)."
    )

    def explain_risk(self, risk_dict: dict[str, Any]) -> str:
        """Generate a risk explanation sentence from a risk_explanation dict."""
        v = float(risk_dict.get("V", 0.0))
        drivers = risk_dict.get("top_drivers", [])
        driver_str = ", ".join(
            f"{d.get('feature', d.get('token', '?'))} ({d.get('attribution', 0):.3f})"
            for d in drivers[:3]
        )
        attention = "recommended" if v > 0.5 else "not immediately required"
        return self.RISK_TEMPLATE.format(v=v, drivers=driver_str, attention_level=attention)

    def explain_intervention(self, delta_q_dict: dict[str, Any]) -> str:
        """Generate an intervention explanation sentence from a delta_q dict."""
        delta_q = float(delta_q_dict.get("delta_q", 0.0))
        a_star = delta_q_dict.get("a_star", "unknown action")
        top_driver = ""
        if delta_q_dict.get("top_drivers"):
            top_driver = delta_q_dict["top_drivers"][0].get("feature", "")
        return self.INTERVENTION_TEMPLATE.format(
            a_star=a_star, delta_q=delta_q, top_driver=top_driver
        )

    def explain_recourse(self, recourse: Any) -> str:
        """Generate a recourse explanation from a RecourseResult."""
        return self.RECOURSE_TEMPLATE.format(
            recourse_text=recourse.explanation_text,
            feasibility_pct=recourse.feasibility_score * 100,
        )

    def full_explanation(
        self,
        risk_dict: dict[str, Any] | None = None,
        delta_q_dict: dict[str, Any] | None = None,
        recourse: Any | None = None,
    ) -> str:
        """Compose all available explanations into a single paragraph."""
        parts = []
        if risk_dict:
            parts.append(self.explain_risk(risk_dict))
        if delta_q_dict:
            parts.append(self.explain_intervention(delta_q_dict))
        if recourse:
            parts.append(self.explain_recourse(recourse))
        return " ".join(parts) if parts else "No explanation data available."


class LLMNLExplainer:
    """LLM-based natural language explanation generation.

    Requires an LLM API (OpenAI, Anthropic, or local model via Ollama).

    Installation::
        pip install openai        # for OpenAI GPT-4
        pip install anthropic     # for Claude
        pip install ollama        # for local Llama

    Config keys (config.yaml → xai_extensions.nl_explanation):
        provider:    "openai" | "anthropic" | "ollama"
        model:       model identifier (e.g. "gpt-4o", "claude-sonnet-4-6", "llama3")
        api_key_env: environment variable name for API key
        max_tokens:  int — max response length  (default: 200)
        language:    "en" | "es"               (default: "en")
    """

    def generate(
        self,
        structured_explanation: dict[str, Any],
        config: dict[str, Any] | None = None,
    ) -> str:
        """Generate fluent NL explanation via LLM.

        TODO: implement. Steps:
            1. Build a structured prompt:
               - System: "You are an AI assistant explaining loan decisions."
               - User: JSON of the structured_explanation dict.
            2. Call LLM API (provider from config).
            3. Return generated text.
            4. Optionally cache results (same state → same explanation).

        Args:
            structured_explanation: dict with risk, delta_q, recourse sections.
            config:                 dict with keys: provider, model, max_tokens, language.

        Returns:
            Fluent explanation string.
        """
        raise NotImplementedError(
            "LLMNLExplainer.generate() is not yet implemented. "
            "Consider starting with TemplateNLExplainer for a no-dependency baseline. "
            "Reference: Ehsan et al. (2019) IUI."
        )
