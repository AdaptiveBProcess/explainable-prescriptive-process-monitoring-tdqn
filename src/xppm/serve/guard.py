"""Policy Guard with fallback mechanisms."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class PolicyGuard:
    """
    Policy Guard con fallbacks de seguridad.

    Checks:
    1. Override humano (prioridad máxima)
    2. Action mask (hard constraint)
    3. OOD detection (features fuera de rango)
    4. Uncertainty threshold (soft constraint)
    """

    def __init__(self, config_path: Path):
        """
        Args:
            config_path: Path to policy_guard_config.json
        """
        with open(config_path) as f:
            self.config = json.load(f)

        # Thresholds
        self.tau_uncertainty = self.config.get("tau_uncertainty", 0.3)
        self.tau_ood_z = self.config.get("tau_ood_z", 3.0)
        self.max_ood_features = self.config.get("max_ood_features", 2)

        # Feature stats (para OOD)
        self.feature_stats = self.config.get("feature_stats", {})

        # Fallback action
        self.fallback_action = self.config.get(
            "fallback_action", {"action_id": 0, "action_name": "do_nothing"}
        )

        logger.info(
            f"PolicyGuard initialized with tau_unc={self.tau_uncertainty}, "
            f"tau_ood_z={self.tau_ood_z}"
        )

    def check_override(self, request: Dict) -> Tuple[bool, Optional[Dict]]:
        """Check 1: Override humano."""
        if request.get("override"):
            logger.info(f"Human override detected: {request['override']}")
            return True, {
                "action_id": request["override"]["action_id"],
                "action_name": request["override"].get("action_name", "unknown"),
                "source": "override",
                "reason": request["override"].get("reason", "human decision"),
            }
        return False, None

    def check_action_mask(self, action_id: int, valid_actions: Optional[list]) -> bool:
        """Check 2: Action mask (hard constraint)."""
        if valid_actions is None:
            return True  # No mask, todo válido

        is_valid = action_id in valid_actions
        if not is_valid:
            logger.warning(f"Action {action_id} not in valid_actions {valid_actions}")
        return is_valid

    def detect_ood(self, features: Dict[str, float]) -> Tuple[bool, Dict]:
        """
        Check 3: OOD detection via z-score.

        Returns:
            (is_ood, ood_details)
        """
        if not self.feature_stats:
            return False, {}  # No stats, skip OOD

        z_scores = {}
        ood_features = []

        for feat_name, value in features.items():
            if feat_name not in self.feature_stats:
                continue

            stats = self.feature_stats[feat_name]
            mean = stats["mean"]
            std = stats["std"]

            if std == 0:
                z = 0
            else:
                z = abs((value - mean) / std)

            z_scores[feat_name] = z

            if z > self.tau_ood_z:
                ood_features.append(feat_name)

        is_ood = len(ood_features) > self.max_ood_features

        if is_ood:
            logger.warning(
                f"OOD detected: {len(ood_features)} features out of range: {ood_features}"
            )

        return is_ood, {"ood_features": ood_features, "z_scores": z_scores}

    def check_uncertainty(self, uncertainty: float) -> Tuple[bool, str]:
        """
        Check 4: Uncertainty threshold.

        Returns:
            (should_fallback, reason)
        """
        if uncertainty > self.tau_uncertainty:
            logger.info(f"High uncertainty detected: {uncertainty:.3f} > {self.tau_uncertainty}")
            return True, (f"uncertainty={uncertainty:.3f} > threshold={self.tau_uncertainty}")
        return False, ""

    def apply_fallback(self, reason: str, request_id: str) -> Dict:
        """Aplica fallback a baseline."""
        logger.warning(f"[{request_id}] Fallback triggered: {reason}")
        return {
            "action_id": self.fallback_action["action_id"],
            "action_name": self.fallback_action["action_name"],
            "source": "baseline",
            "fallback_reason": reason,
        }

    def process(self, request: Dict, surrogate_decision: Dict) -> Dict:
        """
        Procesa una decisión con guards.

        Args:
            request: DecisionRequest as dict
            surrogate_decision: {action_id, confidence, uncertainty}

        Returns:
            Final decision dict with source and metadata
        """
        request_id = request["request_id"]

        # Check 1: Override
        override, override_decision = self.check_override(request)
        if override:
            return override_decision

        # Check 2: Action mask
        valid_actions = request.get("valid_actions")
        if not self.check_action_mask(surrogate_decision["action_id"], valid_actions):
            # Acción inválida → fallback
            return self.apply_fallback(
                f"invalid_action={surrogate_decision['action_id']}", request_id
            )

        # Check 3: OOD
        features = request["features"]
        is_ood, ood_details = self.detect_ood(features)

        if is_ood:
            return {
                **self.apply_fallback("OOD detected", request_id),
                "ood_details": ood_details,
            }

        # Check 4: Uncertainty
        uncertainty = surrogate_decision.get("uncertainty", 0)
        should_fallback, reason = self.check_uncertainty(uncertainty)

        if should_fallback:
            return self.apply_fallback(reason, request_id)

        # All checks passed → use surrogate
        return {
            **surrogate_decision,
            "source": "surrogate",
            "ood": False,
            "ood_details": ood_details,
        }
