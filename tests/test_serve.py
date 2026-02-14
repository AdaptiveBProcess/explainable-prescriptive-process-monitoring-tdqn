"""Unit tests for serving components."""

from __future__ import annotations

import json

import pytest

from xppm.serve.guard import PolicyGuard
from xppm.serve.schemas import CaseFeatures, DecisionRequest


@pytest.fixture
def sample_features():
    return CaseFeatures(
        amount=15000.0,
        est_quality=0.65,
        unc_quality=0.15,
        cum_cost=250.0,
        elapsed_time=4.5,
        prefix_len=3,
        count_validate_application=2,
        count_skip_contact=1,
        count_contact_headquarters=0,
    )


@pytest.fixture
def sample_request(sample_features):
    return DecisionRequest(
        request_id="test_001",
        case_id="case_123",
        t=3,
        features=sample_features,
    )


def test_schema_validation(sample_request):
    """Test that schema validates correctly."""
    assert sample_request.request_id == "test_001"
    assert sample_request.features.amount == 15000.0


def test_guard_override(tmp_path):
    """Test that override bypasses all checks."""
    config_path = tmp_path / "guard_config.json"
    config = {
        "tau_uncertainty": 0.3,
        "fallback_action": {"action_id": 0, "action_name": "do_nothing"},
    }
    with open(config_path, "w") as f:
        json.dump(config, f)

    guard = PolicyGuard(config_path)

    request = {
        "request_id": "test",
        "override": {
            "action_id": 2,
            "action_name": "contact_hq",
            "reason": "human decision",
        },
    }

    override, decision = guard.check_override(request)
    assert override is True
    assert decision["source"] == "override"
    assert decision["action_id"] == 2


def test_guard_ood_detection(tmp_path):
    """Test OOD detection via z-score."""
    config_path = tmp_path / "guard_config.json"
    config = {
        "tau_ood_z": 3.0,
        "max_ood_features": 0,  # 0 means: if any feature is OOD, trigger
        "feature_stats": {"amount": {"mean": 10000, "std": 2000}},
    }
    with open(config_path, "w") as f:
        json.dump(config, f)

    guard = PolicyGuard(config_path)

    # Normal case
    features = {"amount": 11000}
    is_ood, details = guard.detect_ood(features)
    assert is_ood is False

    # OOD case (z > 3)
    features_ood = {"amount": 20000}  # z = (20000-10000)/2000 = 5
    is_ood, details = guard.detect_ood(features_ood)
    assert is_ood is True
    assert "amount" in details["ood_features"]


def test_guard_uncertainty_threshold(tmp_path):
    """Test uncertainty threshold fallback."""
    config_path = tmp_path / "guard_config.json"
    config = {"tau_uncertainty": 0.3}
    with open(config_path, "w") as f:
        json.dump(config, f)

    guard = PolicyGuard(config_path)

    # Low uncertainty → pass
    should_fallback, reason = guard.check_uncertainty(0.2)
    assert should_fallback is False

    # High uncertainty → fallback
    should_fallback, reason = guard.check_uncertainty(0.5)
    assert should_fallback is True
    assert "uncertainty" in reason


def test_guard_action_mask(tmp_path):
    """Test action mask validation."""
    config_path = tmp_path / "guard_config.json"
    config = {"tau_uncertainty": 0.3}
    with open(config_path, "w") as f:
        json.dump(config, f)

    guard = PolicyGuard(config_path)

    # Valid action
    assert guard.check_action_mask(1, [0, 1, 2]) is True

    # Invalid action
    assert guard.check_action_mask(5, [0, 1, 2]) is False

    # No mask
    assert guard.check_action_mask(5, None) is True
