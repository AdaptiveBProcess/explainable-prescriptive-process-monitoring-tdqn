from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def tiny_log_path(project_root: Path) -> Path:
    """Path to tiny synthetic event log for testing."""
    return project_root / "tests" / "data" / "tiny_log.csv"


@pytest.fixture
def tiny_log_with_outcome_path(project_root: Path) -> Path:
    """Path to tiny synthetic event log with outcome for testing."""
    return project_root / "tests" / "data" / "tiny_log_with_outcome.csv"


@pytest.fixture
def tmp_outdir(tmp_path: Path) -> Path:
    """Temporary output directory for test artifacts."""
    d = tmp_path / "out"
    d.mkdir()
    return d


@pytest.fixture
def test_config() -> dict:
    """Minimal test configuration matching the structure of configs/config.yaml."""
    return {
        "mdp": {
            "behavior_trigger_activity": "contact_headquarters",
            "decision_points": {
                "mode": "all",
            },
            "actions": {
                "id2name": ["do_nothing", "contact_headquarters"],
                "noop_action": "do_nothing",
            },
            "action_mask": {
                "default_valid": ["do_nothing"],
                "by_last_activity": {
                    "A": ["do_nothing", "contact_headquarters"],
                    "B": ["do_nothing", "contact_headquarters"],
                },
            },
            "reward": {
                "type": "terminal_profit_delayed",
                "intermediate_reward": 0.0,
                "terminal_column": "outcome",
            },
            "output": {
                "path": "D_offline.npz",
            },
        },
    }
