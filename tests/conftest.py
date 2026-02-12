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
    """Minimal test configuration for MDP building."""
    return {
        "mdp": {
            "actions": {
                "n_actions": 2,
                "id2name": {0: "do_nothing", 1: "contact_headquarters"},
            },
            "decision_points": {
                "by_last_activity": {"A": [0, 1], "B": [0, 1]},
            },
            "reward": {
                "terminal": "profit",
                "non_terminal": 0.0,
            },
        },
    }
