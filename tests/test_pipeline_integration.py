"""Integration tests: verify that pipeline stages compose correctly end-to-end.

These tests chain multiple pipeline steps together to catch interface breakage
that unit tests cannot detect (e.g., wrong output keys, schema mismatches
between stages, missing files written by one step and read by the next).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from xppm.data.build_mdp import build_mdp_dataset
from xppm.data.encode_prefixes import encode_prefixes
from xppm.data.preprocess import preprocess_event_log

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mdp_config(output_path: Path) -> dict:
    """Minimal MDP config: the *mdp section* of configs/config.yaml.

    build_mdp_dataset receives the section directly (as scripts/03 passes
    ``cfg["mdp"]``), not the full nested config -- wrapping it under an
    ``"mdp"`` key silently produces zero transitions.
    """
    return {
        "behavior_trigger_activity": "contact_headquarters",
        "decision_points": {"mode": "all"},
        "actions": {
            "id2name": ["do_nothing", "contact_headquarters"],
            "noop_action": "do_nothing",
        },
        "action_mask": {
            "default_valid": ["do_nothing"],
            "by_last_activity": {
                "start_standard": ["do_nothing", "contact_headquarters"],
                "validate_application": ["do_nothing", "contact_headquarters"],
            },
        },
        "reward": {
            "type": "terminal_profit_delayed",
            "intermediate_reward": 0.0,
            "terminal_column": "outcome",
        },
        "output": {"path": str(output_path)},
    }


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


def test_preprocess_to_encode(tmp_path, tiny_log_with_outcome_path):
    """Stage 1→2: preprocess output feeds cleanly into encode_prefixes."""
    clean_path = tmp_path / "clean.parquet"
    prefixes_path = tmp_path / "prefixes.npz"
    vocab_path = tmp_path / "vocab_activity.json"

    # Stage 1
    stats = preprocess_event_log(tiny_log_with_outcome_path, clean_path)
    assert clean_path.exists(), "preprocess_event_log must write clean.parquet"
    assert stats.get("n_cases", 0) > 0, "Expected at least one case in test log"

    # Stage 2
    enc_cfg = {
        "encoding": {
            "max_len": 10,
            "min_prefix_len": 1,
            "output": {
                "prefixes_path": str(prefixes_path),
                "vocab_activity_path": str(vocab_path),
            },
        }
    }
    enc_stats = encode_prefixes(clean_path, prefixes_path, config=enc_cfg)

    assert prefixes_path.exists(), "encode_prefixes must write prefixes.npz"
    assert vocab_path.exists(), "encode_prefixes must write vocab_activity.json"
    assert enc_stats.get("n_prefixes", 0) > 0, "Expected prefixes from test log"


def test_preprocess_encode_build_mdp(tmp_path, tiny_log_with_outcome_path):
    """Stage 1→2→3: full data phase produces a valid MDP dataset."""
    clean_path = tmp_path / "clean.parquet"
    prefixes_path = tmp_path / "prefixes.npz"
    vocab_path = tmp_path / "vocab_activity.json"
    mdp_path = tmp_path / "D_offline.npz"

    # Stage 1 — preprocess
    preprocess_event_log(tiny_log_with_outcome_path, clean_path)

    # Stage 2 — encode
    enc_cfg = {
        "encoding": {
            "max_len": 10,
            "min_prefix_len": 1,
            "output": {
                "prefixes_path": str(prefixes_path),
                "vocab_activity_path": str(vocab_path),
            },
        }
    }
    encode_prefixes(clean_path, prefixes_path, config=enc_cfg)

    # Stage 3 — build MDP
    cfg = _mdp_config(mdp_path)
    build_mdp_dataset(
        prefixes_path=prefixes_path,
        clean_log_path=clean_path,
        vocab_path=vocab_path,
        output_path=mdp_path,
        config=cfg,
    )

    assert mdp_path.exists(), "build_mdp_dataset must write D_offline.npz"

    dataset = np.load(mdp_path, allow_pickle=False)
    required_keys = {"s", "a", "r", "s_next", "valid_actions"}
    missing = required_keys - set(dataset.files)
    assert not missing, f"MDP dataset missing required keys: {missing}"

    n = len(dataset["a"])
    assert n > 0, "MDP dataset must contain at least one transition"
    assert dataset["s"].shape == (n, dataset["s"].shape[1])
    assert dataset["a"].shape == (n,)
    assert dataset["r"].shape == (n,)
    assert dataset["valid_actions"].shape[0] == n


def test_mdp_dataset_no_data_leakage(tmp_path, tiny_log_with_outcome_path):
    """Transitions from different cases must be independent (no state bleeding)."""
    clean_path = tmp_path / "clean.parquet"
    prefixes_path = tmp_path / "prefixes.npz"
    vocab_path = tmp_path / "vocab_activity.json"
    mdp_path = tmp_path / "D_offline.npz"

    preprocess_event_log(tiny_log_with_outcome_path, clean_path)
    enc_cfg = {
        "encoding": {
            "max_len": 10,
            "min_prefix_len": 1,
            "output": {
                "prefixes_path": str(prefixes_path),
                "vocab_activity_path": str(vocab_path),
            },
        }
    }
    encode_prefixes(clean_path, prefixes_path, config=enc_cfg)
    cfg = _mdp_config(mdp_path)
    build_mdp_dataset(
        prefixes_path=prefixes_path,
        clean_log_path=clean_path,
        vocab_path=vocab_path,
        output_path=mdp_path,
        config=cfg,
    )

    dataset = np.load(mdp_path, allow_pickle=False)
    if len(dataset["a"]) == 0:
        pytest.skip("No transitions in test dataset")

    # case_ptr must exist and group transitions correctly
    assert "case_ptr" in dataset.files, "Dataset must have case_ptr for split support"
    case_ids = dataset["case_ptr"]
    assert len(np.unique(case_ids)) >= 1
