"""Unit tests for the Def. 4 criterion (pure functions, no model needed)."""

from __future__ import annotations

import numpy as np
import pytest

from xppm.xai.evaluability import (
    BAND_FACTOR,
    MIN_CASES,
    NULL_MARGIN_FACTOR,
    granularity_evidence,
    k_of_p,
)


def test_k_is_ceiling_with_a_floor_of_one():
    lengths = np.array([1, 3, 10, 50])
    assert list(k_of_p(lengths, 0.1)) == [1, 1, 1, 5]
    assert list(k_of_p(lengths, 0.2)) == [1, 1, 2, 10]


def test_granularity_flags_a_configuration_whose_fractions_collapse():
    """Prefixes of 2-3 events give the same k at both headline fractions."""
    ev = granularity_evidence(np.array([2, 2, 3, 3, 2]))
    assert ev["k_bar_0.1"] == pytest.approx(1.0)
    assert ev["k_bar_0.2"] == pytest.approx(1.0)
    assert ev["frac_cases_differ"] == pytest.approx(0.0)


def test_granularity_evidence_separates_knife_edge_from_real_separation():
    """A strict k_bar inequality can hold while almost no case is affected."""
    knife = granularity_evidence(np.array([2] * 95 + [6] * 5))
    real = granularity_evidence(np.array([10] * 100))
    assert knife["k_bar_0.2"] > knife["k_bar_0.1"]  # passes the paper's condition (i)
    assert knife["frac_cases_differ"] < 0.10  # but on almost no case
    assert real["frac_cases_differ"] == pytest.approx(1.0)


def test_constants_match_the_paper():
    assert MIN_CASES == 30
    assert NULL_MARGIN_FACTOR == 3.0
    assert BAND_FACTOR == 3.0


def test_declared_config_matches_the_paper():
    """Every threshold the paper cites must live in configs/config.yaml and be
    the value the pipeline actually reads (not a dead key)."""
    from pathlib import Path

    import yaml

    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "configs/config.yaml").read_text())

    ope = cfg["ope"]
    assert ope["min_ess_fraction"] == 0.10  # support gate epsilon
    assert ope["dr"]["clip_importance_weights"] == 20.0  # per-step ratio cap
    assert ope["dr"]["bootstrap"]["n"] == 200  # paired bootstrap resamples
    assert cfg["xai"]["risk_threshold_percentile"] == 50  # tau = p50

    # keys the code reads for the behavior head (regression: the lookup used
    # to target a root-level "behavior_model" that existed in no config)
    bm = ope["behavior_model"]
    assert set(bm) == {"batch_size", "epochs", "learning_rate", "label_smoothing"}

    # single random-null convention: the declared fidelity values are the ones
    # every paper script imports (evaluability.DEFAULT_*)
    from xppm.xai.evaluability import DEFAULT_N_RANDOM, DEFAULT_SEED

    assert cfg["fidelity"]["n_random"] == DEFAULT_N_RANDOM
    assert cfg["fidelity"]["seed"] == DEFAULT_SEED
