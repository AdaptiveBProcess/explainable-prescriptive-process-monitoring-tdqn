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
