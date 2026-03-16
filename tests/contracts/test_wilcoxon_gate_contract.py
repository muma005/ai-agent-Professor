# tests/contracts/test_wilcoxon_gate_contract.py
# Day 13 Contract: Wilcoxon gate correctness invariants
# 4 contract tests — IMMUTABLE after Day 13

import pytest
from unittest.mock import patch
from tools.wilcoxon_gate import is_significantly_better, MIN_FOLDS_REQUIRED


class TestWilcoxonGateContract:

    def test_returns_bool_never_raises(self):
        """Contract: is_significantly_better always returns bool, never raises."""
        a = [0.85, 0.86, 0.84, 0.85, 0.87]
        b = [0.80, 0.81, 0.80, 0.82, 0.81]
        result = is_significantly_better(a, b)
        assert isinstance(result, bool)

        # With scipy error
        with patch("tools.wilcoxon_gate.wilcoxon", side_effect=RuntimeError("boom")):
            result2 = is_significantly_better(a, b)
            assert isinstance(result2, bool)

    def test_returns_false_on_mismatched_fold_count(self):
        """Contract: mismatched fold counts return False, never raise."""
        a = [0.85, 0.86, 0.87]
        b = [0.84, 0.85]
        result = is_significantly_better(a, b)
        assert result is False

    def test_returns_false_on_zero_differences(self):
        """Contract: identical scores return False."""
        scores = [0.85, 0.86, 0.84, 0.85, 0.87]
        result = is_significantly_better(scores, scores)
        assert result is False

    def test_falls_back_below_min_folds(self):
        """Contract: below MIN_FOLDS_REQUIRED, falls back to mean comparison."""
        # Create lists shorter than MIN_FOLDS_REQUIRED
        a = [0.90] * (MIN_FOLDS_REQUIRED - 1)
        b = [0.80] * (MIN_FOLDS_REQUIRED - 1)
        # mean(a) > mean(b) — should return True via fallback
        result = is_significantly_better(a, b)
        assert result is True
