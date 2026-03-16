# tests/test_day17_quality.py
# -------------------------------------------------------------------------
# Day 17: Wilcoxon feature gate + null importance two-stage filter
# 44 adversarial tests — IMMUTABLE after Day 17
# -------------------------------------------------------------------------

import ast
import gc
import json
import os
import re
import subprocess
import sys
import textwrap
import time

import numpy as np
import polars as pl
import pytest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock, patch, call

from tools.wilcoxon_gate import (
    is_feature_worth_adding,
    feature_gate_result,
    is_significantly_better,
    P_VALUE_THRESHOLD,
    MIN_FOLDS_REQUIRED,
)
from tools.null_importance import (
    NullImportanceResult,
    _run_stage1_permutation_filter,
    _run_stage2_null_importance_persistent_sandbox,
    run_null_importance_filter,
    STAGE2_SCRIPT_TEMPLATE,
    N_STAGE1_SHUFFLES,
    STAGE1_DROP_PERCENTILE,
)
from agents.feature_factory import (
    _evaluate_candidate_feature,
    _apply_null_importance_filter,
    _quick_cv,
)
from core.lineage import log_event, read_lineage


# ── Shared helpers ───────────────────────────────────────────────

def _make_synthetic_dataset(n_samples=500, n_signal=5, n_noise=15, seed=42):
    """Create a dataset with n_signal useful + n_noise random features."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n_samples).astype(float)

    data = {}
    for i in range(n_signal):
        data[f"signal_{i}"] = y * 2 + rng.normal(0, 0.3, n_samples)
    for i in range(n_noise):
        data[f"noise_{i}"] = rng.normal(0, 1, n_samples)

    df = pl.DataFrame(data)
    return df, y


# ═══════════════════════════════════════════════════════════════════
# BLOCK 1 — WILCOXON FEATURE GATE: CORRECTNESS (10 tests)
# ═══════════════════════════════════════════════════════════════════

class TestWilcoxonFeatureGate:

    def test_returns_true_when_feature_clearly_improves_cv(self):
        """1.1: Clear improvement → True."""
        baseline = [0.80, 0.81, 0.80]
        augmented = [0.85, 0.86, 0.85]
        assert is_feature_worth_adding(baseline, augmented, "good_feature") is True

    def test_returns_false_when_feature_adds_noise(self):
        """1.2: Scores differ by < 0.001 → False."""
        baseline  = [0.850, 0.851, 0.850, 0.851, 0.850]
        augmented = [0.8499, 0.8509, 0.8499, 0.8509, 0.8499]
        assert is_feature_worth_adding(baseline, augmented, "noisy") is False

    def test_returns_false_when_feature_hurts_performance(self):
        """1.3: Augmented below baseline → False."""
        baseline = [0.85, 0.86, 0.85]
        augmented = [0.80, 0.81, 0.80]
        assert is_feature_worth_adding(baseline, augmented, "bad_feature") is False

    def test_alternative_is_greater_not_less(self):
        """1.4: Bug guard — must use alternative='greater' not 'less'."""
        # When augmented is WORSE, 'less' would return True (inverted).
        # With 'greater', it correctly returns False.
        baseline = [0.90, 0.91, 0.90, 0.91, 0.90]
        augmented = [0.80, 0.81, 0.80, 0.81, 0.80]
        result = is_feature_worth_adding(baseline, augmented, "should_fail")
        assert result is False, "alternative='less' would wrongly return True here"

        # Also verify via source inspection
        import inspect
        source = inspect.getsource(is_feature_worth_adding)
        assert 'alternative="greater"' in source or "alternative='greater'" in source

    def test_three_fold_quick_cv_falls_back_to_mean_comparison(self):
        """1.5: 3-fold (< MIN_FOLDS) → returns bool without raising, uses fallback."""
        baseline = [0.80, 0.81, 0.80]
        augmented = [0.85, 0.86, 0.85]
        result = is_feature_worth_adding(baseline, augmented, "quick_cv")
        assert isinstance(result, bool)
        # With 3 folds < MIN_FOLDS_REQUIRED=5, it uses mean comparison
        # augmented mean > baseline mean → True
        assert result is True

    def test_feature_gate_result_has_gate_type_field(self):
        """1.6: gate_type must be 'feature_selection', not 'model_comparison'."""
        baseline = [0.80, 0.81, 0.80, 0.81, 0.80]
        augmented = [0.85, 0.86, 0.85, 0.86, 0.85]
        result = feature_gate_result(baseline, augmented, "test_feat")
        assert result["gate_type"] == "feature_selection"
        assert result["gate_type"] != "model_comparison"

    def test_feature_gate_result_has_decision_keep_or_drop(self):
        """1.7: KEEP when passed, DROP when failed."""
        # Passing case
        baseline_good = [0.80, 0.81, 0.80]
        augmented_good = [0.90, 0.91, 0.90]
        result_pass = feature_gate_result(baseline_good, augmented_good, "keep_me")
        assert result_pass["decision"] == "KEEP"

        # Failing case
        baseline_bad = [0.90, 0.91, 0.90]
        augmented_bad = [0.80, 0.81, 0.80]
        result_fail = feature_gate_result(baseline_bad, augmented_bad, "drop_me")
        assert result_fail["decision"] == "DROP"

    def test_feature_gate_result_has_feature_name(self):
        """1.8: feature_name must be present and match input."""
        result = feature_gate_result(
            [0.80, 0.81, 0.80], [0.85, 0.86, 0.85],
            feature_name="target_enc_cabin"
        )
        assert result["feature_name"] == "target_enc_cabin"

    def test_is_feature_worth_adding_never_raises(self):
        """1.9: Mismatched lengths, empty, zero, NaN → all return False."""
        # Mismatched lengths
        assert is_feature_worth_adding([0.8, 0.9], [0.8], "mismatch") is False
        # Empty lists
        assert is_feature_worth_adding([], [], "empty") is False
        # All zero
        assert is_feature_worth_adding([0, 0, 0], [0, 0, 0], "zeros") is False
        # NaN scores
        assert is_feature_worth_adding(
            [float('nan'), 0.8, 0.8], [0.9, 0.9, 0.9], "nan"
        ) is False

    def test_feature_factory_logs_gate_decision_to_lineage(self, tmp_path):
        """1.10: _evaluate_candidate_feature logs to lineage."""
        session_id = "test_gate_lineage"
        out_dir = tmp_path / "outputs" / session_id / "logs"
        out_dir.mkdir(parents=True)

        state = {
            "session_id": session_id,
            "task_type": "binary",
        }

        rng = np.random.default_rng(42)
        n = 200
        y = rng.integers(0, 2, size=n).astype(float)
        X_base = pl.DataFrame({"f1": rng.normal(0, 1, n), "f2": rng.normal(0, 1, n)})
        X_with = X_base.with_columns(pl.Series("f3", y * 2 + rng.normal(0, 0.1, n)))

        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            _evaluate_candidate_feature(state, X_base, X_with, y, "test_feature")
            entries = read_lineage(session_id)
        finally:
            os.chdir(old_cwd)

        gate_entries = [e for e in entries if e["action"] == "wilcoxon_feature_gate"]
        assert len(gate_entries) >= 1
        assert gate_entries[0]["values_changed"]["feature_name"] == "test_feature"


# ═══════════════════════════════════════════════════════════════════
# BLOCK 2 — STAGE 1 PERMUTATION FILTER (10 tests)
# ═══════════════════════════════════════════════════════════════════

class TestStage1PermutationFilter:

    def test_stage1_drops_approximately_65_percent(self):
        """2.1: 20 features (5 signal, 15 noise) → 55-75% dropped."""
        df, y = _make_synthetic_dataset(n_samples=500, n_signal=5, n_noise=15)
        feature_names = df.columns
        survivors, dropped, _ = _run_stage1_permutation_filter(
            df, y, feature_names, n_shuffles=5, drop_percentile=0.65
        )
        drop_frac = len(dropped) / 20
        assert 0.55 <= drop_frac <= 0.80, f"drop fraction {drop_frac:.2f} not in [0.55, 0.80]"

    def test_stage1_actual_importances_from_real_y_not_shuffled(self):
        """2.2: Bug guard — actual importances must come from real-y model."""
        rng = np.random.default_rng(42)
        n = 1000
        y = rng.integers(0, 2, size=n).astype(float)

        data = {}
        # Multiple features perfectly correlated with target
        for i in range(5):
            data[f"signal_{i}"] = y * (i + 2.0) + rng.normal(0, 0.01, n)
        for i in range(10):
            data[f"noise_{i}"] = rng.normal(0, 1, n)

        df = pl.DataFrame(data)
        feature_names = df.columns

        _, _, actual_importances = _run_stage1_permutation_filter(
            df, y, feature_names, n_shuffles=5
        )

        # Sum of signal importances must exceed sum of noise importances
        signal_imp_sum = sum(actual_importances[f"signal_{i}"] for i in range(5))
        noise_imp_sum = sum(actual_importances[f"noise_{i}"] for i in range(10))
        assert signal_imp_sum > noise_imp_sum, (
            f"Signal importance sum ({signal_imp_sum}) should exceed noise sum ({noise_imp_sum}). "
            f"Bug: actual importances may come from shuffled model."
        )

    def test_stage1_uses_importance_ratio_not_absolute(self):
        """2.3: Bug guard — filtering must use ratio, not absolute importance."""
        # Mock the entire Stage 1 to verify ratio-based behavior
        import lightgbm as lgb

        call_count = {"n": 0}
        feature_names = ["feat_A", "feat_B"]
        # feat_A: actual=100, null_mean=95 → ratio≈1.05
        # feat_B: actual=5, null_mean=1 → ratio=5.0
        # Ratio-based: feat_B (5.0) survives, feat_A (1.05) dropped

        importances_sequence = [
            np.array([100.0, 5.0]),   # real y
            np.array([95.0, 1.0]),    # shuffle 1
            np.array([95.0, 1.0]),    # shuffle 2
            np.array([95.0, 1.0]),    # shuffle 3
            np.array([95.0, 1.0]),    # shuffle 4
            np.array([95.0, 1.0]),    # shuffle 5
        ]

        class FakeModel:
            def __init__(self, **kw):
                self.feature_importances_ = None
            def fit(self, X, y):
                self.feature_importances_ = importances_sequence[call_count["n"]]
                call_count["n"] += 1
                return self

        rng = np.random.default_rng(42)
        n = 200
        y = rng.integers(0, 2, size=n).astype(float)
        df = pl.DataFrame({
            "feat_A": rng.normal(0, 1, n),
            "feat_B": y * 2 + rng.normal(0, 0.3, n),
        })

        with patch("lightgbm.LGBMClassifier", FakeModel):
            survivors, dropped, actual_imps = _run_stage1_permutation_filter(
                df, y, feature_names, n_shuffles=5, drop_percentile=0.50
            )

        # feat_B (ratio=5.0) should survive; feat_A (ratio≈1.05) should be dropped
        assert "feat_B" in survivors, "feat_B (high ratio) should survive"
        assert "feat_A" in dropped, "feat_A (low ratio despite high absolute) should be dropped"

    def test_stage1_safety_fallback_when_all_features_dropped(self):
        """2.4: drop_percentile=1.0 → safety fallback returns all features."""
        df, y = _make_synthetic_dataset(n_samples=200, n_signal=3, n_noise=7)
        feature_names = df.columns

        # Mock Stage 1 to return empty survivors
        with patch("tools.null_importance._run_stage1_permutation_filter",
                   return_value=([], list(feature_names), {f: 0.0 for f in feature_names})):
            result = run_null_importance_filter(
                df, y, feature_names,
                n_stage1_shuffles=2,
                stage1_drop_percentile=1.0,
            )
        assert set(result.survivors) == set(feature_names)
        assert result.stage1_drop_count == 0

    def test_stage1_gc_called_after_each_null_model(self):
        """2.5: gc.collect() called ≥ n_shuffles times (once per shuffle)."""
        df, y = _make_synthetic_dataset(n_samples=200, n_signal=5, n_noise=5)
        feature_names = df.columns

        gc_calls = {"count": 0}
        original_gc = gc.collect

        def counting_gc(*args, **kwargs):
            gc_calls["count"] += 1
            return original_gc(*args, **kwargs)

        with patch("tools.null_importance.gc.collect", counting_gc):
            _run_stage1_permutation_filter(df, y, feature_names, n_shuffles=5)

        # At least 5 calls (one per shuffle) + 1 for real model = 6
        assert gc_calls["count"] >= 5, (
            f"gc.collect called {gc_calls['count']} times, expected >= 5 "
            f"(one per shuffle iteration)"
        )

    def test_stage1_uses_fixed_random_seed(self):
        """2.6: Deterministic results with fixed seed."""
        df, y = _make_synthetic_dataset(n_samples=200, n_signal=5, n_noise=5)
        feature_names = df.columns

        _, dropped1, _ = _run_stage1_permutation_filter(df, y, feature_names, n_shuffles=3)
        _, dropped2, _ = _run_stage1_permutation_filter(df, y, feature_names, n_shuffles=3)

        assert dropped1 == dropped2, "Stage 1 results should be deterministic with fixed seed"

    def test_stage1_returns_correct_actual_importances_dict(self):
        """2.7: actual_importances has correct keys and non-negative values."""
        df, y = _make_synthetic_dataset(n_samples=200, n_signal=3, n_noise=7)
        feature_names = df.columns
        _, _, actual_importances = _run_stage1_permutation_filter(
            df, y, feature_names, n_shuffles=2
        )
        assert set(actual_importances.keys()) == set(feature_names)
        assert all(isinstance(v, float) for v in actual_importances.values())
        assert all(v >= 0 for v in actual_importances.values())

    def test_stage1_skipped_on_fewer_than_10_features(self):
        """2.8: <10 features → skip, return all as survivors."""
        rng = np.random.default_rng(42)
        n = 200
        y = rng.integers(0, 2, size=n).astype(float)
        df = pl.DataFrame({f"f{i}": rng.normal(0, 1, n) for i in range(7)})
        feature_names = df.columns

        result = run_null_importance_filter(df, y, feature_names)
        assert len(result.survivors) == 7
        assert result.stage1_drop_count == 0
        assert result.total_features_input == 7
        assert result.total_features_output == 7

    def test_stage1_lgbm_params_n_jobs_is_1(self):
        """2.9: n_jobs must be 1, not -1 (OOM risk)."""
        import inspect
        source = inspect.getsource(_run_stage1_permutation_filter)
        # Find n_jobs in lgb_params
        assert '"n_jobs":        1' in source or '"n_jobs": 1' in source or "'n_jobs': 1" in source

    def test_stage1_result_includes_correct_drop_count(self):
        """2.10: Count consistency: dropped + survivors == total."""
        df, y = _make_synthetic_dataset(n_samples=300, n_signal=5, n_noise=15)
        feature_names = df.columns
        result = run_null_importance_filter(
            df, y, feature_names,
            n_stage1_shuffles=2,
            n_stage2_shuffles=5,
        )
        # Stage 1 counts should be consistent
        survivors_after_s1 = result.total_features_input - result.stage1_drop_count
        assert result.stage1_drop_count == len(result.dropped_stage1)
        assert survivors_after_s1 + result.stage1_drop_count == len(feature_names)


# ═══════════════════════════════════════════════════════════════════
# BLOCK 3 — STAGE 2 NULL IMPORTANCE (10 tests)
# ═══════════════════════════════════════════════════════════════════

class TestStage2NullImportance:

    def test_stage2_runs_in_single_execute_code_call(self):
        """3.1: Bug guard — execute_code called exactly once, not 50 times."""
        mock_result = {
            "success": True,
            "stdout": json.dumps({
                "actual_importances": {"f1": 10.0, "f2": 5.0},
                "null_distributions": {"f1": [1.0] * 10, "f2": [1.0] * 10},
            }),
            "stderr": "",
            "returncode": 0,
            "timed_out": False,
            "backend": "mock",
        }

        rng = np.random.default_rng(42)
        X = pl.DataFrame({"f1": rng.normal(0, 1, 100), "f2": rng.normal(0, 1, 100)})
        y = rng.integers(0, 2, size=100).astype(float)

        import tools.null_importance as ni_module
        with patch.object(ni_module, "execute_code", return_value=mock_result) as mock_exec:
            _run_stage2_null_importance_persistent_sandbox(
                X, y, ["f1", "f2"], n_shuffles=50
            )
            assert mock_exec.call_count == 1, (
                f"execute_code called {mock_exec.call_count} times, expected 1"
            )

    def test_stage2_script_outputs_json_to_stdout(self):
        """3.2: Rendered script produces valid JSON on stdout."""
        rng = np.random.default_rng(42)
        X_list = rng.normal(0, 1, (50, 3)).tolist()
        y_list = rng.integers(0, 2, size=50).tolist()

        script = STAGE2_SCRIPT_TEMPLATE.format(
            X_list=json.dumps(X_list),
            y_list=json.dumps(y_list),
            feature_names=json.dumps(["f1", "f2", "f3"]),
            n_shuffles=2,
            task_type="binary",
            random_seed=42,
        )

        # Execute the rendered script
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, f"Script failed: {result.stderr}"
        payload = json.loads(result.stdout.strip())
        assert "actual_importances" in payload
        assert "null_distributions" in payload

    def test_stage2_script_progress_messages_go_to_stderr(self):
        """3.3: Bug guard — progress messages must NOT be in stdout."""
        rng = np.random.default_rng(42)
        X_list = rng.normal(0, 1, (50, 3)).tolist()
        y_list = rng.integers(0, 2, size=50).tolist()

        script = STAGE2_SCRIPT_TEMPLATE.format(
            X_list=json.dumps(X_list),
            y_list=json.dumps(y_list),
            feature_names=json.dumps(["f1", "f2", "f3"]),
            n_shuffles=10,  # triggers progress at i=10
            task_type="binary",
            random_seed=42,
        )

        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, f"Script failed: {result.stderr}"
        # stdout must be pure JSON — no progress messages
        assert "Progress:" not in result.stdout
        # Progress should be in stderr
        assert "Progress:" in result.stderr or result.stderr == ""
        # stdout must parse as valid JSON
        json.loads(result.stdout.strip())

    def test_stage2_keeps_features_above_95th_percentile_threshold(self):
        """3.4: Feature with actual >> null 95th pct → survives."""
        mock_result = {
            "success": True,
            "stdout": json.dumps({
                "actual_importances": {"strong": 150.0},
                "null_distributions": {"strong": list(range(1, 51))},
            }),
            "stderr": "",
            "returncode": 0,
            "timed_out": False,
            "backend": "mock",
        }

        X = pl.DataFrame({"strong": np.random.default_rng(42).normal(0, 1, 100)})
        y = np.random.default_rng(42).integers(0, 2, size=100).astype(float)

        import tools.null_importance as ni_module
        with patch.object(ni_module, "execute_code", return_value=mock_result):
            survivors, dropped, _, _ = _run_stage2_null_importance_persistent_sandbox(
                X, y, ["strong"], n_shuffles=50
            )
        assert "strong" in survivors
        assert "strong" not in dropped

    def test_stage2_drops_features_below_95th_percentile_threshold(self):
        """3.5: Feature with actual < null 95th pct → dropped."""
        mock_result = {
            "success": True,
            "stdout": json.dumps({
                "actual_importances": {"weak": 10.0},
                "null_distributions": {"weak": list(range(5, 55))},
            }),
            "stderr": "",
            "returncode": 0,
            "timed_out": False,
            "backend": "mock",
        }

        X = pl.DataFrame({"weak": np.random.default_rng(42).normal(0, 1, 100)})
        y = np.random.default_rng(42).integers(0, 2, size=100).astype(float)

        import tools.null_importance as ni_module
        with patch.object(ni_module, "execute_code", return_value=mock_result):
            survivors, dropped, _, _ = _run_stage2_null_importance_persistent_sandbox(
                X, y, ["weak"], n_shuffles=50
            )
        assert "weak" in dropped
        assert "weak" not in survivors

    def test_stage2_graceful_fallback_on_sandbox_failure(self):
        """3.6: Sandbox crash → return all Stage 1 survivors."""
        mock_result = {
            "returncode": -1,
            "stdout": "",
            "stderr": "OOM",
            "timed_out": False,
            "backend": "docker",
        }

        X = pl.DataFrame({
            "f1": np.random.default_rng(42).normal(0, 1, 100),
            "f2": np.random.default_rng(42).normal(0, 1, 100),
        })
        y = np.random.default_rng(42).integers(0, 2, size=100).astype(float)

        import tools.null_importance as ni_module
        with patch.object(ni_module, "execute_code", return_value=mock_result):
            survivors, dropped, _, _ = _run_stage2_null_importance_persistent_sandbox(
                X, y, ["f1", "f2"], n_shuffles=50
            )
        assert set(survivors) == {"f1", "f2"}
        assert dropped == []

    def test_stage2_graceful_fallback_on_invalid_json_output(self):
        """3.7: Invalid JSON stdout → return all survivors."""
        mock_result = {
            "success": True,
            "stdout": "not valid json at all",
            "stderr": "",
            "returncode": 0,
            "timed_out": False,
            "backend": "mock",
        }

        X = pl.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [4.0, 5.0, 6.0]})
        y = np.array([0.0, 1.0, 0.0])

        import tools.null_importance as ni_module
        with patch.object(ni_module, "execute_code", return_value=mock_result):
            survivors, dropped, _, _ = _run_stage2_null_importance_persistent_sandbox(
                X, y, ["f1", "f2"], n_shuffles=10
            )
        assert set(survivors) == {"f1", "f2"}
        assert dropped == []

    def test_stage2_null_distributions_correct_length(self):
        """3.8: n_shuffles=10 → each null distribution has 10 entries."""
        n_shuffles = 10
        null_dist = {
            "f1": [float(i) for i in range(n_shuffles)],
            "f2": [float(i) for i in range(n_shuffles)],
        }
        mock_result = {
            "success": True,
            "stdout": json.dumps({
                "actual_importances": {"f1": 100.0, "f2": 100.0},
                "null_distributions": null_dist,
            }),
            "stderr": "",
            "returncode": 0,
            "timed_out": False,
            "backend": "mock",
        }

        X = pl.DataFrame({"f1": [1.0] * 50, "f2": [2.0] * 50})
        y = np.zeros(50)

        import tools.null_importance as ni_module
        with patch.object(ni_module, "execute_code", return_value=mock_result):
            _, _, distributions, _ = _run_stage2_null_importance_persistent_sandbox(
                X, y, ["f1", "f2"], n_shuffles=n_shuffles
            )
        for feat, dist in distributions.items():
            assert len(dist) == n_shuffles, (
                f"Feature {feat}: expected {n_shuffles} entries, got {len(dist)}"
            )

    def test_stage2_threshold_percentiles_stored_per_feature(self):
        """3.9: threshold_percentiles has one entry per Stage 1 survivor."""
        mock_result = {
            "success": True,
            "stdout": json.dumps({
                "actual_importances": {"f1": 100.0, "f2": 50.0, "f3": 200.0},
                "null_distributions": {
                    "f1": [1.0] * 50,
                    "f2": [1.0] * 50,
                    "f3": [1.0] * 50,
                },
            }),
            "stderr": "",
            "returncode": 0,
            "timed_out": False,
            "backend": "mock",
        }

        X = pl.DataFrame({
            "f1": [1.0] * 100, "f2": [2.0] * 100, "f3": [3.0] * 100,
        })
        y = np.zeros(100)

        import tools.null_importance as ni_module
        with patch.object(ni_module, "execute_code", return_value=mock_result):
            _, _, _, threshold_pcts = _run_stage2_null_importance_persistent_sandbox(
                X, y, ["f1", "f2", "f3"], n_shuffles=50
            )
        assert set(threshold_pcts.keys()) == {"f1", "f2", "f3"}
        assert all(isinstance(v, float) and v >= 0 for v in threshold_pcts.values())

    def test_stage2_actual_vs_threshold_dict_has_all_survivors(self):
        """3.10: actual_vs_threshold has required keys for all Stage 1 survivors."""
        df, y = _make_synthetic_dataset(n_samples=300, n_signal=5, n_noise=15)
        feature_names = df.columns

        # Mock Stage 2 to return controlled data
        mock_result = {
            "success": True,
            "stdout": json.dumps({
                "actual_importances": {f: 10.0 for f in feature_names},
                "null_distributions": {f: [1.0] * 50 for f in feature_names},
            }),
            "stderr": "",
            "returncode": 0,
            "timed_out": False,
            "backend": "mock",
        }

        import tools.null_importance as ni_module
        with patch.object(ni_module, "execute_code", return_value=mock_result):
            result = run_null_importance_filter(
                df, y, feature_names,
                n_stage1_shuffles=2,
                n_stage2_shuffles=50,
            )

        for f in result.actual_vs_threshold:
            entry = result.actual_vs_threshold[f]
            assert "actual" in entry
            assert "threshold" in entry
            assert "ratio" in entry
            assert "passed" in entry


# ═══════════════════════════════════════════════════════════════════
# BLOCK 4 — PERSISTENT SANDBOX PATTERN (6 tests)
# ═══════════════════════════════════════════════════════════════════

class TestPersistentSandboxPattern:

    def test_stage2_script_is_self_contained_python(self):
        """4.1: Rendered script must be valid Python (ast.parse succeeds)."""
        rng = np.random.default_rng(42)
        X_list = rng.normal(0, 1, (20, 3)).tolist()
        y_list = rng.integers(0, 2, size=20).tolist()

        script = STAGE2_SCRIPT_TEMPLATE.format(
            X_list=json.dumps(X_list),
            y_list=json.dumps(y_list),
            feature_names=json.dumps(["a", "b", "c"]),
            n_shuffles=5,
            task_type="binary",
            random_seed=42,
        )

        # Must parse without SyntaxError
        tree = ast.parse(script)
        assert tree is not None

        # Check imports are from standard lib + numpy + lightgbm + json only
        import_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    import_names.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    import_names.add(node.module.split(".")[0])

        allowed = {"json", "gc", "sys", "numpy", "np", "lightgbm", "lgb"}
        unexpected = import_names - allowed
        assert not unexpected, f"Unexpected imports in self-contained script: {unexpected}"

    def test_stage2_script_handles_large_input_without_truncation(self):
        """4.2: 100 features × 1000 rows → script < 5MB."""
        rng = np.random.default_rng(42)
        X_list = rng.normal(0, 1, (1000, 100)).tolist()
        y_list = rng.integers(0, 2, size=1000).tolist()

        script = STAGE2_SCRIPT_TEMPLATE.format(
            X_list=json.dumps(X_list),
            y_list=json.dumps(y_list),
            feature_names=json.dumps([f"f{i}" for i in range(100)]),
            n_shuffles=50,
            task_type="binary",
            random_seed=42,
        )

        script_bytes = len(script.encode("utf-8"))
        assert script_bytes < 5 * 1024 * 1024, (
            f"Script is {script_bytes / 1024 / 1024:.1f} MB, exceeds 5MB limit"
        )

    def test_execute_code_timeout_set_to_600_seconds(self):
        """4.3: timeout must be 600 seconds in the execute_code call."""
        import inspect
        source = inspect.getsource(_run_stage2_null_importance_persistent_sandbox)
        assert "timeout_seconds=600" in source

    def test_stage2_uses_execute_code_not_subprocess_directly(self):
        """4.4: Must use execute_code from tools.e2b_sandbox, not subprocess."""
        import inspect
        source = inspect.getsource(_run_stage2_null_importance_persistent_sandbox)
        assert "execute_code" in source
        assert "subprocess.run" not in source
        assert "from tools.e2b_sandbox import execute_code" in source or \
               "execute_code(" in source

    def test_null_importance_result_excluded_from_redis_checkpoint(self):
        """4.5: NullImportanceResult dataclass is not JSON-serialisable → excluded."""
        from memory.redis_state import _is_serialisable

        result = NullImportanceResult(
            survivors=["f1", "f2"],
            dropped_stage1=["f3"],
            dropped_stage2=["f4"],
            stage1_importances={"f1": 10.0},
            null_distributions={"f1": [1.0, 2.0]},
            threshold_percentiles={"f1": 5.0},
            actual_vs_threshold={"f1": {"actual": 10.0}},
            total_features_input=4,
            total_features_output=2,
            stage1_drop_count=1,
            stage2_drop_count=1,
            elapsed_seconds=5.0,
        )

        # The dataclass itself is not JSON-serialisable
        assert not _is_serialisable(result), (
            "NullImportanceResult should NOT be JSON-serialisable"
        )

        # A state dict with this field should still checkpoint
        # (save_state filters non-serialisable values)
        from memory.redis_state import save_state, load_state
        state = {
            "session_id": "test_redis",
            "null_importance_result": result,
            "survivors": ["f1", "f2"],
        }
        saved = save_state("test_redis", state)
        assert saved is True
        loaded = load_state("test_redis")
        assert loaded is not None
        # null_importance_result should be excluded
        assert "null_importance_result" not in loaded
        assert loaded["survivors"] == ["f1", "f2"]

    def test_stage2_script_gc_collect_inside_shuffle_loop(self):
        """4.6: gc.collect() must be inside the shuffle loop body."""
        rng = np.random.default_rng(42)
        script = STAGE2_SCRIPT_TEMPLATE.format(
            X_list=json.dumps([[1, 2], [3, 4]]),
            y_list=json.dumps([0, 1]),
            feature_names=json.dumps(["a", "b"]),
            n_shuffles=5,
            task_type="binary",
            random_seed=42,
        )

        tree = ast.parse(script)

        # Find the for loop with 'n_shuffles' as upper bound
        found_gc_in_loop = False
        for node in ast.walk(tree):
            if isinstance(node, ast.For):
                # Check if loop body contains gc.collect()
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        func = child.func
                        if (isinstance(func, ast.Attribute)
                                and func.attr == "collect"
                                and isinstance(func.value, ast.Name)
                                and func.value.id == "gc"):
                            found_gc_in_loop = True

        assert found_gc_in_loop, (
            "gc.collect() not found inside the shuffle for-loop body. "
            "Must be inside the loop to prevent OOM in Docker container."
        )


# ═══════════════════════════════════════════════════════════════════
# BLOCK 5 — FULL PIPELINE INTEGRATION (8 tests)
# ═══════════════════════════════════════════════════════════════════

class TestNullImportancePipelineIntegration:

    def test_null_importance_filter_reduces_feature_count(self):
        """5.1: Filter must actually remove features — not a no-op."""
        df, y = _make_synthetic_dataset(n_samples=500, n_signal=5, n_noise=25)
        feature_names = df.columns

        # Mock Stage 2 to pass only signal features
        signal_feats = [f"signal_{i}" for i in range(5)]
        noise_feats = [f"noise_{i}" for i in range(25)]

        mock_result = {
            "success": True,
            "stdout": json.dumps({
                "actual_importances": {
                    **{f: 100.0 for f in signal_feats},
                    **{f: 2.0 for f in noise_feats if f in feature_names},
                },
                "null_distributions": {
                    **{f: [1.0] * 50 for f in signal_feats},
                    **{f: [5.0] * 50 for f in noise_feats if f in feature_names},
                },
            }),
            "stderr": "",
            "returncode": 0,
            "timed_out": False,
            "backend": "mock",
        }

        import tools.null_importance as ni_module
        with patch.object(ni_module, "execute_code", return_value=mock_result):
            result = run_null_importance_filter(
                df, y, feature_names,
                n_stage1_shuffles=2,
                n_stage2_shuffles=50,
            )
        assert result.total_features_output < result.total_features_input

    def test_null_importance_filter_preserves_known_signal_features(self):
        """5.2: 5 strong signal features must all survive."""
        rng = np.random.default_rng(42)
        n = 500
        y = rng.integers(0, 2, size=n).astype(float)

        data = {}
        signal_names = []
        for i in range(5):
            name = f"strong_signal_{i}"
            data[name] = y * (3 + i) + rng.normal(0, 0.1, n)
            signal_names.append(name)
        for i in range(20):
            data[f"noise_{i}"] = rng.normal(0, 1, n)

        df = pl.DataFrame(data)
        feature_names = df.columns

        # Mock Stage 2 to correctly identify signal features
        mock_result = {
            "success": True,
            "stdout": json.dumps({
                "actual_importances": {
                    **{f: 200.0 for f in signal_names},
                    **{f: 3.0 for f in feature_names if f not in signal_names},
                },
                "null_distributions": {
                    **{f: [1.0] * 50 for f in signal_names},
                    **{f: [5.0] * 50 for f in feature_names if f not in signal_names},
                },
            }),
            "stderr": "",
            "returncode": 0,
            "timed_out": False,
            "backend": "mock",
        }

        import tools.null_importance as ni_module
        with patch.object(ni_module, "execute_code", return_value=mock_result):
            result = run_null_importance_filter(
                df, y, feature_names,
                n_stage1_shuffles=2,
                n_stage2_shuffles=50,
            )
        for s in signal_names:
            assert s in result.survivors, (
                f"Signal feature {s} was dropped — true signal must survive"
            )

    def test_null_importance_result_logged_to_lineage(self, tmp_path):
        """5.3: _apply_null_importance_filter logs to lineage."""
        session_id = "test_ni_lineage"
        log_dir = tmp_path / "outputs" / session_id / "logs"
        log_dir.mkdir(parents=True)

        state = {
            "session_id": session_id,
            "task_type": "binary",
            "target_column": "target",
            "id_column": "",
        }

        rng = np.random.default_rng(42)
        n = 200
        y = rng.integers(0, 2, size=n).astype(float)
        data = {f"f{i}": rng.normal(0, 1, n) for i in range(12)}
        data["target"] = y
        X = pl.DataFrame(data)

        # Mock null importance to return quickly
        mock_ni_result = NullImportanceResult(
            survivors=[f"f{i}" for i in range(8)],
            dropped_stage1=[f"f{i}" for i in range(8, 10)],
            dropped_stage2=[f"f{i}" for i in range(10, 12)],
            stage1_importances={},
            null_distributions={},
            threshold_percentiles={},
            actual_vs_threshold={},
            total_features_input=12,
            total_features_output=8,
            stage1_drop_count=2,
            stage2_drop_count=2,
            elapsed_seconds=1.0,
        )

        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            with patch("agents.feature_factory.run_null_importance_filter",
                        return_value=mock_ni_result):
                survivors, updated_state = _apply_null_importance_filter(state, X, y)
            entries = read_lineage(session_id)
        finally:
            os.chdir(old_cwd)

        ni_entries = [e for e in entries if e["action"] == "null_importance_filter_complete"]
        assert len(ni_entries) >= 1
        changed = ni_entries[0]["values_changed"]
        assert "total_input" in changed or str(changed.get("total_input", "")) != ""
        assert "total_output" in changed or str(changed.get("total_output", "")) != ""

    def test_wilcoxon_gate_and_null_importance_both_run_in_feature_factory(self, tmp_path):
        """5.4: Both filters run in run_feature_factory end-to-end."""
        from agents.feature_factory import run_feature_factory

        session_id = "test_both_filters"
        out_dir = tmp_path / "outputs" / session_id
        out_dir.mkdir(parents=True)
        (out_dir / "logs").mkdir()

        # Write schema.json
        schema = {
            "session_id": session_id,
            "columns": [
                {"name": "id", "dtype": "int64", "is_id": True, "is_target": False,
                 "null_fraction": 0, "n_unique": 100, "min": 1},
                {"name": "target", "dtype": "int64", "is_id": False, "is_target": True,
                 "null_fraction": 0, "n_unique": 2, "min": 0},
                {"name": "feat_a", "dtype": "float64", "is_id": False, "is_target": False,
                 "null_fraction": 0.0, "n_unique": 50, "min": 0.0},
                {"name": "feat_b", "dtype": "float64", "is_id": False, "is_target": False,
                 "null_fraction": 0.05, "n_unique": 40, "min": -1.0},
            ]
        }
        (out_dir / "schema.json").write_text(json.dumps(schema))

        # Write competition_brief.json with a domain feature
        brief = {"domain": "spaceship", "task_type": "binary_classification",
                 "known_winning_features": []}
        (out_dir / "competition_brief.json").write_text(json.dumps(brief))

        # Write clean data CSV
        rng = np.random.default_rng(42)
        n = 200
        target = rng.integers(0, 2, size=n)
        data_path = tmp_path / "clean.csv"
        df = pl.DataFrame({
            "id": list(range(n)),
            "target": target.tolist(),
            "feat_a": (target * 2 + rng.normal(0, 0.1, n)).tolist(),
            "feat_b": rng.normal(0, 1, n).tolist(),
        })
        df.write_csv(str(data_path))

        state = {
            "session_id": session_id,
            "task_type": "binary",
            "clean_data_path": str(data_path),
            "target_column": "target",
            "id_column": "id",
        }

        mock_ni_result = NullImportanceResult(
            survivors=["log1p_feat_a", "sqrt_feat_a"],
            dropped_stage1=[], dropped_stage2=[],
            stage1_importances={}, null_distributions={},
            threshold_percentiles={}, actual_vs_threshold={},
            total_features_input=2, total_features_output=2,
            stage1_drop_count=0, stage2_drop_count=0,
            elapsed_seconds=0.1,
        )

        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            with patch("agents.feature_factory.call_llm",
                        return_value='[{"name": "domain_feat", "source_columns": ["feat_a"], "transform_type": "domain", "expression": "test"}]'):
                with patch("agents.feature_factory.run_null_importance_filter",
                            return_value=mock_ni_result):
                    result_state = run_feature_factory(state)
            entries = read_lineage(session_id)
        finally:
            os.chdir(old_cwd)

        actions = [e["action"] for e in entries]
        assert "wilcoxon_feature_gate" in actions, "Wilcoxon gate must run"
        assert "null_importance_filter_complete" in actions, "Null importance must run"

    def test_features_dropped_stage1_stored_in_state(self, tmp_path):
        """5.5: features_dropped_stage1 is stored in state."""
        session_id = "test_s1_state"
        log_dir = tmp_path / "outputs" / session_id / "logs"
        log_dir.mkdir(parents=True)

        state = {
            "session_id": session_id,
            "task_type": "binary",
            "target_column": "target",
            "id_column": "",
        }

        rng = np.random.default_rng(42)
        n = 200
        y = rng.integers(0, 2, size=n).astype(float)
        data = {f"f{i}": rng.normal(0, 1, n) for i in range(12)}
        data["target"] = y
        X = pl.DataFrame(data)

        mock_result = NullImportanceResult(
            survivors=[f"f{i}" for i in range(8)],
            dropped_stage1=["f8", "f9"],
            dropped_stage2=["f10", "f11"],
            stage1_importances={}, null_distributions={},
            threshold_percentiles={}, actual_vs_threshold={},
            total_features_input=12, total_features_output=8,
            stage1_drop_count=2, stage2_drop_count=2,
            elapsed_seconds=1.0,
        )

        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            with patch("agents.feature_factory.run_null_importance_filter",
                        return_value=mock_result):
                _, updated_state = _apply_null_importance_filter(state, X, y)
        finally:
            os.chdir(old_cwd)

        assert isinstance(updated_state["features_dropped_stage1"], list)
        assert "f8" in updated_state["features_dropped_stage1"]

    def test_features_dropped_stage2_stored_in_state(self, tmp_path):
        """5.6: features_dropped_stage2 is stored in state."""
        session_id = "test_s2_state"
        log_dir = tmp_path / "outputs" / session_id / "logs"
        log_dir.mkdir(parents=True)

        state = {
            "session_id": session_id,
            "task_type": "binary",
            "target_column": "target",
            "id_column": "",
        }

        rng = np.random.default_rng(42)
        n = 200
        y = rng.integers(0, 2, size=n).astype(float)
        data = {f"f{i}": rng.normal(0, 1, n) for i in range(12)}
        data["target"] = y
        X = pl.DataFrame(data)

        mock_result = NullImportanceResult(
            survivors=[f"f{i}" for i in range(8)],
            dropped_stage1=["f8", "f9"],
            dropped_stage2=["f10", "f11"],
            stage1_importances={}, null_distributions={},
            threshold_percentiles={}, actual_vs_threshold={},
            total_features_input=12, total_features_output=8,
            stage1_drop_count=2, stage2_drop_count=2,
            elapsed_seconds=1.0,
        )

        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            with patch("agents.feature_factory.run_null_importance_filter",
                        return_value=mock_result):
                _, updated_state = _apply_null_importance_filter(state, X, y)
        finally:
            os.chdir(old_cwd)

        assert isinstance(updated_state["features_dropped_stage2"], list)
        assert "f10" in updated_state["features_dropped_stage2"]

    def test_null_importance_elapsed_seconds_is_reasonable(self):
        """5.7: Small dataset → elapsed < 120 seconds."""
        rng = np.random.default_rng(42)
        n = 1000
        y = rng.integers(0, 2, size=n).astype(float)
        data = {f"f{i}": rng.normal(0, 1, n) for i in range(15)}
        df = pl.DataFrame(data)
        feature_names = df.columns

        # Mock Stage 2 for speed
        mock_result = {
            "success": True,
            "stdout": json.dumps({
                "actual_importances": {f: 10.0 for f in feature_names},
                "null_distributions": {f: [1.0] * 50 for f in feature_names},
            }),
            "stderr": "",
            "returncode": 0,
            "timed_out": False,
            "backend": "mock",
        }

        import tools.null_importance as ni_module
        with patch.object(ni_module, "execute_code", return_value=mock_result):
            result = run_null_importance_filter(
                df, y, feature_names,
                n_stage1_shuffles=3,
                n_stage2_shuffles=50,
            )
        assert result.elapsed_seconds < 120, (
            f"Elapsed {result.elapsed_seconds:.1f}s — should be < 120s on small data"
        )

    def test_day13_wilcoxon_model_contracts_still_pass(self):
        """5.8: Day 13 Wilcoxon contracts must still be green."""
        from tools.wilcoxon_gate import is_significantly_better, MIN_FOLDS_REQUIRED

        # Contract 1: returns bool, never raises
        a = [0.85, 0.86, 0.84, 0.85, 0.87]
        b = [0.80, 0.81, 0.80, 0.82, 0.81]
        assert isinstance(is_significantly_better(a, b), bool)

        # Contract 2: mismatched → False
        assert is_significantly_better([0.85, 0.86], [0.84]) is False

        # Contract 3: identical → False
        scores = [0.85, 0.86, 0.84, 0.85, 0.87]
        assert is_significantly_better(scores, scores) is False

        # Contract 4: below MIN_FOLDS → mean comparison
        short_a = [0.90] * (MIN_FOLDS_REQUIRED - 1)
        short_b = [0.80] * (MIN_FOLDS_REQUIRED - 1)
        assert is_significantly_better(short_a, short_b) is True
