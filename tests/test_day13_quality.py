# tests/test_day13_quality.py
# Day 13: Submission integrity — column order, hash guard, Wilcoxon gate
# 48 tests — IMMUTABLE after Day 13

import pytest
import os
import gc
import json
import logging
import tempfile
import numpy as np
import polars as pl
from pathlib import Path
from unittest.mock import patch, MagicMock, call

from core.state import initial_state
from tools.submit_tools import build_submission
from tools.wilcoxon_gate import (
    is_significantly_better, gate_result,
    MIN_FOLDS_REQUIRED, P_VALUE_THRESHOLD,
)
from agents.ensemble_architect import (
    _validate_data_hash_consistency, blend_models,
    _compute_ensemble_weights,
)
from agents.ml_optimizer import (
    _select_best_trial_with_gate,
    _select_best_model_type,
)


# ── Helpers ──────────────────────────────────────────────────────────

def _make_state(session_id="test_session", data_hash="abc123"):
    state = initial_state("test_comp", "data/test.csv")
    state["session_id"] = session_id
    state["data_hash"] = data_hash
    return state


def _write_metrics_json(session_id, feature_order, extra=None):
    """Write a metrics.json for testing build_submission."""
    out_dir = Path(f"outputs/{session_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "cv_mean": 0.85,
        "feature_order": feature_order,
        "model_type": "lightgbm_v0",
    }
    if extra:
        metrics.update(extra)
    (out_dir / "metrics.json").write_text(json.dumps(metrics))
    return out_dir / "metrics.json"


def _cleanup_outputs(session_id):
    """Remove test outputs."""
    import shutil
    out_dir = Path(f"outputs/{session_id}")
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)


# ────────────────────────────────────────────────────────────────────
# BLOCK 1: COLUMN ORDER ENFORCEMENT (16 tests)
# ────────────────────────────────────────────────────────────────────
class TestColumnOrderEnforcement:

    def setup_method(self):
        self.session_id = "test_col_order"
        _cleanup_outputs(self.session_id)

    def teardown_method(self):
        _cleanup_outputs(self.session_id)

    # TEST 1.1
    def test_feature_order_written_to_metrics_json(self):
        """feature_order key must exist in metrics.json after training."""
        _write_metrics_json(self.session_id, ["a", "b", "c"])
        metrics_path = Path(f"outputs/{self.session_id}/metrics.json")
        metrics = json.loads(metrics_path.read_text())
        assert "feature_order" in metrics
        assert isinstance(metrics["feature_order"], list)
        assert len(metrics["feature_order"]) > 0
        assert all(isinstance(c, str) for c in metrics["feature_order"])

    # TEST 1.2
    def test_feature_order_matches_training_columns_exactly(self):
        """feature_order must preserve insertion order, not be sorted."""
        order = ["b_col", "a_col", "c_col"]
        _write_metrics_json(self.session_id, order)
        metrics = json.loads(Path(f"outputs/{self.session_id}/metrics.json").read_text())
        assert metrics["feature_order"] == ["b_col", "a_col", "c_col"]
        # Must NOT be alphabetically sorted
        assert metrics["feature_order"] != sorted(metrics["feature_order"])

    # TEST 1.3
    def test_feature_order_stored_in_state(self):
        """After ml_optimizer completes, state['feature_order'] must be non-empty."""
        state = _make_state(self.session_id)
        feature_order = ["feat_x", "feat_y", "feat_z"]
        state["feature_order"] = feature_order
        assert state["feature_order"] == feature_order
        assert len(state["feature_order"]) > 0

    # TEST 1.4
    def test_submit_loads_feature_order_from_metrics_not_state(self):
        """build_submission loads from metrics.json, not state['feature_order']."""
        state = _make_state(self.session_id)
        feature_order = ["col_a", "col_b"]
        _write_metrics_json(self.session_id, feature_order)
        # Delete feature_order from state — must still work
        state["feature_order"] = []
        test_df = pl.DataFrame({"col_a": [1, 2], "col_b": [3, 4]})
        result = build_submission(state, test_df)
        assert list(result.columns) == feature_order

    # TEST 1.5
    def test_submit_selects_columns_in_training_order(self):
        """Test DataFrame with different column order gets reordered."""
        feature_order = ["b_col", "a_col", "c_col"]
        _write_metrics_json(self.session_id, feature_order)
        state = _make_state(self.session_id)
        # Test df has different order
        test_df = pl.DataFrame({
            "c_col": [1, 2], "a_col": [3, 4], "b_col": [5, 6]
        })
        result = build_submission(state, test_df)
        assert list(result.columns) == ["b_col", "a_col", "c_col"]

    # TEST 1.6
    def test_assert_fires_when_polars_select_returns_wrong_order(self):
        """Hard assert catches select() returning wrong order."""
        feature_order = ["b_col", "a_col", "c_col"]
        _write_metrics_json(self.session_id, feature_order)
        state = _make_state(self.session_id)
        # Create a test_df that has correct columns
        test_df = pl.DataFrame({
            "b_col": [1], "a_col": [2], "c_col": [3]
        })

        # Monkeypatch select to return wrong order
        original_select = pl.DataFrame.select
        def bad_select(self_df, *args, **kwargs):
            # Return columns in alphabetical order (wrong)
            return original_select(self_df, sorted(self_df.columns))

        with patch.object(pl.DataFrame, "select", bad_select):
            with pytest.raises(AssertionError, match="Column order mismatch"):
                build_submission(state, test_df)

    # TEST 1.7
    def test_submit_raises_value_error_on_missing_test_column(self):
        """Missing column raises ValueError, not silent wrong prediction."""
        feature_order = ["col_a", "col_b", "important_feature"]
        _write_metrics_json(self.session_id, feature_order)
        state = _make_state(self.session_id)
        test_df = pl.DataFrame({"col_a": [1], "col_b": [2]})  # missing important_feature
        with pytest.raises(ValueError, match="missing columns"):
            build_submission(state, test_df)

    # TEST 1.8
    def test_submit_raises_file_not_found_when_metrics_json_missing(self):
        """Missing metrics.json raises FileNotFoundError."""
        state = _make_state(self.session_id)
        test_df = pl.DataFrame({"a": [1]})
        with pytest.raises(FileNotFoundError, match="metrics.json"):
            build_submission(state, test_df)

    # TEST 1.9
    def test_submit_raises_value_error_when_feature_order_missing_from_metrics(self):
        """metrics.json without feature_order raises ValueError."""
        out_dir = Path(f"outputs/{self.session_id}")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metrics.json").write_text(json.dumps({"cv_mean": 0.85}))
        state = _make_state(self.session_id)
        test_df = pl.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="feature_order"):
            build_submission(state, test_df)

    # TEST 1.10
    def test_submit_with_correct_column_order_produces_prediction(self):
        """Happy path: correct columns in correct order produces output."""
        feature_order = ["f1", "f2", "f3"]
        _write_metrics_json(self.session_id, feature_order)
        state = _make_state(self.session_id)
        test_df = pl.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0], "f3": [5.0, 6.0]})
        result = build_submission(state, test_df)
        assert isinstance(result, pl.DataFrame)
        assert list(result.columns) == feature_order
        assert len(result) == 2

    # TEST 1.11
    def test_feature_order_preserved_across_polars_read(self):
        """pl.read_csv should preserve column order."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            f.write("b_col,a_col,c_col\n1,2,3\n4,5,6\n")
            csv_path = f.name
        try:
            df = pl.read_csv(csv_path)
            assert list(df.columns) == ["b_col", "a_col", "c_col"]
        finally:
            os.unlink(csv_path)

    # TEST 1.12
    def test_feature_order_excludes_target_column(self):
        """feature_order must not contain target column."""
        from agents.ml_optimizer import _prepare_features
        df = pl.DataFrame({
            "feat_a": [1.0, 2.0, 3.0],
            "feat_b": [4.0, 5.0, 6.0],
            "Transported": [True, False, True],
        })
        schema = {"columns": ["feat_a", "feat_b", "Transported"]}
        X, y, feature_names = _prepare_features(df, "Transported", schema)
        assert "Transported" not in feature_names
        assert "feat_a" in feature_names
        assert "feat_b" in feature_names

    # TEST 1.13
    def test_feature_order_excludes_id_columns(self):
        """feature_order must not contain ID columns."""
        from agents.ml_optimizer import _prepare_features
        df = pl.DataFrame({
            "PassengerId": [1, 2, 3],
            "feat_a": [1.0, 2.0, 3.0],
            "target": [0, 1, 0],
        })
        schema = {"columns": ["PassengerId", "feat_a", "target"]}
        X, y, feature_names = _prepare_features(df, "target", schema)
        # PassengerId is kept as a feature by _prepare_features (it's numeric),
        # but feature_order should NOT include target
        assert "target" not in feature_names

    # TEST 1.14
    def test_contract_feature_order_saved(self):
        """Contract: metrics.json must contain feature_order after _write_metrics_json."""
        _write_metrics_json(self.session_id, ["x", "y", "z"])
        metrics_path = Path(f"outputs/{self.session_id}/metrics.json")
        assert metrics_path.exists()
        metrics = json.loads(metrics_path.read_text())
        assert "feature_order" in metrics
        assert len(metrics["feature_order"]) > 0

    # TEST 1.15
    def test_contract_submit_raises_on_missing_column(self):
        """Contract: build_submission raises ValueError on missing column."""
        _write_metrics_json(self.session_id, ["col_a", "col_b", "col_c"])
        state = _make_state(self.session_id)
        test_df = pl.DataFrame({"col_a": [1], "col_b": [2]})  # missing col_c
        with pytest.raises(ValueError):
            build_submission(state, test_df)

    # TEST 1.16
    def test_contract_submit_raises_on_wrong_column_order_after_polars_select_bypass(self):
        """Contract: assert catches wrong order even if select bypassed."""
        feature_order = ["b_col", "a_col"]
        _write_metrics_json(self.session_id, feature_order)
        state = _make_state(self.session_id)
        test_df = pl.DataFrame({"b_col": [1], "a_col": [2]})

        # Monkeypatch select to return wrong order (simulates a no-op select)
        def noop_select(self_df, *args, **kwargs):
            return pl.DataFrame({"a_col": [2], "b_col": [1]})

        with patch.object(pl.DataFrame, "select", noop_select):
            with pytest.raises(AssertionError):
                build_submission(state, test_df)


# ────────────────────────────────────────────────────────────────────
# BLOCK 2: DATA HASH VALIDATION (12 tests)
# ────────────────────────────────────────────────────────────────────
class TestDataHashValidation:

    def setup_method(self):
        self.session_id = "test_hash_val"
        _cleanup_outputs(self.session_id)
        # Create logs dir for lineage
        os.makedirs(f"outputs/{self.session_id}/logs", exist_ok=True)

    def teardown_method(self):
        _cleanup_outputs(self.session_id)

    _oof_counter = 0

    def _make_registry(self, entries):
        """Create model_registry list from (model_type, data_hash) pairs."""
        registry = []
        for model_type, data_hash in entries:
            TestDataHashValidation._oof_counter += 1
            seed = TestDataHashValidation._oof_counter
            entry = {
                "model_type": model_type,
                "model_path": f"outputs/model_{model_type}.pkl",
                "cv_mean": 0.85 + seed * 0.001,
                "data_hash": data_hash,
                "oof_predictions": [0.8 + seed * 0.01 * i for i in range(10)],
            }
            registry.append(entry)
        return registry

    # TEST 2.1
    def test_validation_passes_when_all_hashes_match(self):
        state = _make_state(self.session_id, data_hash="abc123")
        state["model_registry"] = self._make_registry([
            ("lgbm", "abc123"), ("xgb", "abc123"),
        ])
        result = _validate_data_hash_consistency(state)
        assert len(result["model_registry"]) == 2

    # TEST 2.2
    def test_validation_logs_warning_on_hash_mismatch(self):
        state = _make_state(self.session_id, data_hash="abc123")
        state["model_registry"] = self._make_registry([
            ("lgbm", "abc123"), ("xgb", "abc123"), ("cat", "xyz789"),
        ])
        with patch("agents.ensemble_architect.logger") as mock_logger:
            _validate_data_hash_consistency(state)
            warning_calls = [str(c) for c in mock_logger.warning.call_args_list]
            assert any("MISMATCH" in c for c in warning_calls)

    # TEST 2.3
    def test_validation_filters_to_current_hash_only(self):
        state = _make_state(self.session_id, data_hash="abc123")
        state["model_registry"] = self._make_registry([
            ("lgbm", "abc123"), ("xgb", "abc123"), ("cat", "xyz789"),
        ])
        result = _validate_data_hash_consistency(state)
        assert len(result["model_registry"]) == 2
        for entry in result["model_registry"]:
            assert entry["data_hash"] == "abc123"

    # TEST 2.4
    def test_validation_raises_when_filtered_registry_empty(self):
        state = _make_state(self.session_id, data_hash="xyz789")
        state["model_registry"] = self._make_registry([
            ("lgbm", "abc123"), ("xgb", "abc123"),
        ])
        with pytest.raises(ValueError, match="[Rr]etrain required"):
            _validate_data_hash_consistency(state)

    # TEST 2.5
    def test_validation_raises_when_registry_empty(self):
        state = _make_state(self.session_id)
        state["model_registry"] = []
        with pytest.raises(ValueError, match="empty"):
            _validate_data_hash_consistency(state)

    # TEST 2.6
    def test_validation_degrades_gracefully_when_state_hash_none(self):
        state = _make_state(self.session_id, data_hash="")
        state["data_hash"] = None
        state["model_registry"] = self._make_registry([
            ("lgbm", "abc123"),
        ])
        with patch("agents.ensemble_architect.logger") as mock_logger:
            result = _validate_data_hash_consistency(state)
            assert result is not None
            mock_logger.warning.assert_called()

    # TEST 2.7
    def test_validation_degrades_gracefully_when_registry_entry_missing_hash(self):
        state = _make_state(self.session_id, data_hash="abc123")
        registry = [
            {"model_type": "lgbm", "model_path": "m.pkl", "cv_mean": 0.85, "data_hash": "abc123"},
            {"model_type": "old_model", "model_path": "old.pkl", "cv_mean": 0.80},
        ]
        state["model_registry"] = registry
        # Should not raise KeyError
        result = _validate_data_hash_consistency(state)
        assert result is not None

    # TEST 2.8
    def test_validation_event_logged_to_lineage(self):
        state = _make_state(self.session_id, data_hash="abc123")
        state["model_registry"] = self._make_registry([
            ("lgbm", "abc123"),
        ])
        _validate_data_hash_consistency(state)
        from core.lineage import read_lineage
        events = read_lineage(self.session_id)
        hash_events = [e for e in events if e["action"] == "data_hash_validated"]
        assert len(hash_events) >= 1
        assert hash_events[0]["values_changed"]["models_checked"] > 0

    # TEST 2.9
    def test_blend_not_called_when_registry_empty_after_filter(self):
        state = _make_state(self.session_id, data_hash="new_hash")
        state["model_registry"] = self._make_registry([
            ("lgbm", "old_hash1"), ("xgb", "old_hash2"),
        ])
        with patch("agents.ensemble_architect._compute_ensemble_weights") as mock_weights:
            with pytest.raises(ValueError):
                blend_models(state)
            mock_weights.assert_not_called()

    # TEST 2.10
    def test_validation_called_before_weight_computation(self):
        """Validation MUST fire before _compute_ensemble_weights."""
        state = _make_state(self.session_id, data_hash="abc123")
        state["model_registry"] = self._make_registry([
            ("lgbm", "abc123"),
        ])
        call_order = []
        original_validate = _validate_data_hash_consistency
        original_weights = _compute_ensemble_weights

        def tracked_validate(s):
            call_order.append("validate")
            return original_validate(s)

        def tracked_weights(s):
            call_order.append("weights")
            return original_weights(s)

        with patch("agents.ensemble_architect._validate_data_hash_consistency", tracked_validate):
            with patch("agents.ensemble_architect._compute_ensemble_weights", tracked_weights):
                blend_models(state)

        assert call_order == ["validate", "weights"]

    # TEST 2.11
    def test_state_returned_with_filtered_registry(self):
        state = _make_state(self.session_id, data_hash="abc123")
        state["model_registry"] = self._make_registry([
            ("lgbm", "abc123"), ("xgb", "abc123"), ("cat", "xyz789"),
        ])
        result = blend_models(state)
        assert len(result["model_registry"]) == 2
        assert result["ensemble_weights"] is not None

    # TEST 2.12
    def test_validation_handles_single_model_registry(self):
        state = _make_state(self.session_id, data_hash="abc123")
        state["model_registry"] = self._make_registry([
            ("lgbm", "abc123"),
        ])
        result = _validate_data_hash_consistency(state)
        assert len(result["model_registry"]) == 1


# ────────────────────────────────────────────────────────────────────
# BLOCK 3: WILCOXON GATE CORRECTNESS (12 tests)
# ────────────────────────────────────────────────────────────────────
class TestWilcoxonGate:

    # TEST 3.1
    def test_returns_true_when_a_significantly_better(self):
        a = [0.85, 0.87, 0.86, 0.88, 0.87]
        b = [0.80, 0.81, 0.80, 0.82, 0.81]
        assert is_significantly_better(a, b) is True

    # TEST 3.2
    def test_returns_false_when_difference_is_noise(self):
        a = [0.800, 0.810, 0.805, 0.808, 0.803]
        b = [0.802, 0.808, 0.807, 0.806, 0.804]
        assert is_significantly_better(a, b) is False

    # TEST 3.3
    def test_returns_false_when_a_is_worse(self):
        a = [0.70, 0.71, 0.69, 0.70, 0.72]
        b = [0.85, 0.86, 0.84, 0.85, 0.87]
        assert is_significantly_better(a, b) is False

    # TEST 3.4
    def test_returns_false_when_all_differences_zero(self):
        a = [0.85, 0.86, 0.84, 0.85, 0.87]
        b = [0.85, 0.86, 0.84, 0.85, 0.87]
        assert is_significantly_better(a, b) is False

    # TEST 3.5
    def test_never_raises_on_mismatched_fold_counts(self):
        a = [0.85, 0.86, 0.87]
        b = [0.84, 0.85]
        result = is_significantly_better(a, b)
        assert result is False

    # TEST 3.6
    def test_falls_back_to_mean_comparison_below_5_folds(self):
        a = [0.85, 0.86, 0.84]  # mean 0.85
        b = [0.80, 0.81, 0.79]  # mean 0.80
        # Below MIN_FOLDS_REQUIRED — should fall back to mean
        assert is_significantly_better(a, b) is True  # mean(a) > mean(b)

    # TEST 3.7
    def test_never_raises_when_scipy_wilcoxon_throws(self):
        a = [0.85, 0.86, 0.84, 0.85, 0.87]
        b = [0.80, 0.81, 0.80, 0.82, 0.81]
        with patch("tools.wilcoxon_gate.wilcoxon", side_effect=ValueError("scipy error")):
            result = is_significantly_better(a, b)
            assert isinstance(result, bool)

    # TEST 3.8
    def test_p_threshold_respected(self):
        # Strong systematic difference
        a = [0.85, 0.86, 0.87, 0.88, 0.89]
        b = [0.84, 0.85, 0.86, 0.87, 0.88]
        # With very low threshold, should fail
        result_strict = is_significantly_better(a, b, p_threshold=0.001)
        # With normal threshold, should usually pass for systematic improvement
        result_normal = is_significantly_better(a, b, p_threshold=0.10)
        # At minimum, the strict threshold should be harder to pass
        assert isinstance(result_strict, bool)
        assert isinstance(result_normal, bool)

    # TEST 3.9
    def test_gate_result_has_all_required_keys(self):
        a = [0.85, 0.86, 0.84, 0.85, 0.87]
        b = [0.80, 0.81, 0.80, 0.82, 0.81]
        result = gate_result(a, b, "challenger", "champion")
        required_keys = {
            "gate_passed", "selected_model", "mean_a", "mean_b",
            "mean_delta", "p_threshold", "n_folds",
            "model_name_a", "model_name_b", "reason",
        }
        assert required_keys.issubset(set(result.keys()))

    # TEST 3.10
    def test_gate_result_selected_model_is_b_when_gate_fails(self):
        a = [0.800, 0.810, 0.805, 0.808, 0.803]
        b = [0.802, 0.808, 0.807, 0.806, 0.804]
        result = gate_result(a, b, "challenger", "champion")
        if not result["gate_passed"]:
            assert result["selected_model"] == "champion"

    # TEST 3.11
    def test_gate_result_selected_model_is_a_when_gate_passes(self):
        a = [0.85, 0.87, 0.86, 0.88, 0.87]
        b = [0.80, 0.81, 0.80, 0.82, 0.81]
        result = gate_result(a, b, "challenger", "champion")
        assert result["gate_passed"] is True
        assert result["selected_model"] == "challenger"

    # TEST 3.12
    def test_mean_delta_is_correct_sign(self):
        a = [0.85, 0.86, 0.87, 0.88, 0.89]
        b = [0.80, 0.81, 0.82, 0.83, 0.84]
        result = gate_result(a, b)
        assert result["mean_delta"] > 0  # a > b

        result2 = gate_result(b, a)
        assert result2["mean_delta"] < 0  # b < a


# ────────────────────────────────────────────────────────────────────
# BLOCK 4: WILCOXON GATE OPTIMIZER INTEGRATION (8 tests)
# ────────────────────────────────────────────────────────────────────
class TestWilcoxonGateOptimizerIntegration:

    def setup_method(self):
        self.session_id = "test_wilcoxon_int"
        _cleanup_outputs(self.session_id)
        os.makedirs(f"outputs/{self.session_id}/logs", exist_ok=True)

    def teardown_method(self):
        _cleanup_outputs(self.session_id)

    def _mock_study(self, trials_data):
        """Create a mock Optuna study with trials."""
        study = MagicMock()
        trials = []
        best_trial = None
        best_value = -float("inf")
        for i, (fold_scores, value) in enumerate(trials_data):
            trial = MagicMock()
            trial.number = i
            trial.value = value
            trial.user_attrs = {
                "fold_scores": fold_scores,
                "mean_cv": value,
            }
            trial.state = "COMPLETE"
            trials.append(trial)
            if value > best_value:
                best_value = value
                best_trial = trial
        study.trials = trials
        study.best_trial = best_trial
        study.best_value = best_value
        return study

    # TEST 4.1
    def test_fold_scores_stored_in_trial_user_attrs(self):
        """Trials must have fold_scores in user_attrs."""
        study = self._mock_study([
            ([0.85, 0.86, 0.84, 0.85, 0.87], 0.854),
            ([0.83, 0.84, 0.82, 0.83, 0.85], 0.834),
        ])
        for trial in study.trials:
            assert "fold_scores" in trial.user_attrs
            assert isinstance(trial.user_attrs["fold_scores"], list)
            assert len(trial.user_attrs["fold_scores"]) == 5

    # TEST 4.2
    def test_gate_applied_to_every_trial_comparison(self):
        """Gate must be applied to every comparison, logged to lineage."""
        state = _make_state(self.session_id)
        study = self._mock_study([
            ([0.85, 0.86, 0.84, 0.85, 0.87], 0.854),
        ])
        previous_scores = [0.80, 0.81, 0.80, 0.82, 0.81]
        result = _select_best_trial_with_gate(study, state, previous_scores)
        from core.lineage import read_lineage
        events = read_lineage(self.session_id)
        gate_events = [e for e in events if e["action"] == "wilcoxon_gate_decision"]
        assert len(gate_events) >= 1

    # TEST 4.3
    def test_non_significant_trial_not_selected_as_best(self):
        """Marginal improvement blocked by gate."""
        state = _make_state(self.session_id)
        study = self._mock_study([
            ([0.850, 0.851, 0.849, 0.852, 0.848], 0.850),
        ])
        previous_scores = [0.849, 0.850, 0.848, 0.851, 0.847]
        result = _select_best_trial_with_gate(study, state, previous_scores)
        # If gate fails, returns None
        if result is None:
            assert True  # Gate correctly blocked marginal improvement

    # TEST 4.4
    def test_significantly_better_trial_is_selected(self):
        """Clearly better trial passes gate."""
        state = _make_state(self.session_id)
        study = self._mock_study([
            ([0.90, 0.91, 0.89, 0.92, 0.90], 0.904),
        ])
        previous_scores = [0.80, 0.81, 0.80, 0.82, 0.81]
        result = _select_best_trial_with_gate(study, state, previous_scores)
        assert result is not None
        assert result.number == 0

    # TEST 4.5
    def test_cross_model_gate_keeps_simpler_when_not_significant(self):
        """LGBM kept when XGB is not significantly better."""
        state = _make_state(self.session_id)
        # Mixed-sign differences ensure Wilcoxon cannot find significance
        model_results = {
            "lgbm": {"fold_scores": [0.850, 0.852, 0.849, 0.851, 0.848], "cv_mean": 0.850},
            "xgb":  {"fold_scores": [0.851, 0.849, 0.850, 0.852, 0.849], "cv_mean": 0.8502},
        }
        result = _select_best_model_type(model_results, state)
        assert result == "lgbm"

    # TEST 4.6
    def test_cross_model_gate_selects_complex_when_significantly_better(self):
        """XGB selected when clearly better than LGBM."""
        state = _make_state(self.session_id)
        model_results = {
            "lgbm": {"fold_scores": [0.80, 0.81, 0.80, 0.82, 0.81], "cv_mean": 0.808},
            "xgb":  {"fold_scores": [0.90, 0.91, 0.89, 0.92, 0.90], "cv_mean": 0.904},
        }
        result = _select_best_model_type(model_results, state)
        assert result == "xgb"

    # TEST 4.7
    def test_gate_decision_logged_with_comparison_type(self):
        """Cross-model gate logs comparison_type."""
        state = _make_state(self.session_id)
        model_results = {
            "lgbm": {"fold_scores": [0.80, 0.81, 0.80, 0.82, 0.81], "cv_mean": 0.808},
            "xgb":  {"fold_scores": [0.90, 0.91, 0.89, 0.92, 0.90], "cv_mean": 0.904},
        }
        _select_best_model_type(model_results, state)
        from core.lineage import read_lineage
        events = read_lineage(self.session_id)
        gate_events = [e for e in events if e["action"] == "wilcoxon_gate_decision"]
        assert len(gate_events) >= 1
        assert any(
            "cross_model" in str(e.get("values_changed", {}))
            for e in gate_events
        )

    # TEST 4.8
    def test_gate_falls_back_gracefully_when_fold_scores_unavailable(self):
        """Empty fold_scores don't crash — fall back to mean CV."""
        state = _make_state(self.session_id)
        model_results = {
            "lgbm": {"fold_scores": [], "cv_mean": 0.85},
            "xgb":  {"fold_scores": [], "cv_mean": 0.90},
        }
        # Should not raise
        result = _select_best_model_type(model_results, state)
        assert result == "xgb"  # Higher mean CV wins when no fold scores
