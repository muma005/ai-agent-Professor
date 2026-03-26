# tests/integration/test_pipeline_integrity.py
"""
Integration tests for the Pipeline Integrity Gate.

Tests all 3 checkpoints with both passing and failing states.
Validates that:
  1. Valid states pass all checks
  2. Missing schema authority fields cause FAIL
  3. Missing files cause FAIL
  4. Warnings are correctly identified
"""

import os
import json
import tempfile
import pytest
from guards.pipeline_integrity import (
    check_post_data_engineer,
    check_post_eda,
    check_post_model,
    run_integrity_gate,
)


@pytest.fixture
def temp_dir():
    """Create a temp directory with required test files."""
    with tempfile.TemporaryDirectory() as d:
        # Create fake cleaned.parquet
        parquet_path = os.path.join(d, "cleaned.parquet")
        with open(parquet_path, "wb") as f:
            f.write(b"fake_parquet")

        # Create fake schema.json
        schema_path = os.path.join(d, "schema.json")
        with open(schema_path, "w") as f:
            json.dump({"columns": ["id", "feature1", "target"], "target_col": "target"}, f)

        # Create fake preprocessor.pkl
        preprocessor_path = os.path.join(d, "preprocessor.pkl")
        with open(preprocessor_path, "wb") as f:
            f.write(b"fake_pkl")

        # Create fake model.pkl
        model_path = os.path.join(d, "model.pkl")
        with open(model_path, "wb") as f:
            f.write(b"fake_model")

        yield {
            "dir": d,
            "parquet_path": parquet_path,
            "schema_path": schema_path,
            "preprocessor_path": preprocessor_path,
            "model_path": model_path,
        }


def _valid_state(temp_dir: dict) -> dict:
    """Build a fully valid state dict for testing."""
    return {
        "session_id": "test-session",
        "target_col": "target",
        "task_type": "binary",
        "id_columns": ["id"],
        "dropped_features": ["leaky_feature"],
        "clean_data_path": temp_dir["parquet_path"],
        "schema_path": temp_dir["schema_path"],
        "preprocessor_path": temp_dir["preprocessor_path"],
        "eda_report_path": os.path.join(temp_dir["dir"], "eda_report.json"),
        "eda_report": {
            "target_distribution": {"imbalance_ratio": 0.85},
            "drop_manifest": ["leaky_feature"],
            "summary": "test",
        },
        "model_registry": [{"model_path": temp_dir["model_path"], "score": 0.92}],
        "best_score": 0.92,
        "pipeline_halted": False,
    }


# ── POST_DATA_ENGINEER tests ────────────────────────────────────

class TestPostDataEngineer:
    def test_valid_state_passes(self, temp_dir):
        state = _valid_state(temp_dir)
        result = check_post_data_engineer(state)
        assert result.all_passed, result.report()
        assert not result.has_failures

    def test_missing_target_col_fails(self, temp_dir):
        state = _valid_state(temp_dir)
        state["target_col"] = ""
        result = check_post_data_engineer(state)
        assert result.has_failures
        failed_names = [c.name for c in result.checks if not c.passed]
        assert "target_col_set" in failed_names

    def test_invalid_task_type_fails(self, temp_dir):
        state = _valid_state(temp_dir)
        state["task_type"] = "unknown"
        result = check_post_data_engineer(state)
        assert result.has_failures
        failed_names = [c.name for c in result.checks if not c.passed]
        assert "task_type_valid" in failed_names

    def test_missing_clean_data_fails(self, temp_dir):
        state = _valid_state(temp_dir)
        state["clean_data_path"] = "/nonexistent/path"
        result = check_post_data_engineer(state)
        assert result.has_failures

    def test_missing_preprocessor_fails(self, temp_dir):
        state = _valid_state(temp_dir)
        state["preprocessor_path"] = "/nonexistent/preprocessor.pkl"
        result = check_post_data_engineer(state)
        assert result.has_failures

    def test_id_columns_wrong_type_fails(self, temp_dir):
        state = _valid_state(temp_dir)
        state["id_columns"] = "PassengerId"  # Should be list, not string
        result = check_post_data_engineer(state)
        assert result.has_failures


# ── POST_EDA tests ───────────────────────────────────────────────

class TestPostEda:
    def test_valid_state_passes(self, temp_dir):
        state = _valid_state(temp_dir)
        result = check_post_eda(state)
        assert result.all_passed, result.report()

    def test_empty_eda_report_fails(self, temp_dir):
        state = _valid_state(temp_dir)
        state["eda_report"] = {}
        result = check_post_eda(state)
        assert result.has_failures

    def test_dropped_features_wrong_type_fails(self, temp_dir):
        state = _valid_state(temp_dir)
        state["dropped_features"] = "feature_a"  # Should be list
        result = check_post_eda(state)
        assert result.has_failures

    def test_missing_imbalance_ratio_warns(self, temp_dir):
        state = _valid_state(temp_dir)
        state["eda_report"] = {"target_distribution": {}, "summary": "test"}
        result = check_post_eda(state)
        assert result.has_warnings
        assert not result.has_failures  # Only warn, not fail


# ── POST_MODEL tests ─────────────────────────────────────────────

class TestPostModel:
    def test_valid_state_passes(self, temp_dir):
        state = _valid_state(temp_dir)
        result = check_post_model(state)
        assert result.all_passed, result.report()

    def test_empty_registry_fails(self, temp_dir):
        state = _valid_state(temp_dir)
        state["model_registry"] = []
        result = check_post_model(state)
        assert result.has_failures

    def test_missing_model_file_fails(self, temp_dir):
        state = _valid_state(temp_dir)
        state["model_registry"] = [{"model_path": "/nonexistent/model.pkl"}]
        result = check_post_model(state)
        assert result.has_failures

    def test_pipeline_halted_fails(self, temp_dir):
        state = _valid_state(temp_dir)
        state["pipeline_halted"] = True
        state["pipeline_halt_reason"] = "test crash"
        result = check_post_model(state)
        assert result.has_failures

    def test_zero_best_score_warns(self, temp_dir):
        state = _valid_state(temp_dir)
        state["best_score"] = 0.0
        result = check_post_model(state)
        assert result.has_warnings


# ── run_integrity_gate integration ───────────────────────────────

class TestRunIntegrityGate:
    def test_valid_state_does_not_raise(self, temp_dir):
        state = _valid_state(temp_dir)
        # Should NOT raise for valid state
        result = run_integrity_gate(state, "POST_DATA_ENGINEER")
        assert result.all_passed

    def test_invalid_state_raises_value_error(self, temp_dir):
        state = _valid_state(temp_dir)
        state["target_col"] = ""
        with pytest.raises(ValueError, match="Pipeline Integrity Gate FAILED"):
            run_integrity_gate(state, "POST_DATA_ENGINEER")

    def test_unknown_checkpoint_raises(self, temp_dir):
        state = _valid_state(temp_dir)
        with pytest.raises(ValueError, match="Unknown checkpoint"):
            run_integrity_gate(state, "UNKNOWN")
