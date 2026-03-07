# tests/regression/test_phase1_regression.py
#
# Written: Day 7 -- 2026-03-07
# Gate CV: 0.8798 (5-fold AUC on Spaceship Titanic)
# Gate session: spaceshi_694d438e
# Commit hash: b60b6150276f84d8fded513cdae17793c1fed431
# Wall clock: 20 seconds
#
# IMMUTABLE: NEVER edit this file after Day 7.
# This is the permanent floor that protects everything built in Phase 1.
# If any of these tests fail after a code change, the change introduced
# a regression and must be reverted or fixed before merging.

import os
import sys
import subprocess
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.state import initial_state
from core.professor import run_professor
from tools.submit_tools import validate_existing_submission


# ── Constants frozen from Day 7 gate run ──────────────────────────
GATE_CV = 0.8798
CV_FLOOR = GATE_CV - 0.03  # 0.8498 -- 0.03 buffer for normal variance
COMPETITION = "spaceship-titanic"
DATA_PATH = "data/spaceship_titanic/train.csv"
SAMPLE_SUBMISSION_PATH = "data/spaceship_titanic/sample_submission.csv"
EXPECTED_SUBMISSION_ROWS = 4277
EXPECTED_SUBMISSION_COLS = ["PassengerId", "Transported"]


@pytest.fixture(scope="module")
def gate_result():
    """Run the full pipeline once for all regression tests."""
    if not os.path.exists(DATA_PATH):
        pytest.skip(f"Spaceship Titanic data not found at {DATA_PATH}")

    state = initial_state(
        competition=COMPETITION,
        data_path=DATA_PATH,
        budget_usd=2.0,
        task_type="auto"
    )
    result = run_professor(state)
    return result


# ── Freeze 1: CV Floor ────────────────────────────────────────────
class TestCVFloor:
    """CV must never drop below the Day 7 gate score minus 0.03."""

    def test_cv_above_floor(self, gate_result):
        cv = gate_result.get("cv_mean", 0.0)
        assert cv >= CV_FLOOR, (
            f"CV regression: {cv:.4f} < floor {CV_FLOOR:.4f} "
            f"(Day 7 gate was {GATE_CV:.4f}, floor = gate - 0.03)"
        )

    def test_cv_above_absolute_minimum(self, gate_result):
        cv = gate_result.get("cv_mean", 0.0)
        assert cv > 0.70, (
            f"CV {cv:.4f} below absolute minimum 0.70 -- pipeline is broken"
        )


# ── Freeze 2: Submission Format ───────────────────────────────────
class TestSubmissionFormat:
    """submission.csv must always match Kaggle's expected format."""

    def test_submission_exists(self, gate_result):
        path = gate_result.get("submission_path")
        assert path is not None, "submission_path is None in result state"
        assert os.path.exists(path), f"submission.csv not found at {path}"

    def test_submission_validates(self, gate_result):
        path = gate_result["submission_path"]
        v = validate_existing_submission(path, SAMPLE_SUBMISSION_PATH)
        assert v["valid"], f"Submission validation failed: {v['errors']}"

    def test_submission_columns(self, gate_result):
        import polars as pl
        path = gate_result["submission_path"]
        df = pl.read_csv(path)
        assert df.columns == EXPECTED_SUBMISSION_COLS, (
            f"Column mismatch: {df.columns} != {EXPECTED_SUBMISSION_COLS}"
        )

    def test_submission_row_count(self, gate_result):
        import polars as pl
        path = gate_result["submission_path"]
        df = pl.read_csv(path)
        assert len(df) == EXPECTED_SUBMISSION_ROWS, (
            f"Row count: {len(df)} != {EXPECTED_SUBMISSION_ROWS}"
        )

    def test_submission_zero_nulls(self, gate_result):
        import polars as pl
        path = gate_result["submission_path"]
        df = pl.read_csv(path)
        null_count = df.null_count().sum_horizontal().item()
        assert null_count == 0, f"Submission has {null_count} null values"


# ── Freeze 3: State Pointer Contract ─────────────────────────────
class TestStatePointerContract:
    """No raw DataFrames in state. Only string file pointers."""

    def test_no_polars_dataframes_in_state(self, gate_result):
        import polars as pl
        for key, value in gate_result.items():
            assert not isinstance(value, pl.DataFrame), (
                f"Raw Polars DataFrame found in state key '{key}' "
                f"-- must be a file pointer (str), not data"
            )

    def test_no_pandas_dataframes_in_state(self, gate_result):
        try:
            import pandas as pd
            for key, value in gate_result.items():
                assert not isinstance(value, pd.DataFrame), (
                    f"Pandas DataFrame found in state key '{key}' "
                    f"-- Pandas not allowed in this pipeline"
                )
        except ImportError:
            pass  # pandas not installed, nothing to check

    def test_no_numpy_arrays_in_state(self, gate_result):
        import numpy as np
        for key, value in gate_result.items():
            assert not isinstance(value, np.ndarray), (
                f"Raw numpy array found in state key '{key}' "
                f"-- must be a file pointer (str), not data"
            )


# ── Freeze 4: Cost Tracker ───────────────────────────────────────
class TestCostTracker:
    """cost_tracker must exist and be properly structured."""

    def test_cost_tracker_exists(self, gate_result):
        assert "cost_tracker" in gate_result, "cost_tracker missing from state"

    def test_cost_tracker_has_llm_calls(self, gate_result):
        ct = gate_result["cost_tracker"]
        assert "llm_calls" in ct, "cost_tracker missing 'llm_calls' key"
        assert isinstance(ct["llm_calls"], int), (
            f"llm_calls should be int, got {type(ct['llm_calls'])}"
        )

    def test_cost_tracker_has_budget(self, gate_result):
        ct = gate_result["cost_tracker"]
        assert "budget_usd" in ct, "cost_tracker missing 'budget_usd' key"
        assert ct["budget_usd"] > 0, "budget_usd must be positive"


# ── Freeze 5: All Contract Tests Still Pass ──────────────────────
class TestContractTestsPass:
    """Meta-test: all contract tests that passed on Day 7 must still pass."""

    def test_all_contract_tests_pass(self):
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/contracts/", "-v", "--tb=short"],
            capture_output=True, text=True
        )
        assert result.returncode == 0, (
            f"Contract tests failed (exit code {result.returncode}):\n"
            f"{result.stdout[-500:]}\n{result.stderr[-500:]}"
        )


# ── Freeze 6: Model Registry ─────────────────────────────────────
class TestModelRegistry:
    """model_registry must have at least one entry after a full run."""

    def test_model_registry_not_empty(self, gate_result):
        registry = gate_result.get("model_registry")
        assert registry is not None, "model_registry is None"
        assert len(registry) >= 1, "model_registry is empty after full run"

    def test_model_file_exists(self, gate_result):
        registry = gate_result["model_registry"]
        for entry in registry:
            assert os.path.exists(entry["model_path"]), (
                f"Model file not found: {entry['model_path']}"
            )
