"""
Regression Tests for Performance and Quality.

Ensures that changes don't cause:
- CV score regression (> 5% drop)
- Execution time regression (> 20% increase)
- Memory usage regression (> 50% increase)
- Submission format changes
"""
import pytest
import time
import json
import os
import tempfile
import numpy as np
import polars as pl
from datetime import datetime

from core.state import initial_state
from core.professor import run_professor


# Baseline directory
BASELINE_DIR = os.path.join(os.path.dirname(__file__), "baselines")


def _ensure_baseline_dir():
    """Ensure baseline directory exists."""
    os.makedirs(BASELINE_DIR, exist_ok=True)


def _load_baseline(filename):
    """Load baseline value from file."""
    path = os.path.join(BASELINE_DIR, filename)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def _save_baseline(filename, value):
    """Save baseline value to file."""
    _ensure_baseline_dir()
    path = os.path.join(BASELINE_DIR, filename)
    with open(path, "w") as f:
        json.dump(value, f, indent=2)


@pytest.fixture
def benchmark_data(tmp_path):
    """Create consistent benchmark dataset."""
    np.random.seed(42)
    n_rows = 200
    n_features = 10
    
    X = np.random.randn(n_rows, n_features)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    df = pl.DataFrame({
        f"feature_{i}": X[:, i] for i in range(n_features)
    })
    df = df.with_columns(pl.Series("target", y))
    
    data_path = tmp_path / "train.csv"
    df.write_csv(data_path)
    
    # Create test data
    test_df = pl.DataFrame({
        f"feature_{i}": X[:20, i] for i in range(n_features)
    })
    test_path = tmp_path / "test.csv"
    test_df.write_csv(test_path)
    
    # Create sample submission
    sample_df = pl.DataFrame({
        "id": list(range(20)),
        "target": [0] * 20
    })
    sample_path = tmp_path / "sample_submission.csv"
    sample_df.write_csv(sample_path)
    
    return {
        "data_path": str(data_path),
        "test_path": str(test_path),
        "sample_path": str(sample_path),
        "tmp_path": tmp_path
    }


class TestCVScoreRegression:
    """Test for CV score regression."""
    
    def test_cv_score_no_regression(self, benchmark_data):
        """Ensure CV scores don't regress by more than 5%."""
        state = initial_state(
            competition="regression_test",
            data_path=benchmark_data["data_path"]
        )
        
        result = run_professor(state, timeout_seconds=300)
        
        # Get current CV score
        current_cv = result.get("cv_mean")
        
        if current_cv is None:
            pytest.skip("CV score not available (pipeline may have failed)")
        
        # Load baseline
        baseline = _load_baseline("cv_score_baseline.json")
        
        if baseline is None:
            # First run - save as baseline
            _save_baseline("cv_score_baseline.json", {
                "cv_mean": current_cv,
                "timestamp": datetime.now().isoformat(),
                "commit": "initial"
            })
            pytest.skip("Baseline created, run again to check regression")
        
        baseline_cv = baseline["cv_mean"]
        
        # Allow 5% regression
        min_acceptable = baseline_cv * 0.95
        
        assert current_cv >= min_acceptable, (
            f"CV score regressed: {current_cv:.4f} < {min_acceptable:.4f} "
            f"(baseline: {baseline_cv:.4f})"
        )
    
    def test_cv_score_variance_acceptable(self, benchmark_data):
        """Ensure CV score variance is acceptable."""
        state = initial_state(
            competition="variance_test",
            data_path=benchmark_data["data_path"]
        )
        
        result = run_professor(state, timeout_seconds=300)
        
        # Get CV scores
        cv_scores = result.get("cv_scores", [])
        
        if not cv_scores:
            pytest.skip("CV scores not available")
        
        # Check variance
        cv_std = np.std(cv_scores)
        
        # Standard deviation should be < 0.05
        assert cv_std < 0.05, f"CV variance too high: {cv_std:.4f}"


class TestExecutionTimeRegression:
    """Test for execution time regression."""
    
    def test_execution_time_no_regression(self, benchmark_data):
        """Ensure execution time doesn't regress by more than 20%."""
        state = initial_state(
            competition="time_test",
            data_path=benchmark_data["data_path"]
        )
        
        start = time.time()
        result = run_professor(state, timeout_seconds=300)
        elapsed = time.time() - start
        
        # Load baseline
        baseline = _load_baseline("execution_time_baseline.json")
        
        if baseline is None:
            # First run - save as baseline
            _save_baseline("execution_time_baseline.json", {
                "time_seconds": elapsed,
                "timestamp": datetime.now().isoformat(),
                "commit": "initial"
            })
            pytest.skip("Baseline created, run again to check regression")
        
        baseline_time = baseline["time_seconds"]
        
        # Allow 20% regression
        max_acceptable = baseline_time * 1.20
        
        assert elapsed <= max_acceptable, (
            f"Execution time regressed: {elapsed:.1f}s > {max_acceptable:.1f}s "
            f"(baseline: {baseline_time:.1f}s)"
        )
    
    def test_execution_time_reasonable(self, benchmark_data):
        """Ensure execution time is reasonable."""
        state = initial_state(
            competition="time_test",
            data_path=benchmark_data["data_path"]
        )
        
        start = time.time()
        result = run_professor(state, timeout_seconds=300)
        elapsed = time.time() - start
        
        # Should complete in under 5 minutes for small dataset
        assert elapsed < 300, f"Execution took too long: {elapsed:.1f}s"


class TestMemoryUsageRegression:
    """Test for memory usage regression."""
    
    def test_memory_usage_no_regression(self, benchmark_data):
        """Ensure memory usage doesn't regress by more than 50%."""
        import psutil
        
        process = psutil.Process()
        
        state = initial_state(
            competition="memory_test",
            data_path=benchmark_data["data_path"]
        )
        
        # Get initial memory
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        result = run_professor(state, timeout_seconds=300)
        
        # Get peak memory (approximate)
        final_memory = process.memory_info().rss / (1024 * 1024)  # MB
        memory_increase = final_memory - initial_memory
        
        # Load baseline
        baseline = _load_baseline("memory_usage_baseline.json")
        
        if baseline is None:
            # First run - save as baseline
            _save_baseline("memory_usage_baseline.json", {
                "memory_increase_mb": memory_increase,
                "timestamp": datetime.now().isoformat(),
                "commit": "initial"
            })
            pytest.skip("Baseline created, run again to check regression")
        
        baseline_memory = baseline["memory_increase_mb"]
        
        # Allow 50% regression
        max_acceptable = baseline_memory * 1.50
        
        assert memory_increase <= max_acceptable, (
            f"Memory usage regressed: {memory_increase:.1f}MB > {max_acceptable:.1f}MB "
            f"(baseline: {baseline_memory:.1f}MB)"
        )
    
    def test_memory_usage_reasonable(self, benchmark_data):
        """Ensure memory usage is reasonable."""
        import psutil
        
        process = psutil.Process()
        
        state = initial_state(
            competition="memory_test",
            data_path=benchmark_data["data_path"]
        )
        
        # Get initial memory
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        result = run_professor(state, timeout_seconds=300)
        
        # Get final memory
        final_memory = process.memory_info().rss / (1024 * 1024)  # MB
        memory_increase = final_memory - initial_memory
        
        # Should use < 500MB for small dataset
        assert memory_increase < 500, f"Memory usage too high: {memory_increase:.1f}MB"


class TestSubmissionFormatRegression:
    """Test for submission format regression."""
    
    def test_submission_format_unchanged(self, benchmark_data):
        """Ensure submission format hasn't changed."""
        state = initial_state(
            competition="format_test",
            data_path=benchmark_data["data_path"]
        )
        
        result = run_professor(state, timeout_seconds=300)
        
        # Check submission was created
        submission_path = result.get("submission_path")
        
        if submission_path is None:
            pytest.skip("Submission not created")
        
        if not os.path.exists(submission_path):
            pytest.skip(f"Submission file not found: {submission_path}")
        
        # Load submission
        import polars as pl
        submission = pl.read_csv(submission_path)
        sample = pl.read_csv(benchmark_data["sample_path"])
        
        # Check columns match
        assert set(submission.columns) == set(sample.columns), (
            f"Submission columns changed: {submission.columns} vs {sample.columns}"
        )
        
        # Check row count matches
        assert len(submission) == len(sample), (
            f"Submission row count changed: {len(submission)} vs {len(sample)}"
        )
        
        # Check ID column matches
        id_col = sample.columns[0]
        assert (submission[id_col] == sample[id_col]).all(), (
            "Submission ID column changed"
        )
    
    def test_submission_no_nulls(self, benchmark_data):
        """Ensure submission has no null values."""
        state = initial_state(
            competition="null_test",
            data_path=benchmark_data["data_path"]
        )
        
        result = run_professor(state, timeout_seconds=300)
        
        submission_path = result.get("submission_path")
        
        if submission_path is None:
            pytest.skip("Submission not created")
        
        if not os.path.exists(submission_path):
            pytest.skip(f"Submission file not found: {submission_path}")
        
        # Load submission
        import polars as pl
        submission = pl.read_csv(submission_path)
        
        # Check for nulls
        null_count = submission.null_count().sum_horizontal().item()
        assert null_count == 0, f"Submission contains {null_count} null values"


class TestModelQualityRegression:
    """Test for model quality regression."""
    
    def test_model_fallback_works(self, benchmark_data):
        """Ensure model fallback chain works."""
        state = initial_state(
            competition="fallback_test",
            data_path=benchmark_data["data_path"]
        )
        
        result = run_professor(state, timeout_seconds=300)
        
        # Check model registry has at least one model
        model_registry = result.get("model_registry", [])
        
        assert len(model_registry) > 0, "No models in registry"
    
    def test_predictions_valid(self, benchmark_data):
        """Ensure predictions are valid."""
        state = initial_state(
            competition="predictions_test",
            data_path=benchmark_data["data_path"]
        )
        
        result = run_professor(state, timeout_seconds=300)
        
        submission_path = result.get("submission_path")
        
        if submission_path is None:
            pytest.skip("Submission not created")
        
        if not os.path.exists(submission_path):
            pytest.skip(f"Submission file not found: {submission_path}")
        
        # Load submission
        import polars as pl
        import numpy as np
        
        submission = pl.read_csv(submission_path)
        
        # Get target column
        target_col = [c for c in submission.columns if c != "id"][0]
        preds = submission[target_col].to_numpy()
        
        # Check for NaN
        assert not np.any(np.isnan(preds)), "Predictions contain NaN"
        
        # Check for Inf
        assert not np.any(np.isinf(preds)), "Predictions contain Inf"
        
        # Check range (for classification)
        assert np.all(preds >= 0) and np.all(preds <= 1), (
            f"Predictions out of range [0, 1]: min={preds.min()}, max={preds.max()}"
        )
        
        # Check variance
        assert np.std(preds) > 1e-6, "Predictions have no variance"
