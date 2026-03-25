# Phase 2: Quality & Reliability - Implementation Plan

**Priority:** HIGH  
**Estimated Effort:** 20-25 hours  
**Status:** 📋 READY TO START  

---

## Phase 2 Overview

Phase 2 focuses on **Quality & Reliability** improvements to make the Professor pipeline more robust, maintainable, and production-ready.

**Duration:** Week 4-5  
**Goal:** Address remaining high-priority flaws and improve overall system quality

---

## Flaws to Fix (19 Remaining)

### Priority 1: Critical Quality Issues (Week 4)

| # | Flaw ID | Severity | Component | Estimated Time |
|---|---------|----------|-----------|----------------|
| 1 | FLAW-5.1 | 🔴 CRITICAL | End-to-End Integration Tests | 4-5 hours |
| 2 | FLAW-5.2 | 🔴 CRITICAL | Regression Tests | 3-4 hours |
| 3 | FLAW-12.1 | 🔴 CRITICAL | Submission Format Validation | 2-3 hours |
| 4 | FLAW-12.2 | 🔴 CRITICAL | Submission Sanity Checks | 2-3 hours |
| 5 | FLAW-11.6 | 🟠 HIGH | Overfitting Detection | 3-4 hours |
| 6 | FLAW-11.7 | 🟠 HIGH | Model Stability Checks | 3-4 hours |

### Priority 2: Reliability Improvements (Week 5)

| # | Flaw ID | Severity | Component | Estimated Time |
|---|---------|----------|-----------|----------------|
| 7 | FLAW-6.1 | 🟠 HIGH | Performance Monitoring | 3-4 hours |
| 8 | FLAW-9.1 | 🟡 MEDIUM | Memory Profiling | 2-3 hours |
| 9 | FLAW-10.1 | 🟡 MEDIUM | Seed Management | 2-3 hours |
| 10 | FLAW-10.2 | 🟡 MEDIUM | Reproducibility Checks | 2-3 hours |
| 11 | FLAW-7.3 | 🟡 MEDIUM | API Key Security | 2-3 hours |
| 12 | FLAW-8.1 | 🟡 MEDIUM | API Response Validation | 2-3 hours |

### Priority 3: Code Quality (Week 5)

| # | Flaw ID | Severity | Component | Estimated Time |
|---|---------|----------|-----------|----------------|
| 13 | FLAW-13.1 | 🟡 MEDIUM | Code Linting | 2-3 hours |
| 14 | FLAW-13.2 | 🟡 MEDIUM | Type Hints | 2-3 hours |
| 15 | FLAW-13.3 | 🟡 MEDIUM | Documentation Standards | 2-3 hours |
| 16 | FLAW-5.3 | 🟡 MEDIUM | Contract Tests | 2-3 hours |
| 17 | FLAW-6.2 | 🟡 MEDIUM | Caching Strategy | 2-3 hours |
| 18 | FLAW-8.2 | 🟡 MEDIUM | API Dependency Graph | 2-3 hours |
| 19 | FLAW-9.2 | 🟡 MEDIUM | GC Optimization | 2-3 hours |

---

## Implementation Order

### Step 1: End-to-End Integration Tests (FLAW-5.1)
**File:** `tests/integration/test_full_pipeline.py`

**Why First:** All other improvements need test coverage to ensure they don't break existing functionality.

**Implementation:**
```python
# tests/integration/test_full_pipeline.py
"""
End-to-end integration tests for full Professor pipeline.
"""
import pytest
import tempfile
import os
import polars as pl
import numpy as np

from core.state import initial_state
from core.professor import run_professor, ProfessorPipelineError


class TestFullPipelineIntegration:
    """Test full pipeline integration."""
    
    @pytest.fixture
    def synthetic_data(self, tmp_path):
        """Create synthetic dataset for testing."""
        np.random.seed(42)
        n_rows = 100
        n_features = 5
        
        X = np.random.randn(n_rows, n_features)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        df = pl.DataFrame({
            f"feature_{i}": X[:, i] for i in range(n_features)
        })
        df = df.with_columns(pl.Series("target", y))
        
        data_path = tmp_path / "train.csv"
        df.write_csv(data_path)
        
        # Create sample submission
        sample_df = pl.DataFrame({
            "id": list(range(20)),
            "target": [0] * 20
        })
        sample_path = tmp_path / "sample_submission.csv"
        sample_df.write_csv(sample_path)
        
        return {
            "data_path": str(data_path),
            "sample_path": str(sample_path),
            "tmp_path": tmp_path
        }
    
    def test_pipeline_completes_successfully(self, synthetic_data):
        """Test pipeline completes end-to-end."""
        state = initial_state(
            competition="integration_test",
            data_path=synthetic_data["data_path"]
        )
        
        # Run with short timeout for integration test
        result = run_professor(state, timeout_seconds=300)
        
        assert result is not None
        assert "cv_mean" in result
        assert result["cv_mean"] > 0  # Should have valid CV score
    
    def test_pipeline_handles_api_failure(self, synthetic_data, monkeypatch):
        """Test pipeline handles API failures gracefully."""
        # Mock API to fail
        def mock_call_llm(*args, **kwargs):
            raise Exception("API down")
        
        monkeypatch.setattr("tools.llm_client.call_llm", mock_call_llm)
        
        state = initial_state(
            competition="integration_test",
            data_path=synthetic_data["data_path"]
        )
        
        # Should fail with proper error
        with pytest.raises(ProfessorPipelineError):
            run_professor(state, timeout_seconds=60)
    
    def test_pipeline_resume_from_checkpoint(self, synthetic_data):
        """Test pipeline can resume from checkpoint."""
        # TODO: Implement checkpoint resume test
        pass
```

**Tests:**
- Test pipeline completes successfully
- Test pipeline handles API failures
- Test pipeline handles timeouts
- Test pipeline handles model training failures
- Test pipeline resume from checkpoint
- Test pipeline with invalid data

---

### Step 2: Regression Tests (FLAW-5.2)
**File:** `tests/regression/test_performance_regression.py`

**Implementation:**
```python
# tests/regression/test_performance_regression.py
"""
Regression tests to ensure performance doesn't degrade.
"""
import pytest
import time
import numpy as np
from datetime import datetime

from core.state import initial_state
from core.professor import run_professor


class TestPerformanceRegression:
    """Test for performance regressions."""
    
    @pytest.fixture
    def benchmark_data(self, tmp_path):
        """Create benchmark dataset."""
        np.random.seed(42)
        n_rows = 200
        n_features = 10
        
        X = np.random.randn(n_rows, n_features)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        # Save to file
        # ... (similar to integration test fixture)
        
        return {"data_path": str(data_path)}
    
    def test_cv_score_no_regression(self, benchmark_data):
        """Ensure CV scores don't regress by more than 5%."""
        state = initial_state(
            competition="regression_test",
            data_path=benchmark_data["data_path"]
        )
        
        result = run_professor(state, timeout_seconds=300)
        
        # Baseline from previous run (stored in file)
        baseline_cv = self._load_baseline_cv()
        
        # Allow 5% regression
        min_acceptable = baseline_cv * 0.95
        
        assert result["cv_mean"] >= min_acceptable, (
            f"CV score regressed: {result['cv_mean']} < {min_acceptable}"
        )
    
    def test_execution_time_no_regression(self, benchmark_data):
        """Ensure execution time doesn't regress by more than 20%."""
        state = initial_state(
            competition="regression_test",
            data_path=benchmark_data["data_path"]
        )
        
        start = time.time()
        run_professor(state, timeout_seconds=300)
        elapsed = time.time() - start
        
        # Baseline from previous run
        baseline_time = self._load_baseline_time()
        
        # Allow 20% regression
        max_acceptable = baseline_time * 1.20
        
        assert elapsed <= max_acceptable, (
            f"Execution time regressed: {elapsed}s > {max_acceptable}s"
        )
    
    def _load_baseline_cv(self):
        """Load baseline CV score from file."""
        # Load from tests/regression/baselines/cv_score.json
        pass
    
    def _load_baseline_time(self):
        """Load baseline execution time from file."""
        # Load from tests/regression/baselines/execution_time.json
        pass
```

---

### Step 3: Submission Format Validation (FLAW-12.1, FLAW-12.2)
**File:** `tools/submission_validator.py` (new)

**Implementation:**
```python
# tools/submission_validator.py
"""
Submission validation before Kaggle upload.

Validates:
1. Format matches sample submission
2. No null values
3. Valid ID column
4. Valid prediction range
5. Non-constant predictions
6. File size limits
"""
import polars as pl
import os
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class ProfessorSubmissionError(Exception):
    """Raised when submission validation fails."""
    pass


def validate_submission_format(
    submission_path: str,
    sample_submission_path: str,
) -> bool:
    """
    Validate submission format against sample submission.
    
    Args:
        submission_path: Path to submission CSV
        sample_submission_path: Path to sample submission CSV
    
    Returns:
        True if valid
    
    Raises:
        ProfessorSubmissionError if invalid
    """
    submission = pl.read_csv(submission_path)
    sample = pl.read_csv(sample_submission_path)
    
    # Check columns match exactly
    if set(submission.columns) != set(sample.columns):
        raise ProfessorSubmissionError(
            f"Column mismatch. Expected: {sample.columns}, Got: {submission.columns}"
        )
    
    # Check row count matches
    if len(submission) != len(sample):
        raise ProfessorSubmissionError(
            f"Row count mismatch: {len(submission)} vs {len(sample)}"
        )
    
    # Check ID column values match
    id_col = sample.columns[0]
    if not (submission[id_col] == sample[id_col]).all():
        mismatches = (submission[id_col] != sample[id_col]).sum()
        raise ProfessorSubmissionError(
            f"ID column mismatch: {mismatches} IDs don't match"
        )
    
    # Check for null values
    null_count = submission.null_count().sum_horizontal().item()
    if null_count > 0:
        raise ProfessorSubmissionError(
            f"Submission contains {null_count} null values"
        )
    
    # Check target column
    target_col = sample.columns[1] if len(sample.columns) > 1 else sample.columns[0]
    
    # Check for NaN/Inf in predictions
    target_data = submission[target_col].to_numpy()
    if np.any(np.isnan(target_data)):
        raise ProfessorSubmissionError("Predictions contain NaN values")
    
    if np.any(np.isinf(target_data)):
        raise ProfessorSubmissionError("Predictions contain Inf values")
    
    # Check for constant predictions
    if np.std(target_data) < 1e-6:
        raise ProfessorSubmissionError(
            "Predictions have no variance (constant predictions)"
        )
    
    # Check file size (Kaggle limit: 100MB)
    file_size_mb = os.path.getsize(submission_path) / (1024 * 1024)
    if file_size_mb > 100:
        raise ProfessorSubmissionError(
            f"Submission file too large: {file_size_mb:.1f}MB > 100MB limit"
        )
    
    logger.info(f"Submission validated: {submission_path}")
    return True


def validate_submission_predictions(
    preds: np.ndarray,
    task_type: str = "binary",
    check_distribution: bool = True,
) -> bool:
    """
    Validate prediction distribution before creating submission.
    
    Args:
        preds: Predictions
        task_type: "binary", "multiclass", "regression"
        check_distribution: Whether to check prediction distribution
    
    Returns:
        True if valid
    
    Raises:
        ProfessorSubmissionError if invalid
    """
    # Check for NaN/Inf
    if np.any(np.isnan(preds)):
        raise ProfessorSubmissionError(
            f"Predictions contain {np.sum(np.isnan(preds))} NaN values"
        )
    
    if np.any(np.isinf(preds)):
        raise ProfessorSubmissionError(
            f"Predictions contain {np.sum(np.isinf(preds))} Inf values"
        )
    
    # Check range for classification
    if task_type in ["binary", "multiclass"]:
        if np.any(preds < 0) or np.any(preds > 1):
            raise ProfessorSubmissionError(
                f"Predictions out of range [0, 1]: "
                f"min={float(preds.min()):.4f}, max={float(preds.max()):.4f}"
            )
    
    # Check distribution (detect all-zeros or all-ones)
    if check_distribution:
        if task_type == "binary":
            if np.mean(preds) < 0.01 or np.mean(preds) > 0.99:
                logger.warning(
                    f"Prediction distribution suspicious: "
                    f"mean={np.mean(preds):.4f} (possible constant predictions)"
                )
    
    return True
```

**Integration in submit node:**
```python
# agents/submission_strategist.py (or core/professor.py)
from tools.submission_validator import validate_submission_format, validate_submission_predictions

def run_submit(state: ProfessorState) -> ProfessorState:
    # ... generate predictions ...
    
    # Validate predictions before creating submission
    validate_submission_predictions(preds, task_type=task_type)
    
    # ... create submission file ...
    
    # Validate submission format
    validate_submission_format(submission_path, sample_submission_path)
    
    return {**state, "submission_path": submission_path}
```

---

### Step 4: Overfitting Detection (FLAW-11.6)
**File:** `agents/ml_optimizer.py` (add to existing)

**Implementation:**
```python
# agents/ml_optimizer.py (add)

def detect_overfitting(
    train_score: float,
    cv_score: float,
    threshold: float = 0.1,
) -> tuple[bool, float]:
    """
    Detect overfitting by comparing train and CV scores.
    
    Args:
        train_score: Training score
        cv_score: Cross-validation score
        threshold: Maximum acceptable gap (default: 0.1 = 10%)
    
    Returns:
        (is_overfitting, gap)
    """
    gap = train_score - cv_score
    
    if gap > threshold:
        logger.warning(
            f"Overfitting detected! "
            f"Train: {train_score:.4f}, CV: {cv_score:.4f}, Gap: {gap:.4f}"
        )
        return True, gap
    
    return False, gap


def check_cv_lb_consistency(
    cv_scores: list[float],
    lb_score: Optional[float] = None,
) -> bool:
    """
    Check if CV scores are consistent with LB score (if available).
    
    Args:
        cv_scores: List of CV fold scores
        lb_score: Leaderboard score (optional)
    
    Returns:
        True if consistent
    """
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    
    if lb_score is not None:
        # LB should be within 2 std of CV mean
        if abs(lb_score - cv_mean) > 2 * cv_std:
            logger.warning(
                f"CV-LB inconsistency detected! "
                f"CV: {cv_mean:.4f}±{cv_std:.4f}, LB: {lb_score:.4f}"
            )
            return False
    
    return True
```

---

### Step 5: Model Stability Checks (FLAW-11.7)
**File:** `agents/ml_optimizer.py` (add to existing)

**Implementation:**
```python
# agents/ml_optimizer.py (add)

def check_model_stability(
    X: np.ndarray,
    y: np.ndarray,
    params: dict,
    model_type: str,
    n_seeds: int = 5,
    max_std: float = 0.05,
) -> tuple[bool, float, float]:
    """
    Check model stability across multiple random seeds.
    
    Args:
        X: Features
        y: Target
        params: Model parameters
        model_type: Model type
        n_seeds: Number of seeds to test
        max_std: Maximum acceptable standard deviation
    
    Returns:
        (is_stable, mean_score, std_score)
    """
    from sklearn.model_selection import cross_val_score
    
    scores = []
    
    for seed in range(n_seeds):
        model = _train_single_model(X, y, {**params, "random_state": seed}, model_type)
        
        # Quick CV to get score
        cv_scores = cross_val_score(model, X, y, cv=3, scoring="roc_auc")
        scores.append(np.mean(cv_scores))
    
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    
    if std_score > max_std:
        logger.warning(
            f"Model instability detected! "
            f"Mean: {mean_score:.4f}, Std: {std_score:.4f} (max: {max_std})"
        )
        return False, mean_score, std_score
    
    return True, mean_score, std_score
```

---

## Testing Strategy

### Unit Tests
Each new component gets unit tests:
- Submission validation tests
- Overfitting detection tests
- Stability check tests
- Performance regression tests

### Integration Tests
Test components work together:
- Full pipeline with validation
- Pipeline with failure scenarios
- Pipeline with resume capability

### Regression Tests
Ensure fixes don't break existing functionality:
- CV scores unchanged
- Submission format unchanged
- Execution time within bounds

---

## Success Criteria

- [ ] All 19 flaws addressed (or documented for Phase 3)
- [ ] All unit tests passing
- [ ] All integration tests passing
- [ ] All regression tests passing
- [ ] Documentation complete
- [ ] Code reviewed

---

## Timeline

### Week 4: Critical Quality Issues
- Day 1-2: End-to-End Integration Tests
- Day 3-4: Regression Tests
- Day 5: Submission Format Validation

### Week 5: Reliability Improvements
- Day 1-2: Overfitting Detection + Stability Checks
- Day 3: Performance Monitoring
- Day 4: Memory Profiling + Seed Management
- Day 5: Code Quality (linting, type hints)

---

**Document Version:** 1.0  
**Created:** 2026-03-25  
**Status:** 📋 READY FOR IMPLEMENTATION
