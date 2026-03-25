# tools/prediction_validator.py
"""
Prediction validation before submission.

Validates:
1. Prediction count matches test set
2. No NaN values
3. No Inf values
4. Valid range for classification
5. Non-constant predictions
"""

import numpy as np
from typing import Union, Optional
import logging

logger = logging.getLogger(__name__)


class ProfessorPredictionError(Exception):
    """Raised when prediction validation fails."""
    pass


def validate_predictions(
    preds: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    expected_count: Optional[int] = None,
    task_type: str = "binary",
    check_variance: bool = True,
) -> bool:
    """
    Validate predictions before submission.
    
    Args:
        preds: Predictions (1D or 2D array)
        X_test: Test features (for count validation)
        expected_count: Expected prediction count (alternative to X_test)
        task_type: "binary", "multiclass", "regression"
        check_variance: Whether to check for constant predictions
    
    Returns:
        True if valid
    
    Raises:
        ProfessorPredictionError if invalid
    """
    # Flatten if 2D (for multiclass probabilities)
    if preds.ndim == 2:
        preds_flat = preds.flatten()
    else:
        preds_flat = preds.flatten()
    
    # Check for NaN
    if np.any(np.isnan(preds_flat)):
        nan_count = int(np.sum(np.isnan(preds_flat)))
        raise ProfessorPredictionError(
            f"Predictions contain {nan_count} NaN values"
        )
    
    # Check for Inf
    if np.any(np.isinf(preds_flat)):
        inf_count = int(np.sum(np.isinf(preds_flat)))
        raise ProfessorPredictionError(
            f"Predictions contain {inf_count} Inf values"
        )
    
    # Check count
    if X_test is not None:
        expected = len(X_test) if preds.ndim == 1 else X_test.shape[0]
        if len(preds_flat) != expected:
            raise ProfessorPredictionError(
                f"Prediction count mismatch: {len(preds_flat)} vs {expected}"
            )
    elif expected_count is not None:
        if len(preds_flat) != expected_count:
            raise ProfessorPredictionError(
                f"Prediction count mismatch: {len(preds_flat)} vs {expected_count}"
            )
    
    # Check range for classification
    if task_type in ["binary", "multiclass"]:
        if np.any(preds_flat < 0) or np.any(preds_flat > 1):
            raise ProfessorPredictionError(
                f"Predictions out of range [0, 1]: "
                f"min={float(preds_flat.min()):.4f}, max={float(preds_flat.max()):.4f}"
            )
    
    # Check variance (detect constant predictions)
    if check_variance:
        if np.std(preds_flat) < 1e-6:
            raise ProfessorPredictionError(
                "Predictions have no variance (constant predictions)"
            )
    
    logger.info(f"Predictions validated: {len(preds_flat)} predictions, task_type={task_type}")
    return True


def validate_submission_file(
    submission_path: str,
    sample_submission_path: str,
) -> bool:
    """
    Validate submission file against sample submission.
    
    Args:
        submission_path: Path to submission CSV
        sample_submission_path: Path to sample submission CSV
    
    Returns:
        True if valid
    
    Raises:
        ProfessorPredictionError if invalid
    """
    import polars as pl
    
    # Load files
    submission = pl.read_csv(submission_path)
    sample = pl.read_csv(sample_submission_path)
    
    # Check columns
    if set(submission.columns) != set(sample.columns):
        raise ProfessorPredictionError(
            f"Column mismatch. Expected: {sample.columns}, Got: {submission.columns}"
        )
    
    # Check row count
    if len(submission) != len(sample):
        raise ProfessorPredictionError(
            f"Row count mismatch: {len(submission)} vs {len(sample)}"
        )
    
    # Check for nulls
    null_count = submission.null_count().sum_horizontal().item()
    if null_count > 0:
        raise ProfessorPredictionError(
            f"Submission contains {null_count} null values"
        )
    
    # Check ID column matches
    id_col = sample.columns[0]
    if not (submission[id_col] == sample[id_col]).all():
        mismatches = (submission[id_col] != sample[id_col]).sum()
        raise ProfessorPredictionError(
            f"ID column mismatch: {mismatches} IDs don't match"
        )
    
    logger.info(f"Submission file validated: {submission_path}")
    return True
