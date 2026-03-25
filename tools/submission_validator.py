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
import numpy as np
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
    if not os.path.exists(submission_path):
        raise ProfessorSubmissionError(f"Submission file not found: {submission_path}")
    
    if not os.path.exists(sample_submission_path):
        raise ProfessorSubmissionError(f"Sample submission not found: {sample_submission_path}")
    
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


def validate_submission_from_state(
    state: dict,
    submission_path: Optional[str] = None,
) -> bool:
    """
    Validate submission using state information.
    
    Args:
        state: Professor state dict
        submission_path: Path to submission (optional, uses state if not provided)
    
    Returns:
        True if valid
    
    Raises:
        ProfessorSubmissionError if invalid
    """
    if submission_path is None:
        submission_path = state.get("submission_path")
    
    if submission_path is None:
        raise ProfessorSubmissionError("No submission path provided")
    
    sample_path = state.get("sample_submission_path")
    
    if sample_path is None:
        raise ProfessorSubmissionError("No sample submission path in state")
    
    task_type = state.get("task_type", "binary")
    
    # Validate format
    validate_submission_format(submission_path, sample_path)
    
    logger.info(f"Submission validated successfully: {submission_path}")
    return True
