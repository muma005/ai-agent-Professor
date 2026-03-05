# tools/submit_tools.py

import os
import polars as pl
import numpy as np
from datetime import datetime
from tools.data_tools import read_csv


class SubmissionValidationError(Exception):
    """Raised when submission.csv fails format validation."""
    pass


def generate_submission(
    predictions: np.ndarray,
    sample_submission_path: str,
    output_path: str,
    target_dtype: str = "auto"
) -> dict:
    """
    Generate and validate submission.csv against the sample submission.

    Validates:
      - Column names match sample_submission.csv exactly
      - Row count matches sample_submission.csv exactly
      - ID column values match sample_submission.csv exactly
      - Zero null values in output
      - Target column dtype matches sample (bool, int, or float)

    Args:
        predictions:           numpy array of predictions (1D)
        sample_submission_path: path to sample_submission.csv
        output_path:           where to write submission.csv
        target_dtype:          "auto", "bool", "int", or "float"

    Returns:
        {"path": str, "rows": int, "columns": list, "validation": dict}

    Raises:
        SubmissionValidationError if any check fails
    """
    if not os.path.exists(sample_submission_path):
        raise FileNotFoundError(
            f"sample_submission.csv not found: {sample_submission_path}"
        )

    sample = read_csv(sample_submission_path)

    # ── Validate prediction length ────────────────────────────────
    if len(predictions) != len(sample):
        raise SubmissionValidationError(
            f"Prediction count mismatch: got {len(predictions)}, "
            f"expected {len(sample)} (from sample_submission.csv)"
        )

    # ── Infer column names from sample ────────────────────────────
    id_col     = sample.columns[0]
    target_col = sample.columns[1]

    # ── Infer target dtype from sample ────────────────────────────
    if target_dtype == "auto":
        sample_dtype = sample[target_col].dtype
        if sample_dtype == pl.Boolean:
            target_dtype = "bool"
        elif sample_dtype in (pl.Float32, pl.Float64):
            target_dtype = "float"
        else:
            target_dtype = "int"

    # ── Cast predictions to correct type ─────────────────────────
    if target_dtype == "bool":
        if predictions.dtype in (np.float64, np.float32):
            preds_cast = (predictions > 0.5).tolist()
        else:
            preds_cast = [bool(p) for p in predictions]
    elif target_dtype == "float":
        preds_cast = [float(p) for p in predictions]
    else:
        preds_cast = [int(round(p)) for p in predictions]

    # ── Build submission DataFrame ────────────────────────────────
    submission = pl.DataFrame({
        id_col:     sample[id_col].to_list(),
        target_col: preds_cast,
    })

    # ── Validate columns ──────────────────────────────────────────
    if set(submission.columns) != set(sample.columns):
        raise SubmissionValidationError(
            f"Column mismatch.\n"
            f"  Expected: {sample.columns}\n"
            f"  Got:      {submission.columns}"
        )

    # ── Validate row count ────────────────────────────────────────
    if len(submission) != len(sample):
        raise SubmissionValidationError(
            f"Row count mismatch: {len(submission)} vs {len(sample)}"
        )

    # ── Validate ID column matches exactly ────────────────────────
    id_matches = (submission[id_col] == sample[id_col]).all()
    if not id_matches:
        mismatches = (submission[id_col] != sample[id_col]).sum()
        raise SubmissionValidationError(
            f"ID column mismatch: {mismatches} IDs do not match sample_submission.csv"
        )

    # ── Validate zero nulls ───────────────────────────────────────
    null_count = submission.null_count().sum_horizontal().item()
    if null_count > 0:
        raise SubmissionValidationError(
            f"Submission contains {null_count} null values -- not allowed"
        )

    # ── Write to disk ─────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    submission.write_csv(output_path)

    validation = {
        "valid":        True,
        "rows":         len(submission),
        "columns":      submission.columns,
        "id_col":       id_col,
        "target_col":   target_col,
        "target_dtype": target_dtype,
        "null_count":   0,
        "validated_at": datetime.utcnow().isoformat(),
    }

    print(f"[SubmitTools] submission.csv valid: {output_path}")
    print(f"[SubmitTools] Rows: {len(submission)} | "
          f"Cols: {submission.columns} | dtype: {target_dtype}")

    return {
        "path":       output_path,
        "rows":       len(submission),
        "columns":    submission.columns,
        "validation": validation,
    }


def validate_existing_submission(
    submission_path: str,
    sample_submission_path: str
) -> dict:
    """
    Validate an already-written submission.csv against sample.
    Returns {"valid": bool, "errors": [list]}. Never raises.
    """
    errors = []

    if not os.path.exists(submission_path):
        return {"valid": False, "errors": [f"File not found: {submission_path}"]}

    try:
        submission = read_csv(submission_path)
        sample     = read_csv(sample_submission_path)
    except Exception as e:
        return {"valid": False, "errors": [f"Failed to read CSV: {e}"]}

    if set(submission.columns) != set(sample.columns):
        errors.append(f"Column mismatch: {submission.columns} vs {sample.columns}")

    if len(submission) != len(sample):
        errors.append(f"Row count: {len(submission)} vs {len(sample)}")

    null_count = submission.null_count().sum_horizontal().item()
    if null_count > 0:
        errors.append(f"Contains {null_count} null values")

    id_col = sample.columns[0]
    if id_col in submission.columns and id_col in sample.columns:
        if not (submission[id_col] == sample[id_col]).all():
            errors.append("ID column values do not match sample_submission.csv")

    return {"valid": len(errors) == 0, "errors": errors}


def save_submission_log(
    session_id: str,
    submission_path: str,
    cv_mean: float,
    lb_score: float = None,
    notes: str = ""
) -> str:
    """
    Append an entry to the session's submission ladder log.
    Used to track every submission and its CV/LB score.
    """
    import json

    log_path = f"outputs/{session_id}/submission_log.jsonl"
    os.makedirs(f"outputs/{session_id}", exist_ok=True)

    entry = {
        "timestamp":       datetime.utcnow().isoformat(),
        "session_id":      session_id,
        "submission_path": submission_path,
        "cv_mean":         cv_mean,
        "lb_score":        lb_score,
        "notes":           notes,
    }

    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")

    print(f"[SubmitTools] Logged submission: CV={cv_mean:.4f} "
          f"LB={lb_score if lb_score else 'pending'}")

    return log_path
