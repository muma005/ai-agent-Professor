# shields/submission_safety.py

import os
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import polars as pl
from scipy.stats import spearmanr

from core.state import ProfessorState

logger = logging.getLogger(__name__)

def calculate_diversity(path_a: str, path_b: str) -> float:
    """Calculates Spearman correlation between two submission files. (v1 compatibility)"""
    if not (path_a and path_b and os.path.exists(path_a) and os.path.exists(path_b)):
        return 0.0
    
    try:
        df_a = pl.read_csv(path_a) if path_a.endswith(".csv") else pl.read_parquet(path_a)
        df_b = pl.read_csv(path_b) if path_b.endswith(".csv") else pl.read_parquet(path_b)
        
        # Second column is usually the target
        col_a = df_a.columns[1]
        col_b = df_b.columns[1]
        
        s_a = df_a[col_a].to_numpy()
        s_b = df_b[col_b].to_numpy()

        if len(s_a) < 2:
            return 1.0
        
        std_a = np.std(s_a)
        std_b = np.std(s_b)

        if std_a == 0 or std_b == 0:
            return 1.0
            
        corr, _ = spearmanr(s_a, s_b)
        return float(corr)
    except Exception as e:
        logger.warning(f"Diversity calculation failed: {e}")
        return 0.0

def _estimate_lb_noise_v1(ewma_initial: Optional[list], ewma_current: Optional[list]) -> float:
    """Estimates LB noise based on divergence of EMA from initial state. (v1 compatibility)"""
    if not ewma_initial or not ewma_current:
        return 0.0
        
    try:
        a = np.array(ewma_initial)
        b = np.array(ewma_current)
        if len(a) != len(b): return 0.0
        
        noise = np.mean(np.abs(a - b)) / (np.mean(np.abs(a)) + 1e-9)
        return float(noise)
    except:
        return 0.0

def check_submission_safety(state: ProfessorState) -> ProfessorState:
    """
    Shield: Performs safety analysis on submissions and EMA stability. (v1 compatibility)
    """
    path_a = state.get("submission_a_path")
    path_b = state.get("submission_b_path")
    diversity_corr = calculate_diversity(path_a, path_b)
    
    # 2. LB Noise Estimation (V1 style)
    noise = _estimate_lb_noise_v1(state.get("ewma_initial"), state.get("ewma_current"))
    
    # 3. Freeze Override Logic
    context = state.get("competition_context") or {}
    days_left = context.get("days_remaining", 99)
    freeze_override_active = (days_left < 7)
    
    # 4. Generate Report
    safety_report = {
        "diversity_spearman": round(diversity_corr, 4),
        "diversity_rating": "HIGH" if diversity_corr < 0.90 else ("MEDIUM" if diversity_corr < 0.98 else "LOW"),
        "lb_noise_estimate": round(noise, 6), # Numerical for tests
        "freeze_override_active": bool(freeze_override_active),
        "days_remaining": int(days_left),
        "risk_level": "CRITICAL" if noise > 0.1 else ("WARNING" if noise > 0.05 else "SAFE")
    }
    
    updates = {
        "submission_safety_report": safety_report,
        "submission_b_correlation_with_a": diversity_corr
    }
    
    return ProfessorState.validated_update(state, "submission_safety", updates)

def verify_submission(
    submission_path: str,
    sample_submission_path: str,
    canonical_test_rows: int,
    competition_type: str = "binary",
) -> dict:
    """
    Verify a submission file is valid before marking as final.
    Runs 7 checks. ALL must pass for the submission to be valid.
    """
    issues = []
    checks = {}
    
    # Load submission
    try:
        submission = pl.read_csv(submission_path)
    except Exception as e:
        return {"valid": False, "checks": {}, "issues": [f"Cannot read submission: {e}"]}
    
    # Load sample for reference
    try:
        sample = pl.read_csv(sample_submission_path) if sample_submission_path and os.path.exists(sample_submission_path) else None
    except Exception:
        sample = None
    
    # Check 1: Row count
    checks["row_count_ok"] = len(submission) == canonical_test_rows
    if not checks["row_count_ok"]:
        issues.append(f"Row count {len(submission)} != expected {canonical_test_rows}")
    
    # Check 2: Column count
    if sample is not None:
        checks["column_count_ok"] = len(submission.columns) == len(sample.columns)
        if not checks["column_count_ok"]:
            issues.append(f"Column count {len(submission.columns)} != expected {len(sample.columns)}")
    else:
        checks["column_count_ok"] = True
    
    # Check 3: Column names
    if sample is not None:
        checks["column_names_ok"] = list(submission.columns) == list(sample.columns)
        if not checks["column_names_ok"]:
            issues.append(f"Columns {list(submission.columns)} != expected {list(sample.columns)}")
    else:
        checks["column_names_ok"] = True
    
    # Check 4: No NaN/null in prediction columns
    pred_cols = [c for c in submission.columns if c.lower() not in ("id", "index", "row_id")]
    nan_count = 0
    for c in pred_cols:
        nan_count += submission[c].null_count()
        
    checks["no_nans"] = nan_count == 0
    if not checks["no_nans"]:
        issues.append(f"{nan_count} NaN values found in prediction columns")
    
    # Check 5: Value range
    range_ok = True
    for col in pred_cols:
        vals = submission[col].to_numpy()
        if competition_type in ("binary",):
            if np.any(vals < 0) or np.any(vals > 1):
                range_ok = False
                issues.append(f"Column '{col}': predictions outside [0,1] — min={vals.min():.4f}, max={vals.max():.4f}")
        elif competition_type == "regression":
            if np.any(np.isinf(vals)):
                range_ok = False
                issues.append(f"Column '{col}': contains inf values")
    checks["range_ok"] = range_ok
    
    # Check 6: Not truncated (last row check)
    try:
        last_row = submission.tail(1)
        checks["not_truncated"] = last_row.null_count().sum_horizontal()[0] == 0
        if not checks["not_truncated"]:
            issues.append("Last row has null values — file may be truncated")
    except Exception:
        checks["not_truncated"] = False
        issues.append("Cannot verify last row — file may be truncated")
    
    # Check 7: Format (CSV delimiter)
    try:
        with open(submission_path, "r") as f:
            first_line = f.readline()
        checks["format_ok"] = "," in first_line
        if not checks["format_ok"]:
            issues.append("File does not appear to be comma-delimited CSV")
    except:
        checks["format_ok"] = False
        issues.append("Cannot read file format")
    
    valid = all(checks.values())
    return {"valid": valid, "checks": checks, "issues": issues}


def check_submission_diversity(submission_1_path: str, submission_2_path: str) -> dict:
    """Check diversity between final 2 submissions."""
    try:
        sub1 = pl.read_csv(submission_1_path)
        sub2 = pl.read_csv(submission_2_path)
        pred_col = [c for c in sub1.columns if c.lower() not in ("id", "index", "row_id")][0]
        preds_1 = sub1[pred_col].to_numpy().astype(float)
        preds_2 = sub2[pred_col].to_numpy().astype(float)
        corr = float(np.corrcoef(preds_1, preds_2)[0, 1])
        
        if corr > 0.995: rating = "warning"
        elif corr > 0.95: rating = "moderate"
        else: rating = "good"
        
        return {"correlation": round(corr, 4), "diversity_rating": rating}
    except Exception as e:
        return {"correlation": 1.0, "diversity_rating": "error", "error": str(e)}


def estimate_lb_noise(n_public_rows: int, n_total_test_rows: int) -> dict:
    """Estimate noise in public LB based on sample fraction."""
    fraction = n_public_rows / max(n_total_test_rows, 1)
    if fraction < 0.20:
        return {"public_fraction": round(fraction, 3), "noise_level": "high", "gap_threshold": 0.01}
    elif fraction < 0.50:
        return {"public_fraction": round(fraction, 3), "noise_level": "moderate", "gap_threshold": 0.005}
    else:
        return {"public_fraction": round(fraction, 3), "noise_level": "low", "gap_threshold": 0.003}


def check_ewma_freeze(days_remaining: int, cv_lb_gap: float, gap_threshold: float) -> dict:
    """Determine whether to freeze submissions based on CV/LB gap."""
    should_freeze = cv_lb_gap > gap_threshold
    if should_freeze and days_remaining <= 7:
        return {"freeze_recommended": True, "freeze_enforced": False, "requires_operator": True}
    elif should_freeze:
        return {"freeze_recommended": True, "freeze_enforced": True, "requires_operator": False}
    else:
        return {"freeze_recommended": False, "freeze_enforced": False, "requires_operator": False}


def backup_submission(submission_path: str, session_dir: str) -> str:
    """Copy submission to timestamped backup. Keep last 20."""
    backup_dir = os.path.join(session_dir, "submission_backups")
    os.makedirs(backup_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(backup_dir, f"submission_{timestamp}.csv")
    shutil.copy2(submission_path, backup_path)
    
    backups = sorted(Path(backup_dir).glob("submission_*.csv"))
    for old in backups[:-20]:
        old.unlink()
    return backup_path


def select_final_submissions(submissions_history: list[dict]) -> dict:
    """Select the 2 final submissions for the competition."""
    if not submissions_history:
        return {"selection_1": None, "selection_2": None}
    
    # Sort by LB score for Selection 1
    by_lb = sorted(submissions_history, key=lambda s: s.get("lb_score", 0), reverse=True)
    selection_1 = by_lb[0]
    
    # Selection 2: best CV from a DIFFERENT run
    other_runs = [s for s in submissions_history if s.get("run_id") != selection_1.get("run_id")]
    if other_runs:
        by_cv = sorted(other_runs, key=lambda s: s.get("cv_score", 0), reverse=True)
        selection_2 = by_cv[0]
    else:
        selection_2 = selection_1
        
    return {"selection_1": selection_1, "selection_2": selection_2}
