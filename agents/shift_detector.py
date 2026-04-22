# agents/shift_detector.py

import os
import json
import logging
import polars as pl
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from core.state import ProfessorState
from core.lineage import log_event
from guards.agent_retry import with_agent_retry
from tools.performance_monitor import timed_node
from scipy.stats import ks_2samp

logger = logging.getLogger(__name__)

AGENT_NAME = "shift_detector"

# ── Helpers ──────────────────────────────────────────────────────────────────

def _calculate_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Calculate Population Stability Index using stable quantile binning."""
    # Use quantiles from expected to define bins
    breakpoints = np.percentile(expected, np.arange(0, 101, 100 / bins))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    
    e_counts = np.histogram(expected, bins=breakpoints)[0]
    a_counts = np.histogram(actual, bins=breakpoints)[0]
    
    e_perc = e_counts / len(expected) + 1e-6
    a_perc = a_counts / len(actual) + 1e-6
    
    psi = np.sum((e_perc - a_perc) * np.log(e_perc / a_perc))
    return float(psi)

def _run_adversarial_validation(train_df: pl.DataFrame, test_df: pl.DataFrame, num_cols: List[str]) -> float:
    """Trains a classifier to distinguish train from test."""
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    
    # 1. Prepare Data
    tr = train_df.select(num_cols).with_columns(pl.lit(0).alias("is_test"))
    te = test_df.select(num_cols).with_columns(pl.lit(1).alias("is_test"))
    combined = pl.concat([tr, te])
    
    X = combined.drop("is_test").to_numpy().astype(np.float32)
    y = combined["is_test"].to_numpy()
    
    # 2. 3-Fold Cross-Validation
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    for tr_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        
        model = lgb.LGBMClassifier(n_estimators=50, verbosity=-1, random_state=42)
        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_val)[:, 1]
        scores.append(roc_auc_score(y_val, probs))
        
    return float(np.mean(scores))

# ── Agent Node ───────────────────────────────────────────────────────────────

@timed_node
@with_agent_retry(AGENT_NAME)
def run_shift_detector(state: ProfessorState) -> ProfessorState:
    """
    Intelligence Layer: Distribution Shift Detector.
    """
    session_id = state.get("session_id", "default")
    output_dir = Path(f"outputs/{session_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"[{AGENT_NAME}] Starting drift analysis...")

    # 1. Load Data
    train_path = state.get("clean_data_path")
    test_path = state.get("test_data_path")
    if not train_path or not test_path or not os.path.exists(test_path):
        logger.warning(f"[{AGENT_NAME}] Test data missing. Skipping detection.")
        return state

    train_df = pl.read_parquet(train_path)
    # Detect separator
    try:
        test_df = pl.read_csv(test_path, n_rows=2000)
    except:
        test_df = pl.read_csv(test_path, n_rows=2000, separator="\t")
    
    target_col = state.get("target_col", "")
    num_cols = [c for c in train_df.columns if train_df[c].dtype.is_numeric() and c != target_col]
    num_cols = [c for c in num_cols if c in test_df.columns]

    if not num_cols:
        logger.warning(f"[{AGENT_NAME}] No numeric columns for drift analysis.")
        return state

    # 2. Adversarial Validation
    try:
        auc = _run_adversarial_validation(train_df, test_df, num_cols)
    except Exception as e:
        logger.error(f"Adversarial validation failed: {e}")
        auc = 0.5

    # 3. Per-Feature KS + PSI
    feature_shifts = {}
    for col in num_cols:
        tr_vals = train_df[col].drop_nulls().to_numpy()
        te_vals = test_df[col].drop_nulls().to_numpy()
        if len(tr_vals) > 10 and len(te_vals) > 10:
            ks_stat, p_val = ks_2samp(tr_vals, te_vals)
            psi = _calculate_psi(tr_vals, te_vals)
            feature_shifts[col] = {
                "ks_p": round(float(p_val), 4),
                "psi": round(float(psi), 4),
                "severity": "HIGH" if psi > 0.2 else ("MEDIUM" if psi > 0.1 else "OK")
            }

    # 4. Severity Classification
    # AUC > 0.85 = Critical (Train/Test are completely different)
    # AUC > 0.70 = High (Significant shift)
    overall_severity = "CRITICAL" if auc > 0.85 else ("HIGH" if auc > 0.70 else "OK")
    
    report = {
        "adversarial_auc": round(auc, 4),
        "overall_severity": overall_severity,
        "feature_shifts": feature_shifts,
        "checked_at": datetime.now(timezone.utc).isoformat()
    }
    
    report_path = output_dir / "drift_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # 5. Update State
    updates = {
        "adversarial_auc": auc,
        "drift_report": report,
        "drift_report_path": str(report_path)
    }

    log_event(
        session_id=session_id,
        agent=AGENT_NAME,
        action="drift_analysis_complete",
        keys_written=list(updates.keys())
    )

    return ProfessorState.validated_update(state, AGENT_NAME, updates)
