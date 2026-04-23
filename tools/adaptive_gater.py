# tools/adaptive_gater.py

import logging
import numpy as np
import polars as pl
from typing import List, Dict, Any, Tuple
from core.state import ProfessorState

logger = logging.getLogger(__name__)

# ── Core Logic ──────────────────────────────────────────────────────────────

def evaluate_feature_performance(
    df: pl.DataFrame,
    target_col: str,
    feature_name: str,
    task_type: str = "classification",
    n_estimators: int = 50,
) -> Dict[str, Any]:
    """
    Train with and without the feature to measure exact gain.
    """
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold, KFold
    from sklearn.metrics import roc_auc_score, mean_squared_error

    # 1. Prepare base and full sets
    other_cols = [c for c in df.columns if c != target_col and c != feature_name]
    
    # 2. Fast 3-fold CV
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42) if task_type != "regression" else KFold(n_splits=3, shuffle=True, random_state=42)
    
    def get_score(cols):
        if not cols:
            # Baseline for empty feature set: dummy mean prediction
            y = df[target_col].to_numpy()
            if task_type == "regression":
                return -mean_squared_error(y, np.full_like(y, np.mean(y)))
            else:
                # 0.5 AUC for random/mean classifier
                return 0.5

        X = df.select(cols).to_numpy()
        y = df[target_col].to_numpy()
        scores = []
        for tr_idx, val_idx in cv.split(X, y):
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]
            
            if task_type == "regression":
                model = lgb.LGBMRegressor(n_estimators=n_estimators, verbosity=-1)
                model.fit(X_tr, y_tr)
                preds = model.predict(X_val)
                scores.append(-mean_squared_error(y_val, preds))
            else:
                model = lgb.LGBMClassifier(n_estimators=n_estimators, verbosity=-1)
                model.fit(X_tr, y_tr)
                probs = model.predict_proba(X_val)[:, 1]
                scores.append(roc_auc_score(y_val, probs))
        return np.mean(scores)

    score_base = get_score(other_cols)
    score_full = get_score(other_cols + [feature_name])
    
    improvement = score_full - score_base
    
    return {
        "feature": feature_name,
        "base_score": float(score_base),
        "full_score": float(score_full),
        "improvement": float(improvement),
        "is_beneficial": bool(improvement > 0.0005) # Cast to native bool
    }

def run_adaptive_gate(state: ProfessorState, features_to_test: List[str]) -> Tuple[List[str], List[Dict]]:
    """
    Evaluate a batch of features and return only those that pass the gate.
    """
    clean_path = state.get("clean_data_path")
    target_col = state.get("target_col")
    task_type = state.get("task_type", "classification")
    
    if not clean_path or not os.path.exists(clean_path):
        return features_to_test, []

    df = pl.read_parquet(clean_path)
    # Minimal categorical encoding for LGBM
    df = df.with_columns([
        pl.col(c).cast(pl.Categorical).to_physical().cast(pl.Int32) 
        for c in df.columns if df[c].dtype == pl.Utf8 or df[c].dtype == pl.String
    ])

    passed = []
    reports = []
    
    for feat in features_to_test:
        if feat not in df.columns: continue
        res = evaluate_feature_performance(df, target_col, feat, task_type)
        reports.append(res)
        if res["is_beneficial"]:
            passed.append(feat)
            
    return passed, reports

import os
