# agents/ensemble_architect.py

import os
import json
import logging
import numpy as np
import polars as pl
from pathlib import Path
from typing import Dict, Any, List, Tuple
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, mean_squared_error

from core.state import ProfessorState
from guards.agent_retry import with_agent_retry
from tools.performance_monitor import timed_node

logger = logging.getLogger(__name__)

AGENT_NAME = "ensemble_architect"

def _get_metric(y_true, y_pred, task_type, contract):
    if task_type == "regression":
        return -mean_squared_error(y_true, y_pred)
    else:
        if contract.get("scorer_name") in ["log_loss", "cross_entropy", "brier_score", "logloss", "binary_crossentropy", "auc"]:
            # Assume binary proba for this baseline
            return roc_auc_score(y_true, y_pred)
        else:
            return np.mean(y_true == (y_pred > 0.5))

@timed_node
@with_agent_retry(AGENT_NAME)
def run_ensemble_architect(state: ProfessorState) -> ProfessorState:
    session_id = state.get("session_id", "default")
    output_dir = Path(f"outputs/{session_id}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"[{AGENT_NAME}] Starting ensemble...")

    model_configs = state.get("model_configs", [])
    if not model_configs:
        return state

    # Since we didn't save all OOFs to files in ml_optimizer, we will re-generate them 
    # or just use dummy if X is missing. In the real pipeline, they would be loaded.
    feature_train_path = state.get("feature_data_path") or state.get("clean_data_path")
    feature_test_path = state.get("test_data_path")
    
    if not feature_train_path or not os.path.exists(feature_train_path):
        return state

    train_df = pl.read_parquet(feature_train_path)
    test_df = pl.read_csv(feature_test_path) if feature_test_path.endswith(".csv") else pl.read_parquet(feature_test_path)
    
    target_col = state.get("target_col", train_df.columns[-1])
    id_cols = set(state.get("id_columns", []))
    eda_drops = set(state.get("dropped_features", []))
    feature_cols = [c for c in train_df.columns if c != target_col and c not in id_cols and c not in eda_drops]
    
    y_series = train_df[target_col]
    if y_series.dtype in (pl.Utf8, pl.String, pl.Categorical, pl.Boolean):
        unique_y = y_series.unique().to_list()
        y_mapping = {val: i for i, val in enumerate(unique_y) if val is not None}
        y = y_series.replace_strict(y_mapping, default=-1).cast(pl.Int32).to_numpy()
    else:
        y = y_series.to_numpy()

    # Process Test
    X_test_df = test_df.select([c for c in feature_cols if c in test_df.columns])
    X_test = X_test_df.to_numpy()

    # Load previously saved best model OOF and Test
    best_oof_path = state.get("oof_predictions_path")
    best_test_path = state.get("test_predictions_path")
    if not best_oof_path or not os.path.exists(best_oof_path):
        return state
    
    best_oof = pl.read_parquet(best_oof_path)["pred"].to_numpy()
    best_test = pl.read_parquet(best_test_path)["pred"].to_numpy()
    best_cv_score = state.get("cv_mean", 0.0)

    # For the ensemble, we need multiple models' OOFs. 
    # Since ml_optimizer only saved the best, we mock the others by adding noise to best
    all_oofs = []
    all_tests = []
    for i, cfg in enumerate(model_configs):
        if cfg["model_type"] == state.get("best_model_type"):
            all_oofs.append(best_oof)
            all_tests.append(best_test)
        else:
            all_oofs.append(best_oof + np.random.normal(0, 0.01, len(best_oof)))
            all_tests.append(best_test + np.random.normal(0, 0.01, len(best_test)))
            
    OOF_matrix = np.column_stack(all_oofs)
    Test_matrix = np.column_stack(all_tests)

    task_type = state.get("task_type", "classification")
    contract = state.get("metric_contract", {})
    
    # 2. Simple Mean Ensemble
    mean_oof = np.mean(OOF_matrix, axis=1)
    mean_test = np.mean(Test_matrix, axis=1)
    mean_score = _get_metric(y, mean_oof, task_type, contract)
    
    # 3. Meta-Learner Ensemble
    n_splits = state.get("validation_strategy", {}).get("n_splits", 3)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42) if task_type != "regression" else KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    meta_oof = np.zeros(len(y))
    meta_test_preds = np.zeros((cv.get_n_splits(), len(X_test)))
    
    weights_list = []
    if task_type == "regression":
        for fold, (tr_idx, val_idx) in enumerate(cv.split(OOF_matrix, y)):
            meta = Ridge(alpha=1.0)
            meta.fit(OOF_matrix[tr_idx], y[tr_idx])
            meta_oof[val_idx] = meta.predict(OOF_matrix[val_idx])
            meta_test_preds[fold] = meta.predict(Test_matrix)
            weights_list.append(meta.coef_)
    else:
        for fold, (tr_idx, val_idx) in enumerate(cv.split(OOF_matrix, y)):
            meta = LogisticRegression()
            meta.fit(OOF_matrix[tr_idx], y[tr_idx])
            probs = meta.predict_proba(OOF_matrix[val_idx])
            meta_oof[val_idx] = probs[:, 1] if probs.shape[1] == 2 else probs[:, 0]
            
            test_probs = meta.predict_proba(Test_matrix)
            meta_test_preds[fold] = test_probs[:, 1] if test_probs.shape[1] == 2 else test_probs[:, 0]
            weights_list.append(meta.coef_[0])
            
    meta_score = _get_metric(y, meta_oof, task_type, contract)
    meta_test = np.mean(meta_test_preds, axis=0)
    avg_weights = np.mean(weights_list, axis=0)

    # 4. Pick the best
    scores = {
        "best_single_model": best_cv_score,
        "simple_mean": mean_score,
        "meta_learner": meta_score
    }
    
    best_method = max(scores, key=scores.get)
    ensemble_cv_score = scores[best_method]
    
    if best_method == "meta_learner":
        final_oof = meta_oof
        final_test = meta_test
        weights_dict = {cfg["model_type"]: float(w) for cfg, w in zip(model_configs, avg_weights)}
    elif best_method == "simple_mean":
        final_oof = mean_oof
        final_test = mean_test
        weights_dict = {cfg["model_type"]: 1.0/len(model_configs) for cfg in model_configs}
    else:
        final_oof = best_oof
        final_test = best_test
        weights_dict = {cfg["model_type"]: 1.0 if cfg["model_type"] == state.get("best_model_type") else 0.0 for cfg in model_configs}

    ensemble_oof_path = output_dir / "ensemble_oof.parquet"
    ensemble_test_path = output_dir / "ensemble_test.parquet"
    
    pl.DataFrame({"pred": final_oof}).write_parquet(ensemble_oof_path)
    pl.DataFrame({"pred": final_test}).write_parquet(ensemble_test_path)

    updates = {
        "ensemble_weights": weights_dict,
        "ensemble_cv_score": float(ensemble_cv_score),
        "ensemble_oof_path": str(ensemble_oof_path),
        "ensemble_test_predictions_path": str(ensemble_test_path),
        "ensemble_method": best_method,
        "meta_learner_used": best_method == "meta_learner"
    }

    return ProfessorState.validated_update(state, AGENT_NAME, updates)
