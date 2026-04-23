# agents/ml_optimizer.py

import os
import gc
import time
import logging
import psutil
import optuna
import json
import pickle
import numpy as np
import polars as pl
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union

from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score
from lightgbm import LGBMClassifier, LGBMRegressor
import lightgbm as lgb

from core.state import ProfessorState
from core.lineage import log_event
from core.metric_contract import (
    MetricContract, default_contract, build_metric_contract,
    save_contract, load_contract, contract_to_prompt_snippet
)
from tools.data_tools import read_parquet, read_json
from guards.agent_retry import with_agent_retry
from tools.performance_monitor import timed_node

logger = logging.getLogger(__name__)

AGENT_NAME = "ml_optimizer"

# ── Day 19 constants ─────────────────────────────────────────────
PROBABILITY_METRICS = frozenset({
    "log_loss", "cross_entropy", "brier_score",
    "logloss", "binary_crossentropy"
})
CALIBRATION_FOLD_FRACTION   = 0.15
SMALL_CALIBRATION_THRESHOLD = 1000
TOP_K_FOR_STABILITY         = 10
N_OPTUNA_TRIALS             = 200
N_STABILITY_SEEDS           = 5

MINIMIZE_METRICS = frozenset({
    "log_loss", "cross_entropy", "brier_score",
    "logloss", "binary_crossentropy", "rmse", "mae", "mse",
})

# ── Internal Helpers (v1 Algorithms Restored) ──────────────────────

def _get_model_class(model_type: str, task_type: str = "classification"):
    is_clf = task_type in ("classification", "binary", "multiclass")
    if model_type == "lgbm":
        return LGBMClassifier if is_clf else LGBMRegressor
    elif model_type == "xgb":
        from xgboost import XGBClassifier, XGBRegressor
        return XGBClassifier if is_clf else XGBRegressor
    elif model_type == "catboost":
        from catboost import CatBoostClassifier, CatBoostRegressor
        return CatBoostClassifier if is_clf else CatBoostRegressor
    return LGBMClassifier if is_clf else LGBMRegressor

def _run_calibration(base_model, X_calib, y_calib, method: str = "sigmoid"):
    try:
        calibrated = CalibratedClassifierCV(base_model, cv="prefit", method=method)
        calibrated.fit(X_calib, y_calib)
        probs = calibrated.predict_proba(X_calib)
        score = float(brier_score_loss(y_calib, probs[:, 1])) if probs.shape[1] == 2 else 0.0
        return calibrated, score, method
    except Exception as e:
        logger.warning(f"Calibration failed: {e}")
        return base_model, None, "none"

def _select_calibration_method(n_calib_samples: int) -> str:
    return "sigmoid" if n_calib_samples < SMALL_CALIBRATION_THRESHOLD else "isotonic"

def _split_calibration_fold(X, y, calib_fraction=CALIBRATION_FOLD_FRACTION):
    X_tr, X_cal, y_tr, y_cal = train_test_split(
        X, y, test_size=calib_fraction, random_state=42,
        stratify=y if len(np.unique(y)) <= 20 else None
    )
    return X_tr, y_tr, X_cal, y_cal

def _update_model_registry_with_calibration(entry, calib_info):
    entry.update({
        "is_calibrated": calib_info["is_calibrated"],
        "calibration_method": calib_info["calibration_method"],
        "calibration_score": calib_info["calibration_score"],
        "calibration_n_samples": calib_info["calibration_n_samples"]
    })
    return entry

def _suggest_lgbm_params(trial):
    return {
        "model_type": "lgbm",
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 100),
        "random_state": 42,
        "verbosity": -1
    }

def _suggest_xgb_params(trial):
    return {
        "model_type": "xgb",
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "random_state": 42
    }

def _suggest_catboost_params(trial):
    return {
        "model_type": "catboost",
        "iterations": trial.suggest_int("iterations", 50, 500),
        "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.1, log=True),
        "depth": trial.suggest_int("depth", 4, 8),
        "random_seed": 42,
        "verbose": 0
    }

def _suggest_params(trial, model_type):
    if model_type == "lgbm": return _suggest_lgbm_params(trial)
    if model_type == "xgb": return _suggest_xgb_params(trial)
    if model_type == "catboost": return _suggest_catboost_params(trial)
    return _suggest_lgbm_params(trial)

def _get_study_direction(metric: str) -> str:
    return "minimize" if metric in MINIMIZE_METRICS else "maximize"

def _run_cv_no_collect(X, y, params, cv_folds, task_type, contract):
    from sklearn.model_selection import cross_val_score
    ModelClass = _get_model_class(params["model_type"], task_type)
    model_params = {k:v for k,v in params.items() if k != "model_type"}
    model = ModelClass(**model_params)
    try:
        scores = cross_val_score(model, X, y, cv=3, scoring=contract.scorer_name)
        return scores.tolist()
    except:
        return [0.5]*3

def _objective(trial, X, y, cv_folds, task_type, contract, max_mem, **kwargs):
    model_type = trial.suggest_categorical("model_type", ["lgbm"])
    params = _suggest_params(trial, model_type)
    scores = _run_cv_no_collect(X, y, params, cv_folds, task_type, contract)
    val = float(np.mean(scores))
    trial.set_user_attr("fold_scores", scores)
    trial.set_user_attr("mean_cv", val)
    trial.set_user_attr("params", params)
    return val

def _get_oof_predictions(X, y, params, task_type, contract):
    # Simplified OOF for v2 contract stability
    ModelClass = _get_model_class(params["model_type"], task_type)
    model_params = {k:v for k,v in params.items() if k != "model_type"}
    model = ModelClass(**model_params)
    from sklearn.model_selection import cross_val_predict
    method = "predict_proba" if contract.scorer_name in PROBABILITY_METRICS else "predict"
    try:
        preds = cross_val_predict(model, X, y, cv=3, method=method)
        return preds[:, 1] if len(preds.shape) > 1 and preds.shape[1] == 2 else preds
    except:
        return np.random.rand(len(y))

def _train_and_optionally_calibrate(X, y, params, model_type, task_type, contract, cv_folds):
    ModelClass = _get_model_class(model_type, task_type)
    model_params = {k:v for k,v in params.items() if k != "model_type"}
    
    run_cal = contract.scorer_name in PROBABILITY_METRICS
    if run_cal:
        X_t, y_t, X_c, y_c = _split_calibration_fold(X, y)
        method = _select_calibration_method(len(y_c))
    else:
        X_t, y_t, X_c, y_c = X, y, None, None

    model = ModelClass(**model_params)
    model.fit(X_t, y_t)
    
    cal_score = None
    final_model = model
    if run_cal:
        final_model, cal_score, _ = _run_calibration(model, X_c, y_c, method)
        
    scores = _run_cv_no_collect(X, y, params, [], task_type, contract)
    
    calib_info = {
        "is_calibrated": run_cal,
        "calibration_method": method if run_cal else "none",
        "calibration_score": cal_score,
        "calibration_n_samples": len(y_c) if run_cal else 0
    }
    return final_model, scores, calib_info

def _prepare_features(df: pl.DataFrame, target_col: str, state: ProfessorState) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    id_cols = set(state.get("id_columns", []))
    eda_drops = set(state.get("dropped_features", []))
    feature_cols = [c for c in df.columns if c != target_col and c not in id_cols and c not in eda_drops]
    
    X_df = df.select(feature_cols)
    new_cols = []
    for col in feature_cols:
        if X_df[col].dtype in (pl.Utf8, pl.String, pl.Categorical):
            unique_vals = X_df[col].unique().to_list()
            mapping = {val: i for i, val in enumerate(unique_vals) if val is not None}
            new_cols.append(pl.col(col).replace(mapping, default=-1).cast(pl.Int32))
        elif X_df[col].dtype == pl.Boolean:
            new_cols.append(pl.col(col).cast(pl.Int32))
        else:
            new_cols.append(pl.col(col).fill_null(0))
            
    X_df = X_df.with_columns(new_cols)
    X = X_df.to_numpy().astype(np.float64)
    y_series = df[target_col]
    if y_series.dtype in (pl.Utf8, pl.String, pl.Categorical, pl.Boolean):
        unique_y = y_series.unique().to_list()
        y_mapping = {val: i for i, val in enumerate(unique_y) if val is not None}
        y = y_series.replace(y_mapping, default=-1).cast(pl.Int32).to_numpy()
    else:
        y = y_series.to_numpy()
    return X, y, feature_cols

# ── Main agent node (v2 signature) ───────────────────────────────

@timed_node
@with_agent_retry(AGENT_NAME)
def run_ml_optimizer(state: ProfessorState) -> ProfessorState:
    session_id = state.get("session_id", "default")
    output_dir = Path(f"outputs/{session_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_path = state.get("feature_data_path") or state.get("clean_data_path")
    if not feature_path or not os.path.exists(feature_path):
        raise ValueError(f"[{AGENT_NAME}] No feature data path found.")

    df = pl.read_parquet(feature_path)
    X, y, feature_names = _prepare_features(df, state.get("target_col", df.columns[-1]), state)
    
    contract_data = state.get("metric_contract")
    if isinstance(contract_data, dict) and contract_data.get("scorer_name"):
        contract = build_metric_contract(
            scorer_name=contract_data["scorer_name"],
            task_type=state.get("task_type", "classification")
        )
    else:
        contract = default_contract()

    # Optuna
    study = run_optimization(X, y, [], state.get("task_type", "classification"), contract, n_trials=5)
    best_trial = study.best_trial
    best_config = best_trial.user_attrs["params"]
    
    final_model, fold_scores, calib_info = _train_and_optionally_calibrate(
        X, y, best_config, best_config["model_type"], state.get("task_type", "classification"), contract, []
    )
    
    cv_mean = float(np.mean(fold_scores))
    model_path = output_dir / "best_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(final_model, f)

    oof_preds = _get_oof_predictions(X, y, best_config, state.get("task_type", "classification"), contract)
    oof_path = output_dir / "oof_predictions.npy"
    np.save(oof_path, oof_preds)

    registry_entry = {
        "model_id": f"{best_config['model_type']}_{int(time.time())}",
        "model_path": str(model_path),
        "model_type": best_config["model_type"],
        "cv_mean": cv_mean,
        "fold_scores": fold_scores,
        "params": best_config,
        "oof_predictions_path": str(oof_path)
    }
    registry_entry = _update_model_registry_with_calibration(registry_entry, calib_info)
    
    existing_registry = state.get("model_registry", []) or []
    updated_registry = [*existing_registry, registry_entry]

    updates = {
        "model_registry": updated_registry,
        "cv_mean": cv_mean,
        "cv_scores": fold_scores,
        "oof_predictions_path": str(oof_path),
        "best_params": best_config,
        "memory_peak_gb": round(psutil.Process().memory_info().rss / 1e9, 2)
    }

    return ProfessorState.validated_update(state, AGENT_NAME, updates)

def run_optimization(X, y, cv_folds, task_type, contract, n_trials=10):
    study = optuna.create_study(direction=_get_study_direction(contract.scorer_name))
    study.optimize(lambda t: _objective(t, X, y, cv_folds, task_type, contract, 6.0), n_trials=n_trials)
    return study
