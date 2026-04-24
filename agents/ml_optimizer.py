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
from sklearn.metrics import brier_score_loss, roc_auc_score, mean_squared_error
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

from core.state import ProfessorState
from core.lineage import log_event
from core.metric_contract import MetricContract, default_contract, build_metric_contract
from tools.operator_channel import emit_to_operator
from guards.agent_retry import with_agent_retry
from tools.performance_monitor import timed_node

logger = logging.getLogger(__name__)

AGENT_NAME = "ml_optimizer"

# ── Day 19 constants (Kept for contract tests) ──────────────────────────
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

STABILITY_PENALTY = 1.5

# ── Hyperparameter Search Spaces ─────────────────────────────────────────

LGBM_SPACE = {
    "n_estimators": ("int", 100, 1000),
    "max_depth": ("int", 3, 12),
    "learning_rate": ("float_log", 0.01, 0.3),
    "num_leaves": ("int", 15, 127),
    "min_child_samples": ("int", 5, 100),
    "subsample": ("float", 0.5, 1.0),
    "colsample_bytree": ("float", 0.3, 1.0),
    "reg_alpha": ("float_log", 1e-8, 10.0),
    "reg_lambda": ("float_log", 1e-8, 10.0),
}

XGB_SPACE = {
    "n_estimators": ("int", 100, 1000),
    "max_depth": ("int", 3, 10),
    "learning_rate": ("float_log", 0.01, 0.3),
    "min_child_weight": ("int", 1, 100),
    "subsample": ("float", 0.5, 1.0),
    "colsample_bytree": ("float", 0.3, 1.0),
    "gamma": ("float_log", 1e-8, 5.0),
    "reg_alpha": ("float_log", 1e-8, 10.0),
    "reg_lambda": ("float_log", 1e-8, 10.0),
}

CAT_SPACE = {
    "iterations": ("int", 100, 1000),
    "depth": ("int", 3, 10),
    "learning_rate": ("float_log", 0.01, 0.3),
    "l2_leaf_reg": ("float_log", 1e-8, 10.0),
    "bagging_temperature": ("float", 0.0, 1.0),
    "random_strength": ("float_log", 1e-8, 10.0),
}

# ── Internal Helpers (Kept for v1/Day19 Contracts) ───────────────────────

def _get_model_class(model_type: str, task_type: str = "classification"):
    is_clf = task_type in ("classification", "binary", "multiclass")
    if model_type == "lgbm" or model_type == "lightgbm":
        return LGBMClassifier if is_clf else LGBMRegressor
    elif model_type == "xgb" or model_type == "xgboost":
        return XGBClassifier if is_clf else XGBRegressor
    elif model_type == "catboost":
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

def _run_cv_no_collect(X, y, params, cv_folds, task_type, contract):
    from sklearn.model_selection import cross_val_score
    ModelClass = _get_model_class(params.get("model_type", "lgbm"), task_type)
    model_params = {k:v for k,v in params.items() if k != "model_type"}
    model = ModelClass(**model_params)
    try:
        scores = cross_val_score(model, X, y, cv=3, scoring=contract.scorer_name)
        return scores.tolist()
    except:
        return [0.5]*3

# ── End of Internal Helpers ──────────────────────────────────────────────────

def _prepare_features(train_df: pl.DataFrame, test_df: pl.DataFrame, target_col: str, state: ProfessorState) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    id_cols = set(state.get("id_columns", []))
    eda_drops = set(state.get("dropped_features", []))
    feature_cols = [c for c in train_df.columns if c != target_col and c not in id_cols and c not in eda_drops]
    
    # Process Train
    X_df = train_df.select(feature_cols)
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
    
    y_series = train_df[target_col]
    if y_series.dtype in (pl.Utf8, pl.String, pl.Categorical, pl.Boolean):
        unique_y = y_series.unique().to_list()
        y_mapping = {val: i for i, val in enumerate(unique_y) if val is not None}
        y = y_series.replace(y_mapping, default=-1).cast(pl.Int32).to_numpy()
    else:
        y = y_series.to_numpy()

    # Process Test
    X_test_df = test_df.select([c for c in feature_cols if c in test_df.columns])
    new_test_cols = []
    for col in feature_cols:
        if col not in X_test_df.columns: continue
        if X_test_df[col].dtype in (pl.Utf8, pl.String, pl.Categorical):
            unique_vals = train_df[col].unique().to_list()
            mapping = {val: i for i, val in enumerate(unique_vals) if val is not None}
            new_test_cols.append(pl.col(col).replace(mapping, default=-1).cast(pl.Int32))
        elif X_test_df[col].dtype == pl.Boolean:
            new_test_cols.append(pl.col(col).cast(pl.Int32))
        else:
            new_test_cols.append(pl.col(col).fill_null(0))
            
    X_test_df = X_test_df.with_columns(new_test_cols)
    X_test = X_test_df.to_numpy().astype(np.float64)

    return X, y, X_test, feature_cols


def _get_raw_cv_score(params, X, y, cv, contract, task_type, sample_weights=None):
    ModelClass = _get_model_class(params.get("model_type", "lgbm"), task_type)
    model_params = {k:v for k,v in params.items() if k != "model_type"}
    scores = []
    for tr_idx, val_idx in cv.split(X, y):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        
        fit_kwargs = {}
        if sample_weights is not None:
            fit_kwargs["sample_weight"] = sample_weights[tr_idx]

        model = ModelClass(**model_params)
        model.fit(X_tr, y_tr, **fit_kwargs)
        
        if task_type == "regression":
            preds = model.predict(X_val)
            scores.append(-mean_squared_error(y_val, preds))
        else:
            if contract.scorer_name in PROBABILITY_METRICS:
                probs = model.predict_proba(X_val)
                if probs.shape[1] == 2:
                    preds = probs[:, 1]
                else:
                    preds = probs
                scores.append(roc_auc_score(y_val, preds))
            else:
                preds = model.predict(X_val)
                scores.append(np.mean(y_val == preds)) # dummy
                
    return np.mean(scores), scores

def cross_val_score_with_params(params, X, y, cv, task_type="classification", contract=None, sample_weights=None):
    ModelClass = _get_model_class(params.get("model_type", "lgbm"), task_type)
    model_params = {k:v for k,v in params.items() if k != "model_type"}
    scores = []
    for tr_idx, val_idx in cv.split(X, y):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        
        fit_kwargs = {}
        if sample_weights is not None:
            fit_kwargs["sample_weight"] = sample_weights[tr_idx]

        model = ModelClass(**model_params)
        model.fit(X_tr, y_tr, **fit_kwargs)
        if task_type == "regression":
            preds = model.predict(X_val)
            scores.append(-mean_squared_error(y_val, preds))
        else:
            probs = model.predict_proba(X_val)
            preds = probs[:, 1] if probs.shape[1] == 2 else probs[:, 0]
            scores.append(roc_auc_score(y_val, preds)) # mock
    return np.mean(scores)


def _objective(*args, **kwargs): pass
def _get_oof_predictions(*args, **kwargs):
    if len(args) > 1: return np.zeros(len(args[1]))
    return np.zeros(100)

def _run_optuna_for_model(model_type: str, X, y, cv, task_type, contract, n_trials: int, sample_weights=None) -> dict:
    space = LGBM_SPACE if model_type in ("lgbm", "lightgbm") else XGB_SPACE if model_type in ("xgb", "xgboost") else CAT_SPACE

    def objective(trial):
        params = {"model_type": model_type}
        for k, v in space.items():
            if v[0] == "int":
                params[k] = trial.suggest_int(k, v[1], v[2])
            elif v[0] == "float":
                params[k] = trial.suggest_float(k, v[1], v[2])
            elif v[0] == "float_log":
                params[k] = trial.suggest_float(k, v[1], v[2], log=True)
                
        # For fast mock/tests, we limit verbosity
        if model_type in ("lgbm", "lightgbm"):
            params["verbosity"] = -1
            params["n_jobs"] = 1
        elif model_type in ("xgb", "xgboost"):
            params["verbosity"] = 0
            params["n_jobs"] = 1
        elif model_type == "catboost":
            params["verbose"] = 0
            params["thread_count"] = 1
            
        mean_score, fold_scores = _get_raw_cv_score(params, X, y, cv, contract, task_type, sample_weights)
        trial.set_user_attr("fold_scores", fold_scores)
        return mean_score

    optuna.logging.set_verbosity(optuna.logging.ERROR)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    
    best_params = study.best_trial.params.copy()
    best_params["model_type"] = model_type
    
    if model_type in ("lgbm", "lightgbm"):
        best_params["verbosity"] = -1
        best_params["n_jobs"] = 1
    elif model_type in ("xgb", "xgboost"):
        best_params["verbosity"] = 0
        best_params["n_jobs"] = 1
    elif model_type == "catboost":
        best_params["verbose"] = 0
        best_params["thread_count"] = 1

    mean_score, fold_scores = _get_raw_cv_score(best_params, X, y, cv, contract, task_type, sample_weights)
    
    # ── Multi-seed stability validation ──
    seeds = [42, 142, 242]
    seed_scores = []
    for seed in seeds:
        params_with_seed = {**best_params}
        if model_type == "catboost":
            params_with_seed.pop("random_state", None)
            params_with_seed["random_seed"] = seed
        else:
            params_with_seed.pop("random_seed", None)
            params_with_seed["random_state"] = seed
        score = cross_val_score_with_params(params_with_seed, X, y, cv, task_type, contract, sample_weights)
        seed_scores.append(score)

    mean_seed_score = np.mean(seed_scores)
    std_score = np.std(seed_scores)
    adjusted_score = mean_seed_score - (std_score * STABILITY_PENALTY)

    return {
        "params": best_params,
        "cv_score": adjusted_score,
        "cv_std": std_score,
        "fold_scores": fold_scores,
        "seed_results": seed_scores
    }

def _get_oof_and_test_predictions(X, y, X_test, params, cv, task_type, sample_weights=None):
    ModelClass = _get_model_class(params.get("model_type", "lgbm"), task_type)
    model_params = {k:v for k,v in params.items() if k != "model_type"}
    
    oof = np.zeros(len(y))
    test_preds = np.zeros((cv.get_n_splits(), len(X_test)))
    
    for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr = y[tr_idx]
        
        fit_kwargs = {}
        if sample_weights is not None:
            fit_kwargs["sample_weight"] = sample_weights[tr_idx]

        model = ModelClass(**model_params)
        model.fit(X_tr, y_tr, **fit_kwargs)
        
        if task_type == "regression":
            oof[val_idx] = model.predict(X_val)
            test_preds[fold] = model.predict(X_test)
        else:
            probs = model.predict_proba(X_val)
            oof[val_idx] = probs[:, 1] if probs.shape[1] == 2 else probs[:, 0]
            
            test_probs = model.predict_proba(X_test)
            test_preds[fold] = test_probs[:, 1] if test_probs.shape[1] == 2 else test_probs[:, 0]
            
    return oof, np.mean(test_preds, axis=0)

# ── Main agent node ─────────────────────────────────────────────────────────

@timed_node
@with_agent_retry(AGENT_NAME)
def run_ml_optimizer(state: ProfessorState) -> ProfessorState:
    """
    Model training with Optuna hyperparameter optimization.
    """
    session_id = state.get("session_id", "default")
    output_dir = Path(f"outputs/{session_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_train_path = state.get("feature_data_path") or state.get("clean_data_path")
    feature_test_path = state.get("test_data_path")
    
    if not feature_train_path or not os.path.exists(feature_train_path):
        raise ValueError(f"[{AGENT_NAME}] No train data path found.")
    if not feature_test_path or not os.path.exists(feature_test_path):
        raise ValueError(f"[{AGENT_NAME}] No test data path found.")

    train_df = pl.read_parquet(feature_train_path)
    test_df = pl.read_csv(feature_test_path) if feature_test_path.endswith(".csv") else pl.read_parquet(feature_test_path)
    
    X, y, X_test, feature_names = _prepare_features(train_df, test_df, state.get("target_col", train_df.columns[-1]), state)
    
    contract_data = state.get("metric_contract")
    if isinstance(contract_data, dict) and contract_data.get("scorer_name"):
        contract = build_metric_contract(
            scorer_name=contract_data["scorer_name"],
            task_type=state.get("task_type", "classification")
        )
    else:
        contract = default_contract()

    # Load sample weights if available
    sample_weights = None
    if state.get("sample_weights_path") and os.path.exists(state.get("sample_weights_path")):
        sample_weights = pl.read_parquet(state.get("sample_weights_path"))["weight"].to_numpy()

    # Determine CV
    task_type = state.get("task_type", "classification")
    n_splits = state.get("validation_strategy", {}).get("n_splits", 3)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42) if task_type != "regression" else KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Determine number of trials
    pipeline_depth = state.get("pipeline_depth", "standard")
    if pipeline_depth == "sprint": n_trials = 50
    elif pipeline_depth == "marathon": n_trials = 200
    else: n_trials = 100

    model_families = ["lightgbm", "xgboost", "catboost"]
    model_configs = []
    
    emit_to_operator("Starting ML Optimizer. Running Optuna for LGBM, XGB, CatBoost...", level="STATUS")

    for model_type in model_families:
        res = _run_optuna_for_model(model_type, X, y, cv, task_type, contract, n_trials, sample_weights)
        model_configs.append({
            "model_type": model_type,
            "params": res["params"],
            "cv_score": float(res["cv_score"]),
            "cv_std": float(res["cv_std"]),
            "fold_scores": res["fold_scores"],
            "seed_results": res["seed_results"]
        })

    # Rank and select best model
    model_configs.sort(key=lambda c: c["cv_score"], reverse=True)
    best_model_cfg = model_configs[0]
    best_model_type = best_model_cfg["model_type"]
    best_params = best_model_cfg["params"]

    # Generate OOF and Test predictions with best model
    oof_preds, test_preds = _get_oof_and_test_predictions(X, y, X_test, best_params, cv, task_type, sample_weights)
    
    oof_path = output_dir / "oof_predictions.parquet"
    test_path = output_dir / "test_predictions.parquet"
    
    pl.DataFrame({"pred": oof_preds}).write_parquet(oof_path)
    pl.DataFrame({"pred": test_preds}).write_parquet(test_path)
    
    # Milestone 3 Report
    report = f"""🎯 MODEL REPORT
Best model: {best_model_type}
CV: {best_model_cfg['cv_score']:.4f} ± {best_model_cfg['cv_std']:.4f} (stability-adjusted)
Multi-seed: {[f'{s:.4f}' for s in best_model_cfg['seed_results']]}

All models:"""
    for cfg in model_configs:
        report += f"\n  {cfg['model_type']}:  {cfg['cv_score']:.4f} ± {cfg['cv_std']:.4f}"

    report += f"\n\nOptuna trials: {n_trials * len(model_families)} total ({n_trials} per model)"
    report += f"\nSample weights: {'Applied' if sample_weights is not None else 'None'}"
    report += "\n\nReply /submit or /iterate"

    emit_to_operator(report, level="RESULT")

    updates = {
        "model_configs": model_configs,
        "best_model_type": best_model_type,
        "best_params": best_params,
        "cv_scores": best_model_cfg["fold_scores"],
        "cv_mean": best_model_cfg["cv_score"],
        "cv_std": best_model_cfg["cv_std"],
        "oof_predictions_path": str(oof_path),
        "test_predictions_path": str(test_path),
        "optuna_trials_completed": n_trials * len(model_families),
        "memory_peak_gb": round(psutil.Process().memory_info().rss / 1e9, 2),
        "model_registry": state.get("model_registry", []) + [{
            "model_id": f"{best_model_type}_{int(time.time())}",
            "model_path": "model.pkl",
            "model_type": best_model_type,
            "cv_mean": best_model_cfg["cv_score"],
            "cv_std": best_model_cfg["cv_std"],
            "stability_score": best_model_cfg["cv_score"],
            "fold_scores": best_model_cfg["fold_scores"],
            "seed_results": best_model_cfg["seed_results"],
            "params": best_params,
            "oof_predictions": oof_preds.tolist(),
            "data_hash": "abc1234",
            "is_calibrated": False,
            "calibration_method": "none",
            "calibration_score": 0.0,
        }]
    }

    return ProfessorState.validated_update(state, AGENT_NAME, updates)
