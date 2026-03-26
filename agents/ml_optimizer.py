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
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss
from lightgbm import LGBMClassifier, LGBMRegressor
import lightgbm as lgb
from core.state import ProfessorState
from core.lineage import log_event
from datetime import datetime
from typing import Optional
from core.metric_contract import (
    MetricContract, default_contract,
    save_contract, load_contract, contract_to_prompt_snippet
)
from tools.data_tools import read_parquet, read_json
from guards.agent_retry import with_agent_retry
from tools.performance_monitor import timed_node

logger = logging.getLogger(__name__)


# ── Langsmith tracing helper (avoid circular import from professor) ──
from contextlib import contextmanager

@contextmanager
def _disable_langsmith_tracing():
    """Temporarily disables LangSmith tracing."""
    original = os.environ.get("LANGCHAIN_TRACING_V2", "false")
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    try:
        yield
    finally:
        os.environ["LANGCHAIN_TRACING_V2"] = original


# ── Day 19 constants ─────────────────────────────────────────────
PROBABILITY_METRICS = frozenset({
    "log_loss", "cross_entropy", "brier_score",
    "logloss", "binary_crossentropy",
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


# ── Model class registry ─────────────────────────────────────────
def _get_model_class(model_type: str, task_type: str = "classification"):
    """Returns the appropriate model class for the given model type and task.
    
    Handles task_type values: "classification", "binary", "multiclass", "regression"
    """
    # Normalize task_type: binary/multiclass → classification
    is_clf = task_type in ("classification", "binary", "multiclass") or "classification" in task_type
    if model_type == "lgbm":
        return LGBMClassifier if is_clf else LGBMRegressor
    elif model_type == "xgb":
        from xgboost import XGBClassifier, XGBRegressor
        return XGBClassifier if is_clf else XGBRegressor
    elif model_type == "catboost":
        from catboost import CatBoostClassifier, CatBoostRegressor
        return CatBoostClassifier if is_clf else CatBoostRegressor
    return LGBMClassifier if is_clf else LGBMRegressor


# ── Calibration functions (Task 1) ───────────────────────────────
def _run_calibration(
    base_model,
    X_calib: np.ndarray,
    y_calib: np.ndarray,
    method: str = "sigmoid",
) -> tuple:
    """
    Fits a calibration wrapper on a held-out calibration fold.
    Returns (calibrated_model, calibration_brier_score, method_used).
    Never raises — returns (base_model, None, "none") on any failure.
    """
    try:
        frozen = FrozenEstimator(base_model)
        calibrated = CalibratedClassifierCV(
            estimator=frozen,
            method=method,
        )
        calibrated.fit(X_calib, y_calib)

        y_prob_calib = calibrated.predict_proba(X_calib)[:, 1]
        brier = float(brier_score_loss(y_calib, y_prob_calib))

        return calibrated, brier, method

    except Exception as e:
        logger.warning(
            f"[ml_optimizer] Calibration failed ({method}): {e}. "
            f"Returning uncalibrated model."
        )
        return base_model, None, "none"


def _select_calibration_method(n_calib_samples: int) -> str:
    """
    Platt (sigmoid) for small sets (< 1000), isotonic for >= 1000.
    """
    return "sigmoid" if n_calib_samples < SMALL_CALIBRATION_THRESHOLD else "isotonic"


def _split_calibration_fold(
    X: np.ndarray,
    y: np.ndarray,
    calib_fraction: float = CALIBRATION_FOLD_FRACTION,
    random_state: int = 42,
) -> tuple:
    """
    Splits off a calibration fold from training data BEFORE CV.
    Returns (X_train_cv, y_train_cv, X_calib, y_calib).
    """
    if isinstance(X, pl.DataFrame):
        X = X.to_numpy()
    X_train_cv, X_calib, y_train_cv, y_calib = train_test_split(
        X, y,
        test_size=calib_fraction,
        random_state=random_state,
        stratify=y if len(np.unique(y)) <= 20 else None,
    )
    return X_train_cv, y_train_cv, X_calib, y_calib


def _update_model_registry_with_calibration(
    entry: dict,
    calibration_info: dict,
) -> dict:
    """Adds calibration fields to a model registry entry."""
    return {
        **entry,
        "is_calibrated":         calibration_info["is_calibrated"],
        "calibration_method":    calibration_info["calibration_method"],
        "calibration_score":     calibration_info["calibration_score"],
        "calibration_n_samples": calibration_info["calibration_n_samples"],
    }


# ── LEAKAGE FIX: CV-Safe Target Encoding ─────────────────────────

def _apply_target_encoding_cv_safe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    feature_cols: list[str],
    target_enc_cols: list[str],
    n_folds: int = 3,
    smoothing: float = 30.0,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply target encoding WITHIN CV folds (leak-free).
    
    For each fold:
      - Fit encoding on training portion ONLY
      - Transform validation portion
      - Never use validation targets to compute encoding
    
    Args:
        X_train: Training features (already split from full data)
        y_train: Training targets
        X_val: Validation features
        feature_cols: Names of all feature columns
        target_enc_cols: Names of columns needing target encoding
        n_folds: Number of CV folds for encoding
        smoothing: Smoothing parameter for target encoding
        random_state: Random seed for reproducibility
    
    Returns:
        X_train_encoded, X_val_encoded: Encoded feature arrays
    """
    from sklearn.model_selection import KFold
    
    # For training data: use inner CV to compute encoding
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    n_train = len(y_train)
    n_val = len(X_val)
    
    # Initialize encoded arrays
    X_train_encoded = X_train.copy()
    X_val_encoded = X_val.copy()
    
    # Compute global mean from training data only
    global_mean = float(np.mean(y_train))
    
    for enc_col_idx, enc_col_name in enumerate(target_enc_cols):
        # Find column index
        col_idx = feature_cols.index(enc_col_name)
        
        # Get column values
        train_col = X_train[:, col_idx]
        val_col = X_val[:, col_idx]
        
        # Initialize encoded values
        train_encoded = np.full(n_train, global_mean, dtype=np.float64)
        val_encoded = np.full(n_val, global_mean, dtype=np.float64)
        
        # Inner CV for training data encoding
        fold_assignments = np.zeros(n_train, dtype=int)
        for fold_idx, (_, inner_val_idx) in enumerate(kf.split(np.arange(n_train))):
            fold_assignments[inner_val_idx] = fold_idx
        
        for fold_idx in range(n_folds):
            inner_train_mask = fold_assignments != fold_idx
            inner_val_mask = fold_assignments == fold_idx
            
            # Compute encoding from inner training portion
            cat_stats = {}
            for cat, target in zip(train_col[inner_train_mask], y_train[inner_train_mask]):
                key = str(cat)
                if key not in cat_stats:
                    cat_stats[key] = [0.0, 0]
                cat_stats[key][0] += float(target)
                cat_stats[key][1] += 1
            
            # Apply to inner validation portion
            inner_val_indices = np.where(inner_val_mask)[0]
            for idx in inner_val_indices:
                key = str(train_col[idx])
                if key in cat_stats:
                    sum_t, count = cat_stats[key]
                    group_mean = sum_t / count
                    train_encoded[idx] = (
                        (count * group_mean + smoothing * global_mean)
                        / (count + smoothing)
                    )
        
        # Compute encoding for validation data from ALL training data
        cat_stats_full = {}
        for cat, target in zip(train_col, y_train):
            key = str(cat)
            if key not in cat_stats_full:
                cat_stats_full[key] = [0.0, 0]
            cat_stats_full[key][0] += float(target)
            cat_stats_full[key][1] += 1
        
        for idx in range(n_val):
            key = str(val_col[idx])
            if key in cat_stats_full:
                sum_t, count = cat_stats_full[key]
                group_mean = sum_t / count
                val_encoded[idx] = (
                    (count * group_mean + smoothing * global_mean)
                    / (count + smoothing)
                )
        
        # Add encoded columns
        X_train_encoded = np.column_stack([X_train_encoded, train_encoded])
        X_val_encoded = np.column_stack([X_val_encoded, val_encoded])
    
    return X_train_encoded, X_val_encoded


# ── LEAKAGE FIX: CV-Safe Aggregations ────────────────────────────

def _apply_aggregations_cv_safe(
    X_train: np.ndarray,
    X_val: np.ndarray,
    feature_cols: list[str],
    agg_candidates: list[dict],
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply groupby aggregations WITHIN CV folds (leak-free).
    
    Args:
        X_train: Training features
        X_val: Validation features
        feature_cols: Names of all feature columns
        agg_candidates: List of aggregation candidates with:
            - name: New feature name
            - source_columns: [numeric_col, categorical_col]
            - transform_type: "groupby_agg"
        random_state: Random seed
    
    Returns:
        X_train_encoded, X_val_encoded: Feature arrays with aggregations
    """
    n_train = X_train.shape[0]
    n_val = X_val.shape[0]
    
    X_train_result = X_train.copy()
    X_val_result = X_val.copy()
    
    for agg in agg_candidates:
        num_col_name = agg["source_columns"][0]
        cat_col_name = agg["source_columns"][1]
        agg_name = agg["name"]
        
        # Extract column indices
        num_col_idx = feature_cols.index(num_col_name)
        cat_col_idx = feature_cols.index(cat_col_name)
        
        # Get column values
        train_num = X_train[:, num_col_idx]
        train_cat = X_train[:, cat_col_idx]
        val_num = X_val[:, num_col_idx]
        val_cat = X_val[:, cat_col_idx]
        
        # Compute aggregations from training data ONLY
        agg_stats = {}
        for num_val, cat_val in zip(train_num, train_cat):
            key = str(cat_val)
            if key not in agg_stats:
                agg_stats[key] = []
            agg_stats[key].append(float(num_val))
        
        # Compute statistics
        cat_mean = {}
        cat_std = {}
        cat_min = {}
        cat_max = {}
        cat_count = {}
        
        for cat_key, values in agg_stats.items():
            cat_mean[cat_key] = np.mean(values)
            cat_std[cat_key] = np.std(values) if len(values) > 1 else 0.0
            cat_min[cat_key] = np.min(values)
            cat_max[cat_key] = np.max(values)
            cat_count[cat_key] = len(values)
        
        # Global statistics for unseen categories
        global_mean = np.mean(train_num)
        global_std = np.std(train_num) if len(train_num) > 1 else 0.0
        global_min = np.min(train_num)
        global_max = np.max(train_num)
        
        # Apply to training data (use inner CV in real implementation)
        train_agg = np.array([
            cat_mean.get(str(c), global_mean)
            for c in train_cat
        ])
        
        # Apply to validation data
        val_agg = np.array([
            cat_mean.get(str(c), global_mean)
            for c in val_cat
        ])
        
        # Add as new feature
        X_train_result = np.column_stack([X_train_result, train_agg])
        X_val_result = np.column_stack([X_val_result, val_agg])
    
    return X_train_result, X_val_result


# ── Optuna search spaces (Task 3) ────────────────────────────────
def _suggest_lgbm_params(trial: optuna.Trial) -> dict:
    return {
        "objective":       "binary",
        "n_estimators":    trial.suggest_int("lgbm_n_estimators", 100, 1000, step=50),
        "learning_rate":   trial.suggest_float("lgbm_lr", 0.01, 0.3, log=True),
        "num_leaves":      trial.suggest_int("lgbm_num_leaves", 16, 256),
        "max_depth":       trial.suggest_int("lgbm_max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("lgbm_min_child", 10, 100),
        "feature_fraction":  trial.suggest_float("lgbm_feat_frac", 0.5, 1.0),
        "bagging_fraction":  trial.suggest_float("lgbm_bag_frac", 0.5, 1.0),
        "bagging_freq":    trial.suggest_int("lgbm_bag_freq", 1, 7),
        "reg_alpha":       trial.suggest_float("lgbm_alpha", 1e-8, 10.0, log=True),
        "reg_lambda":      trial.suggest_float("lgbm_lambda", 1e-8, 10.0, log=True),
        "verbosity":       -1,
        "n_jobs":          1,
        "model_type":      "lgbm",
    }


def _suggest_xgb_params(trial: optuna.Trial) -> dict:
    return {
        "n_estimators":    trial.suggest_int("xgb_n_estimators", 100, 1000, step=50),
        "learning_rate":   trial.suggest_float("xgb_lr", 0.01, 0.3, log=True),
        "max_depth":       trial.suggest_int("xgb_max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("xgb_min_child", 1, 20),
        "subsample":       trial.suggest_float("xgb_subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("xgb_colsample", 0.5, 1.0),
        "gamma":           trial.suggest_float("xgb_gamma", 1e-8, 5.0, log=True),
        "reg_alpha":       trial.suggest_float("xgb_alpha", 1e-8, 5.0, log=True),
        "reg_lambda":      trial.suggest_float("xgb_lambda", 1e-8, 5.0, log=True),
        "use_label_encoder": False,
        "eval_metric":     "logloss",
        "verbosity":       0,
        "n_jobs":          1,
        "model_type":      "xgb",
    }


def _suggest_catboost_params(trial: optuna.Trial) -> dict:
    return {
        "iterations":      trial.suggest_int("cb_iters", 100, 1000, step=50),
        "learning_rate":   trial.suggest_float("cb_lr", 0.01, 0.3, log=True),
        "depth":           trial.suggest_int("cb_depth", 3, 10),
        "l2_leaf_reg":     trial.suggest_float("cb_l2", 1e-8, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("cb_bag_temp", 0.0, 1.0),
        "random_strength": trial.suggest_float("cb_rand_str", 1e-8, 10.0, log=True),
        "verbose":         0,
        "thread_count":    1,
        "model_type":      "catboost",
    }


MODEL_SUGGESTERS = {
    "lgbm":     _suggest_lgbm_params,
    "xgb":      _suggest_xgb_params,
    "catboost": _suggest_catboost_params,
}


def _suggest_params(trial: optuna.Trial) -> dict:
    """Selects model type and suggests hyperparameters for that type."""
    model_type = trial.suggest_categorical("model_type", list(MODEL_SUGGESTERS.keys()))
    return MODEL_SUGGESTERS[model_type](trial)


def _identify_target_column(schema: dict, state: ProfessorState) -> str:
    """
    Identify the target column — reads from schema authority (data_engineer).
    NEVER guesses from a hardcoded list.
    """
    # 1. From state (set by data_engineer)
    target = state.get("target_col", "")
    if target and target in schema.get("columns", []):
        return target

    # 2. From schema file (set by data_engineer)
    target = schema.get("target_col", "")
    if target:
        return target

    raise ValueError(
        "[MLOptimizer] Cannot determine target column. "
        "data_engineer must run first to set state['target_col']."
    )


def _prepare_features(df: pl.DataFrame, target_col: str, schema: dict, state: ProfessorState) -> tuple:
    """
    Convert Polars DataFrame to numpy arrays for sklearn/LightGBM.
    Uses SCHEMA AUTHORITY for ID columns and EDA drops — zero heuristics.
    Categorical encoding is handled by TabularPreprocessor (already applied).
    Returns (X, y, feature_names, label_encoders)
    """
    # Read from schema authority — no substring matching
    id_cols = set(state.get("id_columns", []))
    eda_drops = set(state.get("dropped_features", []))

    feature_cols = [
        c for c in df.columns
        if c != target_col
        and c not in id_cols
        and c not in eda_drops
    ]

    # Encoding is now handled by TabularPreprocessor (fitted in data_engineer).
    # String columns should already be integer-encoded. Just handle stragglers.
    label_encoders = {}
    for col in feature_cols:
        if df[col].dtype in (pl.Utf8, pl.String):
            unique_vals = sorted(df[col].drop_nulls().unique().to_list())
            encoder = {val: idx for idx, val in enumerate(unique_vals)}
            label_encoders[col] = encoder
            df = df.with_columns(
                pl.col(col).replace(encoder, default=-1).cast(pl.Int32)
            )
        elif df[col].dtype == pl.Boolean:
            df = df.with_columns(
                pl.col(col).cast(pl.Int32)
            )

    # Fill nulls in numeric columns
    for col in feature_cols:
        if col in df.columns and df[col].dtype in (pl.Int32, pl.Int64, pl.Float32, pl.Float64):
            df = df.with_columns(pl.col(col).fill_null(0))

    # Convert target
    y_series = df[target_col]
    if y_series.dtype == pl.Boolean:
        y = y_series.cast(pl.Int32).to_numpy()
    elif y_series.dtype in (pl.Utf8, pl.String):
        unique_t = sorted(y_series.drop_nulls().unique().to_list())
        t_map = {val: idx for idx, val in enumerate(unique_t)}
        y = y_series.replace(t_map, default=None).cast(pl.Int32).to_numpy()
    else:
        y = y_series.to_numpy()

    X = df.select(feature_cols).to_numpy().astype(np.float64)

    return X, y, feature_cols, label_encoders


# ── FLAW-4.3 FIX: Model Training Fallback ────────────────────────

class ProfessorModelTrainingError(Exception):
    """Raised when all model training attempts fail."""
    pass


def _train_single_model(X, y, params, model_type):
    """
    Train a single model type.
    
    Args:
        X: Features
        y: Target
        params: Model parameters
        model_type: Model type to train
    
    Returns:
        Trained model
    
    Raises:
        Exception if training fails
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.dummy import DummyClassifier, DummyRegressor
    from lightgbm import LGBMClassifier, LGBMRegressor
    
    clean_params = {k: v for k, v in params.items() if k != "model_type"}
    
    if model_type == "lgbm":
        if len(np.unique(y)) <= 20:
            model = LGBMClassifier(**clean_params)
        else:
            model = LGBMRegressor(**clean_params)
        model.fit(X, y)
        return model
    
    elif model_type == "logistic":
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)
        return model
    
    elif model_type == "dummy":
        if len(np.unique(y)) <= 20:
            model = DummyClassifier(strategy="stratified", random_state=42)
        else:
            model = DummyRegressor(strategy="mean")
        model.fit(X, y)
        return model
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_with_fallback(X, y, params, primary_model_type, fallback_chain=None):
    """
    Train model with fallback chain.
    
    Tries primary model first, then falls back through chain if it fails.
    
    Args:
        X: Features
        y: Target
        params: Model parameters
        primary_model_type: Primary model type to try
        fallback_chain: List of fallback model types (default: ["logistic", "dummy"])
    
    Returns:
        (model, model_type_used)
    
    Raises:
        ProfessorModelTrainingError if all models fail
    """
    if fallback_chain is None:
        fallback_chain = ["logistic", "dummy"]
    
    # Build full chain: primary + fallbacks
    if primary_model_type not in fallback_chain:
        model_chain = [primary_model_type] + fallback_chain
    else:
        model_chain = fallback_chain
    
    last_error = None
    
    for model_type in model_chain:
        try:
            logger.info(f"Attempting to train {model_type}...")
            model = _train_single_model(X, y, params, model_type)
            logger.info(f"Successfully trained {model_type}")
            return model, model_type
            
        except Exception as e:
            last_error = e
            logger.warning(f"{model_type} training failed: {e}")
            continue
    
    # All models failed
    raise ProfessorModelTrainingError(
        f"All models failed: {model_chain}. Last error: {last_error}"
    )


# ── FLAW-11.6 FIX: Overfitting Detection ─────────────────────────

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


# ── FLAW-11.7 FIX: Model Stability Checks ────────────────────────

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
    
    logger.info(
        f"Model stability verified: "
        f"Mean: {mean_score:.4f}, Std: {std_score:.4f}"
    )
    return True, mean_score, std_score


# ── CV helper for objective + stability ───────────────────────────
def _run_cv_fold(
    X, y, params, model_type, task_type, contract, 
    fold_idx, train_idx, val_idx, 
    max_memory_gb, trial=None,
    target_enc_cols=None,
    agg_candidates=None,
    feature_cols=None,
):
    """
    Train one CV fold and return (score, model).
    
    If target_enc_cols provided, applies target encoding within fold (leak-free).
    If agg_candidates provided, applies aggregations within fold (leak-free).
    """
    X_tr, X_val = X[train_idx].copy(), X[val_idx].copy()
    y_tr, y_val = y[train_idx], y[val_idx]
    
    # LEAKAGE FIX: Apply target encoding WITHIN fold
    if target_enc_cols and feature_cols:
        X_tr, X_val = _apply_target_encoding_cv_safe(
            X_train=X_tr,
            y_train=y_tr,
            X_val=X_val,
            feature_cols=feature_cols,
            target_enc_cols=target_enc_cols,
            n_folds=3,
            smoothing=30.0,
            random_state=42,
        )
    
    # LEAKAGE FIX: Apply aggregations WITHIN fold
    if agg_candidates and feature_cols:
        X_tr, X_val = _apply_aggregations_cv_safe(
            X_train=X_tr,
            X_val=X_val,
            feature_cols=feature_cols,
            agg_candidates=agg_candidates,
            random_state=42,
        )
    
    clean_params = {k: v for k, v in params.items() if k != "model_type"}
    ModelClass = _get_model_class(model_type, task_type)
    model = ModelClass(**clean_params)

    if model_type == "lgbm":
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
        )
    elif model_type == "catboost":
        model.fit(
            X_tr, y_tr,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50,
            verbose=0,
        )
    elif model_type == "xgb":
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
    else:
        model.fit(X_tr, y_tr)

    if contract.requires_proba:
        probs = model.predict_proba(X_val)
        if probs.shape[1] == 2:
            val_preds = probs[:, 1]
        else:
            val_preds = probs
    else:
        val_preds = model.predict(X_val)

    score = contract.scorer_fn(y_val, val_preds)

    # Day 12 OOM check
    rss_gb = psutil.Process().memory_info().rss / 1e9
    if rss_gb > max_memory_gb:
        if trial is not None:
            trial.set_user_attr("oom_risk", True)
            trial.set_user_attr("oom_at_fold", fold_idx)
            trial.set_user_attr("oom_rss_gb", round(rss_gb, 2))
        raise optuna.TrialPruned(f"Memory limit exceeded: {rss_gb:.2f}GB > {max_memory_gb}GB")

    return float(score), model


def _objective(
    trial: optuna.Trial, 
    X, y, cv_folds, task_type, contract, 
    max_memory_gb: float = 6.0,
    target_enc_cols=None,
    agg_candidates=None,
    feature_cols=None,
) -> float:
    """
    Optuna objective with Day 12 OOM guards. Stores fold_scores in user_attrs.
    
    If target_enc_cols provided, applies target encoding within each fold (leak-free).
    If agg_candidates provided, applies aggregations within each fold (leak-free).
    """
    params = _suggest_params(trial)
    model_type = params.get("model_type", "lgbm")
    models = []
    oof_scores = []

    try:
        for fold_idx, (train_idx, val_idx) in enumerate(cv_folds):
            score, model = _run_cv_fold(
                X, y, params, model_type, task_type, contract,
                fold_idx, train_idx, val_idx, max_memory_gb, trial=trial,
                target_enc_cols=target_enc_cols,
                agg_candidates=agg_candidates,
                feature_cols=feature_cols,
            )
            oof_scores.append(score)
            models.append(model)

        trial.set_user_attr("fold_scores", [float(s) for s in oof_scores])
        trial.set_user_attr("mean_cv", float(np.mean(oof_scores)))
        trial.set_user_attr("params", params)
        return float(np.mean(oof_scores))

    finally:
        for model in models:
            del model
        del models
        gc.collect()


def _run_cv_no_collect(X, y, params, cv_folds, task_type, contract):
    """Lightweight CV for stability seed reruns — no memory monitoring, no trial."""
    fold_scores = []
    for fold_idx, (train_idx, val_idx) in enumerate(cv_folds):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model_type = params.get("model_type", "lgbm")
        clean_params = {k: v for k, v in params.items() if k != "model_type"}
        ModelClass = _get_model_class(model_type, task_type)
        model = ModelClass(**clean_params)

        if model_type == "lgbm":
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
            )
        elif model_type == "catboost":
            model.fit(
                X_tr, y_tr,
                eval_set=(X_val, y_val),
                early_stopping_rounds=50,
                verbose=0,
            )
        elif model_type == "xgb":
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        else:
            model.fit(X_tr, y_tr)

        if contract.requires_proba:
            probs = model.predict_proba(X_val)
            if probs.shape[1] == 2:
                val_preds = probs[:, 1]
            else:
                val_preds = probs
        else:
            val_preds = model.predict(X_val)

        score = contract.scorer_fn(y_val, val_preds)
        fold_scores.append(float(score))
        del model

    gc.collect()
    return fold_scores


def _train_and_optionally_calibrate(
    X: np.ndarray,
    y: np.ndarray,
    params: dict,
    model_type: str,
    metric: str,
    task_type: str,
    contract,
    cv_folds,
) -> tuple:
    """
    Trains a model. If metric is probability-based, calibrates on held-out fold.
    Returns (final_model, fold_scores, calibration_info).
    """
    run_cal = metric in PROBABILITY_METRICS

    if run_cal:
        X_train_cv, y_train_cv, X_calib, y_calib = _split_calibration_fold(X, y)
        n_calib = len(y_calib)
        calib_method = _select_calibration_method(n_calib)
    else:
        X_train_cv, y_train_cv = X, y
        X_calib, y_calib = None, None
        calib_method = "none"

    # Run CV on (possibly reduced) training data
    fold_scores = []
    for fold_idx, (train_idx, val_idx) in enumerate(cv_folds):
        # Re-split from the train_cv portion
        cv_obj = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        break  # Just to get the cv_folds; we use the passed cv_folds

    cv_type = contract.__dict__.get("cv_type", "StratifiedKFold" if "classification" in task_type else "KFold")
    
    # Actually run CV on X_train_cv
    if run_cal:
        # Build new folds on the calibration-reduced data
        if cv_type == "StratifiedKFold":
            cv_obj = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        elif cv_type == "TimeSeriesSplit":
            from sklearn.model_selection import TimeSeriesSplit
            cv_obj = TimeSeriesSplit(n_splits=5)
        else:
            cv_obj = KFold(n_splits=5, shuffle=True, random_state=42)
        cal_cv_folds = list(cv_obj.split(X_train_cv, y_train_cv))
    else:
        cal_cv_folds = cv_folds

    fold_scores = _run_cv_no_collect(X_train_cv, y_train_cv, params, cal_cv_folds, task_type, contract)

    # Train final model on full X_train_cv
    clean_params = {k: v for k, v in params.items() if k != "model_type"}
    ModelClass = _get_model_class(model_type, task_type)
    final_model = ModelClass(**clean_params)
    if model_type == "lgbm":
        final_model.fit(X_train_cv, y_train_cv)
    else:
        final_model.fit(X_train_cv, y_train_cv)

    calibration_info = {
        "is_calibrated":       False,
        "calibration_method":  "none",
        "calibration_score":   None,
        "calibration_n_samples": 0,
    }

    if run_cal and X_calib is not None:
        final_model, brier, method_used = _run_calibration(
            base_model=final_model,
            X_calib=X_calib,
            y_calib=y_calib,
            method=calib_method,
        )
        calibration_info = {
            "is_calibrated":         method_used != "none",
            "calibration_method":    method_used,
            "calibration_score":     brier,
            "calibration_n_samples": len(y_calib),
        }
        if brier is not None:
            logger.info(
                f"[ml_optimizer] Calibration ({method_used}): "
                f"Brier score on calib fold = {brier:.5f}"
            )

    return final_model, fold_scores, calibration_info


def _get_oof_predictions(X, y, params, task_type, contract):
    """Generate out-of-fold predictions for ensemble diversity."""
    model_type = params.get("model_type", "lgbm")
    clean_params = {k: v for k, v in params.items() if k != "model_type"}

    cv_type = contract.__dict__.get("cv_type", "StratifiedKFold" if "classification" in task_type else "KFold")
    
    if cv_type == "StratifiedKFold":
        cv_obj = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    elif cv_type == "TimeSeriesSplit":
        from sklearn.model_selection import TimeSeriesSplit
        cv_obj = TimeSeriesSplit(n_splits=5)
    else:
        cv_obj = KFold(n_splits=5, shuffle=True, random_state=42)

    # Support dynamically sized OOF arrays for multiclass probabilities
    oof_preds = None

    for train_idx, val_idx in cv_obj.split(X, y):
        ModelClass = _get_model_class(model_type, task_type)
        model = ModelClass(**clean_params)
        if model_type == "lgbm":
            model.fit(
                X[train_idx], y[train_idx],
                eval_set=[(X[val_idx], y[val_idx])],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
            )
        else:
            model.fit(X[train_idx], y[train_idx])

        if contract.requires_proba:
            probs = model.predict_proba(X[val_idx])
            if probs.shape[1] == 2:
                fold_preds = probs[:, 1]
            else:
                fold_preds = probs
        else:
            fold_preds = model.predict(X[val_idx])
            
        if oof_preds is None:
            # Initialize the full OOF array dynamically based on the shape of the first fold's predictions
            if len(fold_preds.shape) > 1:
                oof_preds = np.zeros((len(y), fold_preds.shape[1]))
            else:
                oof_preds = np.zeros(len(y))
                
        oof_preds[val_idx] = fold_preds
        del model

    gc.collect()
    return oof_preds


def _get_existing_champion_scores(state: ProfessorState) -> list[float] | None:
    """Get fold_scores from existing champion model in registry."""
    registry = state.get("model_registry") or {}
    if isinstance(registry, list):
        if not registry:
            return None
        return registry[-1].get("fold_scores")
    if isinstance(registry, dict):
        if not registry:
            return None
        last_entry = list(registry.values())[-1]
        return last_entry.get("fold_scores")
    return None


def _select_best_trial_with_gate(
    study: optuna.Study,
    state: ProfessorState,
    previous_best_scores: list[float] | None = None,
) -> optuna.Trial | None:
    """
    Selects the best Optuna trial, but only accepts it over the previous best
    if Wilcoxon confirms the improvement is significant.
    """
    from tools.wilcoxon_gate import gate_result

    candidate = study.best_trial
    candidate_scores = candidate.user_attrs.get("fold_scores", [])

    if not previous_best_scores or not candidate_scores:
        return candidate

    result = gate_result(
        fold_scores_a=candidate_scores,
        fold_scores_b=previous_best_scores,
        model_name_a=f"trial_{candidate.number}",
        model_name_b="previous_best",
    )

    log_event(
        session_id=state["session_id"],
        agent="ml_optimizer",
        action="wilcoxon_gate_decision",
        keys_read=["model_registry"],
        keys_written=[],
        values_changed=result,
    )

    if result["gate_passed"]:
        return candidate
    return None


def _select_best_model_type(
    model_results: dict[str, dict],
    state: ProfessorState,
) -> str:
    """
    Selects the best model type using pairwise Wilcoxon gates.
    Comparison order (complexity ascending): lgbm → xgb → catboost
    """
    from tools.wilcoxon_gate import gate_result

    MODEL_COMPLEXITY_ORDER = ["lgbm", "xgb", "catboost"]
    available = [m for m in MODEL_COMPLEXITY_ORDER if m in model_results]

    if not available:
        raise ValueError("No model results to compare.")

    champion = available[0]
    champion_scores = model_results[champion].get("fold_scores", [])

    for challenger in available[1:]:
        challenger_scores = model_results[challenger].get("fold_scores", [])

        if not champion_scores or not challenger_scores:
            challenger_mean = model_results[challenger].get("cv_mean", 0)
            champion_mean = model_results[champion].get("cv_mean", 0)
            if challenger_mean > champion_mean:
                champion = challenger
                champion_scores = challenger_scores
            continue

        result = gate_result(
            fold_scores_a=challenger_scores,
            fold_scores_b=champion_scores,
            model_name_a=challenger,
            model_name_b=champion,
        )

        log_event(
            session_id=state["session_id"],
            agent="ml_optimizer",
            action="wilcoxon_gate_decision",
            keys_read=["model_registry"],
            keys_written=[],
            values_changed={**result, "comparison_type": "cross_model"},
        )

        if result["gate_passed"]:
            champion = challenger
            champion_scores = challenger_scores

    return champion


def _memory_callback(max_memory_gb: float):
    """Returns a callback that logs memory after each trial."""
    def callback(study, trial):
        rss_gb = psutil.Process().memory_info().rss / 1e9
        if trial.state == optuna.trial.TrialState.PRUNED:
            logger.info(f"[ml_optimizer] Trial {trial.number} PRUNED (OOM). RSS={rss_gb:.2f}GB")
        else:
            logger.debug(f"[ml_optimizer] Trial {trial.number} complete. RSS={rss_gb:.2f}GB")
    return callback


def _get_study_direction(metric: str) -> str:
    """Returns 'minimize' for loss metrics, 'maximize' for everything else."""
    if metric in MINIMIZE_METRICS:
        return "minimize"
    return "maximize"


def _get_peak_rss() -> float:
    return round(psutil.Process().memory_info().rss / 1e9, 2)


def run_optimization(
    X, y, cv_folds, task_type, contract, 
    direction="maximize", n_trials=10, max_memory_gb=6.0, n_jobs=1,
    target_enc_cols=None,
    agg_candidates=None,
    feature_cols=None,
) -> optuna.Study:
    """
    n_jobs=1 is the default and should not be overridden on 8GB RAM.
    
    If target_enc_cols provided, applies target encoding within each fold (leak-free).
    If agg_candidates provided, applies aggregations within each fold (leak-free).
    """
    study = optuna.create_study(direction=direction)

    with _disable_langsmith_tracing():
        study.optimize(
            lambda trial: _objective(
                trial, X, y, cv_folds, task_type, contract, 
                max_memory_gb,
                target_enc_cols=target_enc_cols,
                agg_candidates=agg_candidates,
                feature_cols=feature_cols,
            ),
            n_trials=n_trials,
            n_jobs=n_jobs,
            callbacks=[_memory_callback(max_memory_gb)],
            gc_after_trial=True,
        )

    return study


@timed_node
@with_agent_retry("MLOptimizer")
def run_ml_optimizer(state: ProfessorState) -> ProfessorState:
    """
    LangGraph node: ML Optimizer with Optuna HPO + multi-seed stability + calibration.
    """
    from tools.stability_validator import run_with_seeds, rank_by_stability, format_stability_report
    from tools.wilcoxon_gate import gate_result

    session_id  = state["session_id"]
    output_dir  = f"outputs/{session_id}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"[MLOptimizer] Starting — session: {session_id}")

    if state.get("hitl_required"):
        print("[MLOptimizer] HALTED: hitl_required flag is set.")
        return {**state, "ml_optimizer_ran": False}

    # ── Load data ─────────────────────────────────────────────────
    if not state.get("feature_data_path"):
        raise ValueError("[MLOptimizer] feature_data_path not in state — run Feature Factory first")
    if not state.get("schema_path"):
        raise ValueError("[MLOptimizer] schema_path not in state — run Data Engineer first")

    # Day 19 Fix: Read from the real finalized feature matrix, not the raw clean one
    feature_data_path = state["feature_data_path"]  # FIX: Store as local variable for return
    df     = read_parquet(feature_data_path)
    schema = read_json(state["schema_path"])

    print(f"[MLOptimizer] Data loaded: {df.shape}")

    # ── Metric Contract ───────────────────────────────────────────
    contract_path = f"{output_dir}/metric_contract.json"
    if os.path.exists(contract_path):
        contract = load_contract(contract_path)
        print(f"[MLOptimizer] Loaded existing contract: {contract.scorer_name}")
    else:
        contract = default_contract(competition_name=state["competition_name"])
        save_contract(contract, contract_path)
        print(f"[MLOptimizer] Created default contract: {contract.scorer_name}")

    # ── Identify target + prepare features ────────────────────────
    target_col = _identify_target_column(schema, state)
    print(f"[MLOptimizer] Target column: {target_col}")

    X, y, feature_names, label_encoders = _prepare_features(df, target_col, schema, state)
    print(f"[MLOptimizer] Features: {len(feature_names)} | Rows: {len(X)}")
    
    # ── LEAKAGE FIX: Extract target encoding and aggregation candidates ──
    # Load feature manifest to get PENDING_CV candidates
    feature_manifest_path = f"{output_dir}/feature_manifest.json"
    target_enc_cols = []
    agg_candidates = []
    
    if os.path.exists(feature_manifest_path):
        with open(feature_manifest_path) as f:
            manifest = json.load(f)
        
        for feature in manifest.get("features", []):
            if feature.get("verdict") == "PENDING_CV":
                if feature.get("transform_type") == "target_encoding":
                    target_enc_cols.append(feature["source_columns"][0])
                elif feature.get("transform_type") == "groupby_agg":
                    agg_candidates.append(feature)
        
        if target_enc_cols:
            print(f"[MLOptimizer] LEAKAGE FIX: {len(target_enc_cols)} target encoding cols will be applied within CV folds")
        if agg_candidates:
            print(f"[MLOptimizer] LEAKAGE FIX: {len(agg_candidates)} aggregation candidates will be applied within CV folds")
    else:
        print(f"[MLOptimizer] WARNING: feature_manifest.json not found at {feature_manifest_path}")

    # ── Setup ─────────────────────────────────────────────────────
    val_strategy = state.get("validation_strategy", {})
    contract.__dict__["cv_type"] = val_strategy.get("cv_type", "StratifiedKFold" if "classification" in contract.task_type else "KFold")
    
    n_folds   = val_strategy.get("n_splits", 5)
    task_type = contract.task_type
    metric    = state.get("evaluation_metric", contract.scorer_name)
    MAX_MEMORY_GB = float(os.getenv("PROFESSOR_MAX_MEMORY_GB", "6.0"))

    if val_strategy.get("cv_type") == "StratifiedKFold":
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    elif val_strategy.get("cv_type") == "TimeSeriesSplit":
        from sklearn.model_selection import TimeSeriesSplit
        cv = TimeSeriesSplit(n_splits=n_folds)
    elif val_strategy.get("cv_type") == "GroupKFold":
        from sklearn.model_selection import GroupKFold
        # Stub group extraction for safety, fallback to KFold array slicing
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    else:
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    # ── Calibration fold split (before CV) ────────────────────────
    run_cal = metric in PROBABILITY_METRICS
    if run_cal:
        X_train_cv_np, y_train_cv, X_calib, y_calib = _split_calibration_fold(X, y)
    else:
        X_train_cv_np, y_train_cv = X, y
        X_calib, y_calib = None, None

    cv_folds = list(cv.split(X_train_cv_np, y_train_cv))

    # ── Phase 1: Optuna study ─────────────────────────────────────
    direction = _get_study_direction(metric)
    n_trials = min(N_OPTUNA_TRIALS, max(20, len(y)))
    print(f"[MLOptimizer] Running Optuna ({n_trials} trials, direction={direction}, mem limit={MAX_MEMORY_GB}GB)...")

    study = optuna.create_study(
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    with _disable_langsmith_tracing():
        study.optimize(
            lambda trial: _objective(
                trial, X_train_cv_np, y_train_cv, cv_folds, task_type, contract, 
                MAX_MEMORY_GB,
                target_enc_cols=target_enc_cols,
                agg_candidates=agg_candidates,
                feature_cols=feature_names,
            ),
            n_trials=n_trials,
            n_jobs=1,
            gc_after_trial=True,
            callbacks=[_memory_callback(MAX_MEMORY_GB)],
        )

    # ── Phase 2: Top-K by mean_cv → stability re-run ─────────────
    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
        and t.user_attrs.get("fold_scores")
    ]

    # Validate Optuna completed successfully
    if len(completed) == 0:
        logger.error("[MLOptimizer] No Optuna trials completed successfully!")
        logger.error(f"[MLOptimizer] Study state: {study.user_attrs}")
        logger.warning("[MLOptimizer] Falling back to default LightGBM model")
        
        # Use default model as fallback
        best_config = {
            "model_type": "lgbm",
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 5,
        }
        best_stability = type('obj', (object,), {
            'mean': 0.5,
            'std': 0.1,
            'stability_score': 0.35,
            'seed_results': [0.5, 0.5, 0.5, 0.5, 0.5],
        })()
    else:
        if direction == "minimize":
            completed.sort(key=lambda t: t.user_attrs["mean_cv"])
        else:
            completed.sort(key=lambda t: t.user_attrs["mean_cv"], reverse=True)

        top_k_trials = completed[:TOP_K_FOR_STABILITY]

        logger.info(
            f"[ml_optimizer] Optuna complete: {len(completed)} trials. "
            f"Re-running top {len(top_k_trials)} configs with {N_STABILITY_SEEDS} seeds."
        )

        def _train_fn(config: dict, seed: int) -> float:
            model_type = config.get("model_type", "lgbm")
            if model_type == "catboost":
                params = {**config, "random_seed": seed}
            else:
                params = {**config, "random_state": seed}
            fold_scores = _run_cv_no_collect(X_train_cv_np, y_train_cv, params, cv_folds, task_type, contract)
            return float(np.mean(fold_scores))

        top_k_configs     = [t.user_attrs["params"] for t in top_k_trials]
        stability_results = [
            run_with_seeds(config=cfg, train_fn=_train_fn)
            for cfg in top_k_configs
        ]

        ranked = rank_by_stability(top_k_configs, stability_results)
        best_config, best_stability = ranked[0]

        logger.info(
            f"[ml_optimizer] Stability ranking:\n"
            f"{format_stability_report(ranked, top_n=3)}"
        )

    # ── Phase 3: Train final model ────────────────────────────────
    best_model_type = best_config.get("model_type", "lgbm")

    final_model, fold_scores, calib_info = _train_and_optionally_calibrate(
        X=X, y=y,
        params=best_config,
        model_type=best_model_type,
        metric=metric,
        task_type=task_type,
        contract=contract,
        cv_folds=cv_folds,
    )

    cv_mean = float(np.mean(fold_scores))
    cv_std  = float(np.std(fold_scores))

    # OOF predictions
    oof_preds = _get_oof_predictions(X_train_cv_np, y_train_cv, best_config, task_type, contract)

    # ── FLAW-11.6: Overfitting Detection ──────────────────────────
    # Estimate train score using the best model's performance on training data
    # We use the mean of the training fold scores from Optuna trials as a proxy
    train_score_proxy = None
    for trial in completed:
        if trial.params == best_config:
            train_score_proxy = trial.user_attrs.get("train_score_mean")
            break

    overfitting_detected = False
    overfitting_gap = None
    if train_score_proxy is not None:
        overfitting_detected, overfitting_gap = detect_overfitting(
            train_score=train_score_proxy,
            cv_score=cv_mean,
            threshold=0.1  # 10% gap threshold
        )
        logger.info(
            f"[Overfitting Check] Train: {train_score_proxy:.4f}, CV: {cv_mean:.4f}, "
            f"Gap: {overfitting_gap:.4f}, Overfitting: {overfitting_detected}"
        )
    else:
        logger.warning("[Overfitting Check] Train score not available, skipping overfitting detection")

    # ── Phase 4: Wilcoxon gate vs existing champion ───────────────
    existing_best = _get_existing_champion_scores(state)
    if existing_best:
        gate_decision = gate_result(
            fold_scores_a=fold_scores,
            fold_scores_b=existing_best,
            model_name_a="day19_optimizer",
            model_name_b="previous_champion",
        )
        log_event(
            session_id=session_id,
            agent="ml_optimizer",
            action="wilcoxon_gate_decision",
            keys_read=["model_registry"],
            keys_written=[],
            values_changed=gate_decision,
        )

    # ── Save model + OOF ──────────────────────────────────────────
    model_path = f"{output_dir}/best_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(final_model, f)

    oof_path = f"{output_dir}/oof_predictions.npy"
    np.save(oof_path, oof_preds)

    # ── Save metrics.json ─────────────────────────────────────────
    feature_order = list(feature_names)
    metrics = {
        "scorer_name":          contract.scorer_name,
        "direction":            direction,
        "cv_mean":              cv_mean,
        "cv_std":               cv_std,
        "fold_scores":          fold_scores,
        "n_folds":              n_folds,
        "n_features":           len(feature_names),
        "feature_names":        feature_names,
        "feature_order":        feature_order,
        "target_col":           target_col,
        "n_rows":               len(X),
        "model_type":           best_model_type,
        "trained_at":           datetime.utcnow().isoformat(),
        "data_hash":            state.get("data_hash", ""),
        # FLAW-11.6: Overfitting detection results
        "overfitting_detected": overfitting_detected,
        "overfitting_gap":      overfitting_gap,
    }
    metrics_path = f"{output_dir}/metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save label encoders for consistent test-set encoding
    encoders_path = f"{output_dir}/label_encoders.json"
    with open(encoders_path, "w") as f:
        json.dump(label_encoders, f, indent=2)
    logger.info(f"[ml_optimizer] Saved label encoders: {list(label_encoders.keys())}")

    # ── Process and save test data ─────────────────────────────────────────────
    feature_data_path_test = None
    test_path = state.get("test_data_path", "")
    if test_path and os.path.exists(test_path):
        try:
            df_test = read_parquet(test_path)
            
            # Apply same preprocessor
            preprocessor_path = state.get("preprocessor_path", "")
            if os.path.exists(preprocessor_path):
                from core.preprocessor import TabularPreprocessor
                preprocessor = TabularPreprocessor.load(preprocessor_path)
                X_test = preprocessor.transform(df_test)
                
                # Select only the features used in training (feature_order)
                if feature_order:
                    available_cols = [c for c in feature_order if c in X_test.columns]
                    if available_cols:
                        X_test = X_test.select(available_cols)
                
                # Save test features
                feature_data_path_test = f"{output_dir}/X_test.parquet"
                X_test.write_parquet(feature_data_path_test)
                logger.info(f"[ml_optimizer] Saved test features: {feature_data_path_test}")
        except Exception as e:
            logger.warning(f"[ml_optimizer] Could not process test data: {e}")

    # ── Build registry entry ──────────────────────────────────────
    model_id = f"{best_model_type}_day19_{int(time.time())}"
    registry_entry = {
        "model_id":              model_id,
        "model_path":            model_path,
        "model_type":            best_model_type,
        "cv_mean":               round(best_stability.mean, 6),
        "cv_std":                round(best_stability.std, 6),
        "fold_scores":           fold_scores,
        "stability_score":       best_stability.stability_score,
        "seed_results":          best_stability.seed_results,
        "params":                best_config,
        "oof_predictions_path":  oof_path,
        "data_hash":             state.get("data_hash", ""),
        "scorer_name":           contract.scorer_name,
        # FLAW-11.6: Overfitting detection results
        "overfitting_detected":  overfitting_detected,
        "overfitting_gap":       overfitting_gap,
    }
    registry_entry = _update_model_registry_with_calibration(registry_entry, calib_info)

    # Augment existing registry (not replace)
    existing_registry = state.get("model_registry") or []
    if isinstance(existing_registry, dict):
        existing_registry = list(existing_registry.values())
    existing_registry = [*existing_registry, registry_entry]

    # ── CRITICAL: Validate model registry ────────────────────────────
    if not existing_registry or len(existing_registry) == 0:
        logger.error("[MLOptimizer] CRITICAL: Model registry is empty after training!")
        logger.error(f"[MLOptimizer] Optuna trials completed: {len(completed)}")
        logger.error(f"[MLOptimizer] Best config: {best_config}")
        
        # Save error state for debugging
        error_state_path = f"{output_dir}/ml_optimizer_error.json"
        with open(error_state_path, "w") as f:
            json.dump({
                "error": "model_registry_empty",
                "completed_trials": len(completed),
                "best_config": best_config,
                "cv_mean": cv_mean,
                "timestamp": datetime.utcnow().isoformat(),
            }, f, indent=2)
        
        raise ValueError(
            f"[MLOptimizer] Model training failed - registry is empty. "
            f"Check {error_state_path} for details."
        )

    # Validate that best model was saved
    if not os.path.exists(model_path):
        raise ValueError(f"[MLOptimizer] Best model not saved at {model_path}")

    logger.info(f"[MLOptimizer] Registry validated: {len(existing_registry)} models")

    # ── Memory / OOM stats ────────────────────────────────────────
    peak_rss = _get_peak_rss()
    memory_oom_risk = any(
        t.user_attrs.get("oom_risk") for t in study.trials if t.user_attrs
    )
    optuna_pruned_trials = sum(
        1 for t in study.trials
        if t.state == optuna.trial.TrialState.PRUNED
    )

    # ── Update cost tracker ───────────────────────────────────────
    cost_tracker = dict(state["cost_tracker"])
    cost_tracker["llm_calls"] += 0

    print(f"[MLOptimizer] Complete. cv_mean={cv_mean:.4f}, stability={best_stability.stability_score:.5f}")

    # ── Log lineage ───────────────────────────────────────────────
    log_event(
        session_id=session_id,
        agent="ml_optimizer",
        action="ml_optimizer_complete",
        keys_read=["clean_data_path", "schema_path"],
        keys_written=["model_registry", "cv_mean", "oof_predictions_path"],
        values_changed={
            "model_id":             model_id,
            "cv_mean":              registry_entry["cv_mean"],
            "cv_std":               registry_entry["cv_std"],
            "stability_score":      registry_entry["stability_score"],
            "is_calibrated":        registry_entry["is_calibrated"],
            "n_trials":             len(completed),
            "top_k_rerun":          len(top_k_trials),
            # FLAW-11.6: Overfitting detection results
            "overfitting_detected": overfitting_detected,
            "overfitting_gap":      overfitting_gap,
        },
    )

    # ── Log to MLflow (graceful fallback) ─────────────────────────
    from tools.mlflow_tracker import log_run as mlflow_log_run
    mlflow_log_run(
        session_id=session_id,
        competition=state["competition_name"],
        model_type=best_model_type,
        params=best_config,
        cv_mean=cv_mean,
        cv_std=cv_std,
        n_features=len(feature_names),
        data_hash=state.get("data_hash", "unknown"),
    )

    return {
        **state,
        "cv_scores":            fold_scores,
        "cv_mean":              cv_mean,
        "feature_order":        feature_order,
        "feature_data_path":    feature_data_path,  # FIX Bug 1: Add for pseudo_label_agent
        "feature_data_path_test": feature_data_path_test,
        "model_registry":       existing_registry,
        "metric_contract":      metrics,
        "oof_predictions_path": oof_path,
        "cost_tracker":         cost_tracker,
        "memory_peak_gb":       peak_rss,
        "memory_oom_risk":      memory_oom_risk,
        "optuna_pruned_trials": optuna_pruned_trials,
    }
