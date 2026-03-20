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
from core.metric_contract import (
    MetricContract, default_contract,
    save_contract, load_contract, contract_to_prompt_snippet
)
from tools.data_tools import read_parquet, read_json
from guards.agent_retry import with_agent_retry

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
    """Returns the appropriate model class for the given model type and task."""
    is_clf = "classification" in task_type
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
    Identify the target column from schema.
    Phase 1: look for common names. Phase 2: LLM-driven detection.
    """
    common_targets = [
        "Transported", "target", "Target", "label", "Label",
        "Survived", "survived", "y", "outcome", "Outcome",
        "price", "Price", "SalePrice", "salary", "Salary"
    ]
    columns = schema["columns"]
    for candidate in common_targets:
        if candidate in columns:
            return candidate

    # Fall back to last column (common convention)
    return columns[-1]


def _prepare_features(df: pl.DataFrame, target_col: str, schema: dict) -> tuple:
    """
    Convert Polars DataFrame to numpy arrays for sklearn/LightGBM.
    Encodes categoricals as integer codes.
    Returns (X, y, feature_names)
    """
    feature_cols = [c for c in df.columns if c != target_col]

    # Encode string columns as integer codes
    for col in feature_cols:
        if df[col].dtype == pl.Utf8 or df[col].dtype == pl.String:
            df = df.with_columns(
                pl.col(col).cast(pl.Categorical).cast(pl.Int32)
            )
        elif df[col].dtype == pl.Boolean:
            df = df.with_columns(
                pl.col(col).cast(pl.Int32)
            )

    # Convert target
    y_series = df[target_col]
    if y_series.dtype == pl.Boolean:
        y = y_series.cast(pl.Int32).to_numpy()
    elif y_series.dtype in (pl.Utf8, pl.String):
        y = y_series.cast(pl.Categorical).cast(pl.Int32).to_numpy()
    else:
        y = y_series.to_numpy()

    X = df.select(feature_cols).to_numpy().astype(np.float64)

    return X, y, feature_cols


# ── CV helper for objective + stability ───────────────────────────
def _run_cv_fold(X, y, params, model_type, task_type, contract, fold_idx, train_idx, val_idx, max_memory_gb, trial=None):
    """Train one CV fold and return (score, model). Raises TrialPruned on OOM."""
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    clean_params = {k: v for k, v in params.items() if k != "model_type"}
    ModelClass = _get_model_class(model_type, task_type)
    model = ModelClass(**clean_params)

    if model_type == "lgbm":
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
        )
    else:
        model.fit(X_tr, y_tr)

    if contract.requires_proba:
        val_preds = model.predict_proba(X_val)[:, 1]
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


def _objective(trial: optuna.Trial, X, y, cv_folds, task_type, contract, max_memory_gb: float = 6.0) -> float:
    """Optuna objective with Day 12 OOM guards. Stores fold_scores in user_attrs."""
    params = _suggest_params(trial)
    model_type = params.get("model_type", "lgbm")
    models = []
    oof_scores = []

    try:
        for fold_idx, (train_idx, val_idx) in enumerate(cv_folds):
            score, model = _run_cv_fold(
                X, y, params, model_type, task_type, contract,
                fold_idx, train_idx, val_idx, max_memory_gb, trial=trial,
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
        else:
            model.fit(X_tr, y_tr)

        if contract.requires_proba:
            val_preds = model.predict_proba(X_val)[:, 1]
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

    # Actually run CV on X_train_cv
    if run_cal:
        # Build new folds on the calibration-reduced data
        if "classification" in task_type:
            cv_obj = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
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

    if "classification" in task_type:
        cv_obj = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    else:
        cv_obj = KFold(n_splits=5, shuffle=True, random_state=42)

    oof_preds = np.zeros(len(y))
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
            oof_preds[val_idx] = model.predict_proba(X[val_idx])[:, 1]
        else:
            oof_preds[val_idx] = model.predict(X[val_idx])
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


def run_optimization(X, y, cv_folds, task_type, contract, direction="maximize", n_trials=10, max_memory_gb=6.0, n_jobs=1) -> optuna.Study:
    """
    n_jobs=1 is the default and should not be overridden on 8GB RAM.
    """
    study = optuna.create_study(direction=direction)

    with _disable_langsmith_tracing():
        study.optimize(
            lambda trial: _objective(trial, X, y, cv_folds, task_type, contract, max_memory_gb),
            n_trials=n_trials,
            n_jobs=n_jobs,
            callbacks=[_memory_callback(max_memory_gb)],
            gc_after_trial=True,
        )

    return study


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
    if not state.get("clean_data_path"):
        raise ValueError("[MLOptimizer] clean_data_path not in state — run Data Engineer first")
    if not state.get("schema_path"):
        raise ValueError("[MLOptimizer] schema_path not in state — run Data Engineer first")

    df     = read_parquet(state["clean_data_path"])
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

    X, y, feature_names = _prepare_features(df, target_col, schema)
    print(f"[MLOptimizer] Features: {len(feature_names)} | Rows: {len(X)}")

    # ── Setup ─────────────────────────────────────────────────────
    n_folds   = 5
    task_type = contract.task_type
    metric    = state.get("evaluation_metric", contract.scorer_name)
    MAX_MEMORY_GB = float(os.getenv("PROFESSOR_MAX_MEMORY_GB", "6.0"))

    if "classification" in task_type:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
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
            lambda trial: _objective(trial, X_train_cv_np, y_train_cv, cv_folds, task_type, contract, MAX_MEMORY_GB),
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
        params = {**config, "random_state": seed, "seed": seed}
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
        "scorer_name":    contract.scorer_name,
        "direction":      direction,
        "cv_mean":        cv_mean,
        "cv_std":         cv_std,
        "fold_scores":    fold_scores,
        "n_folds":        n_folds,
        "n_features":     len(feature_names),
        "feature_names":  feature_names,
        "feature_order":  feature_order,
        "target_col":     target_col,
        "n_rows":         len(X),
        "model_type":     best_model_type,
        "trained_at":     datetime.utcnow().isoformat(),
        "data_hash":      state.get("data_hash", ""),
    }
    metrics_path = f"{output_dir}/metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

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
        "oof_predictions":       oof_preds.tolist(),
        "data_hash":             state.get("data_hash", ""),
        "scorer_name":           contract.scorer_name,
    }
    registry_entry = _update_model_registry_with_calibration(registry_entry, calib_info)

    # Augment existing registry (not replace)
    existing_registry = state.get("model_registry") or []
    if isinstance(existing_registry, dict):
        existing_registry = list(existing_registry.values())
    existing_registry = [*existing_registry, registry_entry]

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
            "model_id":        model_id,
            "cv_mean":         registry_entry["cv_mean"],
            "cv_std":          registry_entry["cv_std"],
            "stability_score": registry_entry["stability_score"],
            "is_calibrated":   registry_entry["is_calibrated"],
            "n_trials":        len(completed),
            "top_k_rerun":     len(top_k_trials),
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
        "model_registry":       existing_registry,
        "metric_contract":      metrics,
        "oof_predictions_path": oof_path,
        "cost_tracker":         cost_tracker,
        "memory_peak_gb":       peak_rss,
        "memory_oom_risk":      memory_oom_risk,
        "optuna_pruned_trials": optuna_pruned_trials,
    }
