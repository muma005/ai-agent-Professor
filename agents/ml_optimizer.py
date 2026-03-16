# agents/ml_optimizer.py

import os
import gc
import logging
import psutil
import optuna
import json
import pickle
import numpy as np
import polars as pl
from sklearn.model_selection import StratifiedKFold, KFold
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


def _objective(trial: optuna.Trial, X, y, cv_folds, task_type, contract, max_memory_gb: float = 6.0) -> float:
    models = []
    oof_scores = []
    
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 63),
        "random_state": 42,
        "n_jobs": 1,
        "verbose": -1,
    }
    
    try:
        for fold_idx, (train_idx, val_idx) in enumerate(cv_folds):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            
            if "classification" in task_type:
                model = lgb.LGBMClassifier(**params)
            else:
                model = lgb.LGBMRegressor(**params)
                
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
            )
            
            if contract.requires_proba:
                val_preds = model.predict_proba(X_val)[:, 1]
            else:
                val_preds = model.predict(X_val)
                
            score = contract.scorer_fn(y_val, val_preds)
            oof_scores.append(float(score))
            models.append(model)
            
            # ── Memory check after each fold, not just after the trial ──
            rss_gb = psutil.Process().memory_info().rss / 1e9
            if rss_gb > max_memory_gb:
                print(
                    f"[ml_optimizer] Trial {trial.number} fold {fold_idx}: "
                    f"RSS={rss_gb:.2f}GB exceeds limit {max_memory_gb}GB. "
                    f"Pruning trial to prevent OOM."
                )
                trial.set_user_attr("oom_risk", True)
                trial.set_user_attr("oom_at_fold", fold_idx)
                trial.set_user_attr("oom_rss_gb", round(rss_gb, 2))
                raise optuna.TrialPruned(f"Memory limit exceeded: {rss_gb:.2f}GB > {max_memory_gb}GB")
        
        trial.set_user_attr("fold_scores", [float(s) for s in oof_scores])
        trial.set_user_attr("mean_cv", float(np.mean(oof_scores)))
        return float(np.mean(oof_scores))
    
    finally:
        # ── Always runs — whether trial completed, pruned, or raised ──
        for model in models:
            del model
        del models
        gc.collect()


def _select_best_trial_with_gate(
    study: optuna.Study,
    state: ProfessorState,
    previous_best_scores: list[float] | None = None,
) -> optuna.Trial | None:
    """
    Selects the best Optuna trial, but only accepts it over the previous best
    if Wilcoxon confirms the improvement is significant.
    Returns None if the candidate is not significantly better (caller keeps existing).
    """
    from tools.wilcoxon_gate import gate_result

    candidate = study.best_trial
    candidate_scores = candidate.user_attrs.get("fold_scores", [])

    if not previous_best_scores or not candidate_scores:
        logger.info(
            "[ml_optimizer] No previous best or fold scores unavailable — "
            "accepting study best trial without gate."
        )
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
        logger.info(
            f"[ml_optimizer] Wilcoxon gate PASSED — "
            f"accepting trial_{candidate.number} "
            f"(delta={result['mean_delta']:+.5f}, p<0.05)"
        )
        return candidate
    else:
        logger.info(
            f"[ml_optimizer] Wilcoxon gate FAILED — "
            f"keeping previous best. "
            f"trial_{candidate.number} not significantly better "
            f"(delta={result['mean_delta']:+.5f})"
        )
        return None


def _select_best_model_type(
    model_results: dict[str, dict],
    state: ProfessorState,
) -> str:
    """
    Selects the best model type using pairwise Wilcoxon gates.
    Returns the name of the significantly best model, or the simplest
    model if no significant differences found.

    Comparison order (complexity ascending): lgbm → xgb → catboost
    A more complex model must beat the simpler one to be selected.
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
            # Fall back to mean CV if fold scores not available
            challenger_mean = model_results[challenger].get("cv_mean", 0)
            champion_mean = model_results[champion].get("cv_mean", 0)
            if challenger_mean > champion_mean:
                logger.warning(
                    f"[ml_optimizer] fold_scores unavailable — "
                    f"falling back to mean CV. {challenger} ({challenger_mean:.5f}) > "
                    f"{champion} ({champion_mean:.5f})"
                )
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
            logger.info(
                f"[ml_optimizer] {challenger} significantly beats {result['model_name_b']} "
                f"— new champion."
            )
        else:
            logger.info(
                f"[ml_optimizer] {challenger} does NOT significantly beat {champion} "
                f"— keeping simpler model."
            )

    return champion


def run_optimization(X, y, cv_folds, task_type, contract, direction="maximize", n_trials=10, max_memory_gb=6.0, n_jobs=1) -> optuna.Study:
    """
    n_jobs=1 is the default and should not be overridden on 8GB RAM.
    n_jobs=-1 means each worker holds its own model copy — instant OOM.
    """
    study = optuna.create_study(direction=direction)
    
    def memory_callback(study, trial):
        rss_gb = psutil.Process().memory_info().rss / 1e9
        if trial.state == optuna.trial.TrialState.PRUNED:
            print(f"[ml_optimizer] Trial {trial.number} PRUNED (OOM). RSS={rss_gb:.2f}GB")
        else:
            print(f"[ml_optimizer] Trial {trial.number} complete. RSS={rss_gb:.2f}GB")
    
    from core.professor import _disable_langsmith_tracing
    
    with _disable_langsmith_tracing():
        study.optimize(
            lambda trial: _objective(trial, X, y, cv_folds, task_type, contract, max_memory_gb),
            n_trials=n_trials,
            n_jobs=n_jobs,          # never change this default on 8GB
            callbacks=[memory_callback],
            gc_after_trial=True,    # Optuna's own GC flag — belt AND braces
        )
    
    return study


@with_agent_retry("MLOptimizer")
def run_ml_optimizer(state: ProfessorState) -> ProfessorState:
    """
    LangGraph node: ML Optimizer v0.

    Reads:  state["clean_data_path"]  — cleaned.parquet
            state["schema_path"]      — schema.json
    Writes: state["model_registry"]   — list with model entry
            state["cv_mean"]          — float
            state["cv_scores"]        — list of fold scores
            state["oof_predictions_path"] — str pointer
            state["cost_tracker"]     — updated

    Phase 1: single LightGBM, default params, StratifiedKFold(5).
    """
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

    # ── Cross-validation ──────────────────────────────────────────
    n_folds   = 5
    task_type = contract.task_type

    if "classification" in task_type:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    else:
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    cv_folds = list(cv.split(X, y))
    
    # ── Day 12 Memory Overrides ──────────────────────────────────
    MAX_MEMORY_GB = float(os.getenv("PROFESSOR_MAX_MEMORY_GB", "6.0"))
    
    # Check overrides from HITL
    lgbm_override = state.get("lgbm_override", {})
    if lgbm_override:
        print(f"[MLOptimizer] Applied lgbm_override: {lgbm_override}")

    print(f"[MLOptimizer] Running Optuna Optimization (Memory Limit: {MAX_MEMORY_GB}GB)...")
    
    study = run_optimization(
        X, y, cv_folds, task_type, contract, 
        direction=contract.direction, 
        n_trials=20,  # Keeping low for stability in this phase
        max_memory_gb=MAX_MEMORY_GB
    )

    peak_rss = max(
        (t.user_attrs.get("oom_rss_gb", 0) for t in study.trials if t.user_attrs),
        default=psutil.Process().memory_info().rss / 1e9
    )
    
    memory_oom_risk = any(t.user_attrs.get("oom_risk") for t in study.trials)
    optuna_pruned_trials = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
    
    # Retrain best models on cv_folds to obtain fold models and OOF preds
    best_params = study.best_params
    best_params.update({"random_state": 42, "n_jobs": 1, "verbose": -1})
    best_params.update(lgbm_override)

    fold_scores    = []
    oof_preds      = np.zeros(len(y))
    trained_models = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv_folds):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        if "classification" in task_type:
            model = LGBMClassifier(**best_params)
        else:
            model = LGBMRegressor(**best_params)

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
        )

        if contract.requires_proba:
            val_preds = model.predict_proba(X_val)[:, 1]
        else:
            val_preds = model.predict(X_val)

        oof_preds[val_idx] = val_preds
        fold_score = contract.scorer_fn(y_val, val_preds)
        fold_scores.append(float(fold_score))
        trained_models.append(model)

    cv_mean = float(np.mean(fold_scores))
    cv_std  = float(np.std(fold_scores))
    print(f"[MLOptimizer] Best Optuna {contract.scorer_name.upper()}: {cv_mean:.4f} (+/- {cv_std:.4f})")

    # ── Save best model ───────────────────────────────────────────
    if contract.direction == "maximize":
        best_fold_idx = int(np.argmax(fold_scores))
    else:
        best_fold_idx = int(np.argmin(fold_scores))

    best_model = trained_models[best_fold_idx]
    model_path = f"{output_dir}/best_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)
    print(f"[MLOptimizer] Best model (fold {best_fold_idx + 1}) saved: {model_path}")

    # ── Save OOF predictions ──────────────────────────────────────
    oof_path = f"{output_dir}/oof_predictions.npy"
    np.save(oof_path, oof_preds)

    # ── Save metrics.json ─────────────────────────────────────────
    feature_order = list(feature_names)   # exact training column order
    metrics = {
        "scorer_name":    contract.scorer_name,
        "direction":      contract.direction,
        "cv_mean":        cv_mean,
        "cv_std":         cv_std,
        "fold_scores":    fold_scores,
        "n_folds":        n_folds,
        "best_fold":      best_fold_idx + 1,
        "n_features":     len(feature_names),
        "feature_names":  feature_names,
        "feature_order":  feature_order,
        "target_col":     target_col,
        "n_rows":         len(X),
        "model_type":     "lightgbm_v0",
        "trained_at":     datetime.utcnow().isoformat(),
        "data_hash":      state.get("data_hash", ""),
    }
    metrics_path = f"{output_dir}/metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Build new model registry entry ─────────────────────────────
    # Manually append the new model since we use _replace reducer
    existing_registry = list(state.get("model_registry") or [])
    new_registry_entry = {
        "model_path":  model_path,
        "model_type":  "lightgbm_v0",
        "cv_mean":     cv_mean,
        "cv_std":      cv_std,
        "scorer_name": contract.scorer_name,
        "data_hash":   state.get("data_hash", ""),
        "fold_scores": fold_scores,
    }
    existing_registry.append(new_registry_entry)

    # ── Update cost tracker ───────────────────────────────────────
    cost_tracker = dict(state["cost_tracker"])
    cost_tracker["llm_calls"] += 0  # v0 uses no LLM calls

    print(f"[MLOptimizer] Complete.")

    # ── Log lineage ──────────────────────────────────────────────
    log_event(
        session_id=session_id,
        agent="ml_optimizer",
        action="trained_and_scored",
        keys_read=["clean_data_path", "schema_path"],
        keys_written=["model_registry", "cv_mean", "oof_predictions_path"],
        values_changed={"cv_mean": cv_mean, "cv_std": cv_std},
    )

    # ── Log to MLflow (graceful fallback) ─────────────────────────
    from tools.mlflow_tracker import log_run as mlflow_log_run
    mlflow_log_run(
        session_id=session_id,
        competition=state["competition_name"],
        model_type="lightgbm_v0",
        params={"n_estimators": 500, "learning_rate": 0.05, "num_leaves": 31},
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
        "memory_peak_gb":       round(peak_rss, 2),
        "memory_oom_risk":      memory_oom_risk,
        "optuna_pruned_trials": optuna_pruned_trials,
    }
