# agents/ml_optimizer.py

import os
import json
import pickle
import numpy as np
import polars as pl
from sklearn.model_selection import StratifiedKFold, KFold
from lightgbm import LGBMClassifier, LGBMRegressor
import lightgbm as lgb
from core.state import ProfessorState
from core.metric_contract import (
    MetricContract, default_contract,
    save_contract, load_contract, contract_to_prompt_snippet
)
from tools.data_tools import read_parquet, read_json


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

    if task_type == "classification":
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    else:
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_scores    = []
    oof_preds      = np.zeros(len(y))
    trained_models = []

    print(f"[MLOptimizer] Running {n_folds}-fold CV...")

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Build model — Phase 1: default params
        if task_type == "classification":
            model = LGBMClassifier(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=31,
                random_state=42,
                verbose=-1,
                n_jobs=-1
            )
        else:
            model = LGBMRegressor(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=31,
                random_state=42,
                verbose=-1,
                n_jobs=-1
            )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(0)
            ]
        )

        # Score this fold
        if contract.requires_proba:
            val_preds = model.predict_proba(X_val)[:, 1]
        else:
            val_preds = model.predict(X_val)

        oof_preds[val_idx] = val_preds

        fold_score = contract.scorer_fn(y_val, val_preds)
        fold_scores.append(float(fold_score))
        trained_models.append(model)

        print(f"[MLOptimizer] Fold {fold}: {contract.scorer_name.upper()} = {fold_score:.4f}")

    cv_mean = float(np.mean(fold_scores))
    cv_std  = float(np.std(fold_scores))
    print(f"[MLOptimizer] CV {contract.scorer_name.upper()}: {cv_mean:.4f} (+/- {cv_std:.4f})")

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
        "target_col":     target_col,
        "n_rows":         len(X),
    }
    metrics_path = f"{output_dir}/metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Build new model registry entry ─────────────────────────────
    # NOTE: With Annotated[list, operator.add] in ProfessorState,
    # LangGraph concatenates the returned list with existing state.
    # So we return ONLY the new entry, not a copy of the full list.
    new_registry_entry = [{
        "model_path":  model_path,
        "model_type":  "lightgbm_v0",
        "cv_mean":     cv_mean,
        "cv_std":      cv_std,
        "scorer_name": contract.scorer_name,
        "data_hash":   state.get("data_hash"),
        "fold_scores": fold_scores,
    }]

    # ── Update cost tracker ───────────────────────────────────────
    cost_tracker = dict(state["cost_tracker"])
    cost_tracker["llm_calls"] += 0  # v0 uses no LLM calls

    print(f"[MLOptimizer] Complete.")

    # ── Log lineage ──────────────────────────────────────────────
    from core.lineage import log_event
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
        "model_registry":       new_registry_entry,
        "metric_contract":      metrics,
        "oof_predictions_path": oof_path,
        "cost_tracker":         cost_tracker,
    }
