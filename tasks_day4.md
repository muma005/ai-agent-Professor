# Day 4 Tasks
I've now checked every relevant search. Notion confirms exactly 3 tasks on Day 4 — all Phase 1, all Critical, all Never Cut.

Day 4 Tasks — Confirmed From Notion
┌────┬──────────────────────────────────────────┬─────────────────────────┬──────────┬───────────┐
│ #  │ Task                                     │ Phase                   │ Priority │ Cuttable  │
├────┼──────────────────────────────────────────┼─────────────────────────┼──────────┼───────────┤
│ 1  │ Build core/metric_contract.py            │ 🚀 Phase 1: Make It Run │ Critical │ Never Cut │
│ 2  │ Build agents/ml_optimizer.py — v0        │ 🚀 Phase 1: Make It Run │ Critical │ Never Cut │
│ 3  │ Write contract test — ML Optimizer       │ 🚀 Phase 1: Make It Run │ Critical │ Never Cut │
└────┴──────────────────────────────────────────┴─────────────────────────┴──────────┴───────────┘
Notion notes verbatim:

metric_contract.py: MetricContract dataclass: scorer_fn, direction, forbidden_metrics. Injected into every agent system prompt. Hardcode AUC for now.
ml_optimizer.py: Single LightGBM with default params. Reads schema.json + cleaned.parquet pointer. Outputs best_model.pkl + metrics.json. Uses same CV folds (StratifiedKFold 5).
contract test: INPUT: cleaned.parquet pointer, schema.json, metric_contract.json. OUTPUT: best_model.pkl exists, metrics.json has cv_mean/cv_std/fold_scores. STATE: model_registry updated, cost_tracker incremented. CV score must never use forbidden metrics.

The ONE thing that must work by end of today:
Feed cleaned.parquet into ml_optimizer.py and get back a real CV score in metrics.json. First number that means something. This is the floor everything from Day 5 onward must beat.
Build order matters: Task 1 before Task 2 — the optimizer imports from metric_contract.

Task 1 — Build core/metric_contract.py
python# core/metric_contract.py

from dataclasses import dataclass, field
from typing import Callable, Optional
import json
import os
from sklearn import metrics as skmetrics


# ── All supported scorers ─────────────────────────────────────────
SCORER_REGISTRY = {
    # Classification
    "accuracy":          (skmetrics.accuracy_score,          "maximize"),
    "auc":               (skmetrics.roc_auc_score,           "maximize"),
    "roc_auc":           (skmetrics.roc_auc_score,           "maximize"),
    "log_loss":          (skmetrics.log_loss,                "minimize"),
    "f1":                (skmetrics.f1_score,                "maximize"),
    "f1_macro":          (lambda y, p: skmetrics.f1_score(y, p, average="macro"), "maximize"),
    "f1_weighted":       (lambda y, p: skmetrics.f1_score(y, p, average="weighted"), "maximize"),
    "matthews_corrcoef": (skmetrics.matthews_corrcoef,       "maximize"),
    # Regression
    "rmse":              (lambda y, p: skmetrics.root_mean_squared_error(y, p), "minimize"),
    "mae":               (skmetrics.mean_absolute_error,     "minimize"),
    "r2":                (skmetrics.r2_score,                "maximize"),
    "rmsle":             (lambda y, p: skmetrics.mean_squared_log_error(y, p) ** 0.5, "minimize"),
    "mape":              (skmetrics.mean_absolute_percentage_error, "minimize"),
}

# Metrics that require predict_proba instead of predict
PROBABILITY_METRICS = {"auc", "roc_auc", "log_loss"}

# Metrics that are FORBIDDEN — never optimise toward these
# (proxies that look good but don't reflect true performance)
FORBIDDEN_METRICS = {"accuracy_on_train", "train_loss", "overfit_score"}


@dataclass
class MetricContract:
    """
    The single source of truth for what Professor is optimising toward.
    Written once per competition. Injected into every agent system prompt.
    Never changed mid-pipeline without explicit user approval.
    """
    scorer_name:       str              # e.g. "auc", "rmse"
    direction:         str              # "maximize" or "minimize"
    scorer_fn:         Callable         # the actual sklearn function
    requires_proba:    bool             # True if predict_proba needed
    forbidden_metrics: list             # metrics never to optimise toward
    task_type:         str              # "classification" or "regression"
    competition_name:  str = ""
    locked:            bool = False     # True after first submission
    notes:             str = ""


def build_metric_contract(
    scorer_name: str,
    task_type: str,
    competition_name: str = "",
    notes: str = ""
) -> MetricContract:
    """
    Build a MetricContract from a scorer name string.
    Used by the Validation Architect (Phase 2). For Phase 1: hardcode AUC.

    Args:
        scorer_name:      one of the keys in SCORER_REGISTRY
        task_type:        "classification" or "regression"
        competition_name: for logging
        notes:            any additional context

    Returns:
        MetricContract ready to inject into agent prompts
    """
    scorer_name = scorer_name.lower().strip()

    if scorer_name not in SCORER_REGISTRY:
        raise ValueError(
            f"Unknown scorer: '{scorer_name}'. "
            f"Supported: {list(SCORER_REGISTRY.keys())}"
        )

    scorer_fn, direction = SCORER_REGISTRY[scorer_name]

    return MetricContract(
        scorer_name=scorer_name,
        direction=direction,
        scorer_fn=scorer_fn,
        requires_proba=scorer_name in PROBABILITY_METRICS,
        forbidden_metrics=list(FORBIDDEN_METRICS),
        task_type=task_type,
        competition_name=competition_name,
        locked=False,
        notes=notes
    )


def default_contract(competition_name: str = "") -> MetricContract:
    """
    Phase 1 default: AUC for binary classification.
    Replaced by Validation Architect auto-detection in Phase 2.
    """
    return build_metric_contract(
        scorer_name="auc",
        task_type="classification",
        competition_name=competition_name,
        notes="Phase 1 default — hardcoded AUC. Auto-detected from Day 8."
    )


def save_contract(contract: MetricContract, path: str) -> str:
    """Save MetricContract as metric_contract.json."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        "scorer_name":       contract.scorer_name,
        "direction":         contract.direction,
        "requires_proba":    contract.requires_proba,
        "forbidden_metrics": contract.forbidden_metrics,
        "task_type":         contract.task_type,
        "competition_name":  contract.competition_name,
        "locked":            contract.locked,
        "notes":             contract.notes
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def load_contract(path: str) -> MetricContract:
    """Load MetricContract from metric_contract.json."""
    with open(path) as f:
        data = json.load(f)
    scorer_fn, _ = SCORER_REGISTRY[data["scorer_name"]]
    return MetricContract(
        scorer_name=data["scorer_name"],
        direction=data["direction"],
        scorer_fn=scorer_fn,
        requires_proba=data["requires_proba"],
        forbidden_metrics=data["forbidden_metrics"],
        task_type=data["task_type"],
        competition_name=data["competition_name"],
        locked=data.get("locked", False),
        notes=data.get("notes", "")
    )


def contract_to_prompt_snippet(contract: MetricContract) -> str:
    """
    Returns a string injected into every agent system prompt.
    Makes every agent aware of what it is optimising toward.
    """
    better = "higher" if contract.direction == "maximize" else "lower"
    return f"""
METRIC CONTRACT (read-only — never change this mid-pipeline):
  Competition:     {contract.competition_name}
  Optimise for:    {contract.scorer_name.upper()} ({better} is better)
  Task type:       {contract.task_type}
  Requires proba:  {contract.requires_proba}
  FORBIDDEN:       Never report or optimise toward: {contract.forbidden_metrics}
  Locked:          {contract.locked}
"""

Task 2 — Build agents/ml_optimizer.py — v0
python# agents/ml_optimizer.py

import os
import json
import pickle
import numpy as np
import polars as pl
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier, LGBMRegressor
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

    # Convert target
    y_series = df[target_col]
    if y_series.dtype == pl.Boolean:
        y = y_series.cast(pl.Int32).to_numpy()
    elif y_series.dtype in (pl.Utf8, pl.String):
        y = y_series.cast(pl.Categorical).cast(pl.Int32).to_numpy()
    else:
        y = y_series.to_numpy()

    X = df.select(feature_cols).to_numpy()

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
    Phase 3: upgraded with Optuna HPO.
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

    fold_scores   = []
    oof_preds     = np.zeros(len(y))
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
                __import__("lightgbm").early_stopping(50, verbose=False),
                __import__("lightgbm").log_evaluation(0)
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

    # ── Save best model (highest/lowest score depending on direction) ──
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

    # ── Update model registry in state ───────────────────────────
    model_registry = list(state.get("model_registry") or [])
    model_registry.append({
        "model_path":  model_path,
        "model_type":  "lightgbm_v0",
        "cv_mean":     cv_mean,
        "cv_std":      cv_std,
        "scorer_name": contract.scorer_name,
        "data_hash":   state.get("data_hash"),
        "fold_scores": fold_scores,
    })

    # ── Update cost tracker ───────────────────────────────────────
    cost_tracker = dict(state["cost_tracker"])
    cost_tracker["llm_calls"] += 0  # v0 uses no LLM calls

    print(f"[MLOptimizer] Complete.")

    return {
        **state,
        "cv_scores":            fold_scores,
        "cv_mean":              cv_mean,
        "model_registry":       model_registry,
        "metric_contract":      metrics,
        "oof_predictions_path": oof_path,
        "cost_tracker":         cost_tracker,
    }

Task 3 — Write Contract Test (Immutable From Today)
python# tests/contracts/test_ml_optimizer_contract.py
# ─────────────────────────────────────────────────────────────────
# Written: Day 4
# Status:  IMMUTABLE — never edit this file after today
#
# CONTRACT: run_ml_optimizer()
#   INPUT:   state["clean_data_path"] — must exist
#            state["schema_path"]     — must exist
#   OUTPUT:  outputs/{session_id}/best_model.pkl — must exist
#            outputs/{session_id}/metrics.json   — must have
#              cv_mean (float), cv_std (float), fold_scores (list)
#   STATE:   model_registry — list, at least 1 entry after run
#            cv_mean        — float, > 0
#            cv_scores      — list of length n_folds
#            cost_tracker   — not None
#   NEVER:   optimise toward forbidden metrics
#            put raw model object in state (only file pointer)
# ─────────────────────────────────────────────────────────────────
import pytest
import os
import json
import pickle
import numpy as np
from core.state import initial_state
from core.metric_contract import FORBIDDEN_METRICS
from agents.data_engineer import run_data_engineer
from agents.ml_optimizer import run_ml_optimizer

FIXTURE_CSV = "tests/fixtures/tiny_train.csv"


@pytest.fixture(scope="module")
def optimized_state():
    """Run Data Engineer → ML Optimizer pipeline once for all tests."""
    state = initial_state(
        competition="test-titanic",
        data_path=FIXTURE_CSV,
        budget_usd=2.0
    )
    state = run_data_engineer(state)
    state = run_ml_optimizer(state)
    return state


class TestMLOptimizerContract:

    def test_runs_without_error(self, optimized_state):
        assert optimized_state is not None

    def test_best_model_pkl_exists(self, optimized_state):
        assert os.path.exists(optimized_state["model_registry"][0]["model_path"]), \
            "best_model.pkl must exist on disk"

    def test_model_is_loadable(self, optimized_state):
        model_path = optimized_state["model_registry"][0]["model_path"]
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        assert hasattr(model, "predict"), "Loaded object must have predict()"

    def test_metrics_json_exists(self, optimized_state):
        session_id   = optimized_state["session_id"]
        metrics_path = f"outputs/{session_id}/metrics.json"
        assert os.path.exists(metrics_path), "metrics.json must exist"

    def test_metrics_has_cv_mean(self, optimized_state):
        session_id   = optimized_state["session_id"]
        metrics      = json.load(open(f"outputs/{session_id}/metrics.json"))
        assert "cv_mean" in metrics
        assert isinstance(metrics["cv_mean"], float)

    def test_metrics_has_cv_std(self, optimized_state):
        session_id = optimized_state["session_id"]
        metrics    = json.load(open(f"outputs/{session_id}/metrics.json"))
        assert "cv_std" in metrics
        assert isinstance(metrics["cv_std"], float)

    def test_metrics_has_fold_scores(self, optimized_state):
        session_id = optimized_state["session_id"]
        metrics    = json.load(open(f"outputs/{session_id}/metrics.json"))
        assert "fold_scores" in metrics
        assert isinstance(metrics["fold_scores"], list)
        assert len(metrics["fold_scores"]) == 5

    def test_cv_mean_is_positive(self, optimized_state):
        assert optimized_state["cv_mean"] > 0, \
            "CV mean must be positive"

    def test_cv_mean_is_reasonable(self, optimized_state):
        # AUC should be above 0.5 (random baseline) on any real data
        assert optimized_state["cv_mean"] > 0.5, \
            f"CV mean {optimized_state['cv_mean']} is below random baseline (0.5)"

    def test_cv_scores_length_matches_folds(self, optimized_state):
        assert len(optimized_state["cv_scores"]) == 5

    def test_model_registry_updated(self, optimized_state):
        assert optimized_state["model_registry"] is not None
        assert len(optimized_state["model_registry"]) >= 1

    def test_model_registry_entry_has_required_fields(self, optimized_state):
        entry = optimized_state["model_registry"][0]
        for field in ["model_path", "model_type", "cv_mean", "scorer_name"]:
            assert field in entry, f"model_registry entry missing '{field}'"

    def test_model_path_is_string_not_object(self, optimized_state):
        entry = optimized_state["model_registry"][0]
        assert isinstance(entry["model_path"], str), \
            "model_path must be a str pointer — never a model object"

    def test_no_model_object_in_state(self, optimized_state):
        import lightgbm as lgb
        for key, value in optimized_state.items():
            assert not isinstance(value, (lgb.LGBMClassifier, lgb.LGBMRegressor)), \
                f"Model object found in state['{key}'] — only file pointers allowed"

    def test_oof_predictions_path_exists(self, optimized_state):
        assert optimized_state.get("oof_predictions_path") is not None
        assert os.path.exists(optimized_state["oof_predictions_path"])

    def test_oof_predictions_loadable(self, optimized_state):
        oof = np.load(optimized_state["oof_predictions_path"])
        assert len(oof) > 0

    def test_never_optimises_forbidden_metrics(self, optimized_state):
        session_id = optimized_state["session_id"]
        metrics    = json.load(open(f"outputs/{session_id}/metrics.json"))
        scorer     = metrics["scorer_name"]
        assert scorer not in FORBIDDEN_METRICS, \
            f"Scorer '{scorer}' is in FORBIDDEN_METRICS — never optimise toward this"

    def test_requires_clean_data_path(self):
        state = initial_state("test", "tests/fixtures/tiny_train.csv")
        state = {**state, "clean_data_path": None}
        with pytest.raises((ValueError, TypeError)):
            run_ml_optimizer(state)

End of Day 4 Checklist
bash# 1. Run today's contract tests
pytest tests/contracts/test_ml_optimizer_contract.py -v
# All tests must be green

# 2. Run ALL contracts together (Days 2 + 3 + 4)
pytest tests/contracts/ -v
# Everything still green — no regressions introduced today

# 3. Quick real-data smoke test
python -c "
from core.state import initial_state
from agents.data_engineer import run_data_engineer
from agents.ml_optimizer import run_ml_optimizer

state = initial_state('spaceship-titanic', 'data/spaceship_titanic/train.csv')
state = run_data_engineer(state)
state = run_ml_optimizer(state)
print(f'CV {state[\"cv_mean\"]:.4f}')
print(f'Model: {state[\"model_registry\"][0][\"model_path\"]}')
# CV should be around 0.78-0.82 on Spaceship Titanic with default LightGBM
"

# 4. Commit
git add .
git commit -m "Day 4: metric_contract + ml_optimizer v0 + contract test — all tests pass"
git push origin phase-1
Expected CV score today: ~0.78–0.82 on Spaceship Titanic with default LightGBM. If it's below 0.70 something is wrong in the data pipeline from Day 3 — check cleaned.parquet for null contamination first.

Day 5 Preview
Tomorrow's ONE thing: agents/semantic_router.py v0 — simple linear routing only (Supervisor → Data Engineer → Optimizer → Submit, no DAG yet) and core/professor.py — the LangGraph graph that wires all three nodes together for the first time. First time python main.py run does something real.