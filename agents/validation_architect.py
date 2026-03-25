# agents/validation_architect.py

import os
import json
import polars as pl
from typing import Optional
from core.state import ProfessorState
from core.metric_contract import build_metric_contract, save_contract
from tools.data_tools import read_json
from core.lineage import log_event
from guards.agent_retry import with_agent_retry
from tools.performance_monitor import timed_node


_CV_STRATEGY_RULES = {
    "group":       "GroupKFold",
    "timeseries":  "TimeSeriesSplit",
    "stratified":  "StratifiedKFold",
    "kfold":       "KFold",
}

_DATETIME_DTYPES = {
    "Date", "Datetime", "Time", "Duration",
    pl.Date, pl.Datetime, pl.Time, pl.Duration,
}


def _detect_group_column(schema: dict) -> Optional[str]:
    """Return the name of a group/ID column if one exists in schema."""
    group_keywords = ["group", "patient", "user_id", "customer_id",
                      "store_id", "site_id", "subject", "household"]
    for col in schema.get("columns", []):
        try:
            col_name = str(col.get("name", "")) if isinstance(col, dict) else str(col)
        except Exception:
            continue
        if any(kw in col_name.lower() for kw in group_keywords):
            return col_name
    return None


def _detect_datetime_column(schema: dict) -> Optional[str]:
    """Return the name of a datetime column if one exists."""
    time_keywords = ["date", "time", "timestamp", "year", "month", "week"]
    types = schema.get("types", {})
    for col, dtype in types.items():
        if any(kw in col.lower() for kw in time_keywords):
            return col
        if str(dtype) in {"Date", "Datetime", "Time"}:
            return col
    return None


def _detect_target_type(schema: dict, target_col: str) -> str:
    """Returns 'binary', 'multiclass', or 'continuous'."""
    types = schema.get("types", {})
    dtype = str(types.get(target_col, ""))

    if dtype in {"Boolean", "bool"}:
        return "binary"

    n_unique = schema.get("cardinality", {}).get(target_col)
    if n_unique is not None:
        if n_unique == 2:
            return "binary"
        if 2 < n_unique <= 20:
            return "multiclass"
        return "continuous"

    # Fall back to dtype heuristics
    if "Int" in dtype or "UInt" in dtype:
        return "multiclass"
    if "Float" in dtype:
        return "continuous"
    return "binary"  # safe default for unknown


def _detect_cv_mismatch_risk(
    cv_type: str,
    datetime_col: Optional[str],
    group_col: Optional[str],
    brief: dict,
) -> Optional[str]:
    """
    Returns a mismatch reason string if risk is detected, else None.
    These are the patterns that produce inflated CV scores that collapse on LB.
    """
    if datetime_col:
        return (
            f"Datetime column '{datetime_col}' detected. "
            f"This splits time-ordered data randomly, leaking future information into training folds "
            f"if standard KFold is used. Mismatch risk flagged."
        )

    if group_col:
        return (
            f"Group column '{group_col}' detected. "
            f"Rows from the same group will appear in both train and validation, inflating CV "
            f"if GroupKFold is not properly handled."
        )

    # Check if brief explicitly mentions LB shakeup risk
    if brief.get("known_pitfalls"):
        pitfalls = str(brief["known_pitfalls"]).lower()
        if "shake" in pitfalls or "lb gap" in pitfalls or "public lb" in pitfalls:
            return (
                "Competition brief flags known public/private LB gap risk. "
                "Validate CV strategy carefully before trusting public scores."
            )

    return None


@timed_node
@with_agent_retry("ValidationArchitect")
def run_validation_architect(state: ProfessorState) -> ProfessorState:
    """
    LangGraph node: Validation Architect.

    Reads:  state["schema_path"]          — schema.json from Data Engineer
            state["competition_brief_path"] — competition_brief.json from Intel Agent (optional)
    Writes: validation_strategy.json      — cv_type, n_splits, group_col, datetime_col
            metric_contract.json          — scorer_fn, direction, forbidden_metrics
            state["validation_strategy"]  — dict
            state["hitl_required"]        — True if CV/LB mismatch detected
    """
    session_id = state["session_id"]
    output_dir = f"outputs/{session_id}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"[ValidationArchitect] Starting — session: {session_id}")

    # ── Load schema ────────────────────────────────────────────────────────────
    if not state.get("schema_path") or not os.path.exists(state["schema_path"]):
        raise ValueError(
            "[ValidationArchitect] schema_path missing or file not found. "
            "Run Data Engineer first."
        )
    schema = read_json(state["schema_path"])

    # ── Load competition brief (optional — may not exist in Phase 1) ──────────
    brief = {}
    brief_path = state.get("competition_brief_path", "")
    if brief_path and os.path.exists(brief_path):
        brief = read_json(brief_path)
        print(f"[ValidationArchitect] Competition brief loaded: {brief_path}")
    else:
        print("[ValidationArchitect] No competition brief — using schema only")

    # ── Determine target column ────────────────────────────────────────────────
    target_col = state.get("target_col") or schema.get("target_col")
    if not target_col:
        # Fall back: last column in schema
        cols = schema.get("columns", [])
        target_col = cols[-1] if cols else None
    if not target_col:
        raise ValueError("[ValidationArchitect] Cannot determine target column.")

    # ── CV strategy detection ──────────────────────────────────────────────────
    group_col    = _detect_group_column(schema)
    datetime_col = _detect_datetime_column(schema)
    target_type  = _detect_target_type(schema, target_col)

    if group_col:
        cv_type  = "GroupKFold"
        n_splits = 5
    elif datetime_col:
        cv_type  = "TimeSeriesSplit"
        n_splits = 5
    elif target_type in ("binary", "multiclass"):
        cv_type  = "StratifiedKFold"
        n_splits = 5
    else:
        cv_type  = "KFold"
        n_splits = 5

    print(f"[ValidationArchitect] CV strategy: {cv_type}(n_splits={n_splits})")
    if group_col:
        print(f"[ValidationArchitect] Group column: {group_col}")
    if datetime_col:
        print(f"[ValidationArchitect] Datetime column: {datetime_col}")

    # ── CV/LB mismatch detection ───────────────────────────────────────────────
    mismatch_reason = _detect_cv_mismatch_risk(cv_type, datetime_col, group_col, brief)
    if mismatch_reason:
        print(f"[ValidationArchitect] CV/LB MISMATCH RISK: {mismatch_reason}")
        validation_strategy = {
            "cv_type":          cv_type,
            "n_splits":         n_splits,
            "group_col":        group_col,
            "datetime_col":     datetime_col,
            "target_col":       target_col,
            "target_type":      target_type,
            "cv_strategy_hint": brief.get("cv_strategy_hint", ""),
            "mismatch_risk":    mismatch_reason,
            "hitl_required":    True,
        }
        strategy_path = f"{output_dir}/validation_strategy.json"
        with open(strategy_path, "w") as f:
            json.dump(validation_strategy, f, indent=2)

        log_event(
            session_id=session_id,
            agent="validation_architect",
            action="halted_cv_mismatch",
            keys_read=["schema_path"],
            keys_written=["validation_strategy"],
            values_changed={"mismatch_reason": mismatch_reason},
        )

        return {
            **state,
            "validation_strategy":      validation_strategy,
            "validation_strategy_path": strategy_path,
            "hitl_required":            True,
            "hitl_reason":              f"CV/LB mismatch risk: {mismatch_reason}",
        }

    # ── Determine metric ───────────────────────────────────────────────────────
    # Priority: competition_brief > task_type in state > target_type heuristic
    scorer_name = brief.get("evaluation_metric", "").lower().strip()
    task_type   = state.get("task_type", "unknown")

    if not scorer_name:
        # Heuristic fallback
        if target_type == "binary":
            scorer_name = "auc"
        elif target_type == "multiclass":
            scorer_name = "f1_weighted"
        else:
            scorer_name = "rmse"

    if task_type == "unknown":
        task_type = "classification" if target_type in ("binary", "multiclass") else "regression"

    print(f"[ValidationArchitect] Metric: {scorer_name} | Task: {task_type}")

    # ── Build and save MetricContract ──────────────────────────────────────────
    contract      = build_metric_contract(scorer_name, task_type, state["competition_name"])
    contract_path = f"{output_dir}/metric_contract.json"
    save_contract(contract, contract_path)
    print(f"[ValidationArchitect] MetricContract saved: {contract_path}")

    # ── Build and save validation strategy ────────────────────────────────────
    validation_strategy = {
        "cv_type":          cv_type,
        "n_splits":         n_splits,
        "group_col":        group_col,
        "datetime_col":     datetime_col,
        "target_col":       target_col,
        "target_type":      target_type,
        "scorer_name":      scorer_name,
        "task_type":        task_type,
        "cv_strategy_hint": brief.get("cv_strategy_hint", ""),
        "mismatch_risk":    None,
        "hitl_required":    False,
    }
    strategy_path = f"{output_dir}/validation_strategy.json"
    with open(strategy_path, "w") as f:
        json.dump(validation_strategy, f, indent=2)

    log_event(
        session_id=session_id,
        agent="validation_architect",
        action="strategy_decided",
        keys_read=["schema_path"],
        keys_written=["validation_strategy", "metric_contract"],
        values_changed={
            "cv_type": cv_type,
            "scorer_name": scorer_name,
            "task_type": task_type,
        },
    )

    print(f"[ValidationArchitect] Complete.")

    return {
        **state,
        "validation_strategy":      validation_strategy,
        "validation_strategy_path": strategy_path,
        "metric_contract_path":     contract_path,
        "task_type":                task_type,
        "hitl_required":            False,
    }
