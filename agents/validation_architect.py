# agents/validation_architect.py

import os
import json
import logging
import polars as pl
from pathlib import Path
from typing import Optional, Dict, Any, List
from core.state import ProfessorState
from core.metric_contract import build_metric_contract, save_contract
from tools.data_tools import read_json
from core.lineage import log_event
from guards.agent_retry import with_agent_retry
from tools.performance_monitor import timed_node

logger = logging.getLogger(__name__)

AGENT_NAME = "validation_architect"

# ── Rules & Patterns ─────────────────────────────────────────────────────────

_CV_STRATEGY_RULES = {
    "group":       "GroupKFold",
    "timeseries":  "TimeSeriesSplit",
    "stratified":  "StratifiedKFold",
    "kfold":       "KFold",
}

# ── Internal Helpers ─────────────────────────────────────────────────────────

def _detect_group_column(schema: dict) -> Optional[str]:
    """Return the name of a group/ID column if one exists in schema."""
    group_keywords = ["group", "patient", "user_id", "customer_id",
                      "store_id", "site_id", "subject", "household"]
    for col in schema.get("columns", []):
        col_name = str(col.get("name", "")) if isinstance(col, dict) else str(col)
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

    if "Int" in dtype or "UInt" in dtype:
        return "multiclass"
    if "Float" in dtype:
        return "continuous"
    return "binary"

def _detect_cv_mismatch_risk(
    cv_type: str,
    datetime_col: Optional[str],
    group_col: Optional[str],
    brief: dict,
) -> Optional[str]:
    """Flag patterns that produce inflated CV scores."""
    if datetime_col and cv_type not in ("TimeSeriesSplit"):
        return f"Datetime column '{datetime_col}' detected with {cv_type}. Temporal leakage risk."
    if group_col and cv_type not in ("GroupKFold"):
        return f"Group column '{group_col}' detected with {cv_type}. Data dependency leak risk."
    return None

# ── Agent Node ───────────────────────────────────────────────────────────────

@timed_node
@with_agent_retry(AGENT_NAME)
def run_validation_architect(state: ProfessorState) -> ProfessorState:
    """
    LangGraph node: Validation Architect — Selects CV strategy and Metric Contract.
    """
    session_id = state.get("session_id", "default")
    output_dir = Path(f"outputs/{session_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"[{AGENT_NAME}] Starting — session: {session_id}")

    # 1. Load schema
    schema_path = state.get("schema_path", "")
    if not schema_path or not os.path.exists(schema_path):
        raise ValueError(f"[{AGENT_NAME}] schema_path missing or invalid.")
    schema = read_json(schema_path)

    # 2. Load competition brief (optional)
    brief = state.get("competition_brief", {})

    # 3. Determine target column
    target_col = state.get("target_col") or schema.get("target_col")
    if not target_col:
        raise ValueError(f"[{AGENT_NAME}] Cannot determine target column.")

    # 4. CV strategy detection
    group_col    = _detect_group_column(schema)
    datetime_col = _detect_datetime_column(schema)
    target_type  = _detect_target_type(schema, target_col)

    if state.get("task_type") == "timeseries":
        cv_type  = "TimeSeriesSplit"
        n_splits = 5
    elif group_col:
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

    # 5. CV/LB mismatch detection
    mismatch_reason = _detect_cv_mismatch_risk(cv_type, datetime_col, group_col, brief)
    
    # 6. Determine metric
    scorer_name = str(brief.get("evaluation_metric", "")).lower().strip()
    if not scorer_name:
        if target_type == "binary": scorer_name = "auc"
        elif target_type == "multiclass": scorer_name = "f1_weighted"
        else: scorer_name = "rmse"

    task_type = state.get("task_type", "unknown")
    if task_type == "unknown":
        task_type = "classification" if target_type in ("binary", "multiclass") else "regression"

    # 7. Build and save MetricContract
    contract = build_metric_contract(scorer_name, task_type, state.get("competition_name", ""))
    contract_path = output_dir / "metric_contract.json"
    save_contract(contract, str(contract_path))

    # 8. Build and save validation strategy
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
        "mismatch_risk":    mismatch_reason,
        "hitl_required":    mismatch_reason is not None,
    }
    strategy_path = output_dir / "validation_strategy.json"
    with open(strategy_path, "w") as f:
        json.dump(validation_strategy, f, indent=2)

    # 9. Update State
    # Only pass serializable parts of contract to validated_update
    contract_dict = {
        "scorer_name":       contract.scorer_name,
        "direction":         contract.direction,
        "requires_proba":    contract.requires_proba,
        "task_type":         contract.task_type,
        "forbidden_metrics": contract.forbidden_metrics,
        "locked":            contract.locked
    }

    updates = {
        "validation_strategy":      validation_strategy,
        "validation_strategy_path": str(strategy_path),
        "metric_contract":          contract_dict,
        "task_type":                task_type,
        "hitl_required":            mismatch_reason is not None,
    }
    
    if mismatch_reason:
        updates["hitl_reason"] = f"CV/LB mismatch risk: {mismatch_reason}"

    log_event(
        session_id=session_id,
        agent=AGENT_NAME,
        action="strategy_decided",
        keys_read=["schema_path"],
        keys_written=["validation_strategy", "metric_contract"],
        values_changed={"cv_type": cv_type, "scorer_name": scorer_name}
    )

    return ProfessorState.validated_update(state, AGENT_NAME, updates)

# To handle asdict for MetricContract
from dataclasses import asdict
