# guards/circuit_breaker.py

import os
import json
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Dict, Any, Union
from core.state import ProfessorState
from core.lineage import log_event

logger = logging.getLogger(__name__)


class EscalationLevel(str, Enum):
    MICRO   = "micro"    # patch the failing node only
    MACRO   = "macro"    # rewrite the DAG
    HITL    = "hitl"     # pause, save state, alert human
    TRIAGE  = "triage"   # budget/time exhausted, protect rank


ERROR_CLASS_MAP = {
    "KeyError":            "data_quality",
    "ValueError":          "data_quality",
    "AttributeError":      "data_quality",
    "MemoryError":         "memory",
    "RuntimeError":        "model_failure",
    "optuna":              "model_failure",   # match in repr(error)
    "TimeoutError":        "api_timeout",
    "httpx":               "api_timeout",     # match in repr(type(error))
    "groq":                "api_timeout",
}

INTERVENTION_TEMPLATES = {
    "data_quality": [
        {"id": 1, "label": "Skip validation, proceed with raw features.",
         "action_type": "AUTO", "risk": "LOW",
         "description": "Bypasses all type-checking and missing-value validation. "
                        "Proceeds directly with whatever columns exist. May produce worse CV.",
         "code_hint": None},
        {"id": 2, "label": "Drop columns with > 30% nulls, fill rest with median.",
         "action_type": "AUTO", "risk": "LOW",
         "description": "Conservative imputation. Loses high-null features entirely. "
                        "Safe starting point for any tabular competition.",
         "code_hint": None},
        {"id": 3, "label": "Inspect data manually, then rerun.",
         "action_type": "MANUAL", "risk": "LOW",
         "description": "Open the raw CSV and check column names, dtypes, and sample rows. "
                        "Common cause: target column name differs from expected.",
         "code_hint": "import polars as pl; print(pl.read_csv('data/train.csv').head(5))"},
    ],
    "model_failure": [
        {"id": 1, "label": "Reduce LightGBM to 100 trees, disable early stopping.",
         "action_type": "AUTO", "risk": "LOW",
         "description": "Minimal model to prove the pipeline runs end-to-end. "
                        "CV score will be suboptimal but submission will complete.",
         "code_hint": None},
        {"id": 2, "label": "Switch to LogisticRegression fallback model.",
         "action_type": "AUTO", "risk": "MEDIUM",
         "description": "Replaces all boosting models with sklearn LogisticRegression. "
                        "Will underfit complex competitions but always converges.",
         "code_hint": None},
        {"id": 3, "label": "Check training data shape and feature count.",
         "action_type": "MANUAL", "risk": "LOW",
         "description": "Log X_train.shape before the failing model call to rule out "
                        "zero-feature or zero-row edge cases.",
         "code_hint": "print(f'X_train shape: {X_train.shape}, y shape: {y_train.shape}')"},
    ],
    "memory": [
        {"id": 1, "label": "Sample training data to 50% and retry.",
         "action_type": "AUTO", "risk": "MEDIUM",
         "description": "Randomly samples 50% of training rows. CV score will degrade "
                        "slightly. Prevents OOM for most datasets up to 2GB.",
         "code_hint": None},
        {"id": 2, "label": "Switch to n_jobs=1, reduce n_estimators to 200.",
         "action_type": "AUTO", "risk": "LOW",
         "description": "Forces single-threaded training and limits tree count. "
                        "Halves peak memory usage at ~15% CV score cost.",
         "code_hint": None},
        {"id": 3, "label": "Free memory manually, then rerun.",
         "action_type": "MANUAL", "risk": "LOW",
         "description": "Close all other applications. Restart the Python process to "
                        "clear fragmented memory before retrying.",
         "code_hint": "import gc; gc.collect()  # then restart: python -m professor run"},
    ],
    "api_timeout": [
        {"id": 1, "label": "Retry with 2x timeout and exponential backoff.",
         "action_type": "AUTO", "risk": "LOW",
         "description": "Doubles the API timeout and retries with 2s/4s/8s delays. "
                        "Handles transient Groq rate limits and network blips.",
         "code_hint": None},
        {"id": 2, "label": "Switch to local LLM (ollama/llama3) for this session.",
         "action_type": "AUTO", "risk": "MEDIUM",
         "description": "Falls back to a local model. Slower but zero API cost and "
                        "no rate limits. Requires ollama running on localhost:11434.",
         "code_hint": "ollama pull llama3; export PROFESSOR_LLM_PROVIDER=local"},
        {"id": 3, "label": "Check API status and keys, then resume.",
         "action_type": "MANUAL", "risk": "LOW",
         "description": "Verify GROQ_API_KEY in .env is valid and not expired. "
                        "Check status.groq.com for incidents.",
         "code_hint": "cat .env | grep GROQ_API_KEY"},
    ],
    "unknown": [
        {"id": 1, "label": "Retry this agent with extra debug logging.",
         "action_type": "AUTO", "risk": "LOW",
         "description": "Re-runs the failed agent with LOG_LEVEL=DEBUG to capture "
                        "the full traceback and intermediate state.",
         "code_hint": None},
        {"id": 2, "label": "Skip this agent and continue with defaults.",
         "action_type": "AUTO", "risk": "MEDIUM",
         "description": "Bypasses the failing agent entirely. Pipeline continues "
                        "with whatever state exists before the failure.",
         "code_hint": None},
        {"id": 3, "label": "Inspect session state and rerun manually.",
         "action_type": "MANUAL", "risk": "LOW",
         "description": "Load the Redis checkpoint and inspect state before the failure.",
         "code_hint": "professor inspect --session <id>"},
    ],
}

def _get_as_dict(state: Union[ProfessorState, Dict[str, Any]]) -> Dict[str, Any]:
    return state.model_dump() if isinstance(state, ProfessorState) else dict(state)

def _get_as_object(data: Dict[str, Any]) -> ProfessorState:
    valid_keys = ProfessorState.model_fields.keys()
    filtered = {k: v for k, v in data.items() if k in valid_keys}
    return ProfessorState(**filtered)

def get_escalation_level(state: ProfessorState) -> EscalationLevel:
    budget_remaining = state.get("budget_remaining_usd", 2.0)
    budget_limit     = state.get("budget_limit_usd", 2.0)
    
    if budget_limit > 0 and budget_remaining <= budget_limit * 0.05:
        return EscalationLevel.TRIAGE

    failure_count = state.get("current_node_failure_count", 0)
    if failure_count >= 3: return EscalationLevel.HITL
    if failure_count == 2: return EscalationLevel.MACRO
    return EscalationLevel.MICRO


def handle_escalation(
    state: ProfessorState,
    level: EscalationLevel,
    agent_name: str,
    error: Exception,
    traceback_str: str,
) -> ProfessorState:
    logger.error(f"[CircuitBreaker] {agent_name} escalating to {level.value}. Error: {error}")
    
    # Audit log
    try:
        log_event(
            session_id=state.get("session_id", "unknown"),
            agent="circuit_breaker",
            action=f"escalation_{level.value}",
            values_changed={"agent": agent_name, "error": str(error)}
        )
    except: pass

    state_dict = _get_as_dict(state)

    if level == EscalationLevel.MICRO:
        error_context = list(state_dict.get("error_context", []))
        error_context.append({
            "agent":     agent_name,
            "attempt":   state_dict.get("current_node_failure_count", 1),
            "error":     str(error),
            "traceback": traceback_str,
        })
        state_dict.update({
            "error_context": error_context,
            "current_node_failure_count": state_dict.get("current_node_failure_count", 0) + 1
        })

    elif level == EscalationLevel.MACRO:
        state_dict.update({
            "dag_version": state_dict.get("dag_version", 1) + 1,
            "macro_replan_requested": True,
            "macro_replan_reason": f"{agent_name} failed twice: {error}",
            "current_node_failure_count": state_dict.get("current_node_failure_count", 0) + 1
        })

    elif level == EscalationLevel.HITL:
        state_dict.update({
            "hitl_required": True,
            "pipeline_halted": True,
            "hitl_reason": f"Pipeline paused after {agent_name} failed 3 times: {error}"
        })

    elif level == EscalationLevel.TRIAGE:
        state_dict.update({
            "triage_mode": True,
            "pipeline_halted": True,
            "triage_reason": "Budget/time exhausted."
        })

    return _get_as_object(state_dict)

def reset_failure_count(state: ProfessorState) -> ProfessorState:
    """Call this on successful node completion."""
    state_dict = _get_as_dict(state)
    state_dict.update({"current_node_failure_count": 0, "error_context": []})
    return _get_as_object(state_dict)

def resume_from_checkpoint(session_id: str, intervention_id: int) -> ProfessorState:
    # Logic remains same but returns _get_as_object(state_dict)
    # Stub for now
    return ProfessorState(session_id=session_id)
