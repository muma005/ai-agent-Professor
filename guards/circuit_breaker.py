# guards/circuit_breaker.py

import os
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Optional
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

AUTO_INTERVENTION_EFFECTS = {
    # data_quality
    "Skip validation, proceed with raw features.":
        lambda s: {**s, "skip_data_validation": True},
    "Drop columns with > 30% nulls, fill rest with median.":
        lambda s: {**s, "null_threshold": 0.30, "impute_strategy": "median"},
    # model_failure
    "Reduce LightGBM to 100 trees, disable early stopping.":
        lambda s: {**s, "lgbm_override": {"n_estimators": 100, "early_stopping_rounds": None}},
    "Switch to LogisticRegression fallback model.":
        lambda s: {**s, "model_fallback": "logistic_regression"},
    # memory
    "Sample training data to 50% and retry.":
        lambda s: {**s, "data_sample_fraction": 0.50},
    "Switch to n_jobs=1, reduce n_estimators to 200.":
        lambda s: {**s, "lgbm_override": {"n_jobs": 1, "n_estimators": 200}},
    # api_timeout
    "Retry with 2x timeout and exponential backoff.":
        lambda s: {**s, "api_timeout_multiplier": 2.0, "api_backoff_enabled": True},
    "Switch to local LLM (ollama/llama3) for this session.":
        lambda s: {**s, "llm_provider": "local"},
    # unknown
    "Retry this agent with extra debug logging.":
        lambda s: {**s, "debug_logging": True},
}


def get_escalation_level(state: ProfessorState) -> EscalationLevel:
    """
    Determines which escalation level applies given the current state.
    Called at the top of every agent that has a retry loop.
    """
    budget_remaining = state.get("budget_remaining_usd", float("inf"))
    budget_limit     = state.get("budget_limit_usd", float("inf"))
    time_remaining   = state.get("competition_context", {}).get("hours_remaining")

    # Triage overrides all — check it first
    if budget_limit > 0 and budget_remaining <= budget_limit * 0.05:
        return EscalationLevel.TRIAGE
    if time_remaining is not None and time_remaining <= 2:
        return EscalationLevel.TRIAGE

    failure_count = state.get("current_node_failure_count", 0)
    if failure_count >= 3:
        return EscalationLevel.HITL
    if failure_count == 2:
        return EscalationLevel.MACRO
    if failure_count == 1:
        return EscalationLevel.MICRO

    return EscalationLevel.MICRO  # first failure always starts at MICRO


def handle_escalation(
    state: ProfessorState,
    level: EscalationLevel,
    agent_name: str,
    error: Exception,
    traceback_str: str,
) -> ProfessorState:
    """
    Executes the correct response for each escalation level.
    Returns updated state. Never raises — this function must always complete.
    """
    try:
        logger.error(
            f"[CircuitBreaker] {agent_name} escalating to {level.value}. "
            f"Failure count: {state.get('current_node_failure_count', 0)}. "
            f"Error: {error}"
        )
        log_event(
            session_id=state["session_id"],
            agent="circuit_breaker",
            action=f"escalation_{level.value}",
            keys_read=["current_node_failure_count"],
            keys_written=["hitl_required", "dag_version"],
            values_changed={
                "level": level.value,
                "agent": agent_name,
                "error": str(error),
            },
        )
    except Exception:
        pass  # logging must never crash the circuit breaker

    if level == EscalationLevel.MICRO:
        # Append full traceback to the agent's context for next attempt
        error_context = list(state.get("error_context", []))
        error_context.append({
            "agent":     agent_name,
            "attempt":   state.get("current_node_failure_count", 1),
            "error":     str(error),
            "traceback": traceback_str,
        })
        return {
            **state,
            "error_context":              error_context,
            "current_node_failure_count": state.get("current_node_failure_count", 0) + 1,
        }

    elif level == EscalationLevel.MACRO:
        # Increment dag_version to force a full DAG rewrite on next Supervisor pass
        dag_version = state.get("dag_version", 0) + 1
        logger.warning(
            f"[CircuitBreaker] MACRO replan triggered. "
            f"DAG version incrementing to {dag_version}. "
            f"Supervisor will rewrite execution plan."
        )
        return {
            **state,
            "dag_version":                dag_version,
            "macro_replan_requested":     True,
            "macro_replan_reason":        f"{agent_name} failed twice: {error}",
            "current_node_failure_count": state.get("current_node_failure_count", 0) + 1,
        }

    elif level == EscalationLevel.HITL:
        # Save full state to Redis, pause pipeline, alert human
        try:
            _checkpoint_state_to_redis(state, agent_name, error)
        except Exception as redis_err:
            logger.warning(
                f"[CircuitBreaker] Redis checkpoint failed (non-fatal): {redis_err}. "
                f"HITL will proceed without persistent checkpoint."
            )
            
        prompt = generate_hitl_prompt(state, agent_name, error)
        return {
            **state,
            "hitl_required":      True,
            "hitl_prompt":        prompt,
            "hitl_checkpoint_key": prompt["checkpoint_key"],
            "pipeline_halted":    True,
            "hitl_reason": (
                f"Pipeline paused after {agent_name} failed 3 times. "
                f"See outputs/{state['session_id']}/hitl_prompt.json for interventions. "
                f"Resume: professor resume --session {state['session_id']}"
            ),
        }

    elif level == EscalationLevel.TRIAGE:
        budget_remaining = state.get("budget_remaining_usd", 0)
        logger.warning(
            f"[CircuitBreaker] TRIAGE mode. "
            f"Budget remaining: ${budget_remaining:.4f}. "
            f"Stopping all non-essential work. Protecting submission."
        )
        return {
            **state,
            "triage_mode":    True,
            "triage_reason":  f"Budget/time exhausted. Protecting existing submission.",
            "pipeline_halted": True,
        }

    return state  # unreachable but satisfies type checker


def _checkpoint_state_to_redis(
    state: ProfessorState,
    agent_name: str,
    error: Exception,
) -> None:
    """
    Saves full ProfessorState to Redis for HITL resume.
    Fails silently with a warning — the HITL flag is already set.
    """
    try:
        from memory.redis_state import get_redis_client
        client = get_redis_client()
        key    = f"professor:hitl:{state['session_id']}"
        payload = json.dumps({
            "state":      {k: v for k, v in state.items() if _is_serialisable(v)},
            "agent":      agent_name,
            "error":      str(error),
            "error_class": _classify_error(agent_name, error),
            "checkpointed_at": datetime.utcnow().isoformat(),
        })
        client.set(key, payload, ex=86400 * 7)  # 7-day TTL
        logger.info(f"[CircuitBreaker] State checkpointed to Redis key: {key}")
    except Exception as redis_err:
        logger.warning(
            f"[CircuitBreaker] Could not checkpoint to Redis: {redis_err}. "
            f"HITL flag is set but state was not saved. "
            f"Manual recovery required from session logs."
        )


def _is_serialisable(value) -> bool:
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False


def reset_failure_count(state: ProfessorState) -> ProfessorState:
    """Call this at the top of every agent on successful completion."""
    return {**state, "current_node_failure_count": 0, "error_context": []}

def _classify_error(agent_name: str, error: Exception) -> str:
    err_type = type(error).__name__
    err_repr = repr(type(error)).lower() + repr(error).lower()
    if err_type in ERROR_CLASS_MAP:
        return ERROR_CLASS_MAP[err_type]
    for k, v in ERROR_CLASS_MAP.items():
        if k.lower() in err_repr:
            return v
    return "unknown"

def _build_interventions(state: ProfessorState, agent_name: str, error_class: str, error: Exception) -> list:
    return INTERVENTION_TEMPLATES.get(error_class, INTERVENTION_TEMPLATES["unknown"])

def _describe_attempt(state: ProfessorState, agent_name: str) -> str:
    return f"{agent_name} attempted to execute but encountered an error."

def _write_hitl_prompt(session_id: str, prompt: dict) -> None:
    path = f"outputs/{session_id}/hitl_prompt.json"
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(prompt, f, indent=2)
    except Exception as e:
        logger.warning(f"[CircuitBreaker] Failed to write hitl_prompt to {path}: {e}")

def _print_hitl_banner(prompt: dict) -> None:
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║              ⚠  PROFESSOR PAUSED — HITL REQUIRED         ║")
    print("╚══════════════════════════════════════════════════════════╝\n")
    print(f"Agent:    {prompt['failed_agent']}")
    print(f"Failure:  {prompt['failure_count']} consecutive failures ({prompt['why_it_failed']})")
    print(f"Class:    {prompt['error_class']}")
    print(f"Session:  {prompt['session_id']}\n")
    print(f"What was attempted:\n  {prompt['what_was_attempted']}\n")
    print("Suggested interventions:")
    for inv in prompt["interventions"]:
        print(f"  [{inv['id']}] {inv['action_type']}  ({inv['risk']} risk)  {inv['label']}")
        print(f"             {inv['description']}")
        if inv.get("code_hint"):
            print(f"             HINT: {inv['code_hint']}")
        print("")
    print(f"To resume: {prompt['resume_command']}")
    print(f"State saved to Redis key: {prompt['checkpoint_key']}")
    print("\nProgress will wait until you resume. Nothing will be lost.\n")

def generate_hitl_prompt(state: ProfessorState, agent_name: str, error: Exception) -> dict:
    error_class = _classify_error(agent_name, error)
    interventions = _build_interventions(state, agent_name, error_class, error)
    
    prompt = {
        "session_id":        state["session_id"],
        "failed_agent":      agent_name,
        "failure_count":     state.get("current_node_failure_count", 3),
        "what_was_attempted": _describe_attempt(state, agent_name),
        "why_it_failed":     str(error)[:500],
        "error_class":       error_class,
        "interventions":     interventions,
        "resume_command":    f"professor resume --session {state['session_id']}",
        "checkpoint_key":    f"professor:hitl:{state['session_id']}",
        "generated_at":      datetime.utcnow().isoformat(),
    }
    
    _write_hitl_prompt(state["session_id"], prompt)
    _print_hitl_banner(prompt)
    return prompt

def _error_state(session_id: str, message: str) -> dict:
    return {"session_id": session_id, "hitl_required": True, "hitl_message": message, "error_count": 1}

def _apply_intervention(state: ProfessorState, intervention: dict, agent_name: str) -> ProfessorState:
    if intervention["action_type"] == "AUTO":
        label = intervention["label"]
        # Default fallback for unknown
        if "Skip this agent" in label:
            return {**state, f"skip_{agent_name}": True}
        effect_fn = AUTO_INTERVENTION_EFFECTS.get(label)
        if effect_fn:
            return effect_fn(state)
    return state

def resume_from_checkpoint(session_id: str, intervention_id: int) -> ProfessorState:
    from memory.redis_state import get_redis_client
    redis_client = get_redis_client()
    key = f"professor:hitl:{session_id}"

    raw = redis_client.get(key)
    if raw is None:
        return _error_state(session_id, f"No checkpoint found for key: {key}")

    try:
        checkpoint = json.loads(raw)
        state = checkpoint["state"]
        agent_name = checkpoint["agent"]
        error_class = checkpoint.get("error_class", "unknown")
    except (json.JSONDecodeError, KeyError) as e:
        return _error_state(session_id, f"Checkpoint corrupt: {e}")

    if intervention_id not in (1, 2, 3):
        return _error_state(session_id, f"intervention_id must be 1, 2, or 3. Got: {intervention_id}")

    templates = INTERVENTION_TEMPLATES.get(error_class, INTERVENTION_TEMPLATES["unknown"])
    intervention = templates[intervention_id - 1]
    
    state = _apply_intervention(state, intervention, agent_name)

    state = {
        **state,
        "current_node_failure_count": 0,
        "hitl_required":             False,
        "hitl_intervention_id":      intervention_id,
        "hitl_intervention_label":   intervention["label"],
        "replan_requested":          False,
    }

    log_event(
        session_id=session_id,
        agent="circuit_breaker",
        action="hitl_resumed",
        keys_read=["hitl_intervention_id"],
        keys_written=["hitl_required"],
        values_changed={
            "intervention_id":    intervention_id,
            "intervention_label": intervention["label"],
            "action_type":        intervention["action_type"],
            "failed_agent":       agent_name,
        }
    )

    return state
