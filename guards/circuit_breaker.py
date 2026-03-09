# guards/circuit_breaker.py

import json
import logging
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
        return {
            **state,
            "hitl_required":  True,
            "hitl_reason":    (
                f"Circuit breaker HITL: {agent_name} failed 3 times. "
                f"Last error: {error}. "
                f"Full state checkpointed to Redis. "
                f"Resume with: professor resume --session {state['session_id']}"
            ),
            "pipeline_halted": True,
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
            "checkpointed_at": __import__("datetime").datetime.utcnow().isoformat(),
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
