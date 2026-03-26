# guards/agent_retry.py
"""
Reusable inner retry wrapper for all LangGraph agent nodes.
Wraps an agent function with 3-attempt retry + circuit breaker escalation.
"""

import traceback
import functools
from core.state import ProfessorState
from guards.circuit_breaker import (
    get_escalation_level, handle_escalation, reset_failure_count,
)

MAX_INNER_ATTEMPTS = 3


def with_agent_retry(agent_name: str):
    """
    Decorator factory. Wraps a LangGraph node function with inner retry loop.

    Usage:
        @with_agent_retry("DataEngineer")
        def run_data_engineer(state: ProfessorState) -> ProfessorState:
            ...
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(state: ProfessorState) -> ProfessorState:
            for attempt in range(1, MAX_INNER_ATTEMPTS + 1):
                try:
                    result = fn(state)
                    # Success: reset failure count before returning
                    return reset_failure_count(result)

                except Exception as e:
                    tb = traceback.format_exc()
                    print(
                        f"[{agent_name}] Attempt {attempt}/{MAX_INNER_ATTEMPTS} "
                        f"failed. Error: {e}"
                    )

                    if attempt == MAX_INNER_ATTEMPTS:
                        # All attempts exhausted — run circuit breaker, then RETURN
                        # modified state instead of re-raising (which kills LangGraph).
                        level = get_escalation_level(state)
                        escalated_state = handle_escalation(
                            state=state,
                            level=level,
                            agent_name=agent_name,
                            error=e,
                            traceback_str=tb,
                        )
                        # Mark state as halted — downstream nodes can check this
                        if isinstance(escalated_state, dict):
                            escalated_state["pipeline_halted"] = True
                            escalated_state["pipeline_halt_reason"] = (
                                f"{agent_name} failed after {MAX_INNER_ATTEMPTS} attempts: {e}"
                            )
                            return escalated_state
                        # Fallback if handle_escalation doesn't return a dict
                        return {
                            **state,
                            "pipeline_halted": True,
                            "pipeline_halt_reason": (
                                f"{agent_name} failed after {MAX_INNER_ATTEMPTS} attempts: {e}"
                            ),
                        }

                    # Not yet exhausted — append error context and retry
                    state = {
                        **state,
                        "current_node_failure_count": attempt,
                        "error_context": list(state.get("error_context", [])) + [{
                            "agent":     agent_name,
                            "attempt":   attempt,
                            "error":     str(e),
                            "traceback": tb,
                        }],
                    }

            return state  # unreachable

        return wrapper
    return decorator


def build_error_prompt_block(state: ProfessorState, attempt: int) -> str:
    """
    Builds a prompt block from error_context for LLM retry awareness.
    Call this at the start of agents that use LLMs, on attempt > 1.
    """
    if attempt <= 1:
        return ""

    error_context = state.get("error_context", [])
    if not error_context:
        return ""

    block = "\n\n---\nPREVIOUS ATTEMPT FAILED. DO NOT REPEAT THE SAME MISTAKE.\n"
    for ctx in error_context[-2:]:  # last 2 errors max
        block += f"\nAttempt {ctx['attempt']} error:\n{ctx['traceback']}\n"
    block += "---\nRevise your approach based on the above errors.\n"
    return block
