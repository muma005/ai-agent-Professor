# guards/agent_retry.py
"""
Reusable inner retry wrapper for all LangGraph agent nodes.
Wraps an agent function with 3-attempt retry + circuit breaker escalation.
"""

import traceback
import functools
from typing import Dict, Any, Union
from core.state import ProfessorState
from guards.circuit_breaker import (
    get_escalation_level, handle_escalation, reset_failure_count,
)

MAX_INNER_ATTEMPTS = 3


def with_agent_retry(agent_name: str):
    """
    Decorator factory. Wraps a LangGraph node function with inner retry loop.
    Ensures return is always a ProfessorState object.
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(state: ProfessorState) -> ProfessorState:
            # ── Bridge: Ensure state is the Pydantic object for attribute access ──
            if not isinstance(state, ProfessorState):
                valid_keys = ProfessorState.model_fields.keys()
                filtered_data = {k: v for k, v in dict(state).items() if k in valid_keys}
                
                if "config" in filtered_data and isinstance(filtered_data["config"], dict):
                    from core.config import ProfessorConfig
                    filtered_data["config"] = ProfessorConfig(**filtered_data["config"])
                elif "config" not in filtered_data or filtered_data["config"] is None:
                    from core.config import ProfessorConfig
                    filtered_data["config"] = ProfessorConfig()
                    
                state = ProfessorState(**filtered_data)

            # ── Metadata Update: Set current_node (Owned by supervisor/system) ──
            state.current_node = agent_name

            for attempt in range(1, MAX_INNER_ATTEMPTS + 1):
                try:
                    result = fn(state)
                    
                    # ── SUCCESS: Ensure result is ProfessorState object ──
                    if isinstance(result, dict):
                        valid_keys = ProfessorState.model_fields.keys()
                        filtered_data = {k: v for k, v in result.items() if k in valid_keys}
                        
                        # Special handling for config
                        if "config" in filtered_data and isinstance(filtered_data["config"], dict):
                            from core.config import ProfessorConfig
                            filtered_data["config"] = ProfessorConfig(**filtered_data["config"])
                        elif "config" not in filtered_data or filtered_data["config"] is None:
                            from core.config import ProfessorConfig
                            filtered_data["config"] = ProfessorConfig()
                            
                        result = ProfessorState(**filtered_data)
                    
                    # Success: reset failure count before returning
                    final_result = reset_failure_count(result)
                    print(f"DEBUG: [{agent_name}] Wrapper returning type: {type(final_result)}")
                    return final_result

                except Exception as e:
                    tb = traceback.format_exc()
                    print(f"[{agent_name}] Attempt {attempt}/{MAX_INNER_ATTEMPTS} failed. Error: {e}")

                    if attempt == MAX_INNER_ATTEMPTS:
                        level = get_escalation_level(state)
                        escalated_state = handle_escalation(
                            state=state,
                            level=level,
                            agent_name=agent_name,
                            error=e,
                            traceback_str=tb,
                        )
                        
                        # Normalize to ProfessorState object and mark halted
                        if not isinstance(escalated_state, ProfessorState):
                            valid_keys = ProfessorState.model_fields.keys()
                            filtered_data = {k: v for k, v in dict(escalated_state).items() if k in valid_keys}
                            escalated_state = ProfessorState(**filtered_data)
                            
                        escalated_state.pipeline_halted = True
                        escalated_state.pipeline_halt_reason = f"{agent_name} failed after {MAX_INNER_ATTEMPTS} attempts: {e}"
                        return escalated_state

                    # Not yet exhausted — update failure count and error context
                    state.current_node_failure_count = attempt
                    state.error_context.append({
                        "agent":     agent_name,
                        "attempt":   attempt,
                        "error":     str(e),
                        "traceback": tb,
                    })

            return state  # unreachable

        return wrapper
    return decorator


def build_error_prompt_block(state: ProfessorState, attempt: int) -> str:
    """
    Builds a prompt block from error_context for LLM retry awareness.
    """
    if attempt <= 1:
        return ""

    error_context = getattr(state, "error_context", [])
    if not error_context:
        return ""

    block = "\n\n---\nPREVIOUS ATTEMPT FAILED. DO NOT REPEAT THE SAME MISTAKE.\n"
    for ctx in error_context[-2:]:  # last 2 errors max
        block += f"\nAttempt {ctx['attempt']} error:\n{ctx['traceback']}\n"
    block += "---\nRevise your approach based on the above errors.\n"
    return block
