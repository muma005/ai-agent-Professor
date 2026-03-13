import pytest
from core.state import initial_state
from guards.circuit_breaker import generate_hitl_prompt

def test_hitl_prompt_contract():
    # Setup
    state = initial_state("TestComp", "data.csv")
    state["session_id"] = "test-session"
    state["current_node_failure_count"] = 3
    
    error = ValueError("Missing expected column 'target'")
    
    prompt = generate_hitl_prompt(state, "data_engineer", error)
    
    # Check 1: Dictionary structure includes all 9 required keys
    expected_keys = {
        "session_id", "failed_agent", "failure_count", 
        "what_was_attempted", "why_it_failed", "error_class", 
        "interventions", "resume_command", "checkpoint_key", "generated_at"
    }
    assert expected_keys.issubset(prompt.keys())
    
    # Check 2: Exactly 3 interventions
    assert len(prompt["interventions"]) == 3
    
    # Check 3: Truncates why_it_failed to 500 chars
    long_error = Exception("A" * 1000)
    prompt_long = generate_hitl_prompt(state, "agent", long_error)
    assert len(prompt_long["why_it_failed"]) <= 500
    
    # Check 4: Interventions structural integrity
    for intervention in prompt["interventions"]:
        assert {"id", "label", "action_type", "risk", "description", "code_hint"}.issubset(intervention.keys())
        assert intervention["id"] in (1, 2, 3)
