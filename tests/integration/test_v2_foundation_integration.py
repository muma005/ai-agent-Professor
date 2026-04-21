# tests/integration/test_v2_foundation_integration.py

import pytest
from core.state import ProfessorState
from tools.operator_channel import init_hitl, _MANAGER
from tools.sandbox import run_in_sandbox
from unittest.mock import patch

@pytest.fixture
def hitl_cli():
    """Init CLI HITL for integration test."""
    import tools.operator_channel
    tools.operator_channel._MANAGER = None
    manager = init_hitl(["cli"], {})
    yield manager
    manager.stop_listener()
    tools.operator_channel._MANAGER = None

def test_v2_foundation_synergy(hitl_cli, capsys):
    """
    Verify:
    1. Sandbox runs code
    2. Failure triggers error classification
    3. State is updated via validated_update
    4. HITL would emit (verified by captured stdout)
    """
    state = ProfessorState(session_id="integration_test")
    
    # Code that fails with a KeyError (column_missing)
    code = """
import polars as pl
df = pl.DataFrame({"a": [1, 2]})
print(df["non_existent"])
"""
    
    # We run the sandbox
    res = run_in_sandbox(
        code, 
        agent_name="data_engineer", 
        purpose="testing integration"
    )
    
    assert res["success"] is False
    assert res["error_class"] == "column_missing"
    
    # Verify state update logic (simulated since run_in_sandbox returns dict)
    state = ProfessorState.validated_update(state, "sandbox", {
        "debug_error_class": res["error_class"],
        "debug_diagnostics": res["diagnostics"]
    })
    
    assert state.debug_error_class == "column_missing"
    assert "dataframes" in state.debug_diagnostics
    assert len(state.state_mutations_log) >= 2 # created_at + 2 fields
    
    # Verify HITL message emission
    from tools.operator_channel import emit_to_operator
    emit_to_operator(f"Sandbox Failed: {res['error_class']}", level="STATUS")
    
    captured = capsys.readouterr()
    assert "Sandbox Failed: column_missing" in captured.out
