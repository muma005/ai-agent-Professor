# tests/contracts/test_metric_gate_contract.py

import pytest
from shields.metric_gate import verify_metric, run_metric_verification_gate
from core.state import ProfessorState

def test_verify_known_metric():
    """Verify common metrics pass verification."""
    success, msg = verify_metric("auc", "binary")
    assert success is True
    assert "Verified" in msg
    
    success, msg = verify_metric("rmse", "regression")
    assert success is True

def test_verify_unknown_metric():
    """Verify unknown metrics fail verification."""
    success, msg = verify_metric("non_existent_metric", "binary")
    assert success is False
    assert "Unknown metric" in msg

def test_run_gate_pass():
    """Verify LangGraph node passes state through on success."""
    state = ProfessorState(
        metric_contract={"metric_name": "auc"},
        task_type="binary"
    )
    
    new_state = run_metric_verification_gate(state)
    assert new_state.preflight_passed is False # Default is False
    assert len(new_state.preflight_warnings) == 0

def test_run_gate_fail_triggers_warning():
    """Verify failure adds warning and keeps preflight_passed False."""
    state = ProfessorState(
        metric_contract={"metric_name": "broken_metric"},
        task_type="binary"
    )
    
    new_state = run_metric_verification_gate(state)
    assert any("Metric Verification Failed" in w for w in new_state.preflight_warnings)
