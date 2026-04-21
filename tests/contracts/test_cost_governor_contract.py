# tests/contracts/test_cost_governor_contract.py

import pytest
from shields.cost_governor import CostGovernor, run_cost_governor_check
from core.state import ProfessorState

def test_budget_enforcement():
    """Verify budget checks and updates."""
    gov = CostGovernor(budget_limit_usd=1.0, budget_remaining_usd=1.0)
    
    # Check within budget
    can_proceed, _ = gov.check_budget(0.5)
    assert can_proceed is True
    
    # Update cost
    gov.update_cost(0.5)
    assert gov.budget_remaining_usd == 0.5
    
    # Check exceeding budget (0.5 * 1.2 = 0.6 > 0.5)
    can_proceed, reason = gov.check_budget(0.5)
    assert can_proceed is False
    assert "Budget exceeded" in reason

def test_rate_limiting():
    """Verify rate limit enforcement."""
    gov = CostGovernor(max_calls_per_minute=2)
    
    gov.update_cost(0.01)
    gov.update_cost(0.01)
    
    can_proceed, reason = gov.check_budget(0.01)
    assert can_proceed is False
    assert "Rate limit reached" in reason

def test_run_governor_node_pass():
    """Verify LangGraph node allows state to pass when budget OK."""
    state = ProfessorState(budget_remaining_usd=1.0)
    new_state = run_cost_governor_check(state)
    assert new_state.pipeline_halted is False

def test_run_governor_node_halt():
    """Verify LangGraph node halts pipeline when budget exhausted."""
    state = ProfessorState(budget_remaining_usd=0.0)
    new_state = run_cost_governor_check(state)
    assert new_state.pipeline_halted is True
    assert "Cost Governor" in new_state.pipeline_halt_reason
