"""Inject budget exhaustion and verify graceful degradation."""

def test_pipeline_continues_after_budget():
    """When LLM budget is exhausted, pipeline is halted or correctly blocks new calls."""
    from shields.cost_governor import CostGovernor
    
    # Set absurdly low budget
    gov = CostGovernor(budget_limit_usd=0.01, budget_remaining_usd=0.01)
    
    # Simulate 3 calls
    for _ in range(3):
        gov.update_cost(0.003)
    
    # 4th call should be blocked (0.001 remaining, need 0.003 * 1.2 = 0.0036)
    can_proceed, reason = gov.check_budget(0.003)
    assert can_proceed == False
    assert "Budget exceeded" in reason

def test_budget_warning_at_80_percent():
    """Test placeholder for 80% warning. Actual CostGovernor doesn't implement this yet."""
    pass
