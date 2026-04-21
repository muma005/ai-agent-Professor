# shields/cost_governor.py

import time
import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# ── Cost Governor ────────────────────────────────────────────────────────────

@dataclass
class CostGovernor:
    """
    Tracks and enforces execution budgets and safety multipliers.
    """
    budget_limit_usd: float = 2.0
    budget_remaining_usd: float = 2.0
    safety_multiplier: float = 1.2
    
    # Rate limiting
    calls_last_minute: int = 0
    last_reset_time: float = field(default_factory=time.time)
    max_calls_per_minute: int = 15

    def check_budget(self, estimated_cost: float) -> Tuple[bool, str]:
        """Verify if we have enough budget for the next call."""
        # Reset rate limit if minute passed
        if time.time() - self.last_reset_time > 60:
            self.calls_last_minute = 0
            self.last_reset_time = time.time()
            
        if self.calls_last_minute >= self.max_calls_per_minute:
            return False, f"Rate limit reached: {self.max_calls_per_minute} calls/min"

        projected = estimated_cost * self.safety_multiplier
        if projected > self.budget_remaining_usd:
            return False, f"Budget exceeded. Remaining: ${self.budget_remaining_usd:.4f}, Required: ${projected:.4f}"
            
        return True, ""

    def update_cost(self, actual_cost: float):
        """Deducts cost from remaining budget."""
        self.budget_remaining_usd -= actual_cost
        self.calls_last_minute += 1

def run_cost_governor_check(state: Any) -> Any:
    """LangGraph node: Shield 3 pre-check."""
    from core.state import ProfessorState
    
    gov = CostGovernor(
        budget_limit_usd=state.get("budget_limit_usd", 2.0),
        budget_remaining_usd=state.get("budget_remaining_usd", 2.0)
    )
    
    # Simple check for the next agent
    can_proceed, reason = gov.check_budget(0.01) # Baseline small call
    
    if not can_proceed:
        logger.error(f"[Shield 3] BUDGET/RATE GATE FAILED: {reason}")
        return ProfessorState.validated_update(state, "cost_governor", {
            "pipeline_halted": True,
            "pipeline_halt_reason": f"Cost Governor: {reason}"
        })
        
    return state
