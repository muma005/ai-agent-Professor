# agents/semantic_router.py

import os
import logging
from typing import Dict, Any, List
from core.state import ProfessorState
from guards.agent_retry import with_agent_retry
from tools.performance_monitor import timed_node

logger = logging.getLogger(__name__)

AGENT_NAME = "semantic_router"

@timed_node
@with_agent_retry(AGENT_NAME)
def run_semantic_router(state: ProfessorState) -> ProfessorState:
    """
    Analyzes competition details and defines the DAG.
    """
    comp_name = state.get("competition_name", "unknown")
    print(f"[SemanticRouter] Competition: {comp_name}")
    
    # 1. Logic to determine task type (Stubbed)
    task_type = "tabular_classification"
    print(f"[SemanticRouter] Task type: {task_type}")
    
    # 2. Define the execution DAG
    dag = [
        "competition_intel",
        "data_engineer",
        "eda_agent",
        "validation_architect",
        "feature_factory",
        "ml_optimizer",
        "red_team_critic",
        "submission_strategist"
    ]
    print(f"[SemanticRouter] Route: {' -> '.join(dag)}")
    
    # 3. Update State
    updates = {
        "dag": dag,
        "task_type": task_type
    }
    
    return ProfessorState.validated_update(state, AGENT_NAME, updates)
