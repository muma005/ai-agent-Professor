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
    comp_name = str(state.get("competition_name", "") or "").lower()
    session_id = str(state.get("session_id", "") or "").lower()
    
    # 1. Logic to determine task type
    task_type = state.get("task_type", "unknown")
    if not task_type or task_type == "unknown":
        combined_text = comp_name + " " + session_id
        if any(k in combined_text for k in ["regression", "house price", "demand", "price", "predict"]):
            task_type = "tabular_regression"
        else:
            # DEFAULT to classification if no regression signals
            task_type = "tabular_classification"
        
    print(f"[SemanticRouter] Competition: {comp_name}, Session: {session_id} -> Task type: {task_type}")
    
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
        "task_type": task_type,
        "next_node": dag[0] if dag else None
    }
    
    return ProfessorState.validated_update(state, AGENT_NAME, updates)
