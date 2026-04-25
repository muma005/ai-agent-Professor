# agents/supervisor.py

import logging
from typing import Dict, Any, List, Optional
from core.state import ProfessorState
from core.lineage import log_event
from tools.operator_channel import emit_to_operator

logger = logging.getLogger(__name__)

AGENT_NAME = "supervisor"

MAX_REPLAN_ATTEMPTS = 3

# Node priority order (earlier = lower number)
NODE_PRIORITY = {
    "competition_intel":    0,
    "domain_research":      1,
    "data_engineer":        2,
    "eda_agent":            3,
    "validation_architect": 4,
    "creative_hypothesis":  5,
    "feature_factory":      6,
    "ml_optimizer":         7,
    "ensemble_architect":   8,
    "red_team_critic":      9,
    "submission_strategist": 10
}

def run_supervisor(state: ProfessorState) -> ProfessorState:
    """
    Orchestrates the pipeline by determining the next node to execute.
    """
    session_id = state.get("session_id", "default")
    current_node = state.get("current_node")
    dag = state.get("dag") or []
    
    print(f"DEBUG: [supervisor] Entering with current_node='{current_node}', dag={dag}")

    # Check for replan trigger first
    if state.get("replan_requested"):
        return run_supervisor_replan(state)

    # Handle Normal Sequencing
    if not dag:
        next_node = "semantic_router"
    else:
        try:
            curr_idx = dag.index(current_node)
            if curr_idx + 1 < len(dag):
                next_node = dag[curr_idx + 1]
            else:
                next_node = "publisher"
        except (ValueError, TypeError):
            if current_node == "preflight":
                next_node = "semantic_router"
            elif current_node == "semantic_router":
                next_node = dag[0] if dag else "publisher"
            elif current_node == "supervisor":
                # Special case: supervisor was just set as current_node during replan
                next_node = state.get("next_node") or "semantic_router"
            else:
                print(f"DEBUG: [supervisor] Node '{current_node}' not in DAG. Falling back to publisher.")
                next_node = "publisher"

    updates = {"next_node": next_node}
    print(f"DEBUG: [supervisor] next_node determined: '{next_node}'")
    return ProfessorState.validated_update(state, AGENT_NAME, updates)

def run_supervisor_replan(state: ProfessorState) -> ProfessorState:
    """
    Reads replan instructions and constructs a new execution path.
    Increments dag_version. If max attempts reached, escalates to HITL.
    """
    dag_version = state.get("dag_version", 1)

    if dag_version >= MAX_REPLAN_ATTEMPTS:
        emit_to_operator(f"🚨 Supervisor exhausted {MAX_REPLAN_ATTEMPTS} replan attempts. Pausing.", level="ESCALATION")
        return ProfessorState.validated_update(state, AGENT_NAME, {
            "hitl_required": True,
            "pipeline_halted": True,
            "pipeline_halt_reason": f"Max replan attempts ({MAX_REPLAN_ATTEMPTS}) reached."
        })

    remove_features = list(state.get("replan_remove_features", []))
    rerun_nodes     = list(state.get("replan_rerun_nodes", []))
    features_dropped = list(set(list(state.get("features_dropped", [])) + remove_features))

    new_dag_version = dag_version + 1
    emit_to_operator(f"🔄 Supervisor: Replan v{new_dag_version}. Rerunning: {rerun_nodes}", level="STATUS")

    log_event(
        session_id=state.get("session_id", "default"),
        agent=AGENT_NAME,
        action="dag_replan",
        values_changed={"dag_version": new_dag_version, "nodes_to_rerun": rerun_nodes}
    )

    updates = {
        "dag_version":            new_dag_version,
        "features_dropped":       features_dropped,
        "replan_requested":       False,
        "replan_remove_features": [],
        "replan_rerun_nodes":     rerun_nodes,
        "hitl_required":          False,
        "critic_severity":        "unchecked",
        "next_node":              get_replan_target(state)
    }
    
    return ProfessorState.validated_update(state, AGENT_NAME, updates)

def get_replan_target(state: ProfessorState) -> str:
    """Returns the earliest node to re-enter from."""
    rerun_nodes = state.get("replan_rerun_nodes", [])
    if not rerun_nodes:
        return "feature_factory"
    return min(rerun_nodes, key=lambda n: NODE_PRIORITY.get(n, 99))
