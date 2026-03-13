# agents/supervisor.py
# -------------------------------------------------------------------------
# Day 11 — Supervisor Replan Node
# Triggered when critic returns CRITICAL verdict.
# Reads replan instructions, drops bad features, increments dag_version.
# If max attempts exhausted → escalate to HITL.
# -------------------------------------------------------------------------

import logging
from core.state import ProfessorState
from core.lineage import log_event

logger = logging.getLogger(__name__)

MAX_REPLAN_ATTEMPTS = 3

# Node priority order (earlier = lower number)
NODE_PRIORITY = {
    "data_engineer":        1,
    "eda_agent":            2,
    "validation_architect": 3,
    "feature_factory":      4,
    "ml_optimizer":         5,
    "red_team_critic":      6,
}


def run_supervisor_replan(state: ProfessorState) -> ProfessorState:
    """
    Called when critic returns CRITICAL verdict.
    Reads replan instructions and constructs a new execution path.
    Increments dag_version. If max attempts reached, escalates to HITL.
    Never a retry — this is a new DAG, not a repeat of the old one.
    """
    dag_version = state.get("dag_version", 1)

    if dag_version >= MAX_REPLAN_ATTEMPTS:
        # Self-healing exhausted — escalate to human
        logger.warning(
            f"[Supervisor] dag_version={dag_version} >= MAX_REPLAN_ATTEMPTS={MAX_REPLAN_ATTEMPTS}. "
            f"Escalating to HITL."
        )
        return {
            **state,
            "hitl_required": True,
            "hitl_reason": (
                f"Supervisor exhausted {MAX_REPLAN_ATTEMPTS} replan attempts. "
                f"Critic still finding CRITICAL issues. Manual review required. "
                f"Last verdict: {state.get('hitl_reason', 'unknown')}"
            ),
            "pipeline_halted": True,
        }

    # Read replan instructions from critic verdict
    remove_features = list(state.get("replan_remove_features", []))
    rerun_nodes     = list(state.get("replan_rerun_nodes", []))

    # Build the new execution context — accumulate dropped features
    features_dropped = list(set(
        list(state.get("features_dropped", [])) + remove_features
    ))

    new_dag_version = dag_version + 1
    print(
        f"[Supervisor] Replan v{new_dag_version}. "
        f"Dropping features: {remove_features}. "
        f"Rerunning nodes: {rerun_nodes}."
    )

    log_event(
        session_id=state["session_id"],
        agent="supervisor",
        action="dag_replan",
        keys_read=["replan_remove_features", "replan_rerun_nodes"],
        keys_written=["dag_version", "features_dropped"],
        values_changed={
            "dag_version_before": dag_version,
            "dag_version_after":  new_dag_version,
            "features_dropped":   remove_features,
            "nodes_to_rerun":     rerun_nodes,
        },
    )

    return {
        **state,
        "dag_version":            new_dag_version,
        "features_dropped":       features_dropped,
        "replan_requested":       False,       # consumed
        "replan_remove_features": [],           # cleared
        "replan_rerun_nodes":     rerun_nodes,  # kept for routing
        "hitl_required":          False,        # critic set this — clear it for replan pass
        "critic_severity":        "unchecked",  # critic will re-run after replan
        "critic_verdict":         {},
    }


def get_replan_target(state: ProfessorState) -> str:
    """
    Returns the earliest node to re-enter from based on replan_rerun_nodes.
    Used by the LangGraph conditional edge after supervisor_replan.
    """
    rerun_nodes = state.get("replan_rerun_nodes", [])
    if not rerun_nodes:
        return "feature_factory"  # default: re-run from feature factory
    # Re-enter at the earliest affected node
    earliest = min(rerun_nodes, key=lambda n: NODE_PRIORITY.get(n, 99))
    return earliest
