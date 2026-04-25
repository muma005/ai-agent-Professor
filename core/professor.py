# core/professor.py

import os
import time
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

from langgraph.graph import StateGraph, END

from core.state import ProfessorState, initial_state
from core.lineage import log_event
from core.checkpoint import save_node_checkpoint
from tools.operator_channel import init_hitl, process_pending_injections, emit_to_operator

# ── Agent Imports ──────────────────────────────────────────────────
from agents.semantic_router import run_semantic_router
from agents.competition_intel import run_competition_intel
from agents.domain_research import run_domain_research
from agents.data_engineer import run_data_engineer
from agents.eda_agent import run_eda_agent
from tools.eda_plots import run_eda_visualizer
from agents.shift_detector import run_shift_detector
from agents.validation_architect import run_validation_architect
from agents.ml_optimizer import run_ml_optimizer
from agents.red_team_critic import run_red_team_critic
from agents.feature_hypothesis import run_feature_hypothesis
from agents.feature_factory import run_feature_factory
from agents.ensemble_architect import run_ensemble_architect
from agents.pseudo_label import pseudo_label_architect
from agents.supervisor import run_supervisor
from agents.submission_strategist import run_submission_strategist
from agents.publisher import run_publisher
from agents.qa_gate import run_qa_gate
from shields.preflight import run_preflight_checks

logger = logging.getLogger(__name__)

# ── Routing functions (conditional edges) ─────────────────────────

def route_after_node(state: ProfessorState) -> str:
    """Uses state.next_node (set by supervisor or router) to decide where to go."""
    if state.get("pipeline_halted") or state.get("triage_mode"):
        return END
        
    next_n = state.get("next_node")
    if not next_n:
        return END
        
    return next_n

# ── Professor Pipeline Builder ─────────────────────────────────────

def build_professor_graph():
    """Constructs the LangGraph for Professor v2.0."""
    workflow = StateGraph(ProfessorState)

    # 1. Define Nodes
    workflow.add_node("preflight", run_preflight_checks)
    workflow.add_node("semantic_router", run_semantic_router)
    workflow.add_node("competition_intel", run_competition_intel)
    workflow.add_node("domain_researcher", run_domain_research)
    workflow.add_node("data_engineer", run_data_engineer)
    workflow.add_node("eda_agent", run_eda_agent)
    workflow.add_node("shift_detector", run_shift_detector)
    workflow.add_node("validation_architect", run_validation_architect)
    workflow.add_node("creative_hypothesis", run_feature_hypothesis)
    workflow.add_node("feature_factory", run_feature_factory)
    workflow.add_node("ml_optimizer", run_ml_optimizer)
    workflow.add_node("ensemble_architect", run_ensemble_architect)
    workflow.add_node("pseudo_label", pseudo_label_architect)
    workflow.add_node("red_team_critic", run_red_team_critic)
    workflow.add_node("submission_strategist", run_submission_strategist)
    workflow.add_node("publisher", run_publisher)
    workflow.add_node("supervisor", run_supervisor)
    workflow.add_node("qa_gate", run_qa_gate)

    # 2. Define Edges
    workflow.set_entry_point("preflight")
    workflow.add_edge("preflight", "supervisor")
    
    managed_nodes = [
        "semantic_router", "competition_intel", "domain_researcher",
        "data_engineer", "eda_agent", "shift_detector",
        "validation_architect", "creative_hypothesis", "feature_factory",
        "ml_optimizer", "ensemble_architect", "pseudo_label", "red_team_critic",
        "submission_strategist"
    ]
    for node in managed_nodes:
        workflow.add_edge(node, "supervisor")

    # Supervisor uses conditional routing
    workflow.add_conditional_edges(
        "supervisor",
        route_after_node,
        {
            "semantic_router": "semantic_router",
            "competition_intel": "competition_intel",
            "domain_researcher": "domain_researcher",
            "data_engineer": "data_engineer",
            "eda_agent": "eda_agent",
            "shift_detector": "shift_detector",
            "validation_architect": "validation_architect",
            "creative_hypothesis": "creative_hypothesis",
            "feature_factory": "feature_factory",
            "ml_optimizer": "ml_optimizer",
            "ensemble_architect": "ensemble_architect",
            "pseudo_label": "pseudo_label",
            "red_team_critic": "red_team_critic",
            "submission_strategist": "submission_strategist",
            "publisher": "publisher",
            "__end__": END
        }
    )
    
    workflow.add_edge("publisher", "qa_gate")
    workflow.add_edge("qa_gate", END)

    return workflow.compile()

def get_graph_cache_clear():
    """Teardown helper for tests."""
    pass

# ── Entry Point ────────────────────────────────────────────────────

def run_professor(
    initial_state_data: Dict[str, Any], 
    timeout_seconds: int = 1800,
    use_hitl: bool = True
) -> ProfessorState:
    """
    Main loop to execute the Professor pipeline.
    """
    session_id = initial_state_data.get("session_id", "default")
    logger.info(f"🚀 Starting Professor v2.0 — Session: {session_id}")

    if use_hitl:
        init_hitl(["cli"], initial_state_data.get("config", {}))

    app = build_professor_graph()
    current_state = initial_state_data
    
    try:
        current_state = process_pending_injections(current_state)
        
        events = app.stream(current_state)
        for event in events:
            node_name = list(event.keys())[0]
            current_state = event[node_name]
            save_node_checkpoint(current_state, session_id, node_name)
            current_state = process_pending_injections(current_state)
            if node_name == "qa_gate" or current_state.get("pipeline_halted"):
                break
                
    except Exception as e:
        logger.error(f"💥 Pipeline CRASH in run_professor: {e}")
        import traceback
        traceback.print_exc()
            
    logger.info("🏁 Professor execution complete.")
    return current_state
