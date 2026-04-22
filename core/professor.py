# core/professor.py

import os
from dotenv import load_dotenv
load_dotenv()

import pickle
import contextlib
import logging
import threading
import traceback
import functools
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from datetime import datetime, timezone
from langgraph.graph import StateGraph, END
from core.state import ProfessorState

# ── Phase 1 Imports: Core Stability ────────────────────────────────
from core.error_context import ErrorContextManager
from core.checkpoint import save_node_checkpoint, load_last_checkpoint
from core.circuit_breaker import CircuitBreakerError
from core.timeout import timeout

# ── Shield Imports (v2.0) ──────────────────────────────────────────
from shields.preflight import run_preflight_checks
from shields.metric_gate import run_metric_verification_gate
from shields.cost_governor import run_cost_governor_check

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
from agents.feature_factory import run_feature_factory
from agents.ensemble_architect import run_ensemble_architect
from agents.pseudo_label_agent import run_pseudo_label_agent
from agents.supervisor import run_supervisor_replan, get_replan_target, MAX_REPLAN_ATTEMPTS
from agents.submission_strategist import run_submission_strategist
from agents.publisher import run_publisher
from agents.qa_gate import run_qa_gate

logger = logging.getLogger(__name__)

# ── Routing functions (conditional edges) ─────────────────────────

def route_after_router(state: ProfessorState) -> str:
    """After router runs: go to first node in DAG or halt."""
    if state.pipeline_halted or state.triage_mode:
        return END
    dag = state.dag or []
    if not dag:
        return END
    return dag[0]

def route_after_node(state: ProfessorState, node_name: str) -> str:
    """Generic DAG-driven routing with shield checks."""
    if state.pipeline_halted or state.triage_mode:
        return END
        
    dag = state.dag or []
    
    # ── Handle Shield Transition ──
    if node_name == "metric_gate":
        reference_node = "validation_architect"
    else:
        reference_node = node_name

    if reference_node not in dag:
        return END
        
    idx = dag.index(reference_node)
    if idx + 1 >= len(dag):
        return END
        
    return dag[idx + 1]

def route_after_critic(state: ProfessorState) -> str:
    """After Critic: route based on severity."""
    if state.pipeline_halted or state.triage_mode:
        return END

    if state.critic_severity == "CRITICAL":
        if state.dag_version >= MAX_REPLAN_ATTEMPTS:
            return END
        return "supervisor_replan"
        
    return route_after_node(state, "red_team_critic")

def route_after_supervisor_replan(state: ProfessorState) -> str:
    """After supervisor_replan: re-enter at earliest affected node."""
    if state.pipeline_halted:
        return END
    return get_replan_target(state)

# ── Build the graph ───────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Assemble the Professor LangGraph using full Pydantic Orchestration.
    """
    # Use ProfessorState Pydantic model directly
    graph = StateGraph(ProfessorState)

    # ── Add Foundation & Shield Nodes ─────────────────────────────
    graph.add_node("preflight",           run_preflight_checks)
    graph.add_node("cost_governor",      run_cost_governor_check)
    graph.add_node("metric_gate",        run_metric_verification_gate)
    
    # ── Add Agent Nodes ───────────────────────────────────────────
    graph.add_node("semantic_router",      run_semantic_router)
    graph.add_node("competition_intel",    run_competition_intel)
    graph.add_node("domain_researcher",    run_domain_research)
    graph.add_node("data_engineer",        run_data_engineer)
    graph.add_node("eda_agent",            run_eda_agent)
    graph.add_node("eda_visualizer",       run_eda_visualizer)
    graph.add_node("shift_detector",       run_shift_detector)
    graph.add_node("validation_architect", run_validation_architect)
    graph.add_node("feature_factory",      run_feature_factory)
    graph.add_node("ml_optimizer",         run_ml_optimizer)
    graph.add_node("ensemble_architect",   run_ensemble_architect)
    graph.add_node("red_team_critic",      run_red_team_critic)
    graph.add_node("pseudo_label_agent",   run_pseudo_label_agent)
    graph.add_node("supervisor_replan",    run_supervisor_replan)
    graph.add_node("submission_strategist", run_submission_strategist)
    graph.add_node("publisher",            run_publisher)
    graph.add_node("qa_gate",              run_qa_gate)

    # ── Set entry point ───────────────────────────────────────────
    graph.set_entry_point("preflight")

    # ── Add Edges ─────────────────────────────────────────────────
    graph.add_edge("preflight", "semantic_router")
    
    graph.add_conditional_edges(
        "semantic_router",
        route_after_router,
        {node: node for node in [
            "competition_intel", "domain_researcher", "data_engineer", "eda_agent", 
            "eda_visualizer", "shift_detector",
            "validation_architect", "feature_factory", "ml_optimizer",
            "ensemble_architect", "red_team_critic", "pseudo_label_agent",
            "submission_strategist", "publisher", "qa_gate"
        ]} | {END: END}
    )

    # Helper for generic DAG-driven routing
    def _get_router(node_name):
        def _route(state: ProfessorState):
            return route_after_node(state, node_name)
        return _route

    # All nodes follow the DAG except special cases
    for node in ["competition_intel", "domain_researcher", "data_engineer", "eda_agent", 
                 "eda_visualizer", "shift_detector",
                 "feature_factory", "ml_optimizer", "ensemble_architect",
                 "pseudo_label_agent", "submission_strategist", "publisher"]:
        graph.add_conditional_edges(
            node, 
            _get_router(node), 
            {n: n for n in [
                "competition_intel", "domain_researcher", "data_engineer", "eda_agent", 
                "eda_visualizer", "shift_detector",
                "validation_architect", "feature_factory", "ml_optimizer",
                "ensemble_architect", "red_team_critic", "pseudo_label_agent",
                "submission_strategist", "publisher", "qa_gate"
            ]} | {END: END}
        )

    # Special transition: validation_architect must run metric_gate
    graph.add_edge("validation_architect", "metric_gate")
    graph.add_conditional_edges(
        "metric_gate", 
        _get_router("metric_gate"),
        {n: n for n in [
            "competition_intel", "domain_researcher", "data_engineer", "eda_agent", 
            "eda_visualizer", "shift_detector",
            "validation_architect", "feature_factory", "ml_optimizer",
            "ensemble_architect", "red_team_critic", "pseudo_label_agent",
            "submission_strategist", "publisher", "qa_gate"
        ]} | {END: END}
    )

    # Special transition: red_team_critic can trigger replan
    graph.add_conditional_edges(
        "red_team_critic", 
        route_after_critic,
        {n: n for n in [
            "supervisor_replan", "competition_intel", "domain_researcher", "data_engineer", "eda_agent", 
            "eda_visualizer", "shift_detector",
            "validation_architect", "feature_factory", "ml_optimizer",
            "ensemble_architect", "pseudo_label_agent",
            "submission_strategist", "publisher", "qa_gate"
        ]} | {END: END}
    )
    
    # Special transition: supervisor_replan re-enters
    graph.add_conditional_edges(
        "supervisor_replan", 
        route_after_supervisor_replan,
        {n: n for n in [
            "competition_intel", "domain_researcher", "data_engineer", "eda_agent", 
            "eda_visualizer", "shift_detector",
            "validation_architect", "feature_factory", "ml_optimizer",
            "ensemble_architect", "red_team_critic", "pseudo_label_agent",
            "submission_strategist", "publisher", "qa_gate"
        ]} | {END: END}
    )

    graph.add_edge("qa_gate", END)

    return graph.compile()


# ── Graph Singleton ───────────────────────────────────────────────────

_GRAPH = None
_GRAPH_LOCK = threading.Lock()

def get_graph():
    global _GRAPH
    if _GRAPH is None:
        with _GRAPH_LOCK:
            if _GRAPH is None:
                _GRAPH = build_graph()
    return _GRAPH

# ── Main Runner ───────────────────────────────────────────────────────

def run_professor(
    state_input: Union[ProfessorState, Dict[str, Any]],
    resume_from: str = None,
    timeout_seconds: int = 1200,
) -> ProfessorState:
    """
    The main entry point for the Professor v2.0 Autonomous Pipeline.
    """
    from core.state import generate_session_id
    
    # 1. Normalize State
    if isinstance(state_input, dict):
        # Handle flexible initial state
        from core.state import initial_state
        state_dict = initial_state(**state_input)
        state = ProfessorState(**state_dict)
    else:
        state = state_input

    if not state.session_id:
        state.session_id = generate_session_id(state.competition_name or "unknown")

    session_id = state.session_id
    logger.info(f"[Professor] Starting session: {session_id}")

    # 2. Setup environment
    output_dir = Path(f"outputs/{session_id}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    error_context = ErrorContextManager(session_id)
    error_context.start()

    try:
        # 3. Invoke Graph
        graph = get_graph()
        with timeout(timeout_seconds, "Professor Pipeline"):
            final_state = graph.invoke(state)
            
        error_context.success()
        logger.info(f"[Professor] Pipeline successful.")
        return final_state

    except Exception as e:
        error_context.record_error(e, traceback_str=traceback.format_exc())
        error_context.fail()
        save_node_checkpoint(state, session_id, "FAILURE")
        logger.error(f"[Professor] Pipeline failed: {e}")
        raise e
