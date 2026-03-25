# core/professor.py

import os
from dotenv import load_dotenv
load_dotenv()

# ── LangSmith tracing — must be set before LangGraph is imported ──
_tracing_enabled = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
if _tracing_enabled:
    print(f"[Professor] LangSmith tracing ON -- project: "
          f"{os.getenv('LANGCHAIN_PROJECT', 'default')}")

# Day 12: Default tracing sampling rate to 10% to save costs
os.environ.setdefault("LANGCHAIN_TRACING_SAMPLING_RATE", os.getenv("LANGCHAIN_TRACING_SAMPLING_RATE", "0.10"))

import random
import numpy as np

def _set_global_seed(seed=42):
    """Enforce deterministic behaviour at process start."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass

_set_global_seed(42)

import pickle
import contextlib
import logging
import threading
import traceback
import numpy as np
import polars as pl
import polars.selectors as cs
from typing import Optional
from datetime import datetime, timezone
from langgraph.graph import StateGraph, END
from core.state import ProfessorState

# ── Phase 1 Imports: Core Stability ────────────────────────────────
from core.error_context import ErrorContextManager
from core.checkpoint import save_node_checkpoint, load_last_checkpoint
from core.circuit_breaker import llm_circuit_breaker, CircuitBreakerError
from core.timeout import timeout
from tools.prediction_validator import validate_predictions

# ── Agent Imports ──────────────────────────────────────────────────
from agents.semantic_router import run_semantic_router
from agents.competition_intel import run_competition_intel
from agents.data_engineer import run_data_engineer
from agents.eda_agent import run_eda_agent
from agents.validation_architect import run_validation_architect
from agents.ml_optimizer import run_ml_optimizer
from agents.red_team_critic import run_red_team_critic
from agents.feature_factory import run_feature_factory
from agents.ensemble_architect import blend_models
from agents.pseudo_label_agent import run_pseudo_label_agent
from agents.supervisor import run_supervisor_replan, get_replan_target, MAX_REPLAN_ATTEMPTS, NODE_PRIORITY

logger = logging.getLogger(__name__)


# ── Routing functions (conditional edges) ─────────────────────────

def route_after_router(state: ProfessorState) -> str:
    """After router runs: go to first node in DAG."""
    if state.get("pipeline_halted") or state.get("triage_mode"):
        print("[Professor] Pipeline halted (circuit breaker). Ending.")
        return END
    dag = state.get("dag", [])
    if not dag:
        print("[Professor] WARNING: DAG is empty after router. Ending.")
        return END

    next_node = dag[0]
    print(f"[Professor] Routing to: {next_node}")
    return next_node


def route_after_intel(state: ProfessorState) -> str:
    """After Competition Intel: advance to Data Engineer."""
    return _advance_dag(state, current="competition_intel")


def route_after_data_engineer(state: ProfessorState) -> str:
    """After Data Engineer: run integrity gate, then advance to EDA Agent."""
    from guards.pipeline_integrity import run_integrity_gate
    run_integrity_gate(state, "POST_DATA_ENGINEER")
    return _advance_dag(state, current="data_engineer")


def route_after_eda(state: ProfessorState) -> str:
    """After EDA Agent: run integrity gate, then advance to Validation Architect."""
    from guards.pipeline_integrity import run_integrity_gate
    run_integrity_gate(state, "POST_EDA")
    return _advance_dag(state, current="eda_agent")


def route_after_validation(state: ProfessorState) -> str:
    """After Validation Architect: halt if HITL required, else advance."""
    if state.get("hitl_required"):
        print("[Professor] HITL required. Halting execution.")
        return END
    return _advance_dag(state, current="validation_architect")


def route_after_feature_factory(state: ProfessorState) -> str:
    """After Feature Factory: advance to ML Optimizer."""
    return _advance_dag(state, current="feature_factory")


def route_after_optimizer(state: ProfessorState) -> str:
    """After Optimizer: run integrity gate, then advance to ensemble_architect."""
    from guards.pipeline_integrity import run_integrity_gate
    run_integrity_gate(state, "POST_MODEL")
    return _advance_dag(state, current="ml_optimizer")


def route_after_ensemble(state: ProfessorState) -> str:
    """After Ensemble Architect: advance to red_team_critic."""
    return _advance_dag(state, current="ensemble_architect")


def route_after_critic(state: ProfessorState) -> str:
    """After Critic: route based on severity.
    CRITICAL → supervisor_replan (or hitl if exhausted).
    HIGH/MEDIUM/OK → submit.
    """
    if state.get("pipeline_halted") or state.get("triage_mode"):
        print("[Professor] Pipeline halted after critic. Ending.")
        return END

    severity = state.get("critic_severity", "unchecked")
    if severity == "CRITICAL":
        dag_version = state.get("dag_version", 1)
        if dag_version >= MAX_REPLAN_ATTEMPTS:
            print(f"[Professor] CRITICAL + dag_version={dag_version} >= {MAX_REPLAN_ATTEMPTS}. HITL required.")
            return END  # hitl_handler — pipeline halted
        print(f"[Professor] CRITICAL verdict. Routing to supervisor_replan (dag_version={dag_version}).")
        return "supervisor_replan"
    # HIGH, MEDIUM, OK: continue to submit
    print(f"[Professor] Critic verdict: {severity}. Continuing to submit.")
    return "submit"


def route_after_supervisor_replan(state: ProfessorState) -> str:
    """After supervisor_replan: re-enter at earliest affected node."""
    if state.get("pipeline_halted"):
        print("[Professor] Pipeline halted by supervisor. Ending.")
        return END
    target = get_replan_target(state)
    print(f"[Professor] Supervisor replan complete. Re-entering at: {target}")
    return target


def _advance_dag(state: ProfessorState, current: str) -> str:
    """
    Find current node in DAG and return the next one.
    If current is last node, return END.
    Checks pipeline_halted / triage_mode before advancing.
    """
    if state.get("pipeline_halted") or state.get("triage_mode"):
        print(f"[Professor] Pipeline halted after '{current}'. Ending.")
        return END

    dag = state.get("dag", [])

    if current not in dag:
        print(f"[Professor] '{current}' not in DAG -- ending.")
        return END

    idx = dag.index(current)

    if idx + 1 >= len(dag):
        print(f"[Professor] '{current}' is last node -- ending.")
        return END

    next_node = dag[idx + 1]
    print(f"[Professor] DAG advance: {current} -> {next_node}")
    return next_node


# -- Parallel execution groups (Day 9) ------------------------------------

def _fan_out_intelligence(state: ProfessorState) -> list:
    """
    Fan-out node: dispatches to competition_intel and data_engineer in parallel.
    Returns a list of Send objects -- LangGraph executes them concurrently.
    """
    from langgraph.types import Send
    return [
        Send("competition_intel", state),
        Send("data_engineer",     state),
    ]


def _fan_out_model_trials(state: ProfessorState) -> list:
    """
    Fan-out node: dispatches one trial per model type.
    Each trial runs in its own E2B sandbox subprocess -- true parallelism.
    """
    from langgraph.types import Send
    model_types = ["lgbm", "xgb", "catboost"]
    return [
        Send("run_model_trial", {**state, "trial_model_type": model})
        for model in model_types
    ]


def _fan_out_critic_vectors(state: ProfessorState) -> list:
    """
    Fan-out node: all 4 critic vectors are fully independent.
    Order does not matter. Slowest vector determines total critic time.
    """
    from langgraph.types import Send
    vectors = [1, 2, 3, 4]
    return [
        Send("run_critic_vector", {**state, "critic_vector_id": v})
        for v in vectors
    ]


def _intelligence_fan_join(state: ProfessorState) -> ProfessorState:
    """
    Fan-join after competition_intel + data_engineer.
    Verifies both outputs exist before handing off to eda_agent.
    """
    missing = []
    if not state.get("schema_path") or not os.path.exists(state.get("schema_path", "")):
        missing.append("schema.json (data_engineer)")
    if not state.get("competition_brief_path"):
        missing.append("competition_brief.json (competition_intel)")

    if missing:
        raise ValueError(
            f"[FanJoin:intelligence] Expected outputs missing: {missing}. "
            f"One or more parallel branches did not complete."
        )

    parallel_groups = dict(state.get("parallel_groups", {}))
    intelligence = dict(parallel_groups.get("intelligence", {}))
    intelligence["status"] = "complete"
    parallel_groups["intelligence"] = intelligence

    return {**state, "parallel_groups": parallel_groups}


# -- Submit node --- validated submission via submit_tools ----------------


def run_submit(state: ProfessorState) -> ProfessorState:
    """
    Submit node: generates validated submission.csv using submit_tools.
    Validates against sample_submission.csv for correct columns, row count,
    ID match, and zero nulls.
    """
    from tools.data_tools import read_csv, read_parquet, read_json
    from tools.submit_tools import generate_submission, save_submission_log
    from core.metric_contract import load_contract
    from core.lineage import log_event

    session_id  = state["session_id"]
    output_dir  = f"outputs/{session_id}"
    competition = state["competition_name"]
    os.makedirs(output_dir, exist_ok=True)

    print(f"[Submit] Generating submission -- session: {session_id}")

    # ── Load test data (from paths set by data_engineer — no string replacement) ──
    test_path   = state.get("test_data_path", "")
    sample_path = state.get("sample_submission_path", "")

    if not test_path or not os.path.exists(test_path):
        print(f"[Submit] WARNING: test data not found at '{test_path}'")
        return {**state, "submission_path": None}
    if not sample_path or not os.path.exists(sample_path):
        print(f"[Submit] WARNING: sample_submission not found at '{sample_path}'")
        return {**state, "submission_path": None}

    test_df = read_csv(test_path)

    # ── Load model ────────────────────────────────────────────────
    if not state.get("model_registry"):
        raise ValueError("[Submit] No model in registry -- run ML Optimizer first")

    model_path = state["model_registry"][0]["model_path"]
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # ── Prepare test features (using schema authority — no guessing) ──
    schema     = read_json(state["schema_path"])
    target_col = state["target_col"]

    from core.preprocessor import TabularPreprocessor
    preprocessor_path = f"{output_dir}/preprocessor.pkl"
    if not os.path.exists(preprocessor_path):
        raise ValueError("[Submit] TabularPreprocessor not found. Run Feature Factory first.")
        
    preprocessor = TabularPreprocessor.load(preprocessor_path)
    test_features = preprocessor.transform(test_df)

    # Slice features using schema authority — no substring heuristics
    id_cols = set(state.get("id_columns", []))
    eda_drops = set(state.get("dropped_features", []))
    feature_cols = [
        c for c in test_features.columns
        if c != target_col
        and c not in id_cols
        and c not in eda_drops
    ]
    
    X_test = test_features.select(feature_cols).to_numpy().astype(np.float64)

    # ── Generate predictions ──────────────────────────────────────
    contract_path = f"{output_dir}/metric_contract.json"
    if os.path.exists(contract_path):
        contract = load_contract(contract_path)
        if contract.requires_proba:
            probs = model.predict_proba(X_test)
            if probs.shape[1] == 2:
                preds = probs[:, 1]
            else:
                preds = probs
        else:
            preds = model.predict(X_test)
    else:
        preds = model.predict(X_test)

    # ── Generate + validate submission via submit_tools ────────────
    submission_path = f"{output_dir}/submission.csv"

    result = generate_submission(
        predictions=preds,
        sample_submission_path=sample_path,
        output_path=submission_path,
        target_dtype="auto"
    )

    # ── Log to submission ladder ──────────────────────────────────
    entry = save_submission_log(
        session_id=session_id,
        submission_path=submission_path,
        cv_mean=state.get("cv_mean", 0.0),
        notes=f"Phase 1 baseline -- {competition}"
    )
    
    existing_log = list(state.get("submission_log") or [])
    existing_log.append(entry)

    # ── Log lineage ──────────────────────────────────────────────
    log_event(
        session_id=session_id,
        agent="submit",
        action="generated_submission",
        keys_read=["model_registry", "clean_data_path"],
        keys_written=["submission_path"],
        values_changed={"submission_path": submission_path},
    )

    print(f"[Submit] Done. Upload to Kaggle:")
    print(f"  kaggle competitions submit -c {competition} "
          f"-f {submission_path} -m 'Professor Phase 1 baseline'")

    return {
        **state,
        "submission_path": submission_path,
        "submission_log": existing_log,
    }


# ── Build the graph ───────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Assemble the Professor LangGraph StateGraph.

    Phase 1 graph:
      semantic_router -> data_engineer -> ml_optimizer -> submit -> END

    Phase 2+: conditional edges, parallel branches, Critic loop added here.
    """
    graph = StateGraph(ProfessorState)

    # ── Add nodes ─────────────────────────────────────────────────
    graph.add_node("semantic_router", run_semantic_router)
    graph.add_node("competition_intel", run_competition_intel)
    graph.add_node("data_engineer",   run_data_engineer)
    graph.add_node("eda_agent",       run_eda_agent)
    graph.add_node("validation_architect", run_validation_architect)
    graph.add_node("ml_optimizer",    run_ml_optimizer)
    graph.add_node("red_team_critic", run_red_team_critic)
    graph.add_node("feature_factory", run_feature_factory)
    graph.add_node("ensemble_architect", blend_models)
    graph.add_node("supervisor_replan", run_supervisor_replan)
    graph.add_node("submit",          run_submit)
    graph.add_node("pseudo_label_agent", run_pseudo_label_agent)

    # ── Set entry point ───────────────────────────────────────────
    graph.set_entry_point("semantic_router")

    # All possible node targets for DAG-driven conditional edges
    _all_nodes = {
        "competition_intel": "competition_intel",
        "data_engineer":     "data_engineer",
        "eda_agent":         "eda_agent",
        "validation_architect": "validation_architect",
        "feature_factory":   "feature_factory",
        "ml_optimizer":      "ml_optimizer",
        "red_team_critic":   "red_team_critic",
        "ensemble_architect": "ensemble_architect",
        "supervisor_replan": "supervisor_replan",
        "pseudo_label_agent": "pseudo_label_agent",
        "submit":            "submit",
        END:                 END,
    }

    # ── Add edges ─────────────────────────────────────────────────
    graph.add_conditional_edges(
        "semantic_router",
        route_after_router,
        _all_nodes,
    )

    graph.add_conditional_edges(
        "competition_intel",
        route_after_intel,
        _all_nodes,
    )

    graph.add_conditional_edges(
        "data_engineer",
        route_after_data_engineer,
        _all_nodes,
    )

    graph.add_conditional_edges(
        "eda_agent",
        route_after_eda,
        _all_nodes,
    )

    graph.add_conditional_edges(
        "validation_architect",
        route_after_validation,
        _all_nodes,
    )

    graph.add_conditional_edges(
        "ml_optimizer",
        route_after_optimizer,
        _all_nodes,
    )

    # Day 16: Ensemble Architect → advance to critic
    graph.add_conditional_edges(
        "ensemble_architect",
        route_after_ensemble,
        _all_nodes,
    )

    # Day 11: Critic → conditional routing
    graph.add_conditional_edges(
        "red_team_critic",
        route_after_critic,
        _all_nodes,
    )

    # Feature factory → conditional advance via DAG
    graph.add_conditional_edges(
        "feature_factory",
        route_after_feature_factory,
        _all_nodes,
    )

    # Day 11: Supervisor replan → re-enter at earliest affected node
    graph.add_conditional_edges(
        "supervisor_replan",
        route_after_supervisor_replan,
        _all_nodes,
    )

    graph.add_edge("submit", END)

    return graph.compile()


# ── Day 15: Graph singleton ─────────────────────────────────────────

_GRAPH = None
_GRAPH_LOCK = threading.Lock()


def get_graph():
    """
    Returns the compiled LangGraph graph, building it once per process.
    Thread-safe via double-checked locking.
    Resets on process restart (intended — code may have changed).
    """
    global _GRAPH
    if _GRAPH is None:
        with _GRAPH_LOCK:
            if _GRAPH is None:
                logger.info("[professor] Compiling LangGraph graph (first invocation)...")
                _GRAPH = build_graph()
                logger.info("[professor] Graph compiled and cached.")
    return _GRAPH


def get_graph_cache_clear() -> None:
    """
    Forces graph recompilation on next invocation.
    Use in tests that modify the graph structure between runs.
    Never call in production code.
    """
    global _GRAPH
    with _GRAPH_LOCK:
        _GRAPH = None
    logger.debug("[professor] Graph cache cleared (testing only).")


# ── Cost Management (Day 12) ──────────────────────────────────────

@contextlib.contextmanager
def _disable_langsmith_tracing():
    """Temporarily disables LangSmith tracing. Restores original value on exit, even on exception."""
    original = os.environ.get("LANGCHAIN_TRACING_V2", "false")
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    try:
        yield
    finally:
        os.environ["LANGCHAIN_TRACING_V2"] = original


def _log_estimated_cost(state: ProfessorState) -> None:
    """
    Rough cost estimate. Logged at end of each run.
    Formula: outer_nodes * avg_tokens_per_node * cost_per_1k_tokens
    """
    tracker = state.get("cost_tracker", {})
    budget = tracker.get("budget_usd", 0.0)
    
    in_tokens = tracker.get("groq_tokens_in", 0)
    out_tokens = tracker.get("groq_tokens_out", 0)
    
    # very rough standard cost
    current_cost = (in_tokens * 0.003 / 1000) + (out_tokens * 0.015 / 1000)
    
    print(
        f"[Professor] Actual tracked token cost this run: ${current_cost:.4f} "
        f"(Optuna tracing disabled/exempt). "
        f"Adjust LANGCHAIN_TRACING_SAMPLING_RATE in .env to change trace coverage."
    )

    if budget > 0:
        ratio = current_cost / budget
        if ratio >= tracker.get("triage_threshold", 0.95):
            logger.error(f"[Budget] CRITICAL: Budget usage {ratio:.1%} exceeds hitl threshold! Halt recommended.")
        elif ratio >= tracker.get("throttle_threshold", 0.85):
            logger.warning(f"[Budget] THROTTLE: Budget usage {ratio:.1%} exceeds throttle threshold.")
        elif ratio >= tracker.get("warning_threshold", 0.70):
            logger.warning(f"[Budget] WARNING: Budget usage {ratio:.1%} exceeds warning threshold.")


# ── Day 15: LangFuse observability ─────────────────────────────────

_LANGFUSE_CLIENT: Optional[object] = None
_LANGFUSE_ENABLED = False


def _init_langfuse() -> bool:
    """
    Initialises LangFuse client if keys are present.
    Returns True if LangFuse is active, False if degraded to JSONL.
    Never raises.
    """
    global _LANGFUSE_CLIENT
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    if not public_key or not secret_key:
        logger.info(
            "[observability] LangFuse keys not found. "
            "Tracing disabled — using JSONL lineage only. "
            "Add LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY to .env to enable."
        )
        return False

    try:
        from langfuse import Langfuse
        _LANGFUSE_CLIENT = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
        )
        logger.info(f"[observability] LangFuse connected ({host}).")
        return True
    except Exception as e:
        logger.warning(f"[observability] LangFuse init failed: {e}. Falling back to JSONL.")
        _LANGFUSE_CLIENT = None
        return False


# Call once at module load:
_LANGFUSE_ENABLED = _init_langfuse()


@contextlib.contextmanager
def _langfuse_trace(session_id: str, competition_name: str):
    """
    Context manager: wraps a full pipeline run in a LangFuse trace.
    Yields a trace object (or None if LangFuse disabled).
    """
    if not _LANGFUSE_ENABLED or _LANGFUSE_CLIENT is None:
        yield None
        return

    trace = _LANGFUSE_CLIENT.trace(
        name="professor_pipeline_run",
        session_id=session_id,
        metadata={
            "competition": competition_name,
            "started_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    try:
        yield trace
    except Exception as e:
        trace.update(status="ERROR", status_message=str(e)[:500])
        raise
    finally:
        trace.update(metadata={"ended_at": datetime.now(timezone.utc).isoformat()})
        _LANGFUSE_CLIENT.flush()


def _trace_node(trace, node_name: str, input_summary: dict, output_summary: dict) -> None:
    """
    Records a single node execution as a LangFuse span.
    No-op if trace is None (LangFuse disabled).
    """
    if trace is None:
        return
    try:
        trace.span(
            name=node_name,
            input=input_summary,
            output=output_summary,
        )
    except Exception:
        pass  # never let tracing break the pipeline


# ── Convenience runner ────────────────────────────────────────────

class ProfessorPipelineError(Exception):
    """Custom exception for pipeline failures with context."""
    def __init__(self, message, node=None, state_snapshot=None):
        super().__init__(message)
        self.node = node
        self.state_snapshot = state_snapshot
        self.timestamp = datetime.now(timezone.utc).isoformat()


def run_professor(
    state: ProfessorState,
    resume_from: str = None,
    timeout_seconds: int = 600,
) -> ProfessorState:
    """
    Run the full Professor graph with comprehensive error handling,
    checkpointing, and timeout.
    
    Args:
        state: Initial ProfessorState
        resume_from: Path to checkpoint to resume from (optional)
        timeout_seconds: Maximum execution time (default: 10 minutes)
    
    Returns:
        Final ProfessorState
    
    Raises:
        ProfessorPipelineError: If pipeline fails
    """
    session_id = state.get("session_id", "unknown")
    competition_name = state.get("competition_name", "unknown")
    
    # Initialize error context manager
    error_context = ErrorContextManager(session_id)
    
    # Resume from checkpoint if provided
    if resume_from and os.path.exists(resume_from):
        logger.info(f"Resuming from checkpoint: {resume_from}")
        checkpoint = load_last_checkpoint(session_id)
        if checkpoint:
            state.update(checkpoint["state"])
            state["resumed_from_checkpoint"] = resume_from
    
    try:
        # Start error context tracking
        error_context.start()
        logger.info(f"[Professor] Starting pipeline — session: {session_id}")
        
        # Run with timeout
        with timeout(timeout_seconds, "Pipeline execution"):
            with _langfuse_trace(session_id, competition_name) as trace:
                # NOTE: trace is NOT injected into state (non-serializable)
                graph = get_graph()
                result = graph.invoke(state)
        
        # ── Capture final LLM Tokens ───────────────────────────────
        from tools.llm_client import get_token_usage
        usage = get_token_usage()
        cost_tracker = dict(result.get("cost_tracker", {}))
        cost_tracker["groq_tokens_in"] = usage.get("prompt", 0)
        cost_tracker["groq_tokens_out"] = usage.get("completion", 0)
        result["cost_tracker"] = cost_tracker
        
        _log_estimated_cost(result)
        
        # Mark success
        error_context.success()
        logger.info(f"[Professor] Pipeline completed successfully — session: {session_id}")
        
        return result
        
    except CircuitBreakerError as e:
        # API circuit breaker opened
        error_context.record_error(e, traceback_str=traceback.format_exc())
        error_context.fail()
        
        # Save failure checkpoint
        save_node_checkpoint(state, session_id, "FAILURE")
        
        raise ProfessorPipelineError(
            f"API circuit breaker opened: {e}",
            node="circuit_breaker",
            state_snapshot=state
        ) from e
        
    except TimeoutError as e:
        # Pipeline timeout
        error_context.record_error(e, traceback_str=traceback.format_exc())
        error_context.fail()
        
        # Save failure checkpoint
        save_node_checkpoint(state, session_id, "FAILURE")
        
        raise ProfessorPipelineError(
            f"Pipeline timeout: {e}",
            node="timeout",
            state_snapshot=state
        ) from e
        
    except Exception as e:
        # General pipeline failure
        error_context.record_error(e, node=error_context.context.get("current_node"), traceback_str=traceback.format_exc())
        error_context.fail()
        
        # Save failure checkpoint for recovery
        save_node_checkpoint(state, session_id, "FAILURE")
        
        raise ProfessorPipelineError(
            f"Pipeline failed: {e}",
            node=error_context.context.get("current_node"),
            state_snapshot=state
        ) from e

