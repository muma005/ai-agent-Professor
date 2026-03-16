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

import pickle
import contextlib
import logging
import threading
import numpy as np
import polars as pl
import polars.selectors as cs
from typing import Optional
from datetime import datetime, timezone
from langgraph.graph import StateGraph, END
from core.state import ProfessorState
from agents.semantic_router import run_semantic_router
from agents.competition_intel import run_competition_intel
from agents.data_engineer import run_data_engineer
from agents.eda_agent import run_eda_agent
from agents.validation_architect import run_validation_architect
from agents.ml_optimizer import run_ml_optimizer
from agents.red_team_critic import run_red_team_critic
from agents.feature_factory import run_feature_factory
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
    """After Data Engineer: advance to EDA Agent."""
    return _advance_dag(state, current="data_engineer")


def route_after_eda(state: ProfessorState) -> str:
    """After EDA Agent: advance to Validation Architect."""
    return _advance_dag(state, current="eda_agent")


def route_after_validation(state: ProfessorState) -> str:
    """After Validation Architect: halt if HITL required, else advance."""
    if state.get("hitl_required"):
        print("[Professor] HITL required. Halting execution.")
        return END
    return _advance_dag(state, current="validation_architect")


def route_after_optimizer(state: ProfessorState) -> str:
    """After Optimizer: advance to red_team_critic."""
    return _advance_dag(state, current="ml_optimizer")


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
    from agents.ml_optimizer import _identify_target_column
    from core.metric_contract import load_contract
    from core.lineage import log_event

    session_id  = state["session_id"]
    output_dir  = f"outputs/{session_id}"
    competition = state["competition_name"]
    os.makedirs(output_dir, exist_ok=True)

    print(f"[Submit] Generating submission -- session: {session_id}")

    # ── Load test data ────────────────────────────────────────────
    test_path   = state["raw_data_path"].replace("train.csv", "test.csv")
    sample_path = state["raw_data_path"].replace("train.csv", "sample_submission.csv")

    if not os.path.exists(test_path):
        print(f"[Submit] WARNING: test.csv not found at {test_path}")
        return {**state, "submission_path": None}
    if not os.path.exists(sample_path):
        print(f"[Submit] WARNING: sample_submission.csv not found at {sample_path}")
        return {**state, "submission_path": None}

    test_df = read_csv(test_path)

    # ── Load model ────────────────────────────────────────────────
    if not state.get("model_registry"):
        raise ValueError("[Submit] No model in registry -- run ML Optimizer first")

    model_path = state["model_registry"][0]["model_path"]
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # ── Prepare test features (same encoding as training) ────────
    schema     = read_json(state["schema_path"])
    train_df   = read_parquet(state["clean_data_path"])
    target_col = _identify_target_column(schema, state)

    feature_cols = [c for c in train_df.columns
                    if c != target_col and c in test_df.columns]
    test_subset = test_df.select(feature_cols)

    for col in test_subset.columns:
        if test_subset[col].dtype in (pl.Utf8, pl.String):
            test_subset = test_subset.with_columns(
                pl.col(col).cast(pl.Categorical).cast(pl.Int32)
            )
        elif test_subset[col].dtype == pl.Boolean:
            test_subset = test_subset.with_columns(
                pl.col(col).cast(pl.Int32)
            )

    for col in test_subset.select(cs.numeric()).columns:
        test_subset = test_subset.with_columns(
            pl.col(col).fill_null(0)
        )

    X_test = test_subset.to_numpy().astype(np.float64)

    # ── Generate predictions ──────────────────────────────────────
    contract_path = f"{output_dir}/metric_contract.json"
    if os.path.exists(contract_path):
        contract = load_contract(contract_path)
        if contract.requires_proba:
            preds = model.predict_proba(X_test)[:, 1]
        else:
            preds = model.predict(X_test).astype(float)
    else:
        preds = model.predict(X_test).astype(float)

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
    graph.add_node("supervisor_replan", run_supervisor_replan)
    graph.add_node("submit",          run_submit)

    # ── Set entry point ───────────────────────────────────────────
    graph.set_entry_point("semantic_router")

    # ── Add edges ─────────────────────────────────────────────────
    graph.add_conditional_edges(
        "semantic_router",
        route_after_router,
        {
            "competition_intel": "competition_intel",
            END:             END,
        }
    )

    graph.add_conditional_edges(
        "competition_intel",
        route_after_intel,
        {
            "data_engineer": "data_engineer",
            END:             END,
        }
    )

    graph.add_conditional_edges(
        "data_engineer",
        route_after_data_engineer,
        {
            "eda_agent": "eda_agent",
            END:             END,
        }
    )

    graph.add_conditional_edges(
        "eda_agent",
        route_after_eda,
        {
            "validation_architect": "validation_architect",
            END:             END,
        }
    )

    graph.add_conditional_edges(
        "validation_architect",
        route_after_validation,
        {
            "ml_optimizer": "ml_optimizer",
            END:             END,
        }
    )

    graph.add_conditional_edges(
        "ml_optimizer",
        route_after_optimizer,
        {
            "red_team_critic": "red_team_critic",
            END:               END,
        }
    )

    # Day 11: Critic → conditional routing
    graph.add_conditional_edges(
        "red_team_critic",
        route_after_critic,
        {
            "supervisor_replan": "supervisor_replan",
            "submit":            "submit",
            END:                 END,
        }
    )

    # Feature factory stub → advance to ml_optimizer
    graph.add_edge("feature_factory", "ml_optimizer")

    # Day 11: Supervisor replan → re-enter at earliest affected node
    graph.add_conditional_edges(
        "supervisor_replan",
        route_after_supervisor_replan,
        {
            **{node: node for node in NODE_PRIORITY},
            END: END,
        }
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
    OUTER_NODES = [
        "semantic_router", "competition_intel", "data_engineer", "eda_agent",
        "validation_architect", "feature_factory", "red_team_critic",
        "ensemble_architect", "submission_strategist"
    ]
    AVG_TOKENS_PER_NODE = 3000
    COST_PER_1K = 0.003  # claude-sonnet approximate

    estimated_cost = len(OUTER_NODES) * AVG_TOKENS_PER_NODE * COST_PER_1K / 1000
    print(
        f"[Professor] Estimated LLM cost this run: ${estimated_cost:.4f} "
        f"(outer pipeline only, Optuna tracing disabled). "
        f"Adjust LANGCHAIN_TRACING_SAMPLING_RATE in .env to change trace coverage."
    )


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

def run_professor(state: ProfessorState) -> ProfessorState:
    """Run the full Professor graph from an initial state. Uses cached graph."""
    session_id = state.get("session_id", "unknown")
    competition_name = state.get("competition_name", "unknown")

    with _langfuse_trace(session_id, competition_name) as trace:
        state = {**state, "_langfuse_trace": trace}
        graph = get_graph()
        result = graph.invoke(state)

    _log_estimated_cost(result)
    return result

