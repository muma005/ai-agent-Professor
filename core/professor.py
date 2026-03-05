# core/professor.py

import os
import pickle
import numpy as np
import polars as pl
import polars.selectors as cs
from langgraph.graph import StateGraph, END
from core.state import ProfessorState
from agents.semantic_router import run_semantic_router
from agents.data_engineer import run_data_engineer
from agents.ml_optimizer import run_ml_optimizer


# ── Routing functions (conditional edges) ─────────────────────────

def route_after_router(state: ProfessorState) -> str:
    """After router runs: go to first node in DAG."""
    next_node = state.get("next_node")
    dag       = state.get("dag", [])

    if not dag:
        print("[Professor] WARNING: DAG is empty after router. Ending.")
        return END

    print(f"[Professor] Routing to: {next_node}")
    return next_node


def route_after_data_engineer(state: ProfessorState) -> str:
    """After Data Engineer: advance to next node in DAG."""
    return _advance_dag(state, current="data_engineer")


def route_after_optimizer(state: ProfessorState) -> str:
    """After Optimizer: advance to next node in DAG."""
    return _advance_dag(state, current="ml_optimizer")


def _advance_dag(state: ProfessorState, current: str) -> str:
    """
    Find current node in DAG and return the next one.
    If current is last node, return END.
    """
    dag = state.get("dag", [])

    if current not in dag:
        print(f"[Professor] '{current}' not in DAG — ending.")
        return END

    idx = dag.index(current)

    if idx + 1 >= len(dag):
        print(f"[Professor] '{current}' is last node — ending.")
        return END

    next_node = dag[idx + 1]
    print(f"[Professor] DAG advance: {current} -> {next_node}")
    return next_node


# ── Submit node — Phase 1 stub ────────────────────────────────────

def run_submit(state: ProfessorState) -> ProfessorState:
    """
    Phase 1 stub: generate submission.csv from best model.
    Full implementation Day 6 with tools/submit_tools.py.
    """
    from tools.data_tools import read_csv, read_parquet, read_json
    from agents.ml_optimizer import _identify_target_column
    from core.metric_contract import load_contract, PROBABILITY_METRICS

    session_id = state["session_id"]
    output_dir = f"outputs/{session_id}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"[Submit] Generating submission — session: {session_id}")

    # Load test data
    test_path = state["raw_data_path"].replace("train.csv", "test.csv")
    if not os.path.exists(test_path):
        print(f"[Submit] WARNING: test.csv not found at {test_path}")
        print(f"[Submit] Stub: skipping submission generation.")
        return {**state, "submission_path": None}

    test_df = read_csv(test_path)

    # Load best model
    if not state.get("model_registry"):
        raise ValueError("[Submit] No model in registry — run ML Optimizer first")

    model_path = state["model_registry"][0]["model_path"]
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Identify target and feature columns from the training schema
    schema     = read_json(state["schema_path"])
    train_df   = read_parquet(state["clean_data_path"])
    target_col = _identify_target_column(schema, state)

    # Prepare test set — use same feature columns as training
    test_features = [c for c in train_df.columns if c != target_col
                     and c in test_df.columns]
    test_subset = test_df.select(test_features)

    # Encode string columns as integer codes (same as _prepare_features)
    for col in test_subset.columns:
        if test_subset[col].dtype in (pl.Utf8, pl.String):
            test_subset = test_subset.with_columns(
                pl.col(col).cast(pl.Categorical).cast(pl.Int32)
            )
        elif test_subset[col].dtype == pl.Boolean:
            test_subset = test_subset.with_columns(
                pl.col(col).cast(pl.Int32)
            )

    # Fill nulls
    for col in test_subset.select(cs.numeric()).columns:
        test_subset = test_subset.with_columns(
            pl.col(col).fill_null(0)
        )
    for col in test_subset.select(cs.string()).columns:
        test_subset = test_subset.with_columns(
            pl.col(col).fill_null("missing")
        )

    X_test = test_subset.to_numpy().astype(np.float64)

    # Predict
    contract_path = f"{output_dir}/metric_contract.json"
    if os.path.exists(contract_path):
        contract = load_contract(contract_path)
        if contract.requires_proba:
            preds = model.predict_proba(X_test)[:, 1]
            preds = preds > 0.5  # convert to bool for classification
        else:
            preds = model.predict(X_test)
    else:
        preds = model.predict(X_test)

    # Build submission
    id_col = test_df.columns[0]  # first column is usually ID
    submission = pl.DataFrame({
        id_col:     test_df[id_col].to_list(),
        target_col: preds.tolist(),
    })

    submission_path = f"{output_dir}/submission.csv"
    submission.write_csv(submission_path)

    print(f"[Submit] submission.csv saved: {submission_path}")
    print(f"[Submit] Rows: {len(submission)} | Columns: {submission.columns}")

    return {
        **state,
        "submission_path": submission_path,
    }


# ── Build the graph ───────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Assemble the Professor LangGraph StateGraph.

    Phase 1 graph:
      semantic_router → data_engineer → ml_optimizer → submit → END

    Phase 2+: conditional edges, parallel branches, Critic loop added here.
    """
    graph = StateGraph(ProfessorState)

    # ── Add nodes ─────────────────────────────────────────────────
    graph.add_node("semantic_router", run_semantic_router)
    graph.add_node("data_engineer",   run_data_engineer)
    graph.add_node("ml_optimizer",    run_ml_optimizer)
    graph.add_node("submit",          run_submit)

    # ── Set entry point ───────────────────────────────────────────
    graph.set_entry_point("semantic_router")

    # ── Add edges ─────────────────────────────────────────────────
    graph.add_conditional_edges(
        "semantic_router",
        route_after_router,
        {
            "data_engineer": "data_engineer",
            END:              END,
        }
    )

    graph.add_conditional_edges(
        "data_engineer",
        route_after_data_engineer,
        {
            "ml_optimizer": "ml_optimizer",
            END:             END,
        }
    )

    graph.add_conditional_edges(
        "ml_optimizer",
        route_after_optimizer,
        {
            "submit": "submit",
            END:       END,
        }
    )

    graph.add_edge("submit", END)

    return graph.compile()


# ── Convenience runner ────────────────────────────────────────────

def run_professor(state: ProfessorState) -> ProfessorState:
    """Run the full Professor graph from an initial state."""
    graph  = build_graph()
    result = graph.invoke(state)
    return result
