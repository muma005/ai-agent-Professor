# Day 5 Tasks
Confirmed. Notion has exactly **3 tasks on Day 5** — all Phase 1, all Critical, all Never Cut.

---

## Day 5 Tasks — Pulled From Notion

```
┌────┬──────────────────────────────────────────────┬─────────────────────────┬──────────┬───────────┐
│ #  │ Task                                         │ Phase                   │ Priority │ Cuttable  │
├────┼──────────────────────────────────────────────┼─────────────────────────┼──────────┼───────────┤
│ 1  │ Build agents/semantic_router.py — v0         │ 🚀 Phase 1: Make It Run │ Critical │ Never Cut │
│ 2  │ Build core/professor.py — LangGraph wiring   │ 🚀 Phase 1: Make It Run │ Critical │ Never Cut │
│ 3  │ Write contract test — Semantic Router        │ 🚀 Phase 1: Make It Run │ Critical │ Never Cut │
└────┴──────────────────────────────────────────────┴─────────────────────────┴──────────┴───────────┘
```

Notion notes verbatim:
- **semantic_router.py**: v0: simple linear routing only. Supervisor → Data Engineer → Optimizer → Submit. No DAG yet. Add DAG in Phase 2.
- **professor.py**: Wire nodes into LangGraph StateGraph. State flows correctly between nodes. Add edges: router → data_engineer → optimizer → submit.
- **contract test**: INPUT: problem_statement, raw_data_path. OUTPUT: dag populated in state, task_type set, metric_contract initialized. CONSTRAINT: Router must never write code or touch data directly. Verify it only mutates routing fields in state.

**The ONE thing that must work by end of today:**
`python main.py run --competition spaceship-titanic --data ./data/spaceship_titanic/` runs the full LangGraph graph end to end — router → data engineer → optimizer — without crashing. First time the pipeline runs as a connected system.

Build order: Task 1 → Task 2 → Task 3. Professor.py imports from semantic_router, so router must exist first.

---

## Task 1 — Build `agents/semantic_router.py` — v0

v0 is intentionally simple. No LLM calls, no DAG building, no complex routing logic. Linear only. The DAG and LLM-driven routing come in Phase 2.

```python
# agents/semantic_router.py

from core.state import ProfessorState


# ── v0 linear route ───────────────────────────────────────────────
# Phase 1: fixed sequence, no branching, no LLM involvement.
# Phase 2: replaced with LLM-driven DAG construction.

LINEAR_ROUTE_V0 = [
    "data_engineer",
    "ml_optimizer",
    "submit",
]


def run_semantic_router(state: ProfessorState) -> ProfessorState:
    """
    LangGraph node: Semantic Router v0.

    Phase 1 behaviour:
      - Sets task_type to "tabular_classification" (hardcoded)
      - Populates state["dag"] with the linear route
      - Sets state["next_node"] to first node in route
      - Never writes code, never touches data files

    Reads:   state["competition_name"], state["task_type"]
    Writes:  state["dag"], state["task_type"], state["next_node"],
             state["current_node"]
    NEVER:   writes code, reads/writes data files, calls external APIs
    """
    competition = state["competition_name"]
    task_type   = state.get("task_type", "auto")

    print(f"[SemanticRouter] Competition: {competition}")

    # ── Task type detection — v0: rule-based, v1: LLM-driven ─────
    if task_type == "auto":
        task_type = _detect_task_type(competition)

    print(f"[SemanticRouter] Task type: {task_type}")
    print(f"[SemanticRouter] Route: {' → '.join(LINEAR_ROUTE_V0)}")

    return {
        **state,
        "task_type":    task_type,
        "dag":          LINEAR_ROUTE_V0.copy(),
        "current_node": "semantic_router",
        "next_node":    LINEAR_ROUTE_V0[0],
    }


def _detect_task_type(competition_name: str) -> str:
    """
    Rule-based task type detection for v0.
    Phase 2: replaced with LLM parsing of competition description.
    """
    name = competition_name.lower()

    # Known time-series patterns
    if any(kw in name for kw in ["forecast", "time-series", "timeseries",
                                  "temporal", "sales", "demand", "predict-future"]):
        return "timeseries"

    # Known regression patterns
    if any(kw in name for kw in ["price", "cost", "revenue", "salary",
                                  "amount", "value", "regression", "house"]):
        return "tabular_regression"

    # Default: classification
    return "tabular_classification"
```

---

## Task 2 — Build `core/professor.py` — LangGraph Graph Wiring

This is the most important file in the project. Everything built so far connects here.

```python
# core/professor.py

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
    print(f"[Professor] DAG advance: {current} → {next_node}")
    return next_node


# ── Submit node — Phase 1 stub ────────────────────────────────────

def run_submit(state: ProfessorState) -> ProfessorState:
    """
    Phase 1 stub: generate submission.csv from OOF predictions.
    Full implementation Day 6 with tools/submit_tools.py.
    """
    import os
    import numpy as np
    import polars as pl
    from tools.data_tools import read_csv

    session_id = state["session_id"]
    output_dir = f"outputs/{session_id}"

    print(f"[Submit] Generating submission — session: {session_id}")

    # Load test data
    test_path = state["raw_data_path"].replace("train.csv", "test.csv")
    if not os.path.exists(test_path):
        print(f"[Submit] WARNING: test.csv not found at {test_path}")
        print(f"[Submit] Stub: writing empty submission.csv")
        submission_path = f"{output_dir}/submission.csv"
        pl.DataFrame({"stub": []}).write_csv(submission_path)
        return {**state, "submission_path": submission_path}

    test_df = read_csv(test_path)

    # Load best model and generate predictions
    import pickle
    if not state.get("model_registry"):
        raise ValueError("[Submit] No model in registry — run ML Optimizer first")

    model_path = state["model_registry"][0]["model_path"]
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Prepare test features (same logic as optimizer)
    from tools.data_tools import read_parquet, read_json
    from agents.ml_optimizer import _identify_target_column, _prepare_features

    schema     = read_json(state["schema_path"])
    train_df   = read_parquet(state["clean_data_path"])
    target_col = _identify_target_column(schema, state)

    # Prepare test set — drop target if it exists, encode same way
    test_features = [c for c in train_df.columns if c != target_col
                     and c in test_df.columns]

    test_subset = test_df.select(test_features)

    # Encode string columns
    for col in test_subset.columns:
        if test_subset[col].dtype in (pl.Utf8, pl.String):
            test_subset = test_subset.with_columns(
                pl.col(col).cast(pl.Categorical).cast(pl.Int32)
            )

    # Fill nulls
    import polars.selectors as cs
    for col in test_subset.select(cs.numeric()).columns:
        test_subset = test_subset.with_columns(
            pl.col(col).fill_null(0)
        )

    X_test = test_subset.to_numpy()

    # Predict
    from core.metric_contract import load_contract, PROBABILITY_METRICS
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
```

Now wire it into `main.py` — replace the `_run` stub from Day 1:

```python
# main.py — update _run() only, everything else stays the same

def _run(args):
    from core.state import initial_state
    from core.professor import run_professor

    if not os.path.exists(args.data):
        print(f"[ERROR] Data path does not exist: {args.data}")
        sys.exit(1)

    state = initial_state(
        competition=args.competition,
        data_path=args.data,
        budget_usd=args.budget,
        task_type=args.task_type
    )

    print(f"[Professor] Session:     {state['session_id']}")
    print(f"[Professor] Competition: {state['competition_name']}")
    print(f"[Professor] Data:        {state['raw_data_path']}")
    print(f"[Professor] Budget:      ${state['cost_tracker']['budget_usd']:.2f}")
    print()

    result = run_professor(state)

    print()
    print(f"[Professor] ✓ Complete")
    print(f"[Professor] CV score:    {result.get('cv_mean', 'N/A')}")
    print(f"[Professor] Submission:  {result.get('submission_path', 'N/A')}")
    print(f"[Professor] LLM calls:   {result['cost_tracker']['llm_calls']}")
```

---

## Task 3 — Write Contract Test — Semantic Router (Immutable From Today)

```python
# tests/contracts/test_semantic_router_contract.py
# ─────────────────────────────────────────────────────────────────
# Written: Day 5
# Status:  IMMUTABLE — never edit this file after today
#
# CONTRACT: run_semantic_router()
#   INPUT:   state["competition_name"] — str
#            state["task_type"]        — str ("auto" or explicit)
#   OUTPUT:  state["dag"]         — non-empty list of node names
#            state["task_type"]   — str, one of known task types
#            state["next_node"]   — str, first node in DAG
#            state["current_node"]— "semantic_router"
#   NEVER:   writes code
#            reads or writes data files
#            calls external APIs
#            mutates raw_data_path, clean_data_path, model_registry
#            or any non-routing field in state
# ─────────────────────────────────────────────────────────────────
import pytest
from core.state import initial_state
from agents.semantic_router import run_semantic_router

KNOWN_TASK_TYPES = {
    "tabular_classification",
    "tabular_regression",
    "timeseries"
}

KNOWN_NODES = {
    "data_engineer",
    "ml_optimizer",
    "submit",
    "eda_agent",
    "feature_factory",
    "red_team_critic",
    "ensemble_architect",
    "validation_architect",
    "publisher",
}


@pytest.fixture
def base_state():
    return initial_state(
        competition="spaceship-titanic",
        data_path="tests/fixtures/tiny_train.csv",
        budget_usd=2.0
    )


class TestSemanticRouterContract:

    def test_runs_without_error(self, base_state):
        result = run_semantic_router(base_state)
        assert result is not None

    def test_dag_is_populated(self, base_state):
        result = run_semantic_router(base_state)
        assert result.get("dag") is not None
        assert isinstance(result["dag"], list)
        assert len(result["dag"]) > 0, "DAG must not be empty"

    def test_dag_contains_valid_node_names(self, base_state):
        result  = run_semantic_router(base_state)
        for node in result["dag"]:
            assert isinstance(node, str), f"DAG node must be str, got {type(node)}"

    def test_task_type_is_set(self, base_state):
        result = run_semantic_router(base_state)
        assert result.get("task_type") is not None
        assert result["task_type"] in KNOWN_TASK_TYPES, \
            f"task_type '{result['task_type']}' not in {KNOWN_TASK_TYPES}"

    def test_next_node_is_first_dag_node(self, base_state):
        result = run_semantic_router(base_state)
        assert result["next_node"] == result["dag"][0], \
            "next_node must be the first node in the DAG"

    def test_current_node_is_semantic_router(self, base_state):
        result = run_semantic_router(base_state)
        assert result["current_node"] == "semantic_router"

    def test_explicit_task_type_preserved(self, base_state):
        state  = {**base_state, "task_type": "tabular_regression"}
        result = run_semantic_router(state)
        assert result["task_type"] == "tabular_regression", \
            "Explicit task_type must not be overridden by auto-detection"

    def test_auto_detects_classification(self):
        state  = initial_state("spaceship-titanic", "tests/fixtures/tiny_train.csv")
        result = run_semantic_router(state)
        assert result["task_type"] == "tabular_classification"

    def test_auto_detects_regression(self):
        state  = initial_state("house-price-prediction", "tests/fixtures/tiny_train.csv")
        result = run_semantic_router(state)
        assert result["task_type"] == "tabular_regression"

    def test_never_touches_data_paths(self, base_state):
        before_clean  = base_state.get("clean_data_path")
        before_schema = base_state.get("schema_path")
        result        = run_semantic_router(base_state)
        assert result.get("clean_data_path") == before_clean, \
            "Router must never modify clean_data_path"
        assert result.get("schema_path") == before_schema, \
            "Router must never modify schema_path"

    def test_never_modifies_model_registry(self, base_state):
        before = base_state.get("model_registry")
        result = run_semantic_router(base_state)
        assert result.get("model_registry") == before, \
            "Router must never modify model_registry"

    def test_never_modifies_cost_tracker_llm_calls(self, base_state):
        """Router v0 makes no LLM calls — llm_calls must not increase."""
        before = base_state["cost_tracker"]["llm_calls"]
        result = run_semantic_router(base_state)
        after  = result["cost_tracker"]["llm_calls"]
        assert after == before, \
            f"Router v0 must not make LLM calls. Before: {before}, After: {after}"

    def test_raw_data_path_unchanged(self, base_state):
        before = base_state["raw_data_path"]
        result = run_semantic_router(base_state)
        assert result["raw_data_path"] == before, \
            "Router must never modify raw_data_path"
```

---

## End of Day 5 Checklist

```bash
# 1. Run today's contract test
pytest tests/contracts/test_semantic_router_contract.py -v
# All tests green

# 2. Run ALL contracts — Days 2 + 3 + 4 + 5
pytest tests/contracts/ -v
# Zero regressions from today's wiring

# 3. THE BIG TEST — first full pipeline run
python main.py run \
  --competition spaceship-titanic \
  --data ./data/spaceship_titanic/

# Expected output:
# [Professor] Session:     spaceship_abc123de
# [Professor] Competition: spaceship-titanic
# [SemanticRouter] Task type: tabular_classification
# [SemanticRouter] Route: data_engineer → ml_optimizer → submit
# [DataEngineer] Loaded: 8693 rows, 14 columns
# [DataEngineer] Complete. data_hash: a3f9c21d
# [MLOptimizer] CV AUC: 0.8XXX (+/- 0.00XX)
# [Submit] submission.csv saved: outputs/spaceship_abc123de/submission.csv
# [Professor] ✓ Complete
# [Professor] CV score:   0.8XXX
# [Professor] Submission: outputs/spaceship_abc123de/submission.csv

# 4. Verify submission.csv exists and has correct format
python -c "
import polars as pl
df = pl.read_csv('outputs/spaceship-titanic_abc123/submission.csv')
print('Columns:', df.columns)
print('Shape:  ', df.shape)
print('Sample: ')
print(df.head(3))
# Must have: PassengerId + Transported columns, 4277 rows
"

# 5. Commit
git add .
git commit -m "Day 5: semantic_router + professor graph wiring + contract test — full pipeline runs, CV: X.XXXX"
git push origin phase-1
```

---

## Day 6 Preview

Tomorrow: `tools/submit_tools.py` (proper submission generator replacing the Phase 1 stub), then the full vertical slice test — `python main.py run` must produce a properly validated submission.csv that can be uploaded directly to Kaggle. Day 7 is the Phase 1 gate, so Day 6 is the polish day before the gate.