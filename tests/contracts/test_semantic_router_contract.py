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
