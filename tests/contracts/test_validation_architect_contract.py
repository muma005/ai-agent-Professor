# tests/contracts/test_validation_architect_contract.py

import pytest
import os
import json
from pathlib import Path
from core.state import initial_state
from agents.validation_architect import run_validation_architect

FIXTURE_SCHEMA = "outputs/test-va/schema.json"

# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def validation_architect_state():
    """State after DataEngineer."""
    # Ensure a dummy schema exists
    os.makedirs("outputs/test-va", exist_ok=True)
    schema = {
        "target_col": "target",
        "columns": ["id", "feature1", "target"],
        "types": {"id": "Int64", "feature1": "Float64", "target": "Int64"},
        "cardinality": {"target": 2}
    }
    with open(FIXTURE_SCHEMA, "w") as f:
        json.dump(schema, f)

    return initial_state(
        session_id="test-va",
        schema_path=FIXTURE_SCHEMA,
        target_col="target"
    )

# ── Tests ───────────────────────────────────────────────────────────────────

class TestValidationArchitectContract:
    """
    Contract: Validation Architect Agent
    Ensures correct CV selection and MetricContract generation.
    """

    def test_validation_strategy_has_cv_type(self, validation_architect_state):
        """Verify cv_type is chosen and valid."""
        state = run_validation_architect(validation_architect_state)
        strategy = state["validation_strategy"]
        assert strategy["cv_type"] in {"StratifiedKFold", "KFold", "GroupKFold", "TimeSeriesSplit"}

    def test_validation_strategy_has_n_splits(self, validation_architect_state):
        """Verify n_splits is an integer >= 2."""
        state = run_validation_architect(validation_architect_state)
        assert isinstance(state["validation_strategy"]["n_splits"], int)
        assert state["validation_strategy"]["n_splits"] >= 2

    def test_validation_strategy_has_group_col_key(self, validation_architect_state):
        """Verify group_col key exists (even if None)."""
        state = run_validation_architect(validation_architect_state)
        assert "group_col" in state["validation_strategy"]

    def test_metric_contract_json_exists(self, validation_architect_state):
        """Verify metric_contract.json is persisted."""
        state = run_validation_architect(validation_architect_state)
        assert Path(state["metric_contract_path"]).exists()

    def test_metric_contract_has_scorer_name(self, validation_architect_state):
        """Verify scorer_name is chosen and non-empty."""
        state = run_validation_architect(validation_architect_state)
        assert state["validation_strategy"]["scorer_name"] != ""

    def test_metric_contract_has_direction(self, validation_architect_state):
        """Verify optimization direction is decided."""
        state = run_validation_architect(validation_architect_state)
        # We check the dict in state
        contract = state["metric_contract"]
        assert contract["direction"] in {"minimize", "maximize"}

    def test_metric_contract_has_forbidden_metrics(self, validation_architect_state):
        """Verify forbidden_metrics list exists."""
        state = run_validation_architect(validation_architect_state)
        assert isinstance(state["metric_contract"]["forbidden_metrics"], list)

    def test_task_type_written_to_state(self, validation_architect_state):
        """Verify task_type is updated in state."""
        state = run_validation_architect(validation_architect_state)
        assert state["task_type"] in {"classification", "regression"}

    def test_no_hitl_on_clean_tabular_data(self, validation_architect_state):
        """Verify HITL not required for standard StratifiedKFold data."""
        state = run_validation_architect(validation_architect_state)
        assert state["hitl_required"] is False
