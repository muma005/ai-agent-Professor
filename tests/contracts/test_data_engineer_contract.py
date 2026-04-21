# tests/contracts/test_data_engineer_contract.py

import pytest
import os
import json
import polars as pl
from pathlib import Path
from core.state import initial_state, ProfessorState
from agents.data_engineer import run_data_engineer

FIXTURE_TRAIN = "data/spaceship_titanic/train.csv"

# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def data_engineer_state():
    """Valid initial state for DataEngineer."""
    return initial_state(
        session_id="test-de",
        competition_name="spaceship-titanic",
        raw_data_path=FIXTURE_TRAIN
    )

# ── Tests ───────────────────────────────────────────────────────────────────

class TestDataEngineerContract:
    """
    Contract: Data Engineer Agent
    Ensures structural data analysis and preprocessor persistence.
    """

    def test_target_column_detection(self, data_engineer_state):
        """Verify target column is detected or read from state."""
        state = run_data_engineer(data_engineer_state)
        # For Spaceship Titanic, target is 'Transported'
        assert state["target_col"] == "Transported"

    def test_id_column_detection(self, data_engineer_state):
        """Verify ID columns are identified correctly."""
        state = run_data_engineer(data_engineer_state)
        # Spaceship Titanic has 'PassengerId' as ID
        assert "PassengerId" in state["id_columns"]
        # Target must not be in ID columns
        assert state["target_col"] not in state["id_columns"]

    def test_task_type_detection(self, data_engineer_state):
        """Verify task type is detected correctly."""
        state = run_data_engineer(data_engineer_state)
        # Transported is boolean -> binary
        assert state["task_type"] == "binary"

    def test_clean_data_parquet_exists(self, data_engineer_state):
        """Verify cleaned data is persisted as Parquet."""
        state = run_data_engineer(data_engineer_state)
        path = Path(state["clean_data_path"])
        assert path.exists()
        assert path.suffix == ".parquet"

    def test_preprocessor_pkl_exists(self, data_engineer_state):
        """Verify preprocessor is persisted as PKL."""
        state = run_data_engineer(data_engineer_state)
        path = Path(state["preprocessor_path"])
        assert path.exists()
        assert path.suffix == ".pkl"

    def test_schema_json_exists_and_valid(self, data_engineer_state):
        """Verify schema.json is written and contains authority fields."""
        state = run_data_engineer(data_engineer_state)
        path = Path(state["schema_path"])
        assert path.exists()
        
        schema = json.loads(path.read_text())
        assert schema["target_col"] == state["target_col"]
        assert schema["task_type"] == state["task_type"]
        assert "columns" in schema

    def test_data_hash_is_consistent(self, data_engineer_state):
        """Verify data hash is generated and non-empty."""
        state = run_data_engineer(data_engineer_state)
        assert len(state["data_hash"]) > 0
        assert isinstance(state["data_hash"], str)

    def test_immutability_enforced(self, data_engineer_state):
        """Verify [IMMUTABLE] fields cannot be overwritten by other agents."""
        state = run_data_engineer(data_engineer_state)
        
        # Now try to overwrite as another agent
        with pytest.raises(Exception): # OwnershipError or ImmutableFieldError
            ProfessorState.validated_update(state, "eda_agent", {
                "canonical_train_rows": 999999
            })

    def test_missing_input_raises_error(self):
        """Verify pipeline is halted when raw_data_path is missing."""
        state = initial_state(session_id="err", raw_data_path="missing.csv")
        new_state = run_data_engineer(state)
        
        assert new_state["pipeline_halted"] is True
        assert "not valid: missing.csv" in new_state["pipeline_halt_reason"]
