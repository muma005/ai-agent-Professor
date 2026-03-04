# tests/contracts/test_data_engineer_contract.py
# ─────────────────────────────────────────────────────────────────
# Written: Day 3
# Status:  IMMUTABLE — never edit this file after today
#
# CONTRACT: run_data_engineer()
#   INPUT:   state["raw_data_path"] — str, must exist on disk
#   OUTPUT:  outputs/{session_id}/cleaned.parquet — must exist
#            outputs/{session_id}/schema.json — must have:
#              columns (list), types (dict), missing_rates (dict)
#   STATE:   clean_data_path — str pointer (not DataFrame)
#            schema_path     — str pointer
#            data_hash       — 16-char hex string
#            cost_tracker    — llm_calls incremented
#   NEVER:   raw DataFrame in state
#            raw DataFrame in any state field
# ─────────────────────────────────────────────────────────────────
import pytest
import os
import json
import polars as pl
from pathlib import Path
from core.state import initial_state
from agents.data_engineer import run_data_engineer

# ── Fixture: minimal CSV the tests always use ─────────────────────
FIXTURE_CSV = "tests/fixtures/tiny_train.csv"

@pytest.fixture(scope="session", autouse=True)
def create_fixture_csv():
    """Create a minimal CSV fixture for contract tests."""
    os.makedirs("tests/fixtures", exist_ok=True)
    if not os.path.exists(FIXTURE_CSV):
        df = pl.DataFrame({
            "PassengerId": ["0001_01", "0002_01", "0003_01",
                            "0004_01", "0005_01"],
            "HomePlanet":  ["Europa", "Earth", None, "Mars", "Earth"],
            "Age":         [39.0, 24.0, None, 58.0, 33.0],
            "RoomService": [0.0, 109.0, None, 43.0, 0.0],
            "Transported": [False, True, True, False, True],
        })
        df.write_csv(FIXTURE_CSV)


@pytest.fixture
def base_state():
    return initial_state(
        competition="test-titanic",
        data_path=FIXTURE_CSV,
        budget_usd=2.0
    )


class TestDataEngineerContract:

    def test_accepts_valid_raw_data_path(self, base_state):
        result = run_data_engineer(base_state)
        assert result is not None

    def test_rejects_nonexistent_path(self, base_state):
        bad_state = {**base_state, "raw_data_path": "/nonexistent/train.csv"}
        with pytest.raises(FileNotFoundError):
            run_data_engineer(bad_state)

    def test_produces_cleaned_parquet(self, base_state):
        result = run_data_engineer(base_state)
        assert os.path.exists(result["clean_data_path"]), \
            "cleaned.parquet must exist after run"

    def test_produces_schema_json(self, base_state):
        result = run_data_engineer(base_state)
        assert os.path.exists(result["schema_path"]), \
            "schema.json must exist after run"

    def test_schema_has_columns_field(self, base_state):
        result = run_data_engineer(base_state)
        schema = json.loads(Path(result["schema_path"]).read_text())
        assert "columns" in schema, "schema.json must have 'columns'"
        assert isinstance(schema["columns"], list)
        assert len(schema["columns"]) > 0

    def test_schema_has_types_field(self, base_state):
        result = run_data_engineer(base_state)
        schema = json.loads(Path(result["schema_path"]).read_text())
        assert "types" in schema, "schema.json must have 'types'"
        assert isinstance(schema["types"], dict)

    def test_schema_has_missing_rates_field(self, base_state):
        result = run_data_engineer(base_state)
        schema = json.loads(Path(result["schema_path"]).read_text())
        assert "missing_rates" in schema, "schema.json must have 'missing_rates'"
        assert isinstance(schema["missing_rates"], dict)

    def test_clean_data_path_is_string_not_dataframe(self, base_state):
        result = run_data_engineer(base_state)
        assert isinstance(result["clean_data_path"], str), \
            "clean_data_path must be a str pointer — never a DataFrame"

    def test_no_raw_data_in_state(self, base_state):
        result = run_data_engineer(base_state)
        for key, value in result.items():
            assert not isinstance(value, pl.DataFrame), \
                f"DataFrame found in state['{key}'] — only pointers allowed"

    def test_data_hash_set_in_state(self, base_state):
        result = run_data_engineer(base_state)
        assert "data_hash" in result
        assert isinstance(result["data_hash"], str)
        assert len(result["data_hash"]) == 16, \
            "data_hash must be 16-char hex string"

    def test_cost_tracker_llm_calls_incremented(self, base_state):
        before = base_state["cost_tracker"]["llm_calls"]
        result = run_data_engineer(base_state)
        after  = result["cost_tracker"]["llm_calls"]
        assert after >= before, \
            "cost_tracker.llm_calls must be incremented after run"

    def test_parquet_is_polars_readable(self, base_state):
        result = run_data_engineer(base_state)
        df = pl.read_parquet(result["clean_data_path"])
        assert isinstance(df, pl.DataFrame), \
            "cleaned.parquet must be readable as Polars DataFrame"

    def test_parquet_has_no_object_dtype(self, base_state):
        result = run_data_engineer(base_state)
        df = pl.read_parquet(result["clean_data_path"])
        object_cols = [c for c in df.columns if df[c].dtype == pl.Object]
        assert len(object_cols) == 0, \
            f"Object dtype columns detected (Pandas contamination): {object_cols}"

    def test_no_nulls_in_cleaned_parquet(self, base_state):
        result = run_data_engineer(base_state)
        df = pl.read_parquet(result["clean_data_path"])
        total_nulls = df.null_count().sum_horizontal().item()
        assert total_nulls == 0, \
            f"cleaned.parquet should have 0 nulls after cleaning, found {total_nulls}"

    def test_session_id_namespacing(self, base_state):
        """Output files must live under outputs/{session_id}/"""
        result = run_data_engineer(base_state)
        session_id = base_state["session_id"]
        assert session_id in result["clean_data_path"], \
            "clean_data_path must be namespaced under session_id"
        assert session_id in result["schema_path"], \
            "schema_path must be namespaced under session_id"
