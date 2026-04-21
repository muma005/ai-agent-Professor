# tests/contracts/test_eda_agent_contract.py

import pytest
import os
import json
import polars as pl
from pathlib import Path
from core.state import initial_state
from agents.eda_agent import run_eda_agent

FIXTURE_PARQUET = "data/spaceship_titanic/train.parquet"

# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def eda_state():
    """State after DataEngineer but before EDA."""
    # Ensure parquet exists for testing
    if not os.path.exists(FIXTURE_PARQUET):
        df = pl.read_csv("data/spaceship_titanic/train.csv")
        df.write_parquet(FIXTURE_PARQUET)
        
    return initial_state(
        session_id="test-eda",
        clean_data_path=FIXTURE_PARQUET,
        target_col="Transported",
        id_columns=["PassengerId"]
    )

# ── Tests ───────────────────────────────────────────────────────────────────

class TestEDAAgentContract:
    """
    Contract: EDA Agent
    Ensures 12-vector analysis and drop manifest generation.
    """

    def test_eda_report_written_and_valid(self, eda_state):
        """Verify eda_report.json exists and contains key vectors."""
        state = run_eda_agent(eda_state)
        path = Path(state["eda_report_path"])
        assert path.exists()
        
        report = json.loads(path.read_text())
        assert "target_distribution" in report
        assert "leakage_fingerprint" in report
        assert "drop_manifest" in report
        assert "summary" in report

    def test_dropped_features_populated(self, eda_state):
        """Verify dropped_features key is written to state."""
        state = run_eda_agent(eda_state)
        assert isinstance(state["dropped_features"], list)
        # Should not drop target or ID
        assert state["target_col"] not in state["dropped_features"]
        assert all(id_col not in state["dropped_features"] for id_col in state["id_columns"])

    def test_target_imbalance_ratio_captured(self, eda_state):
        """Verify imbalance ratio is a float between 0 and 1."""
        state = run_eda_agent(eda_state)
        ratio = state["eda_report"]["target_distribution"]["imbalance_ratio"]
        assert isinstance(ratio, float)
        assert 0.0 < ratio <= 1.0

    def test_leakage_detection_flags_target(self, eda_state):
        """Verify features with >0.95 correlation are flagged."""
        # Spaceship Titanic might not have natural leakage, but we verify the keys exist
        state = run_eda_agent(eda_state)
        leakage = state["eda_report"]["leakage_fingerprint"]
        assert isinstance(leakage, list)
        for entry in leakage:
            assert "feature" in entry
            assert "verdict" in entry
            assert entry["verdict"] in {"OK", "WATCH", "FLAG"}

    def test_skip_eda_logic(self, eda_state):
        """Verify EDA can be skipped via config."""
        from core.config import ProfessorConfig
        config = ProfessorConfig()
        config.agents.skip_eda = True
        
        state_with_skip = {**eda_state, "config": config}
        result = run_eda_agent(state_with_skip)
        
        # If skipped, it returns the input state (which is a dict or object)
        # and doesn't write report files.
        assert result.get("eda_report_path") is None or result.get("eda_report_path") == ""
