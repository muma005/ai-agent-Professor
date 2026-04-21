# tests/contracts/test_submission_strategist_contract.py

import pytest
import os
import json
import polars as pl
from pathlib import Path
from core.state import initial_state
from agents.submission_strategist import run_submission_strategist

# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def strategist_state(tmp_path):
    """State with a diverse model registry and sample sub."""
    sample_path = tmp_path / "sample.csv"
    pl.DataFrame({"id": [1, 2, 3], "target": [0.5, 0.5, 0.5]}).write_csv(sample_path)
    
    return initial_state(
        session_id="test-strat",
        model_registry=[
            {
                "model_id": "best_1",
                "stability_score": 0.9,
                "test_predictions": [0.9, 0.1, 0.8]
            },
            {
                "model_id": "diverse_1",
                "stability_score": 0.8,
                "test_predictions": [0.2, 0.7, 0.3]
            }
        ],
        sample_submission_path=str(sample_path)
    )

# ── Tests ───────────────────────────────────────────────────────────────────

class TestSubmissionStrategistContract:
    """
    Contract: Submission Strategist Agent
    Ensures EWMA monitoring and valid CSV generation.
    """

    def test_submission_path_populated(self, strategist_state):
        """Verify submission.csv is created."""
        state = run_submission_strategist(strategist_state)
        assert "submission_path" in state
        assert Path(state["submission_path"]).exists()

    def test_submission_log_in_state(self, strategist_state):
        """Verify submission_log key exists."""
        state = run_submission_strategist(strategist_state)
        assert "submission_log" in state
        assert isinstance(state["submission_log"], list)

    def test_freeze_active_is_boolean(self, strategist_state):
        """Verify freeze status is captured."""
        state = run_submission_strategist(strategist_state)
        assert "submission_freeze_active" in state
        assert isinstance(state["submission_freeze_active"], bool)

    def test_empty_registry_safety(self):
        """Verify safety when no models exist."""
        state = initial_state(session_id="empty", model_registry=[])
        result = run_submission_strategist(state)
        assert result.get("submission_path") is None or result.get("submission_path") == ""
