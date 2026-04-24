# tests/contracts/test_submission_strategist_contract.py

import pytest
import os
import numpy as np
import polars as pl
from pathlib import Path
from unittest.mock import patch, MagicMock
from core.state import ProfessorState, initial_state
from agents.submission_strategist import run_submission_strategist

@pytest.fixture
def strategist_state(tmp_path):
    ensemble_preds_path = str(tmp_path / "ensemble_test.parquet")
    sample_sub_path = str(tmp_path / "sample_sub.csv")
    
    pl.DataFrame({"pred": [0.1, 0.9, 0.4]}).write_parquet(ensemble_preds_path)
    
    pl.DataFrame({
        "id": [1, 2, 3],
        "target": [0.0, 0.0, 0.0]
    }).write_csv(sample_sub_path)
    
    state_dict = initial_state(
        session_id="test-strategist",
        sample_submission_path=sample_sub_path,
        task_type="classification"
    )
    state = ProfessorState(**state_dict)
    state.ensemble_test_predictions_path = ensemble_preds_path
    
    return state

class TestSubmissionStrategistContract:

    @patch("agents.submission_strategist.apply_submission_freeze")
    def test_submission_freeze_applied(self, mock_freeze, strategist_state):
        mock_freeze.return_value = (np.array([0.1, 0.9, 0.4]), True)
        state = run_submission_strategist(strategist_state)
        assert mock_freeze.called
        assert state["submission_freeze_active"] is True

    @patch("agents.submission_strategist.validate_submission")
    def test_submission_validated(self, mock_validate, strategist_state):
        mock_validate.return_value = {"is_valid": True, "errors": []}
        state = run_submission_strategist(strategist_state)
        assert mock_validate.called

    @patch("agents.submission_strategist.validate_submission")
    def test_fallback_on_validation_failure(self, mock_validate, strategist_state):
        mock_validate.return_value = {"is_valid": False, "errors": ["Mock error"]}
        state = run_submission_strategist(strategist_state)
        
        # If it falls back, the submission written should exactly match sample_submission
        final_df = pl.read_csv(state["submission_path"])
        sample_df = pl.read_csv(strategist_state["sample_submission_path"])
        assert final_df.equals(sample_df)

    def test_submission_saved_to_disk(self, strategist_state):
        state = run_submission_strategist(strategist_state)
        assert state["submission_path"] is not None
        assert os.path.exists(state["submission_path"])
        assert state["submission_path"].endswith(".csv")

    def test_n_submissions_incremented(self, strategist_state):
        initial_n = strategist_state["n_submissions_with_lb"] or 0
        state = run_submission_strategist(strategist_state)
        assert state["n_submissions_with_lb"] == initial_n + 1
