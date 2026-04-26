# tests/contracts/test_publisher_contract.py

import pytest
import os
from unittest.mock import patch, MagicMock
from core.state import ProfessorState, initial_state
from agents.publisher import run_publisher

@pytest.fixture
def publisher_state():
    state_dict = initial_state(
        session_id="test-publisher",
        task_type="classification"
    )
    state = ProfessorState(**state_dict)
    state.cv_mean = 0.85
    state.best_model_type = "lightgbm"
    state.feature_order = ["f1", "f2"]
    state.pseudo_label_activated = True
    state.cost_tracker = {"llm_calls": 10, "api_cost_usd": 0.05}
    state.memory_peak_gb = 1.2
    return state

class TestPublisherContract:
    """
    Contract: Final Publisher Agent (Component 6)
    """

    @patch("agents.publisher.emit_to_operator")
    def test_milestone_4_emission(self, mock_emit, publisher_state):
        """Verify Milestone 4 summary is emitted."""
        run_publisher(publisher_state)
        assert mock_emit.called
        # Check if the "FINAL REPORT" header is in the last emission
        last_call_msg = mock_emit.call_args_list[-1][0][0]
        assert "FINAL REPORT" in last_call_msg

    @patch("agents.publisher.emit_to_operator")
    def test_cost_report_correctness(self, mock_emit, publisher_state):
        """Verify cost and LLM calls appear correctly in the report."""
        run_publisher(publisher_state)
        last_call_msg = mock_emit.call_args_list[-1][0][0]
        assert "LLM Calls: 10" in last_call_msg
        assert "API Cost: $0.0500" in last_call_msg

    @patch("agents.publisher.emit_to_operator")
    def test_provenance_links_verified(self, mock_emit, publisher_state, tmp_path):
        """Verify report correctly flags presence of solution artifacts."""
        # Setup paths to mock existence
        sol_dir = tmp_path / "outputs" / "test-publisher" / "solution"
        sol_dir.mkdir(parents=True, exist_ok=True)
        (sol_dir / "solution_notebook.py").write_text("code")
        
        # Override project root for test
        with patch("agents.publisher.Path") as mock_path:
            mock_path.return_value = tmp_path / "outputs" / "test-publisher"
            
            run_publisher(publisher_state)
            last_call_msg = mock_emit.call_args_list[-1][0][0]
            assert "Standalone Notebook: ✅ CREATED" in last_call_msg
            # In V2, run_publisher calls assemble which creates writeup if possible.
            # The test setup ensures it exists.
            assert "Documentation: ✅ CREATED" in last_call_msg

    @patch("agents.publisher.emit_to_operator")
    def test_submission_validation_status(self, mock_emit, publisher_state, tmp_path):
        """Verify report flags missing submission.csv correctly."""
        publisher_state.submission_path = str(tmp_path / "submission.csv")
        # File does not exist yet
        run_publisher(publisher_state)
        last_call_msg = mock_emit.call_args_list[-1][0][0]
        assert "Submission: ❌ MISSING" in last_call_msg

    def test_final_report_persisted(self, publisher_state, tmp_path):
        """Verify final_report.txt is saved to disk."""
        with patch("agents.publisher.Path") as mock_path:
            out_dir = tmp_path / "outputs" / "test-publisher"
            out_dir.mkdir(parents=True, exist_ok=True)
            mock_path.return_value = out_dir
            
            res = run_publisher(publisher_state)
            assert os.path.exists(res["report_path"])
            with open(res["report_path"], "r", encoding="utf-8") as f:
                assert "FINAL REPORT" in f.read()

    def test_post_mortem_marked_complete(self, publisher_state):
        res = run_publisher(publisher_state)
        assert res["post_mortem_completed"] is True
