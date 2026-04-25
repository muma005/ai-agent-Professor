# tests/contracts/test_pseudo_label_contract.py

import pytest
import os
import json
import polars as pl
import numpy as np
from unittest.mock import patch, MagicMock
from core.state import ProfessorState, initial_state
from agents.pseudo_label import pseudo_label_architect

# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def large_test_data(tmp_path):
    """Test set 2x larger than train — pseudo-labels should activate."""
    train_path = str(tmp_path / "train.parquet")
    test_path = str(tmp_path / "test.parquet")
    preds_path = str(tmp_path / "test_preds.parquet")
    
    pl.DataFrame({"feat": range(100), "target": [0, 1]*50}).write_parquet(train_path)
    pl.DataFrame({"feat": range(200)}).write_parquet(test_path)
    pl.DataFrame({"pred": [0.5]*200}).write_parquet(preds_path)
    
    return train_path, test_path, preds_path

@pytest.fixture
def small_test_data(tmp_path):
    """Test set smaller than train — pseudo-labels should NOT activate."""
    train_path = str(tmp_path / "train.parquet")
    test_path = str(tmp_path / "test.parquet")
    preds_path = str(tmp_path / "test_preds.parquet")
    
    pl.DataFrame({"feat": range(100), "target": [0, 1]*50}).write_parquet(train_path)
    pl.DataFrame({"feat": range(50)}).write_parquet(test_path)
    pl.DataFrame({"pred": [0.5]*50}).write_parquet(preds_path)
    
    return train_path, test_path, preds_path

@pytest.fixture
def pseudo_state(large_test_data):
    tr, te, pr = large_test_data
    state_dict = initial_state(
        session_id="test-pseudo",
        feature_data_path=tr,
        test_data_path=te,
        target_col="target",
        task_type="classification"
    )
    state = ProfessorState(**state_dict)
    state.canonical_train_rows = 100
    state.canonical_test_rows = 200
    state.test_predictions_path = pr
    state.cv_mean = 0.8
    state.cv_scores = [0.78, 0.82, 0.80, 0.79, 0.81]
    return state

# ── Tests ───────────────────────────────────────────────────────────────────

class TestPseudoLabelContract:
    """
    Contract: Pseudo-Label Architect (Component 1)
    """

    @patch("agents.pseudo_label.run_in_sandbox")
    @patch("agents.pseudo_label.emit_to_operator")
    def test_activates_when_test_larger(self, mock_emit, mock_sb, pseudo_state):
        mock_sb.return_value = {
            "success": True, 
            "stdout": "PSEUDO_RESULT:{\"mean_cv\": 0.85, \"scores\": [0.84, 0.86, 0.85, 0.85, 0.85], \"n_pseudo\": 20}"
        }
        final_state = pseudo_label_architect(pseudo_state)
        assert final_state["pseudo_label_activated"] is True
        assert final_state["pseudo_label_cv_delta"] > 0

    @patch("agents.pseudo_label.emit_to_operator")
    def test_skips_when_test_smaller(self, mock_emit, small_test_data):
        tr, te, pr = small_test_data
        state = ProfessorState(**initial_state(feature_data_path=tr, test_data_path=te))
        state.canonical_train_rows = 100
        state.canonical_test_rows = 50
        state.test_predictions_path = pr
        
        final_state = pseudo_label_architect(state)
        assert final_state["pseudo_label_activated"] is False

    @patch("agents.pseudo_label.emit_to_operator")
    def test_skips_when_critic_critical(self, mock_emit, pseudo_state):
        pseudo_state.critic_severity = "CONFIRMED_CRITICAL"
        final_state = pseudo_label_architect(pseudo_state)
        assert final_state["pseudo_label_activated"] is False

    @patch("agents.pseudo_label.emit_to_operator")
    def test_skips_in_sprint_mode(self, mock_emit, pseudo_state):
        pseudo_state.pipeline_depth = "sprint"
        final_state = pseudo_label_architect(pseudo_state)
        assert final_state["pseudo_label_activated"] is False

    @patch("agents.pseudo_label._run_pseudo_round")
    @patch("agents.pseudo_label.emit_to_operator")
    def test_max_2_rounds(self, mock_emit, mock_round, pseudo_state):
        mock_round.return_value = (True, 0.9, [0.85, 0.95, 0.90, 0.91, 0.89], {"n_pseudo": 10})
        pseudo_label_architect(pseudo_state)
        assert mock_round.call_count == 2

    @patch("agents.pseudo_label.run_in_sandbox")
    @patch("agents.pseudo_label.emit_to_operator")
    def test_pseudo_fraction_capped_at_30pct(self, mock_emit, mock_sb, pseudo_state):
        pseudo_state.canonical_test_rows = 10000
        # Mock successful rounds
        mock_sb.return_value = {
            "success": True, 
            "stdout": "PSEUDO_RESULT:{\"mean_cv\": 0.95, \"scores\": [0.94, 0.96, 0.95, 0.95, 0.95], \"n_pseudo\": 30}"
        }
        final_state = pseudo_label_architect(pseudo_state)
        assert final_state["pseudo_label_fraction"] == 0.30

    @patch("agents.pseudo_label.run_in_sandbox")
    @patch("agents.pseudo_label.emit_to_operator")
    def test_wilcoxon_gate_applied(self, mock_emit, mock_sb, pseudo_state):
        # Mock Round 1 showing minor improvement but high p-value
        # (3 better, 2 worse)
        mock_sb.return_value = {
            "success": True, 
            "stdout": "PSEUDO_RESULT:{\"mean_cv\": 0.801, \"scores\": [0.781, 0.819, 0.801, 0.789, 0.811], \"n_pseudo\": 10}"
        }
        final_state = pseudo_label_architect(pseudo_state)
        # Should be rejected because p > 0.05
        assert final_state["pseudo_label_activated"] is False

    @patch("agents.pseudo_label.run_in_sandbox")
    @patch("agents.pseudo_label.emit_to_operator")
    def test_wilcoxon_gate_passes(self, mock_emit, mock_sb, pseudo_state):
        # Mock significant improvement (all 5 greater)
        mock_sb.return_value = {
            "success": True, 
            "stdout": "PSEUDO_RESULT:{\"mean_cv\": 0.95, \"scores\": [0.94, 0.96, 0.95, 0.95, 0.95], \"n_pseudo\": 10}"
        }
        final_state = pseudo_label_architect(pseudo_state)
        assert final_state["pseudo_label_activated"] is True

    @patch("agents.pseudo_label.run_in_sandbox")
    @patch("agents.pseudo_label.emit_to_operator")
    def test_revert_on_round2_degradation(self, mock_emit, mock_sb, pseudo_state):
        # Round 1: Good
        # Round 2: Worse than Round 1 (but still better than original)
        mock_sb.side_effect = [
            {"success": True, "stdout": "PSEUDO_RESULT:{\"mean_cv\": 0.90, \"scores\": [0.89, 0.91, 0.90, 0.90, 0.90], \"n_pseudo\": 10}"},
            {"success": True, "stdout": "PSEUDO_RESULT:{\"mean_cv\": 0.85, \"scores\": [0.84, 0.86, 0.85, 0.85, 0.85], \"n_pseudo\": 5}"}
        ]
        final_state = pseudo_label_architect(pseudo_state)
        # Should keep Round 1 (+0.10)
        assert abs(final_state["pseudo_label_cv_delta"] - 0.10) < 1e-6
        assert final_state["pseudo_label_fraction"] == 0.10

    def test_never_halts_pipeline(self, pseudo_state):
        """Verify any internal exception results in clean exit with False activated."""
        # This will trigger our internal try/except block
        with patch("agents.pseudo_label._run_pseudo_round", side_effect=RuntimeError("Sandbox crash")):
            final_state = pseudo_label_architect(pseudo_state)
            assert final_state["pseudo_label_activated"] is False
            # Check pipeline still continues (not halted by with_agent_retry)
            assert final_state.pipeline_halted is False
