# tests/contracts/test_feature_factory_v2_contract.py

import pytest
import json
import polars as pl
from unittest.mock import patch, MagicMock
from core.state import ProfessorState, initial_state
from agents.feature_factory import run_feature_factory

# ── Tests ───────────────────────────────────────────────────────────────────

class TestFeatureFactoryV2Contract:
    """
    Contract: Iterative Feature Factory (Component 5)
    """

    def test_5_round_loop_standard_depth(self):
        """Verify 5 rounds are executed for 'standard' depth."""
        state_dict = initial_state(
            feature_candidates=[{"name": "f1", "logic": "df*2"}],
            pipeline_depth="standard",
            session_id="test-5-round"
        )
        state = ProfessorState(**state_dict)
        
        with patch("agents.feature_factory.llm_call", return_value="df=df") as mock_llm:
            with patch("agents.feature_factory.run_in_sandbox") as mock_sb:
                mock_sb.return_value = {"success": True, "entry": {"entry_id": "1", "code": "df=df", "success": True}}
                
                final_state = run_feature_factory(state)
                
                # Check LLM was called 5 times
                assert mock_llm.call_count == 5
                assert final_state["round5_features"] is not None

    def test_2_round_loop_sprint_depth(self):
        """Verify loop is reduced to 2 rounds for 'sprint' depth."""
        state_dict = initial_state(
            feature_candidates=[{"name": "f1", "logic": "df*2"}],
            pipeline_depth="sprint",
            session_id="test-sprint"
        )
        state = ProfessorState(**state_dict)
        
        with patch("agents.feature_factory.llm_call", return_value="df=df") as mock_llm:
            with patch("agents.feature_factory.run_in_sandbox") as mock_sb:
                mock_sb.return_value = {"success": True, "entry": {"entry_id": "1", "code": "df=df", "success": True}}
                
                run_feature_factory(state)
                assert mock_llm.call_count == 2

    def test_skips_when_no_hypotheses(self):
        """Verify agent returns early if no hypotheses exist."""
        state = ProfessorState(**initial_state(feature_candidates=[]))
        with patch("agents.feature_factory.llm_call") as mock_llm:
            run_feature_factory(state)
            assert mock_llm.call_count == 0

    def test_adaptive_gating_integration(self, tmp_path):
        """Verify that passed/dropped features are tracked correctly."""
        # Create minimal parquet for gater
        df = pl.DataFrame({"target": [0, 1]*10, "f1": [1]*20})
        p = tmp_path / "clean.parquet"
        df.write_parquet(p)
        
        state_dict = initial_state(
            feature_candidates=[{"name": "f1"}],
            clean_data_path=str(p),
            target_col="target",
            pipeline_depth="sprint",
            session_id="test-gate-integration"
        )
        state = ProfessorState(**state_dict)
        
        with patch("agents.feature_factory.llm_call", return_value="df=df"):
            with patch("agents.feature_factory.run_in_sandbox") as mock_sb:
                mock_sb.return_value = {"success": True, "entry": {"entry_id": "1", "code": "df=df", "success": True}}
                
                # Mock gater to return f1 as passed
                with patch("agents.feature_factory.run_adaptive_gate", return_value=(["f1"], [{"feature": "f1", "is_beneficial": True}])):
                    final_state = run_feature_factory(state)
                    assert "f1" in final_state["features_gate_passed"]
                    assert len(final_state["features_gate_dropped"]) == 0
