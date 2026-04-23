# tests/contracts/test_feature_factory_v2_contract.py

import pytest
import json
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

    def test_failure_in_round_captured_in_ledger(self):
        """Verify failing rounds are still recorded in history (implied by ledger growth)."""
        state_dict = initial_state(
            feature_candidates=[{"name": "f1"}],
            session_id="test-fail-round"
        )
        state = ProfessorState(**state_dict)
        
        with patch("agents.feature_factory.llm_call", return_value="bad code"):
            with patch("agents.feature_factory.run_in_sandbox") as mock_sb:
                # Mock a failure
                mock_sb.return_value = {
                    "success": False, 
                    "stderr": "SyntaxError", 
                    "entry": {"entry_id": "fail", "code": "bad code", "success": False}
                }
                
                final_state = run_feature_factory(state)
                # Successful features order should be empty
                assert len(final_state["feature_order"]) == 0
                # But round1 code should still be stored
                assert final_state["round1_features"] == "bad code"
