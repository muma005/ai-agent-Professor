# tests/contracts/test_feature_hypothesis_contract.py

import pytest
import json
from unittest.mock import patch
from core.state import ProfessorState, initial_state
from agents.feature_hypothesis import run_feature_hypothesis

# ── Tests ───────────────────────────────────────────────────────────────────

class TestFeatureHypothesisContract:
    """
    Contract: Feature Hypothesis Node (Component 4)
    """

    def test_structured_brief_returned(self):
        """Verify the node returns a structured JSON brief in state."""
        state_dict = initial_state(
            target_col="target",
            session_id="test-hypo"
        )
        state = ProfessorState(**state_dict)
        
        mock_response = json.dumps({
            "hypotheses": [
                {"name": "ratio_a_b", "logic": "A/B", "signal_type": "interaction", "complexity": 3}
            ],
            "creative_direction": "Focus on ratios."
        })
        
        with patch("agents.feature_hypothesis.llm_call", return_value=mock_response):
            final_state = run_feature_hypothesis(state)
            
            assert isinstance(final_state["feature_candidates"], list)
            assert len(final_state["feature_candidates"]) > 0
            assert final_state["feature_candidates"][0]["name"] == "ratio_a_b"
            assert "creative_direction" in final_state["feature_manifest"]

    def test_fallback_on_llm_failure(self):
        """Verify the node provides a fallback hypothesis brief on error."""
        state = ProfessorState(**initial_state(session_id="test-fail"))
        
        with patch("agents.feature_hypothesis.llm_call", side_effect=Exception("LLM Down")):
            final_state = run_feature_hypothesis(state)
            assert len(final_state["feature_candidates"]) > 0
            assert "Fallback" in final_state["feature_manifest"]["creative_direction"]

    def test_historical_context_included_in_prompt(self):
        """Verify recent state_mutations_log is passed to the reasoning engine."""
        # This test verifies the prompt assembly logic (indirectly via mock)
        state_dict = initial_state(session_id="test-context")
        state = ProfessorState(**state_dict)
        state.state_mutations_log.append({"agent": "test", "field": "f", "old_hash": "1", "new_hash": "2"})
        
        with patch("agents.feature_hypothesis.llm_call", return_value='{"hypotheses": []}') as mock_llm:
            run_feature_hypothesis(state)
            # Check if ledger data appears in the call
            args, kwargs = mock_llm.call_args
            assert "PREVIOUS ATTEMPT" in args[0]

    def test_state_updated_via_validated_update(self):
        """Verify ownership rules are respected (implicitly via validated_update)."""
        state = ProfessorState(**initial_state(session_id="test-ownership"))
        with patch("agents.feature_hypothesis.llm_call", return_value='{"hypotheses": []}'):
            final_state = run_feature_hypothesis(state)
            assert final_state["current_node"] == "creative_hypothesis"
