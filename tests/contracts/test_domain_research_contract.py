# tests/contracts/test_domain_research_contract.py

import pytest
import os
import json
from pathlib import Path
from unittest.mock import patch
from core.state import ProfessorState, initial_state
from agents.domain_research import run_domain_research, _classify_domain

# ── Tests ───────────────────────────────────────────────────────────────────

class TestDomainResearchContract:
    """
    Contract: Domain Research Agent (Component 4)
    Ensures classification and acquisition are operational.
    """

    def test_classification_logic(self):
        """Verify keyword-based domain classification."""
        assert _classify_domain("Jane Street Market Prediction", "") == "tabular_finance"
        assert _classify_domain("Home Credit Default Risk", "banking data") == "tabular_finance"
        assert _classify_domain("Mayo Clinic healthcare challenge", "") == "healthcare"
        assert _classify_domain("H&M Personalized Recommendations", "") == "ecommerce"
        assert _classify_domain("Generic Competition", "some text") == "generic_tabular"

    def test_brief_structured_correctly(self):
        """Verify the agent node returns a structured brief in state."""
        state_dict = initial_state(
            competition_name="Jane Street Finance",
            session_id="test-research"
        )
        state = ProfessorState(**state_dict)
        
        mock_response = json.dumps({
            "engineering_ideas": ["lag_1", "volatility"],
            "critical_risks": ["leakage"],
            "validation_strategy": "TimeSeriesSplit",
            "external_data_suggestions": []
        })
        with patch("agents.domain_research.llm_call", return_value=mock_response):
            final_state = run_domain_research(state)

            assert final_state["competition_context"]["domain"] == "tabular_finance"
            assert "knowledge" in final_state["competition_context"]
            assert "volatility" in final_state["competition_context"]["knowledge"]["engineering_ideas"]
            assert os.path.exists(final_state["intel_brief_path"])

    def test_fallback_on_llm_failure(self):
        """Verify template-based fallback when LLM fails."""
        state_dict = initial_state(
            competition_name="Generic data challenge",
            session_id="test-fallback"
        )
        state = ProfessorState(**state_dict)

        with patch("agents.domain_research.llm_call", side_effect=Exception("LLM Down")):
            final_state = run_domain_research(state)
            # Should have "generic_tabular" or similar
            assert "domain" in final_state["competition_context"]
            assert final_state["competition_context"]["knowledge"]["validation_strategy"] != ""
