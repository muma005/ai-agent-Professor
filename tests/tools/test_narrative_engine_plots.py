import pytest
import os
import json
from unittest.mock import patch, MagicMock
from tools.narrative_engine import generate_thesis_visualizations
from core.state import ProfessorState

@pytest.fixture
def narrative_state():
    """State ready for visualization generation."""
    return ProfessorState(
        session_id="test-narrative",
        hackathon_mode=True,
        active_thesis={
            "statement": "ESI undertriages elderly patients",
            "condition_variable": "age_group",
            "hypothesis": "Elderly patients have higher triage error rates"
        },
        hackathon_effort_plan={"visualization_count": 2},
        clean_data_path="train.parquet",
        data_schema={"age": "Float64", "triage_level": "Int64"}
    )

@pytest.fixture
def mock_narrative_tools():
    with patch("tools.narrative_engine.llm_call") as mock_llm, \
         patch("tools.narrative_engine.run_in_sandbox") as mock_sandbox, \
         patch("tools.narrative_engine.emit_to_operator") as mock_emit, \
         patch("tools.narrative_engine._validate_plot_output") as mock_val:
        
        # 1. Plot planning response
        mock_llm.side_effect = [
            json.dumps([
                {"title": "Age Dist", "type": "distribution", "insight_goal": "Show age groups", "features_needed": ["age"]},
                {"title": "Error Rate", "type": "interaction", "insight_goal": "Compare error by age", "features_needed": ["age", "target"]}
            ]),
            # 2. Execution code responses (one for each plot)
            "import seaborn as sns\nplt.savefig('plot1.png')",
            "import seaborn as sns\nplt.savefig('plot2.png')"
        ]
        
        mock_sandbox.return_value = {"success": True, "stdout": "", "stderr": ""}
        mock_val.return_value = True # Simulate file existence
        
        yield mock_llm, mock_sandbox, mock_emit, mock_val

class TestNarrativeEnginePlots:

    def test_generates_correct_number_of_plots(self, narrative_state, mock_narrative_tools):
        res = generate_thesis_visualizations(narrative_state)
        assert len(res) == 2
        assert res[0]["title"] == "Age Dist"
        assert res[1]["title"] == "Error Rate"

    def test_handles_no_thesis_gracefully(self, narrative_state):
        narrative_state.active_thesis = None
        res = generate_thesis_visualizations(narrative_state)
        assert res == []

    def test_skips_failed_plots(self, narrative_state, mock_narrative_tools):
        _, mock_sandbox, _, _ = mock_narrative_tools
        # First plot fails in sandbox
        mock_sandbox.side_effect = [
            {"success": False, "stderr": "Error"},
            {"success": True, "stdout": ""}
        ]
        res = generate_thesis_visualizations(narrative_state)
        # Should only have one plot successfully generated
        assert len(res) == 1
        assert res[0]["title"] == "Error Rate"

    def test_plot_specs_parsed_robustly(self, narrative_state, mock_narrative_tools):
        mock_llm, _, _, _ = mock_narrative_tools
        # LLM returns markdown fenced JSON
        mock_llm.side_effect = [
            "```json\n[{\"title\": \"M\", \"type\": \"t\", \"insight_goal\": \"i\", \"features_needed\": []}]\n```",
            "code", "code"
        ]
        res = generate_thesis_visualizations(narrative_state)
        assert len(res) == 1
        assert res[0]["title"] == "M"
