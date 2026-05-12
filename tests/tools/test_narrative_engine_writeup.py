import pytest
import os
import json
from unittest.mock import patch, MagicMock
from tools.narrative_engine import generate_hackathon_writeup
from core.state import ProfessorState

@pytest.fixture
def writeup_state():
    """State ready for writeup generation."""
    return ProfessorState(
        session_id="test-writeup",
        competition_name="Triagegeist",
        hackathon_mode=True,
        active_thesis={
            "statement": "ESI undertriages elderly patients",
            "condition_variable": "age_group",
            "hypothesis": "Elderly patients have higher triage error rates"
        },
        hackathon_writeup_template={
            "sections": ["intro", "results"],
            "max_words": 500
        },
        domain_brief={"primary_domain": "healthcare"},
        eda_insights_summary="Summary",
        external_datasets=[{"name": "D1"}],
        thesis_effect_sizes={"error_rate_diff": 0.15},
        ensemble_cv_score=0.85
    )

@pytest.fixture
def mock_writeup_tools():
    with patch("tools.narrative_engine.llm_call") as mock_llm, \
         patch("tools.narrative_engine.emit_to_operator") as mock_emit:
        
        mock_llm.side_effect = [
            "Intro content",
            "Results content"
        ]
        
        yield mock_llm, mock_emit

class TestNarrativeEngineWriteup:

    def test_generates_writeup_file(self, writeup_state, mock_writeup_tools):
        # Target the mock to return False only for the code_ledger check
        def side_effect(path):
            if "code_ledger.jsonl" in path:
                return False
            return os.path.isfile(path) or os.path.isdir(path)

        with patch("os.path.exists", side_effect=side_effect):
            res_path = generate_hackathon_writeup(writeup_state)
            assert "hackathon_writeup.md" in res_path
            
            # The actual file is written, but our mock might say it doesn't exist
            # if it doesn't match the side_effect logic correctly.
            # Let's use the real os.path.exists for the final check by bypassing the mock
            assert os.path.isfile(res_path)

    def test_handles_no_thesis_gracefully(self, writeup_state):
        writeup_state.active_thesis = None
        res = generate_hackathon_writeup(writeup_state)
        assert "No active thesis" in res

    def test_section_failure_insertion(self, writeup_state, mock_writeup_tools):
        mock_llm, _ = mock_writeup_tools
        mock_llm.side_effect = Exception("LLM Error")
        
        with patch("os.path.exists", return_value=False):
            res_path = generate_hackathon_writeup(writeup_state)
            with open(res_path, "r", encoding="utf-8") as f:
                content = f.read()
                assert "[Section generation failed: LLM Error]" in content
