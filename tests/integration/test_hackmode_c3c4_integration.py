import pytest
import os
import json
from unittest.mock import patch, MagicMock
from agents.external_data_scout import external_data_scout, DatasetCandidate
from tools.narrative_engine import generate_thesis_visualizations, generate_hackathon_writeup
from core.state import ProfessorState

@pytest.fixture
def hack_state():
    """State set up for full component integration test."""
    return ProfessorState(
        session_id="integration-test",
        competition_name="Hackathon X",
        hackathon_mode=True,
        active_thesis={
            "statement": "Triage AI is biased against older patients",
            "condition_variable": "age_group",
            "hypothesis": "Error rates 2x higher for patients > 70",
            "data_plan": {"external_needed": ["Census data"]}
        },
        hackathon_rubric={"criteria": [{"name": "Technical", "weight": 50}]},
        hackathon_effort_plan={"external_data_priority": "medium", "visualization_count": 1},
        domain_classification="healthcare",
        data_schema={"age": "int"},
        clean_data_path="train.parquet",
        canonical_train_rows=100
    )

class TestHackmodeC3C4Integration:

    def test_full_component_chain(self, hack_state):
        """Verify data scouting leads to enriched plots and writeup."""
        
        # 1. Mock External Data Scout internal functions to bypass real tool calls
        with patch("agents.external_data_scout._search_all_sources") as mock_search, \
             patch("agents.external_data_scout.llm_call") as mock_scout_llm, \
             patch("agents.external_data_scout.run_in_sandbox") as mock_scout_sandbox, \
             patch("agents.external_data_scout.emit_to_operator") as mock_emit, \
             patch("agents.external_data_scout._download_dataset") as mock_download, \
             patch("polars.read_parquet") as mock_pl, \
             patch("os.path.exists", return_value=True):
            
            # Setup search result
            mock_search.return_value = [
                DatasetCandidate(name="Census", source_url="u", source_type="kaggle", description="d",
                                 join_feasibility=0, relevance_to_thesis=0, size_compatible=True,
                                 license_compatible=True, download_accessible=True, join_key="",
                                 join_type="", estimated_match_rate=0, overall_score=0)
            ]
            
            # Setup evaluation and integration LLM calls
            mock_scout_llm.side_effect = [
                # Evaluation JSON
                json.dumps([
                    {"candidate_index": 1, "relevance": 0.9, "join_feasibility": 0.9, "join_key": "age",
                     "join_type": "left", "match_rate": 1.0, "license_ok": True, "size_ok": True}
                ]),
                # Integration Code
                "print('{\"matched_rows\": 100, \"new_columns\": [\"census\"]}')"
            ]
            
            # Setup sandbox and download
            mock_scout_sandbox.return_value = {"success": True, "stdout": '{"matched_rows": 100, "new_columns": ["census"]}'}
            def download_side_effect(c, d):
                c.downloaded = True
                return c
            mock_download.side_effect = download_side_effect
            mock_emit.return_value = "/continue"
            
            # Setup polars read to return correct length
            mock_df = MagicMock()
            mock_df.__len__.return_value = 100
            mock_pl.return_value = mock_df
            
            # RUN SCOUT
            scout_res = external_data_scout(hack_state)
            
            # Update state with scout results
            hack_state = hack_state.model_copy(update=scout_res)
            assert len(hack_state.external_datasets) == 1
            assert hack_state.enriched_data_path != ""

        # 2. Mock Narrative Engine - Plots
        with patch("tools.narrative_engine.llm_call") as mock_plot_llm, \
             patch("tools.narrative_engine.run_in_sandbox") as mock_plot_sandbox, \
             patch("tools.narrative_engine._validate_plot_output", return_value=True), \
             patch("tools.narrative_engine.emit_to_operator"):
            
            mock_plot_llm.side_effect = [
                json.dumps([{"title": "P1", "type": "dist", "insight_goal": "G", "features_needed": ["age"]}]),
                "code"
            ]
            mock_plot_sandbox.return_value = {"success": True, "stdout": ""}
            
            # RUN PLOTS
            plots_res = generate_thesis_visualizations(hack_state)
            hack_state = hack_state.model_copy(update={"narrative_plots": plots_res})
            
            assert len(hack_state.narrative_plots) == 1

        # 3. Mock Narrative Engine - Writeup
        with patch("tools.narrative_engine.llm_call") as mock_write_llm, \
             patch("tools.narrative_engine.emit_to_operator"), \
             patch("os.path.exists", return_value=False): # For code_ledger check
            
            mock_write_llm.return_value = "Generated Section Content"
            
            # RUN WRITEUP
            writeup_path = generate_hackathon_writeup(hack_state)
            
            # Verify file exists and content (bypass our own mock if needed but here we just check real existence)
            assert "hackathon_writeup.md" in writeup_path
            with open(writeup_path, "r", encoding="utf-8") as f:
                content = f.read()
                assert "Triage AI is biased" in content
                assert "Generated Section Content" in content
                assert "## Problem Statement" in content
