import pytest
import os
import json
from unittest.mock import patch, MagicMock
from agents.external_data_scout import external_data_scout, DatasetCandidate
from core.state import ProfessorState

@pytest.fixture
def scout_state():
    """State with active thesis — ready for scouting."""
    return ProfessorState(
        session_id="test-scout",
        hackathon_mode=True,
        active_thesis={
            "statement": "ESI undertriages elderly patients with atypical cardiac symptoms",
            "data_plan": {
                "primary_dataset": "MIMIC-IV-ED",
                "external_needed": ["AHA cardiac risk thresholds by age", "ESI inter-rater reliability data"],
                "join_strategy": "Lookup table on age_group"
            },
            "condition_variable": "age_group × presentation_type",
        },
        hackathon_rubric={
            "recommended_datasets": [
                {"name": "MIMIC-IV-ED", "url": "physionet.org", "description": "ED data"}
            ],
        },
        hackathon_effort_plan={"external_data_priority": "high"},
        domain_classification="healthcare",
        data_schema={"patient_id": "Int64", "age": "Float64", "chief_complaint": "Utf8"},
        clean_data_path="train.parquet",
        canonical_train_rows=5000,
    )

@pytest.fixture(autouse=True)
def mock_external_tools():
    """Mock LLM, Operator, and Sandbox for all tests."""
    with patch("agents.external_data_scout.llm_call") as mock_llm, \
         patch("agents.external_data_scout.emit_to_operator") as mock_emit, \
         patch("agents.external_data_scout.run_in_sandbox") as mock_sandbox:
        
        # Default LLM evaluations
        mock_llm.return_value = json.dumps([
            {
                "candidate_index": 1,
                "relevance": 0.9,
                "join_feasibility": 0.8,
                "join_key": "age_group",
                "join_type": "lookup_table",
                "match_rate": 0.95,
                "license_ok": True,
                "size_ok": True,
                "reasoning": "Strong match"
            }
        ])
        
        mock_emit.return_value = "/continue"
        
        mock_sandbox.return_value = {
            "success": True,
            "stdout": '{"matched_rows": 4800, "new_columns": ["risk_score"]}',
            "stderr": ""
        }
        
        yield mock_llm, mock_emit, mock_sandbox


class TestExternalDataScoutContract:

    def test_recommended_datasets_searched_first(self, scout_state):
        with patch("agents.external_data_scout._search_all_sources") as mock_search:
            mock_search.return_value = []
            external_data_scout(scout_state)
            # Verify recommended was passed to search
            args = mock_search.call_args[0]
            assert args[1] == scout_state.hackathon_rubric["recommended_datasets"]

    def test_candidates_evaluated_with_scores(self, scout_state):
        res = external_data_scout(scout_state)
        # Check that integrated datasets have overall scores
        for ds in res["external_datasets"]:
            assert ds["overall_score"] > 0

    def test_join_feasibility_populated(self, scout_state):
        res = external_data_scout(scout_state)
        if res["external_datasets"]:
            ds = res["external_datasets"][0]
            assert ds["join_key"] != ""
            assert ds["estimated_match_rate"] > 0

    def test_max_3_integrated(self, scout_state, mock_external_tools):
        mock_llm, _, _ = mock_external_tools
        # Return 10 evaluations
        mock_llm.return_value = json.dumps([
            {"candidate_index": i+1, "relevance": 0.9, "join_feasibility": 0.9, "join_key": "k", 
             "match_rate": 0.9, "license_ok": True, "size_ok": True}
            for i in range(10)
        ])
        
        # Mock search results with 10 candidates
        with patch("agents.external_data_scout._search_all_sources") as mock_search:
            mock_search.return_value = [
                DatasetCandidate(name=f"D{i}", source_url="u", source_type="kaggle", description="d",
                                 join_feasibility=0, relevance_to_thesis=0, size_compatible=True,
                                 license_compatible=True, download_accessible=True, join_key="",
                                 join_type="", estimated_match_rate=0, overall_score=0)
                for i in range(10)
            ]
            res = external_data_scout(scout_state)
            
        assert len(res["external_datasets"]) <= 3

    def test_row_count_preserved_after_integration(self, scout_state, mock_external_tools):
        # Mock pl.read_parquet to return a dummy with correct length
        with patch("polars.read_parquet") as mock_read:
            mock_df = MagicMock()
            mock_df.__len__.return_value = scout_state.canonical_train_rows
            mock_read.return_value = mock_df
            
            with patch("os.path.exists", return_value=True):
                res = external_data_scout(scout_state)
                assert len(res["external_datasets"]) > 0
                assert res["external_datasets"][0]["integrated"] is True

    def test_row_count_change_rejected(self, scout_state, mock_external_tools):
        # Mock pl.read_parquet to return a dummy with WRONG length
        with patch("polars.read_parquet") as mock_read:
            mock_df = MagicMock()
            mock_df.__len__.return_value = scout_state.canonical_train_rows + 10 # Corrupted
            mock_read.return_value = mock_df
            
            with patch("os.path.exists", return_value=True):
                res = external_data_scout(scout_state)
                # None should be marked as successfully integrated if count changed
                for ds in res["external_datasets"]:
                    assert ds["integrated"] is False

    def test_sources_tracked_for_citation(self, scout_state):
        res = external_data_scout(scout_state)
        for ds in res["external_datasets"]:
            assert ds["source_url"] != ""

    def test_reference_table_constructed(self, scout_state, mock_external_tools):
        mock_llm, _, _ = mock_external_tools
        # Isolate to ONE data need and NO other search results
        scout_state.active_thesis["data_plan"]["external_needed"] = ["AHA cardiac risk thresholds by age"]
        
        with patch("agents.external_data_scout._search_all_sources", return_value=[]):
            # Ensure LLM returns SOURCE/CSV/JOIN_KEY for construction, then eval, then integration code
            mock_llm.side_effect = [
                "SOURCE: AHA 2024\nCSV:\nage,min_hr\n65,100\nJOIN_KEY: age", # Table construction
                json.dumps([{"candidate_index": 1, "relevance": 0.9, "join_feasibility": 0.9, 
                            "join_key": "age", "join_type": "lookup_table", "match_rate": 0.9, 
                            "license_ok": True, "size_ok": True}]), # Evaluation
                "import polars as pl\nprint('{\"matched_rows\": 5000, \"new_columns\": [\"c\"]}')" # Integration code
            ]
            
            res = external_data_scout(scout_state)
            
        has_ref = any(ds["source_type"] == "reference_table" for ds in res["external_datasets"])
        assert has_ref is True
        assert len(res["external_datasets"]) == 1

    def test_graceful_when_none_found(self, scout_state, mock_external_tools):
        # Remove threshold keyword to avoid reference table auto-generation
        scout_state.active_thesis["data_plan"]["external_needed"] = ["Generic data 1"]
        
        # Mock search to find nothing
        with patch("agents.external_data_scout._search_all_sources", return_value=[]):
            res = external_data_scout(scout_state)
            assert res["external_datasets"] == []
            assert res["enriched_data_path"] == scout_state.clean_data_path

    def test_graceful_when_search_unavailable(self, scout_state):
        # Mock search to fail
        with patch("agents.external_data_scout._search_all_sources", side_effect=Exception("API Down")):
            res = external_data_scout(scout_state)
            # Should still function and use recommended datasets
            assert isinstance(res["external_datasets"], list)

    def test_license_unclear_flagged(self, scout_state, mock_external_tools):
        mock_llm, mock_emit, _ = mock_external_tools
        mock_llm.return_value = json.dumps([
            {"candidate_index": 1, "relevance": 0.9, "join_feasibility": 0.9, "join_key": "k", 
             "match_rate": 0.9, "license_ok": False, "size_ok": True}
        ])
        
        external_data_scout(scout_state)
        # Check all emit calls for the warning
        found_warning = any("⚠️ unclear" in str(call[0][0]) for call in mock_emit.call_args_list)
        assert found_warning is True

    def test_skip_priority_only_checks_recommended(self, scout_state):
        scout_state.hackathon_effort_plan["external_data_priority"] = "skip"
        with patch("agents.external_data_scout._search_kaggle_datasets") as mock_kaggle, \
             patch("agents.external_data_scout._search_web_sources") as mock_web:
            external_data_scout(scout_state)
            assert not mock_kaggle.called
            assert not mock_web.called

    def test_enriched_path_chains_joins(self, scout_state, mock_external_tools):
        # Integration succeeds for first two
        with patch("agents.external_data_scout._integrate_dataset") as mock_int:
            mock_int.side_effect = [
                {"success": True, "enriched_path": "path1", "new_columns": ["c1"], "matched_rows": 5000, "match_rate": 1.0},
                {"success": True, "enriched_path": "path2", "new_columns": ["c2"], "matched_rows": 5000, "match_rate": 1.0}
            ]
            
            res = external_data_scout(scout_state)
            assert res["enriched_data_path"] == "path2"
            
    def test_non_hackathon_returns_empty(self, scout_state):
        scout_state.hackathon_mode = False
        res = external_data_scout(scout_state)
        assert res == {}

    def test_no_data_needs_returns_immediately(self, scout_state):
        scout_state.active_thesis["data_plan"]["external_needed"] = []
        res = external_data_scout(scout_state)
        assert res["thesis_data_sufficient"] is True
        assert res["external_datasets"] == []
