# tests/contracts/test_eda_v2_contract.py

import pytest
import os
import json
import polars as pl
import numpy as np
from pathlib import Path
from unittest.mock import patch
from core.state import ProfessorState, initial_state
from agents.eda_agent import run_eda_agent

# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_eda_data(tmp_path):
    """1000-row dataset with known properties for testing."""
    np.random.seed(42)
    m1 = np.random.normal(0, 1, 500)
    m2 = np.random.normal(10, 1, 500)
    multimodal = np.concatenate([m1, m2]).tolist()
    
    df = pl.DataFrame({
        "id": range(1000),
        "normal_feat": np.random.normal(0, 1, 1000).tolist(),
        "skewed_feat": np.random.exponential(2, 1000).tolist(),
        "multimodal_feat": multimodal,
        "target": [0, 1] * 500,
    })
    
    parquet_path = tmp_path / "clean_train.parquet"
    df.write_parquet(parquet_path)
    return parquet_path

@pytest.fixture
def eda_state(mock_eda_data):
    """Initial state for EDA v2."""
    state_dict = initial_state(
        clean_data_path=str(mock_eda_data),
        target_col="target",
        session_id="test-eda-v2"
    )
    return ProfessorState(**state_dict)

# ── Tests ───────────────────────────────────────────────────────────────────

class TestEDAV2Contract:
    """
    Contract: Deep EDA Agent (v2 Upgrades)
    Ensures statistical profiling, MI, VIF, and LLM synthesis are operational.
    """

    def test_insights_summary_nonempty(self, eda_state):
        """Verify eda_insights_summary is a meaningful string."""
        with patch("agents.eda_agent.llm_call", return_value="Mocked EDA insights for testing."):
            final_state = run_eda_agent(eda_state)
            # Use dictionary access to be safe for both object and dict return
            assert len(final_state["eda_insights_summary"]) > 10

    def test_mutual_info_captured(self, eda_state):
        """Verify target_mi dict exists and has features."""
        with patch("agents.eda_agent.llm_call", return_value="Mock"):
            final_state = run_eda_agent(eda_state)
            assert len(final_state["eda_mutual_info"]["target_mi"]) > 0

    def test_multimodal_feature_detected(self, eda_state):
        """Verify multimodal_feat appears in modality_flags."""
        with patch("agents.eda_agent.llm_call", return_value="Mock"):
            final_state = run_eda_agent(eda_state)
            assert "multimodal_feat" in final_state["eda_modality_flags"]

    def test_v1_keys_preserved(self, eda_state):
        """Verify all original 8 keys still exist in eda_report."""
        with patch("agents.eda_agent.llm_call", return_value="Mock"):
            final_state = run_eda_agent(eda_state)
            required_keys = {
                "target_distribution", "correlations", "outlier_profile",
                "duplicate_analysis", "temporal_profile", "leakage_fingerprint",
                "drop_candidates", "summary"
            }
            assert required_keys.issubset(final_state["eda_report"].keys())

    def test_eda_report_v2_json_persisted(self, eda_state):
        """Verify the full report is saved to disk."""
        with patch("agents.eda_agent.llm_call", return_value="Mock"):
            final_state = run_eda_agent(eda_state)
            assert os.path.exists(final_state["eda_report_path"])
            
            with open(final_state["eda_report_path"], "r") as f:
                report = json.load(f)
                assert "eda_insights_summary" in report
                assert "eda_report" in report
