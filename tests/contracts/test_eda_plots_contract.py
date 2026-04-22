# tests/contracts/test_eda_plots_contract.py

import pytest
import os
import polars as pl
import numpy as np
from pathlib import Path
from core.state import ProfessorState, initial_state
from tools.eda_plots import generate_eda_plots, run_eda_visualizer

# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_clean_data(tmp_path):
    """Cleaned dataset for plotting test."""
    df = pl.DataFrame({
        "id": range(100),
        "feat1": np.random.normal(0, 1, 100).tolist(),
        "feat2": np.random.uniform(0, 10, 100).tolist(),
        "target": [0, 1] * 50
    })
    path = tmp_path / "clean.parquet"
    df.write_parquet(path)
    return path

@pytest.fixture
def plotting_state(mock_clean_data):
    """State ready for visualization."""
    state_dict = initial_state(
        clean_data_path=str(mock_clean_data),
        target_col="target",
        session_id="test-plotting"
    )
    return ProfessorState(**state_dict)

# ── Tests ───────────────────────────────────────────────────────────────────

class TestEDAPlotsContract:
    """
    Contract: EDA Plotting Tool (Component 3)
    Ensures artifacts are generated and feature importance is captured.
    """

    def test_plot_generation_success(self, plotting_state):
        """Verify the generator returns success and paths."""
        res = generate_eda_plots(plotting_state)
        assert res["success"] is True
        assert len(res["plot_paths"]) > 0

    def test_plot_files_exist(self, plotting_state):
        """Verify image files are actually on disk."""
        res = generate_eda_plots(plotting_state)
        for p in res["plot_paths"]:
            assert os.path.exists(p)
            assert p.endswith(".png")

    def test_baseline_importance_captured(self, plotting_state):
        """Verify importance dict is non-empty."""
        res = generate_eda_plots(plotting_state)
        assert "feat1" in res["importance"]
        assert "feat2" in res["importance"]

    def test_node_updates_state(self, plotting_state):
        """Verify the LangGraph node updates the state correctly."""
        new_state = run_eda_visualizer(plotting_state)
        assert new_state.eda_plots_delivered is True
        assert len(new_state.eda_plots_paths) > 0
        assert "feat1" in new_state.eda_quick_baseline_importance
