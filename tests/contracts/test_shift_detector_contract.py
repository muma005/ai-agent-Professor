# tests/contracts/test_shift_detector_contract.py

import pytest
import os
import polars as pl
import numpy as np
from pathlib import Path
from core.state import ProfessorState, initial_state
from agents.shift_detector import run_shift_detector, _calculate_psi

# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_drift_data(tmp_path):
    """Dataset with artificial drift for testing."""
    train_df = pl.DataFrame({
        "id": range(500),
        "feat1": np.random.normal(0, 1, 500).tolist(),
        "target": [0, 1] * 250
    })
    # Shift feat1 in test
    test_df = pl.DataFrame({
        "id": range(500, 1000),
        "feat1": np.random.normal(5, 1, 500).tolist()
    })
    
    train_path = tmp_path / "clean_train.parquet"
    test_path = tmp_path / "test.csv"
    
    train_df.write_parquet(train_path)
    test_df.write_csv(test_path)
    
    return train_path, test_path

@pytest.fixture
def drift_state(mock_drift_data):
    """Initial state for shift detector."""
    tr, te = mock_drift_data
    state_dict = initial_state(
        clean_data_path=str(tr),
        test_data_path=str(te),
        target_col="target",
        session_id="test-drift"
    )
    return ProfessorState(**state_dict)

# ── Tests ───────────────────────────────────────────────────────────────────

class TestShiftDetectorContract:
    """
    Contract: Distribution Shift Detector (Component 5)
    Ensures adversarial validation and feature PSI are operational.
    """

    def test_psi_calculation_accuracy(self):
        """Verify PSI logic on known distribution shift."""
        e = np.random.normal(0, 1, 1000)
        a = np.random.normal(0.5, 1, 1000)
        psi = _calculate_psi(e, a)
        assert psi > 0 # Some drift
        
        a_same = np.random.normal(0, 1, 1000)
        psi_same = _calculate_psi(e, a_same)
        assert psi_same < 0.1 # Very low drift

    def test_adversarial_auc_detected(self, drift_state):
        """Verify adversarial validation detects the shift in feat1."""
        final_state = run_shift_detector(drift_state)
        # With feat1 shifted significantly, AUC should be very high
        assert final_state["adversarial_auc"] > 0.8
        assert final_state["drift_report"]["overall_severity"] in ("HIGH", "CRITICAL")

    def test_per_feature_psi_captured(self, drift_state):
        """Verify the report contains specific PSI values for feat1."""
        final_state = run_shift_detector(drift_state)
        assert "feat1" in final_state["drift_report"]["feature_shifts"]
        assert final_state["drift_report"]["feature_shifts"]["feat1"]["psi"] > 0.2

    def test_missing_test_data_safety(self):
        """Verify the agent skips and returns state if test.csv is missing."""
        state_dict = initial_state(clean_data_path="train.parquet", test_data_path="missing.csv")
        state = ProfessorState(**state_dict)
        final_state = run_shift_detector(state)
        assert "drift_report" not in final_state or not final_state["drift_report"]
