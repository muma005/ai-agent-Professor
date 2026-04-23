# tests/contracts/test_shift_detector_contract.py

import pytest
import os
import json
import polars as pl
import numpy as np
from pathlib import Path
from core.state import ProfessorState, initial_state
from agents.shift_detector import run_shift_detector

# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def clean_data(tmp_path):
    """Train and test from the SAME distribution — should return 'clean'."""
    np.random.seed(42)
    n = 2000
    df = pl.DataFrame({
        "feat_1": np.random.normal(0, 1, n).tolist(),
        "feat_2": np.random.normal(5, 2, n).tolist(),
        "feat_3": np.random.choice(["A", "B", "C"], n).tolist(),
        "target": np.random.randint(0, 2, n).tolist(),
    })
    train = df.head(1500)
    test = df.tail(500).drop("target")
    
    train_path = tmp_path / "clean_train.parquet"
    test_path = tmp_path / "test.csv"
    
    train.write_parquet(train_path)
    test.write_csv(test_path)
    return train_path, test_path

@pytest.fixture
def shifted_data(tmp_path):
    """Train and test from DIFFERENT distributions — should return 'severe'."""
    np.random.seed(42)
    train = pl.DataFrame({
        "feat_1": np.random.normal(0, 1, 1500).tolist(),
        "feat_2": np.random.normal(5, 2, 1500).tolist(),
        "feat_3": np.random.choice(["A", "B", "C"], 1500).tolist(),
        "target": np.random.randint(0, 2, 1500).tolist(),
    })
    test = pl.DataFrame({
        "feat_1": np.random.normal(3, 1, 500).tolist(),    # SHIFTED mean
        "feat_2": np.random.normal(10, 2, 500).tolist(),   # SHIFTED mean
        "feat_3": np.random.choice(["A", "D", "E"], 500).tolist(),  # SHIFTED categories
    })
    
    train_path = tmp_path / "clean_train.parquet"
    test_path = tmp_path / "test.csv"
    
    train.write_parquet(train_path)
    test.write_csv(test_path)
    return train_path, test_path

# ── Tests ───────────────────────────────────────────────────────────────────

class TestShiftDetectorContract:
    """
    Contract: Distribution Shift Detector (Component 5 - Corrected)
    """

    def test_clean_data_returns_clean(self, clean_data):
        """Verify identical distributions return 'clean'."""
        tr, te = clean_data
        state = ProfessorState(**initial_state(clean_data_path=str(tr), test_data_path=str(te), session_id="test-clean"))
        result = run_shift_detector(state)
        
        assert result["shift_severity"] == "clean"
        assert result["shift_report"]["adversarial_auc"] < 0.55

    def test_shifted_data_returns_severe(self, shifted_data):
        """Verify drifted distributions return 'severe'."""
        tr, te = shifted_data
        state = ProfessorState(**initial_state(clean_data_path=str(tr), test_data_path=str(te), session_id="test-severe"))
        result = run_shift_detector(state)
        
        assert result["shift_severity"] == "severe"
        assert result["shift_report"]["adversarial_auc"] > 0.65

    def test_shift_report_has_all_required_keys(self, clean_data):
        """Verify the 10 core keys in shift_report."""
        tr, te = clean_data
        state = ProfessorState(**initial_state(clean_data_path=str(tr), test_data_path=str(te)))
        result = run_shift_detector(state)
        report = result["shift_report"]
        
        keys = [
            "adversarial_auc", "severity", "drifted_features", "n_drifted",
            "n_total_features", "drift_ratio", "sample_weights_generated",
            "sample_weights_path", "remediation_strategy", "checked_at"
        ]
        for k in keys:
            assert k in report

    def test_adversarial_auc_range(self, clean_data):
        tr, te = clean_data
        state = ProfessorState(**initial_state(clean_data_path=str(tr), test_data_path=str(te)))
        result = run_shift_detector(state)
        assert 0.0 <= result["shift_report"]["adversarial_auc"] <= 1.0

    def test_severity_is_valid_enum(self, shifted_data):
        tr, te = shifted_data
        result = run_shift_detector(ProfessorState(**initial_state(clean_data_path=str(tr), test_data_path=str(te))))
        assert result["shift_severity"] in ["clean", "mild", "severe"]

    def test_sample_weights_generated_when_severe(self, shifted_data):
        tr, te = shifted_data
        result = run_shift_detector(ProfessorState(**initial_state(clean_data_path=str(tr), test_data_path=str(te))))
        assert result["sample_weights_path"] != ""
        assert os.path.exists(result["sample_weights_path"])

    def test_no_weights_when_clean(self, clean_data):
        tr, te = clean_data
        result = run_shift_detector(ProfessorState(**initial_state(clean_data_path=str(tr), test_data_path=str(te))))
        assert result["sample_weights_path"] == ""

    def test_shifted_features_populated(self, shifted_data):
        tr, te = shifted_data
        result = run_shift_detector(ProfessorState(**initial_state(clean_data_path=str(tr), test_data_path=str(te))))
        assert len(result["shifted_features"]) > 0
        assert "feat_1" in result["shifted_features"]

    def test_dual_threshold_for_numeric(self, tmp_path):
        """Verify KS < 0.001 AND PSI > 0.25 requirement."""
        # Create subtle drift: low PSI but high sample size (making KS p low)
        np.random.seed(42)
        n = 50000 # Large sample size
        train = pl.DataFrame({"feat_1": np.random.normal(0, 1, n).tolist()})
        test = pl.DataFrame({"feat_1": np.random.normal(0.01, 1, n).tolist()}) # Tiny shift
        
        train_path = tmp_path / "train.parquet"
        test_path = tmp_path / "test.csv"
        train.write_parquet(train_path)
        test.write_csv(test_path)
        
        state = ProfessorState(**initial_state(clean_data_path=str(train_path), test_data_path=str(test_path)))
        result = run_shift_detector(state)
        
        # Should NOT be flagged as drifted if PSI is low, even if KS p is low
        assert "feat_1" not in result["shifted_features"]

    def test_state_has_no_raw_arrays(self, shifted_data):
        tr, te = shifted_data
        result = run_shift_detector(ProfessorState(**initial_state(clean_data_path=str(tr), test_data_path=str(te))))
        assert isinstance(result["sample_weights_path"], str)

    def test_pipeline_continues_on_severe(self, shifted_data):
        tr, te = shifted_data
        result = run_shift_detector(ProfessorState(**initial_state(clean_data_path=str(tr), test_data_path=str(te))))
        assert result["pipeline_halted"] is False

    def test_graceful_failure(self):
        """Verify 'unchecked' status on total failure."""
        state = ProfessorState(**initial_state(clean_data_path="missing", test_data_path="missing"))
        result = run_shift_detector(state)
        assert result["shift_severity"] == "unchecked"
