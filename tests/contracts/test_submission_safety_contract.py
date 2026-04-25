# tests/contracts/test_submission_safety_contract.py

import pytest
import os
import polars as pl
import numpy as np
from core.state import ProfessorState, initial_state
from shields.submission_safety import check_submission_safety, calculate_diversity

# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_submissions(tmp_path):
    sub_a = tmp_path / "sub_a_mock.csv"
    sub_b = tmp_path / "sub_b_mock.csv"
    
    np.random.seed(42)
    base = np.random.rand(100)
    
    pl.DataFrame({"id": range(100), "target": base}).write_csv(sub_a)
    pl.DataFrame({"id": range(100), "target": base + np.random.normal(0, 0.01, 100)}).write_csv(sub_b)
    
    return str(sub_a), str(sub_b)

@pytest.fixture
def divergent_submissions(tmp_path):
    sub_a = tmp_path / "sub_a_div.csv"
    sub_b = tmp_path / "sub_b_div.csv"
    
    np.random.seed(42)
    pl.DataFrame({"id": range(100), "target": np.random.rand(100)}).write_csv(sub_a)
    pl.DataFrame({"id": range(100), "target": np.random.rand(100)}).write_csv(sub_b)
    
    return str(sub_a), str(sub_b)

# ── Tests ───────────────────────────────────────────────────────────────────

class TestSubmissionSafetyContract:
    """
    Contract: Submission Safety Guard (Component 4)
    """

    def test_diversity_calculation_spearman(self, mock_submissions):
        sa, sb = mock_submissions
        corr = calculate_diversity(sa, sb)
        assert corr > 0.95

    def test_diversity_rating_enum(self, mock_submissions, divergent_submissions):
        # Case 1: LOW diversity (High correlation)
        sa, sb = mock_submissions
        state = ProfessorState(**initial_state(submission_a_path=sa, submission_b_path=sb))
        res = check_submission_safety(state)
        assert res["submission_safety_report"]["diversity_rating"] == "LOW"

        # Case 2: HIGH diversity (Low correlation)
        sa, sb = divergent_submissions
        state = ProfessorState(**initial_state(submission_a_path=sa, submission_b_path=sb))
        res = check_submission_safety(state)
        assert res["submission_safety_report"]["diversity_rating"] == "HIGH"

    def test_lb_noise_estimation_divergence(self):
        state = ProfessorState(**initial_state())
        state.ewma_initial = [0.1] * 100
        state.ewma_current = [0.15] * 100
        res = check_submission_safety(state)
        assert res["submission_safety_report"]["lb_noise_estimate"] > 0

    def test_freeze_override_active_last_7_days(self):
        state = ProfessorState(**initial_state())
        state.competition_context = {"days_remaining": 5}
        res = check_submission_safety(state)
        assert res["submission_safety_report"]["freeze_override_active"] is True

    def test_freeze_override_inactive_early(self):
        state = ProfessorState(**initial_state())
        state.competition_context = {"days_remaining": 15}
        res = check_submission_safety(state)
        assert res["submission_safety_report"]["freeze_override_active"] is False

    def test_risk_level_critical_on_high_noise(self):
        state = ProfessorState(**initial_state())
        # Large jump in EMA
        state.ewma_initial = [0.1] * 100
        state.ewma_current = [0.5] * 100
        res = check_submission_safety(state)
        assert res["submission_safety_report"]["risk_level"] == "CRITICAL"

    def test_state_updated_with_report(self, mock_submissions):
        sa, sb = mock_submissions
        state = ProfessorState(**initial_state(submission_a_path=sa, submission_b_path=sb))
        res = check_submission_safety(state)
        assert "risk_level" in res["submission_safety_report"]

    def test_diversity_correlation_persisted_to_field(self, mock_submissions):
        sa, sb = mock_submissions
        state = ProfessorState(**initial_state(submission_a_path=sa, submission_b_path=sb))
        res = check_submission_safety(state)
        assert res["submission_b_correlation_with_a"] > 0.9
