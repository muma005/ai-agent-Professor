# tests/contracts/test_pseudo_label_agent_contract.py

import pytest
import os
import json
import polars as pl
from pathlib import Path
from core.state import initial_state
from agents.pseudo_label_agent import run_pseudo_label_agent

# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def pseudo_label_state():
    """State that passes all activation gates."""
    return initial_state(
        session_id="test-pl",
        metric_contract={"scorer_name": "auc"},
        canonical_train_rows=1000,
        canonical_test_rows=3000, # > 2x train
        model_registry=[{
            "is_calibrated": True,
            "calibration_score": 0.1, # 1.0 - 0.1 = 0.9 (> 0.8)
            "params": {"model_type": "lgbm"},
            "fold_scores": [0.85]*5
        }]
    )

# ── Tests ───────────────────────────────────────────────────────────────────

class TestPseudoLabelAgentContract:
    """
    Contract: Pseudo Label Agent
    Ensures confidence-gated iteration logic and safety gates.
    """

    def test_activation_gate_requires_probability_metric(self, pseudo_label_state):
        """Verify skip when metric is not probability-based."""
        state = {**pseudo_label_state, "metric_contract": {"scorer_name": "mae"}}
        result = run_pseudo_label_agent(state)
        # Should return state without applying PL
        assert result.get("pseudo_labels_applied") is False

    def test_activation_gate_requires_data_ratio(self, pseudo_label_state):
        """Verify skip when test set is too small."""
        state = {**pseudo_label_state, "canonical_test_rows": 500}
        result = run_pseudo_label_agent(state)
        assert result.get("pseudo_labels_applied") is False

    def test_activation_gate_requires_calibration(self, pseudo_label_state):
        """Verify skip when model is uncalibrated."""
        state = {**pseudo_label_state, "model_registry": [{"is_calibrated": False}]}
        result = run_pseudo_label_agent(state)
        assert result.get("pseudo_labels_applied") is False

    def test_pseudo_label_keys_written_to_state(self, pseudo_label_state):
        """Verify required keys are updated in state."""
        # Note: In mock tests, actual labeling might be skipped due to missing files,
        # but keys should still be initialized.
        state = run_pseudo_label_agent(pseudo_label_state)
        assert "pseudo_labels_applied" in state
        assert "pseudo_label_iterations" in state
        assert "pseudo_label_n_added" in state
