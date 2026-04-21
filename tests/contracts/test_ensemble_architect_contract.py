# tests/contracts/test_ensemble_architect_contract.py

import pytest
import numpy as np
from core.state import initial_state
from agents.ensemble_architect import run_ensemble_architect

# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def ensemble_state():
    """State with multiple diverse models in registry."""
    return initial_state(
        session_id="test-ensemble",
        data_hash="hash-123",
        model_registry=[
            {
                "model_id": "lgbm_1",
                "data_hash": "hash-123",
                "cv_mean": 0.85,
                "fold_scores": [0.84, 0.86, 0.85, 0.85, 0.85],
                "oof_predictions": [0.8, 0.2, 0.7]*100
            },
            {
                "model_id": "xgb_1",
                "data_hash": "hash-123",
                "cv_mean": 0.84,
                "fold_scores": [0.83, 0.85, 0.84, 0.84, 0.84],
                "oof_predictions": [0.7, 0.3, 0.6]*100
            }
        ],
        y_train=[1, 0, 1]*100,
        evaluation_metric="auc",
        task_type="binary"
    )

# ── Tests ───────────────────────────────────────────────────────────────────

class TestEnsembleArchitectContract:
    """
    Contract: Ensemble Architect Agent
    Ensures diversity selection and weight optimization.
    """

    def test_selected_models_populated(self, ensemble_state):
        """Verify selected_models is a non-empty list."""
        state = run_ensemble_architect(ensemble_state)
        assert isinstance(state["selected_models"], list)
        assert len(state["selected_models"]) > 0

    def test_ensemble_weights_sum_to_one(self, ensemble_state):
        """Verify weights are normalized."""
        state = run_ensemble_architect(ensemble_state)
        weights = state["ensemble_weights"]
        assert isinstance(weights, dict)
        assert abs(sum(weights.values()) - 1.0) < 1e-5

    def test_ensemble_accepted_is_boolean(self, ensemble_state):
        """Verify acceptance flag is written."""
        state = run_ensemble_architect(ensemble_state)
        assert isinstance(state["ensemble_accepted"], bool)

    def test_empty_registry_safety(self):
        """Verify safety when registry is empty."""
        state = initial_state(session_id="empty", model_registry=[])
        result = run_ensemble_architect(state)
        # Should skip and return state
        assert "ensemble_weights" not in result or result.get("ensemble_weights") is None

    def test_holdout_score_captured(self, ensemble_state):
        """Verify ensemble performance is measured."""
        state = run_ensemble_architect(ensemble_state)
        assert "ensemble_selection" in state
        assert "holdout_score" in state["ensemble_selection"]
        assert isinstance(state["ensemble_selection"]["holdout_score"], float)
