# tests/contracts/test_ml_optimizer_contract.py

import pytest
import os
import pickle
import numpy as np
import polars as pl
from pathlib import Path
from core.state import initial_state
from agents.ml_optimizer import run_ml_optimizer

FIXTURE_TRAIN = "data/spaceship_titanic/train.parquet"

# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def ml_optimizer_state():
    """State after Feature Factory."""
    # Ensure parquet exists
    if not os.path.exists(FIXTURE_TRAIN):
        df = pl.read_csv("data/spaceship_titanic/train.csv")
        df.write_parquet(FIXTURE_TRAIN)
        
    return initial_state(
        session_id="test-ml",
        feature_data_path=FIXTURE_TRAIN,
        target_col="Transported",
        task_type="binary"
    )

# ── Tests ───────────────────────────────────────────────────────────────────

class TestMLOptimizerContract:
    """
    Contract: ML Optimizer Agent
    Ensures model training, registry updates, and OOF persistence.
    """

    def test_model_registry_populated(self, ml_optimizer_state):
        """Verify model_registry contains at least one entry."""
        state = run_ml_optimizer(ml_optimizer_state)
        assert len(state["model_registry"]) >= 1
        entry = state["model_registry"][-1]
        assert "model_path" in entry
        assert "cv_mean" in entry

    def test_best_model_pkl_exists(self, ml_optimizer_state):
        """Verify the best model is persisted as PKL."""
        state = run_ml_optimizer(ml_optimizer_state)
        path = Path(state["model_registry"][-1]["model_path"])
        assert path.exists()
        assert path.suffix == ".pkl"
        
        # Verify it can be loaded
        with open(path, "rb") as f:
            model = pickle.load(f)
        assert hasattr(model, "predict")

    def test_oof_predictions_exists(self, ml_optimizer_state):
        """Verify OOF predictions are persisted as NPY."""
        state = run_ml_optimizer(ml_optimizer_state)
        path = Path(state["oof_predictions_path"])
        assert path.exists()
        assert path.suffix == ".npy"
        
        # Verify shape
        oof = np.load(path)
        assert len(oof) > 0

    def test_cv_mean_in_state(self, ml_optimizer_state):
        """Verify cv_mean is updated in state."""
        state = run_ml_optimizer(ml_optimizer_state)
        assert isinstance(state["cv_mean"], float)
        assert 0.0 <= state["cv_mean"] <= 1.0

    def test_best_params_captured(self, ml_optimizer_state):
        """Verify best_params dict is written to state."""
        state = run_ml_optimizer(ml_optimizer_state)
        assert isinstance(state["best_params"], dict)
        assert "model_type" in state["best_params"]
