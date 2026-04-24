# tests/contracts/test_ensemble_architect_contract.py

import pytest
import os
import numpy as np
import polars as pl
from pathlib import Path
from unittest.mock import patch, MagicMock
from core.state import ProfessorState, initial_state
from agents.ensemble_architect import run_ensemble_architect

@pytest.fixture
def ensemble_state(tmp_path):
    oof_path = str(tmp_path / "oof.parquet")
    test_path = str(tmp_path / "test.parquet")
    
    pl.DataFrame({"pred": [0.1, 0.9, 0.4, 0.6]}).write_parquet(oof_path)
    pl.DataFrame({"pred": [0.2, 0.8]}).write_parquet(test_path)
    
    train_path = str(tmp_path / "train.parquet")
    test_features_path = str(tmp_path / "test_features.csv")
    
    pl.DataFrame({
        "f1": [1, 2, 3, 4],
        "target": [0, 1, 0, 1]
    }).write_parquet(train_path)
    
    pl.DataFrame({"f1": [4, 5]}).write_csv(test_features_path)

    state_dict = initial_state(
        session_id="test-ensemble",
        feature_data_path=train_path,
        test_data_path=test_features_path,
        target_col="target",
        task_type="classification",
        validation_strategy={"n_splits": 2},
        metric_contract={"scorer_name": "log_loss"}
    )
    state = ProfessorState(**state_dict)
    state.model_configs = [
        {"model_type": "lightgbm", "cv_score": 0.8},
        {"model_type": "xgboost", "cv_score": 0.75}
    ]
    state.best_model_type = "lightgbm"
    state.cv_mean = 0.8
    state.oof_predictions_path = oof_path
    state.test_predictions_path = test_path
    
    return state

class TestEnsembleArchitectContract:

    @patch("agents.ensemble_architect._get_metric")
    def test_loads_all_oof_predictions(self, mock_metric, ensemble_state):
        mock_metric.return_value = 0.9
        state = run_ensemble_architect(ensemble_state)
        # Should calculate metrics and pick one
        assert state["ensemble_method"] in ["meta_learner", "simple_mean", "best_single_model"]

    @patch("agents.ensemble_architect.LogisticRegression")
    def test_computes_meta_learner_and_mean(self, mock_lr, ensemble_state):
        run_ensemble_architect(ensemble_state)
        # Linear Regression fit should be called
        assert mock_lr.return_value.fit.called

    def test_selects_best_ensemble_method(self, ensemble_state):
        # We don't mock get_metric, so it calculates actual AUC
        state = run_ensemble_architect(ensemble_state)
        assert state["ensemble_method"] in ["meta_learner", "simple_mean", "best_single_model"]

    def test_ensemble_oof_path_saved(self, ensemble_state):
        state = run_ensemble_architect(ensemble_state)
        assert os.path.exists(state["ensemble_oof_path"])

    def test_ensemble_test_predictions_saved(self, ensemble_state):
        state = run_ensemble_architect(ensemble_state)
        assert os.path.exists(state["ensemble_test_predictions_path"])

    def test_ensemble_cv_score_calculated(self, ensemble_state):
        state = run_ensemble_architect(ensemble_state)
        assert isinstance(state["ensemble_cv_score"], float)
        assert state["ensemble_cv_score"] > 0

    def test_ensemble_weights_populated(self, ensemble_state):
        state = run_ensemble_architect(ensemble_state)
        assert isinstance(state["ensemble_weights"], dict)
        assert "lightgbm" in state["ensemble_weights"]
        assert "xgboost" in state["ensemble_weights"]
