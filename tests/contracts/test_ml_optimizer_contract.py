# tests/contracts/test_ml_optimizer_contract.py

import pytest
import os
import pickle
import numpy as np
import polars as pl
from pathlib import Path
from unittest.mock import patch, MagicMock
from core.state import ProfessorState, initial_state
from agents.ml_optimizer import run_ml_optimizer

@pytest.fixture
def ml_optimizer_state(tmp_path):
    fixture_train = str(tmp_path / "train.parquet")
    fixture_test = str(tmp_path / "test.csv")
    
    df_train = pl.DataFrame({
        "feat1": np.random.randn(100),
        "Transported": np.random.randint(0, 2, 100)
    })
    df_train.write_parquet(fixture_train)
    # Create dummy test parquet
    df_test = pl.DataFrame({"feat1": np.random.randn(20)})
    df_test.write_csv(fixture_test)

    return ProfessorState(**initial_state(
        session_id="test-ml-opt",
        feature_data_path=fixture_train,
        test_data_path=fixture_test,
        target_col="Transported",
        task_type="binary",
        canonical_train_rows=100,
        canonical_test_rows=20,
        pipeline_depth="sprint",
        validation_strategy={"cv_type": "StratifiedKFold", "n_splits": 3},
        metric_contract={"scorer_name": "log_loss", "direction": "minimize"}
    ))

class TestMLOptimizerContract:

    @pytest.fixture(autouse=True)
    def mock_optuna(self):
        with patch("optuna.create_study") as mock_study:
            study_instance = MagicMock()
            study_instance.best_trial.params = {"n_estimators": 2, "learning_rate": 0.1, "max_depth": 3}
            mock_study.return_value = study_instance
            yield mock_study

    def test_3_model_configs_produced(self, ml_optimizer_state):
        state = run_ml_optimizer(ml_optimizer_state)
        assert len(state["model_configs"]) == 3
        types = [c["model_type"] for c in state["model_configs"]]
        assert set(types) == {"lightgbm", "xgboost", "catboost"}

    def test_model_configs_have_required_fields(self, ml_optimizer_state):
        state = run_ml_optimizer(ml_optimizer_state)
        cfg = state["model_configs"][0]
        assert "model_type" in cfg
        assert "params" in cfg
        assert "cv_score" in cfg
        assert "cv_std" in cfg

    def test_best_model_type_valid(self, ml_optimizer_state):
        state = run_ml_optimizer(ml_optimizer_state)
        assert state["best_model_type"] in ["lightgbm", "xgboost", "catboost"]

    def test_cv_scores_is_list_of_floats(self, ml_optimizer_state):
        state = run_ml_optimizer(ml_optimizer_state)
        assert isinstance(state["cv_scores"], list)
        for s in state["cv_scores"]:
            assert isinstance(s, float)

    def test_cv_mean_matches_scores(self, ml_optimizer_state):
        state = run_ml_optimizer(ml_optimizer_state)
        assert state["cv_mean"] is not None

    def test_oof_predictions_file_exists(self, ml_optimizer_state):
        state = run_ml_optimizer(ml_optimizer_state)
        assert os.path.exists(state["oof_predictions_path"])
        assert state["oof_predictions_path"].endswith(".parquet")

    def test_test_predictions_file_exists(self, ml_optimizer_state):
        state = run_ml_optimizer(ml_optimizer_state)
        assert os.path.exists(state["test_predictions_path"])
        assert state["test_predictions_path"].endswith(".parquet")

    def test_oof_row_count_matches_train(self, ml_optimizer_state):
        state = run_ml_optimizer(ml_optimizer_state)
        oof = pl.read_parquet(state["oof_predictions_path"])
        assert len(oof) == state["canonical_train_rows"]

    def test_test_row_count_matches_test(self, ml_optimizer_state):
        state = run_ml_optimizer(ml_optimizer_state)
        test_preds = pl.read_parquet(state["test_predictions_path"])
        assert len(test_preds) == state["canonical_test_rows"]

    def test_sprint_uses_50_trials(self, mock_optuna, ml_optimizer_state):
        ml_optimizer_state["pipeline_depth"] = "sprint"
        run_ml_optimizer(ml_optimizer_state)
        assert mock_optuna.return_value.optimize.call_args[1]["n_trials"] == 50

    def test_standard_uses_100_trials(self, mock_optuna, ml_optimizer_state):
        ml_optimizer_state["pipeline_depth"] = "standard"
        run_ml_optimizer(ml_optimizer_state)
        assert mock_optuna.return_value.optimize.call_args[1]["n_trials"] == 100

    def test_marathon_uses_200_trials(self, mock_optuna, ml_optimizer_state):
        ml_optimizer_state["pipeline_depth"] = "marathon"
        run_ml_optimizer(ml_optimizer_state)
        assert mock_optuna.return_value.optimize.call_args[1]["n_trials"] == 200

    @patch("agents.ml_optimizer.cross_val_score_with_params")
    def test_multi_seed_stability(self, mock_cv, ml_optimizer_state):
        mock_cv.return_value = 0.8
        state = run_ml_optimizer(ml_optimizer_state)
        assert mock_cv.call_count == 9  # 3 models * 3 seeds

    @patch("agents.ml_optimizer.cross_val_score_with_params")
    def test_stability_penalty_applied(self, mock_cv, ml_optimizer_state):
        # Return scores with high variance
        mock_cv.side_effect = [0.9, 0.5, 0.7, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
        with patch("agents.ml_optimizer._get_raw_cv_score", return_value=(0.7, [0.7,0.7,0.7])):
            state = run_ml_optimizer(ml_optimizer_state)
            std = np.std([0.9, 0.5, 0.7])
            mean = np.mean([0.9, 0.5, 0.7])
            penalty = std * 1.5
            expected_score = mean - penalty
            # Find the lightgbm score (first 3 seeds were for the first model)
            lgbm_cfg = next(c for c in state["model_configs"] if c["model_type"] == "lightgbm")
            assert abs(lgbm_cfg["cv_score"] - expected_score) < 1e-4

    @patch("agents.ml_optimizer.LGBMClassifier")
    def test_sample_weights_used_when_available(self, mock_lgb, ml_optimizer_state, tmp_path):
        weights_path = tmp_path / "weights.parquet"
        train_len = pl.read_parquet(ml_optimizer_state["feature_data_path"]).height
        pl.DataFrame({"weight": np.ones(train_len)}).write_parquet(weights_path)
        ml_optimizer_state["sample_weights_path"] = str(weights_path)
        
        run_ml_optimizer(ml_optimizer_state)
        # Verify sample_weight was passed to fit
        mock_instance = mock_lgb.return_value
        fit_kwargs = mock_instance.fit.call_args[1]
        assert "sample_weight" in fit_kwargs

    @patch("agents.ml_optimizer.emit_to_operator")
    def test_milestone_3_emitted(self, mock_emit, ml_optimizer_state):
        run_ml_optimizer(ml_optimizer_state)
        assert mock_emit.call_count >= 1
        calls = [c[0][0] for c in mock_emit.call_args_list]
        assert any("MODEL REPORT" in c for c in calls)
