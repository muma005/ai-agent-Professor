# tests/integration/test_layer3_integration.py

import pytest
import os
import polars as pl
import numpy as np
from unittest.mock import patch, MagicMock
from core.state import ProfessorState, initial_state

from agents.validation_architect import run_validation_architect
from agents.ml_optimizer import run_ml_optimizer
from agents.ensemble_architect import run_ensemble_architect
from agents.submission_strategist import run_submission_strategist

@pytest.fixture
def layer3_mock_data(tmp_path):
    train_path = str(tmp_path / "train.parquet")
    test_path = str(tmp_path / "test.csv")
    sample_sub_path = str(tmp_path / "sample_sub.csv")
    schema_path = str(tmp_path / "schema.json")
    
    pl.DataFrame({
        "id": range(100),
        "feat1": np.random.randn(100),
        "target": np.random.randint(0, 2, 100)
    }).write_parquet(train_path)
    
    pl.DataFrame({
        "id": range(100, 120),
        "feat1": np.random.randn(20)
    }).write_csv(test_path)
    
    pl.DataFrame({
        "id": range(100, 120),
        "target": [0]*20
    }).write_csv(sample_sub_path)
    
    import json
    with open(schema_path, "w") as f:
        json.dump({"target": "Int64", "feat1": "Float64"}, f)
    
    return train_path, test_path, sample_sub_path, schema_path

class TestLayer3Integration:

    @patch("optuna.create_study")
    def test_layer3_e2e_sequence(self, mock_study, layer3_mock_data):
        train_path, test_path, sample_sub_path, schema_path = layer3_mock_data
        
        # Mock Optuna study to be very fast
        study_instance = MagicMock()
        study_instance.best_trial.params = {"n_estimators": 2, "learning_rate": 0.1, "max_depth": 3}
        mock_study.return_value = study_instance

        # 1. Setup State at boundary of Layer 2 -> Layer 3
        state_dict = initial_state(
            session_id="test-layer3-e2e",
            feature_data_path=train_path,
            test_data_path=test_path,
            sample_submission_path=sample_sub_path,
            target_col="target",
            task_type="binary",
            pipeline_depth="sprint"
        )
        state = ProfessorState(**state_dict)
        state.canonical_train_rows = 100
        state.canonical_test_rows = 20
        state.schema_path = schema_path

        # 2. Run Validation Architect
        state = run_validation_architect(state)
        assert state["validation_strategy"] is not None
        assert state["metric_contract"] is not None
        assert state["gate_config"] is not None

        # 3. Run ML Optimizer
        state = run_ml_optimizer(state)
        assert len(state["model_configs"]) == 3
        assert os.path.exists(state["oof_predictions_path"])

        # 4. Run Ensemble Architect
        state = run_ensemble_architect(state)
        assert state["ensemble_method"] != ""
        assert os.path.exists(state["ensemble_test_predictions_path"])

        # 5. Run Submission Strategist
        state = run_submission_strategist(state)
        assert os.path.exists(state["submission_path"])
        assert state["ewma_current"] is not None
        assert state["n_submissions_with_lb"] == 1
