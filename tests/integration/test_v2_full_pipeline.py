# tests/integration/test_v2_full_pipeline.py

import pytest
import os
import json
import polars as pl
import numpy as np
from unittest.mock import patch, MagicMock
from core.state import ProfessorState, initial_state
from core.professor import run_professor

# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def dummy_competition_data(tmp_path):
    train_path = tmp_path / "train.parquet"
    test_path = tmp_path / "test.csv"
    sample_sub = tmp_path / "sample_submission.csv"
    
    # 100 train, 200 test (to trigger pseudo-labels)
    pl.DataFrame({
        "id": range(100),
        "feat1": np.random.randn(100),
        "feat2": np.random.randn(100),
        "target": np.random.randint(0, 2, 100)
    }).write_parquet(train_path)
    
    pl.DataFrame({
        "id": range(100, 300),
        "feat1": np.random.randn(200),
        "feat2": np.random.randn(200)
    }).write_csv(test_path)
    
    pl.DataFrame({
        "id": range(100, 300),
        "target": [0.5]*200
    }).write_csv(sample_sub)
    
    schema_path = tmp_path / "schema.json"
    with open(schema_path, "w") as f:
        json.dump({"target": "Int64", "id": "Int64", "feat1": "Float64", "feat2": "Float64"}, f)

    return str(train_path), str(test_path), str(sample_sub), str(schema_path)

# ── Tests ───────────────────────────────────────────────────────────────────

class TestV2FullPipeline:
    """
    E2E Integration: Layers 0-4
    """

    @patch("optuna.create_study")
    @patch("tools.llm_client.call_llm")
    @patch("tools.operator_channel.init_hitl")
    def test_end_to_end_success(self, mock_hitl, mock_llm, mock_optuna, dummy_competition_data, tmp_path):
        tr, te, sub, sc = dummy_competition_data
        
        # Mocks
        mock_hitl.return_value = MagicMock()
        
        def llm_side_effect(prompt, **kwargs):
            if "validation" in prompt.lower() or "scorer" in prompt.lower():
                return json.dumps({
                    "scorer_name": "log_loss",
                    "task_type": "binary",
                    "cv_type": "StratifiedKFold",
                    "n_splits": 3
                })
            if "hypothesis" in prompt.lower():
                return json.dumps({"hypotheses": [{"name": "f1", "logic": "df*2"}], "creative_direction": "test"})
            return "{}"
            
        mock_llm.side_effect = llm_side_effect
        
        study_instance = MagicMock()
        study_instance.best_trial.params = {"n_estimators": 2}
        study_instance.best_trial.user_attrs = {"params": {"model_type": "lightgbm", "n_estimators": 2}}
        mock_optuna.return_value = study_instance

        # 1. Setup Initial State
        state_dict = initial_state(
            session_id="test-e2e-full",
            raw_data_path=tr,
            test_data_path=te,
            sample_submission_path=sub,
            target_col="target",
            pipeline_depth="sprint"
        )
        
        # Override DAG to include hypothesis node
        state_dict["dag"] = [
            "data_engineer", "eda_agent", "validation_architect", 
            "creative_hypothesis", "feature_factory", "ml_optimizer", 
            "ensemble_architect", "submission_strategist"
        ]
        with patch("core.professor.Path") as mock_path:
            mock_path.return_value = tmp_path
            
            # 2. Run Professor Pipeline
            # Note: We use a simplified DAG for integration test speed
            # but include the key Layer 4 nodes
            state = run_professor(state_dict, timeout_seconds=600)
            
            # 3. Verify Layer 4 Outcomes
            # Publisher marked complete
            assert state.get("post_mortem_completed") is True
            assert os.path.exists(state.get("report_path"))
            
            # Solution assembled
            # Path logic in publisher: output_dir / "solution"
            # where output_dir = Path(f"outputs/{session_id}")
            # Inside the test, 'outputs' is at the project root.
            sol_dir = os.path.join("outputs", "test-e2e-full", "solution")
            assert os.path.exists(os.path.join(sol_dir, "solution_notebook.py"))
            assert os.path.exists(os.path.join(sol_dir, "solution_writeup.md"))
            
            # Submission generated
            assert os.path.exists(state.get("submission_path"))
            
            # Verify pseudo-label logic was at least attempted (in state)
            # (In sprint mode it skips, but we can verify field existence)
            assert "pseudo_label_activated" in state
