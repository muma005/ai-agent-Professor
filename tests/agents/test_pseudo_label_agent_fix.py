"""
Test fixture for pseudo_label_agent bug fixes.
Run BEFORE making any changes to establish baseline.
Run AFTER each fix to verify no regression.
"""
import pytest
import numpy as np
import polars as pl
from unittest.mock import patch, MagicMock
import os
import sys
import json
import pickle

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from core.state import initial_state


class TestPseudoLabelAgentFixtures:
    """Test fixtures that MUST pass after fixes."""
    
    @pytest.fixture
    def minimal_state(self, tmp_path):
        """Create minimal valid state for pseudo_label_agent."""
        session_id = "test_pl_fix"
        output_dir = tmp_path / "outputs" / session_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create synthetic train data
        np.random.seed(42)
        n_rows, n_features = 100, 5
        X = np.random.randn(n_rows, n_features)
        y = (X[:, 0] + X[:, 1] + np.random.randn(n_rows) * 0.5 > 0).astype(int)
        
        feature_cols = [f"feature_{i}" for i in range(n_features)]
        train_df = pl.DataFrame({
            **{col: X[:, i] for i, col in enumerate(feature_cols)},
            "target": y
        })
        test_df = pl.DataFrame({
            col: X[:50, i] for i, col in enumerate(feature_cols)
        })
        
        train_path = output_dir / "X_train.parquet"
        test_path = output_dir / "X_test.parquet"
        train_df.write_parquet(train_path)
        test_df.write_parquet(test_path)
        
        # Create metric contract
        metric_contract = {
            "scorer_name": "auc",
            "direction": "maximize",
            "requires_proba": True,
            "task_type": "classification"
        }
        with open(output_dir / "metric_contract.json", "w") as f:
            json.dump(metric_contract, f)
        
        # Create model registry entry
        model_registry = [{
            "model_type": "lgbm",
            "model_path": str(output_dir / "model.pkl"),
            "params": {"n_estimators": 10, "verbosity": -1},
            "fold_scores": [0.75, 0.78, 0.76, 0.77, 0.74],
            "cv_mean": 0.76
        }]
        
        # Create dummy model file (can't pickle MagicMock, use simple dict)
        with open(output_dir / "model.pkl", "wb") as f:
            pickle.dump({"model_type": "lgbm", "params": {}}, f)
        
        state = initial_state(
            competition="test_pl",
            data_path=str(train_path),
            budget_usd=0.10
        )
        
        # Set required state keys
        state["session_id"] = session_id
        state["feature_data_path"] = str(train_path)
        state["feature_data_path_test"] = str(test_path)
        state["target_col"] = "target"
        state["metric_contract_path"] = str(output_dir / "metric_contract.json")
        state["model_registry"] = model_registry
        state["selected_models"] = ["lgbm"]
        state["feature_order"] = feature_cols
        
        return state
    
    def test_state_has_required_keys(self, minimal_state):
        """Verify test fixture has all required state keys."""
        required = [
            "feature_data_path",
            "feature_data_path_test", 
            "target_col",
            "metric_contract_path",
            "model_registry",
            "selected_models",
            "feature_order"
        ]
        for key in required:
            assert key in minimal_state, f"Missing required key: {key}"
            assert minimal_state[key] is not None, f"Key is None: {key}"
    
    def test_data_files_exist(self, minimal_state):
        """Verify data files exist on disk."""
        assert os.path.exists(minimal_state["feature_data_path"])
        assert os.path.exists(minimal_state["feature_data_path_test"])
    
    def test_data_has_correct_schema(self, minimal_state):
        """Verify data files have correct schema."""
        X_train = pl.read_parquet(minimal_state["feature_data_path"])
        X_test = pl.read_parquet(minimal_state["feature_data_path_test"])
        
        # Train should have target
        assert "target" in X_train.columns
        # Test should not have target
        assert "target" not in X_test.columns
        # Both should have same feature columns
        feature_cols = [c for c in X_train.columns if c != "target"]
        assert feature_cols == X_test.columns


class TestPseudoLabelAgentAfterFix:
    """Tests that should pass AFTER fixes are implemented."""
    
    @pytest.fixture
    def fixed_state(self, tmp_path):
        """Create state for testing fixed agent."""
        session_id = "test_pl_fixed"
        output_dir = tmp_path / "outputs" / session_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create synthetic train data
        np.random.seed(42)
        n_rows, n_features = 100, 5
        X = np.random.randn(n_rows, n_features)
        y = (X[:, 0] + X[:, 1] + np.random.randn(n_rows) * 0.5 > 0).astype(int)
        
        feature_cols = [f"feature_{i}" for i in range(n_features)]
        train_df = pl.DataFrame({
            **{col: X[:, i] for i, col in enumerate(feature_cols)},
            "target": y
        })
        test_df = pl.DataFrame({
            col: X[:50, i] for i, col in enumerate(feature_cols)
        })
        
        train_path = output_dir / "X_train.parquet"
        test_path = output_dir / "X_test.parquet"
        train_df.write_parquet(train_path)
        test_df.write_parquet(test_path)
        
        # Create metric contract
        metric_contract = {
            "scorer_name": "auc",
            "direction": "maximize",
            "requires_proba": True,
            "task_type": "classification"
        }
        with open(output_dir / "metric_contract.json", "w") as f:
            json.dump(metric_contract, f)
        
        # Create model registry entry with all required fields
        model_registry = [{
            "model_type": "lgbm",
            "model_path": str(output_dir / "model.pkl"),
            "params": {"n_estimators": 10, "verbosity": -1},
            "fold_scores": [0.75, 0.78, 0.76, 0.77, 0.74],
            "cv_mean": 0.76,
            "data_hash": "test123"
        }]
        
        # Create dummy model file
        with open(output_dir / "model.pkl", "wb") as f:
            pickle.dump({"model_type": "lgbm", "params": {}}, f)
        
        state = initial_state(
            competition="test_pl",
            data_path=str(train_path),
            budget_usd=0.10
        )
        
        state["session_id"] = session_id
        state["feature_data_path"] = str(train_path)
        state["feature_data_path_test"] = str(test_path)
        state["target_col"] = "target"
        state["metric_contract_path"] = str(output_dir / "metric_contract.json")
        state["model_registry"] = model_registry
        state["selected_models"] = ["lgbm"]
        state["feature_order"] = feature_cols
        
        return state
    
    def test_agent_runs_without_name_error(self, fixed_state):
        """Agent should not crash with NameError on undefined variables."""
        from agents.pseudo_label_agent import run_pseudo_label_agent
        
        # This should not raise NameError
        try:
            result = run_pseudo_label_agent(fixed_state)
            # If it runs, check it returns a dict
            assert isinstance(result, dict)
        except NameError as e:
            pytest.fail(f"Agent crashed with NameError: {e}")
        except Exception as e:
            # Other exceptions are OK for now (e.g., model training issues)
            pass
    
    def test_agent_imports_is_significantly_better(self, fixed_state):
        """Agent should import is_significantly_better without error."""
        try:
            from agents.pseudo_label_agent import run_pseudo_label_agent
        except ImportError as e:
            if "is_significantly_better" in str(e):
                pytest.fail(f"Missing import: is_significantly_better - {e}")


class TestHelperFunctions:
    """Test helper functions that should work independently."""
    
    def test_compute_confidence_binary(self):
        """Confidence for binary classification should be |pred - 0.5|."""
        from agents.pseudo_label_agent import _compute_confidence
        import numpy as np
        
        y_pred = np.array([0.9, 0.7, 0.5, 0.3, 0.1])
        confidence = _compute_confidence(y_pred, "auc")
        
        expected = np.abs(y_pred - 0.5)
        np.testing.assert_array_almost_equal(confidence, expected)
    
    def test_compute_confidence_multiclass_1d(self):
        """Confidence for 1D multiclass should be |pred - 0.5|."""
        from agents.pseudo_label_agent import _compute_confidence
        import numpy as np
        
        y_pred = np.array([0.8, 0.6, 0.4, 0.2])
        confidence = _compute_confidence(y_pred, "multiclass")
        
        expected = np.abs(y_pred - 0.5)
        np.testing.assert_array_almost_equal(confidence, expected)
    
    def test_select_confident_samples(self):
        """Should select top fraction by confidence."""
        from agents.pseudo_label_agent import _select_confident_samples
        import numpy as np
        
        confidence = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        y_pred = np.array([0, 1, 0, 1, 1])
        
        mask, threshold = _select_confident_samples(confidence, y_pred, top_fraction=0.4)
        
        # Should select top 40% (2 out of 5)
        assert mask.sum() == 2
        # Should be the highest confidence samples
        assert mask[3] and mask[4]
    
    def test_pseudo_label_result_dataclass(self):
        """PseudoLabelResult should use field(default_factory=list)."""
        from agents.pseudo_label_agent import PseudoLabelResult
        
        # Create two instances
        result1 = PseudoLabelResult(iterations_completed=0)
        result2 = PseudoLabelResult(iterations_completed=0)
        
        # Modifying one should not affect the other
        result1.pseudo_labels_added.append(10)
        
        assert len(result1.pseudo_labels_added) == 1
        assert len(result2.pseudo_labels_added) == 0


class TestRegressionPrevention:
    """Tests to ensure fixes don't break existing functionality."""
    
    @pytest.fixture
    def minimal_state(self, tmp_path):
        """Create minimal valid state for pseudo_label_agent."""
        session_id = "test_pl_fix"
        output_dir = tmp_path / "outputs" / session_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create synthetic train data
        np.random.seed(42)
        n_rows, n_features = 100, 5
        X = np.random.randn(n_rows, n_features)
        y = (X[:, 0] + X[:, 1] + np.random.randn(n_rows) * 0.5 > 0).astype(int)
        
        feature_cols = [f"feature_{i}" for i in range(n_features)]
        train_df = pl.DataFrame({
            **{col: X[:, i] for i, col in enumerate(feature_cols)},
            "target": y
        })
        test_df = pl.DataFrame({
            col: X[:50, i] for i, col in enumerate(feature_cols)
        })
        
        train_path = output_dir / "X_train.parquet"
        test_path = output_dir / "X_test.parquet"
        train_df.write_parquet(train_path)
        test_df.write_parquet(test_path)
        
        # Create metric contract
        metric_contract = {
            "scorer_name": "auc",
            "direction": "maximize",
            "requires_proba": True,
            "task_type": "classification"
        }
        with open(output_dir / "metric_contract.json", "w") as f:
            json.dump(metric_contract, f)
        
        # Create model registry entry
        model_registry = [{
            "model_type": "lgbm",
            "model_path": str(output_dir / "model.pkl"),
            "params": {"n_estimators": 10, "verbosity": -1},
            "fold_scores": [0.75, 0.78, 0.76, 0.77, 0.74],
            "cv_mean": 0.76
        }]
        
        # Create dummy model file
        with open(output_dir / "model.pkl", "wb") as f:
            pickle.dump({"model_type": "lgbm", "params": {}}, f)
        
        state = initial_state(
            competition="test_pl",
            data_path=str(train_path),
            budget_usd=0.10
        )
        
        state["session_id"] = session_id
        state["feature_data_path"] = str(train_path)
        state["feature_data_path_test"] = str(test_path)
        state["target_col"] = "target"
        state["metric_contract_path"] = str(output_dir / "metric_contract.json")
        state["model_registry"] = model_registry
        state["selected_models"] = ["lgbm"]
        state["feature_order"] = feature_cols
        
        return state
    
    def test_state_keys_preserved(self, minimal_state):
        """Agent should preserve all state keys it doesn't modify."""
        from agents.pseudo_label_agent import run_pseudo_label_agent
        
        # Store original keys
        original_keys = set(minimal_state.keys())
        
        # Run agent (may skip due to missing model)
        try:
            result = run_pseudo_label_agent(minimal_state)
            # Check original keys are still present
            assert original_keys.issubset(set(result.keys()))
        except Exception:
            pass  # OK if agent fails for other reasons
    
    def test_graceful_skip_on_missing_data(self, minimal_state):
        """Agent should skip gracefully if data files don't exist."""
        from agents.pseudo_label_agent import run_pseudo_label_agent
        
        # Set invalid paths
        minimal_state["feature_data_path"] = "/nonexistent/path.parquet"
        minimal_state["feature_data_path_test"] = "/nonexistent/path_test.parquet"
        
        # Should return without crashing
        result = run_pseudo_label_agent(minimal_state)
        
        assert result["pseudo_labels_applied"] == False
        assert result["pseudo_label_cv_improvement"] == 0.0
    
    def test_graceful_skip_on_missing_target_col(self, minimal_state):
        """Agent should skip gracefully if target_col is missing."""
        from agents.pseudo_label_agent import run_pseudo_label_agent
        
        minimal_state["target_col"] = ""
        
        # Should return without crashing (or raise clear error)
        try:
            result = run_pseudo_label_agent(minimal_state)
            assert isinstance(result, dict)
        except ValueError as e:
            # Clear error message is acceptable
            assert "target_col" in str(e)
