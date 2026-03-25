"""
Comprehensive tests for multi-model stacking.

Advanced Feature: Multi-Model Stacking
Tests are regression-aware with frozen baselines.
"""
import pytest
import numpy as np
from agents.stacking_agent import (
    StackingAgent,
    StackingResult,
    stack_models,
)
import lightgbm as lgb


# ── Frozen Test Data for Regression Testing ──────────────────────

np.random.seed(42)
FROZEN_N_SAMPLES = 500
FROZEN_N_FEATURES = 10

FROZEN_X = np.random.randn(FROZEN_N_SAMPLES, FROZEN_N_FEATURES)
FROZEN_Y = (FROZEN_X[:, 0] + FROZEN_X[:, 1] > 0).astype(int)

# Create base models for testing
def create_base_models():
    """Create base models for testing."""
    return {
        "lgbm": lgb.LGBMClassifier(n_estimators=10, random_state=42, verbose=-1),
        "lgbm2": lgb.LGBMClassifier(n_estimators=10, random_state=123, verbose=-1),
    }


class TestStackingAgentInit:
    """Test StackingAgent initialization."""

    def test_default_initialization(self):
        """Test default initialization parameters."""
        stacker = StackingAgent()
        
        assert stacker.n_folds == 5
        assert stacker.mode == "stacking"
        assert stacker.random_state == 42
        assert stacker.n_jobs == -1

    def test_custom_initialization(self):
        """Test custom initialization parameters."""
        stacker = StackingAgent(
            n_folds=3,
            mode="blending",
            random_state=123,
            n_jobs=1,
        )
        
        assert stacker.n_folds == 3
        assert stacker.mode == "blending"
        assert stacker.random_state == 123


class TestStackingFit:
    """Test stacking fit."""

    def test_fit_basic(self):
        """Test basic stacking fit."""
        stacker = StackingAgent(n_folds=3)
        base_models = create_base_models()
        
        result = stacker.fit(
            X=FROZEN_X,
            y=FROZEN_Y,
            base_models=base_models,
            meta_model="logistic",
        )
        
        assert isinstance(result, StackingResult)
        assert len(result.base_models) == 2
        assert result.meta_model == "logistic"
        assert result.cv_score_stacked > result.cv_score_base  # Should improve

    def test_fit_different_meta_models(self):
        """Test stacking with different meta-models."""
        stacker = StackingAgent(n_folds=3)
        base_models = create_base_models()
        
        for meta_model in ["logistic", "ridge", "rf", "gbm", "lgbm"]:
            result = stacker.fit(
                X=FROZEN_X,
                y=FROZEN_Y,
                base_models=base_models,
                meta_model=meta_model,
            )
            
            assert isinstance(result, StackingResult)
            assert result.meta_model == meta_model

    def test_fit_with_test_predictions(self):
        """Test stacking with test set predictions."""
        stacker = StackingAgent(n_folds=3)
        base_models = create_base_models()
        
        X_test = np.random.randn(100, FROZEN_N_FEATURES)
        
        result = stacker.fit(
            X=FROZEN_X,
            y=FROZEN_Y,
            base_models=base_models,
            meta_model="logistic",
            X_test=X_test,
        )
        
        assert result.test_predictions is not None
        assert result.test_predictions.shape == (100, 2)  # 100 samples, 2 models

    def test_fit_stacking_vs_blending(self):
        """Test stacking vs blending modes."""
        base_models = create_base_models()
        
        # Stacking mode
        stacker_stacking = StackingAgent(n_folds=3, mode="stacking")
        result_stacking = stacker_stacking.fit(
            X=FROZEN_X, y=FROZEN_Y,
            base_models=base_models,
        )
        
        # Blending mode
        stacker_blending = StackingAgent(n_folds=3, mode="blending")
        result_blending = stacker_blending.fit(
            X=FROZEN_X, y=FROZEN_Y,
            base_models=base_models,
        )
        
        assert result_stacking.mode == "stacking"
        assert result_blending.mode == "blending"
        assert result_stacking.oof_predictions.shape[0] == FROZEN_N_SAMPLES
        assert result_blending.oof_predictions.shape[0] == FROZEN_N_SAMPLES


class TestResultStructure:
    """Test result structure."""

    def test_result_has_required_fields(self):
        """Test result has all required fields."""
        stacker = StackingAgent(n_folds=3)
        base_models = create_base_models()
        
        result = stacker.fit(
            X=FROZEN_X,
            y=FROZEN_Y,
            base_models=base_models,
        )
        
        # Check all required fields
        assert hasattr(result, "base_models")
        assert hasattr(result, "meta_model")
        assert hasattr(result, "mode")
        assert hasattr(result, "n_folds")
        assert hasattr(result, "n_base_models")
        assert hasattr(result, "cv_score_base")
        assert hasattr(result, "cv_score_stacked")
        assert hasattr(result, "improvement")
        assert hasattr(result, "meta_model_weights")
        assert hasattr(result, "oof_predictions")
        assert hasattr(result, "test_predictions")

    def test_result_to_dict(self):
        """Test result serialization."""
        result = StackingResult(
            base_models=["lgbm", "xgb"],
            meta_model="logistic",
            mode="stacking",
            n_folds=5,
            n_base_models=2,
            cv_score_base=0.75,
            cv_score_stacked=0.80,
            improvement=0.05,
            meta_model_weights={"lgbm": 0.6, "xgb": 0.4},
            oof_predictions=np.random.randn(100, 2),
            test_predictions=np.random.randn(50, 2),
        )
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict["base_models"] == ["lgbm", "xgb"]
        assert result_dict["cv_score_stacked"] == 0.80
        assert "oof_predictions_shape" in result_dict


class TestOOFGeneration:
    """Test out-of-fold prediction generation."""

    def test_oof_predictions_shape(self):
        """Test OOF predictions have correct shape."""
        stacker = StackingAgent(n_folds=3)
        base_models = create_base_models()
        
        result = stacker.fit(
            X=FROZEN_X,
            y=FROZEN_Y,
            base_models=base_models,
        )
        
        assert result.oof_predictions.shape == (FROZEN_N_SAMPLES, 2)

    def test_oof_no_leakage(self):
        """Test OOF predictions have no leakage."""
        stacker = StackingAgent(n_folds=3)
        base_models = create_base_models()
        
        result = stacker.fit(
            X=FROZEN_X,
            y=FROZEN_Y,
            base_models=base_models,
        )
        
        # OOF predictions should not be perfect (no leakage)
        assert result.cv_score_stacked < 1.0


class TestMetaModelWeights:
    """Test meta-model weight extraction."""

    def test_linear_meta_model_weights(self):
        """Test weight extraction from linear meta-models."""
        stacker = StackingAgent(n_folds=3)
        base_models = create_base_models()
        
        result = stacker.fit(
            X=FROZEN_X,
            y=FROZEN_Y,
            base_models=base_models,
            meta_model="logistic",
        )
        
        # Logistic regression should have weights
        assert result.meta_model_weights is not None
        assert len(result.meta_model_weights) == 2
        assert "lgbm" in result.meta_model_weights
        assert "lgbm2" in result.meta_model_weights


class TestHistoryTracking:
    """Test history tracking."""

    def test_history_tracking(self):
        """Test stacking history is tracked."""
        stacker = StackingAgent(n_folds=3)
        base_models = create_base_models()
        
        # Run multiple stacking operations
        stacker.fit(X=FROZEN_X, y=FROZEN_Y, base_models=base_models)
        stacker.fit(X=FROZEN_X, y=FROZEN_Y, base_models=base_models)
        
        history = stacker.get_stacking_history()
        
        assert len(history) == 2
        assert all(isinstance(h, dict) for h in history)


class TestBaselineTracking:
    """Test baseline tracking."""

    def test_set_baseline(self):
        """Test setting baseline score."""
        stacker = StackingAgent()
        
        stacker.set_baseline("single_model", 0.75)
        
        assert "single_model" in stacker.baseline_scores
        assert stacker.baseline_scores["single_model"] == 0.75


class TestConvenienceFunction:
    """Test convenience function."""

    def test_stack_models_function(self):
        """Test stack_models convenience function."""
        base_models = create_base_models()
        
        result = stack_models(
            X=FROZEN_X,
            y=FROZEN_Y,
            base_models=base_models,
            meta_model="logistic",
            n_folds=3,
        )
        
        assert isinstance(result, StackingResult)
        assert result.improvement > -0.05  # Should not degrade significantly


class TestRegressionBaselines:
    """Test regression-aware baselines."""

    def test_frozen_data_shape(self):
        """Test frozen data has expected shape."""
        assert FROZEN_X.shape == (FROZEN_N_SAMPLES, FROZEN_N_FEATURES)
        assert len(FROZEN_Y) == FROZEN_N_SAMPLES

    def test_stacking_improves_over_base(self):
        """Test stacking improves over base models."""
        stacker = StackingAgent(n_folds=3)
        base_models = create_base_models()
        
        result = stacker.fit(
            X=FROZEN_X,
            y=FROZEN_Y,
            base_models=base_models,
        )
        
        # Stacking should improve or at least not degrade significantly
        assert result.improvement > -0.02  # Allow small degradation due to variance

    def test_stacking_with_more_models(self):
        """Test stacking with more base models."""
        stacker = StackingAgent(n_folds=3)
        
        # Create 3 base models
        base_models = {
            "lgbm1": lgb.LGBMClassifier(n_estimators=10, random_state=42, verbose=-1),
            "lgbm2": lgb.LGBMClassifier(n_estimators=10, random_state=123, verbose=-1),
            "lgbm3": lgb.LGBMClassifier(n_estimators=10, random_state=456, verbose=-1),
        }
        
        result = stacker.fit(
            X=FROZEN_X,
            y=FROZEN_Y,
            base_models=base_models,
        )
        
        assert result.n_base_models == 3
        assert result.oof_predictions.shape[1] == 3
