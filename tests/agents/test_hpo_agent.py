"""
Comprehensive tests for advanced HPO.

Advanced Feature: Advanced HPO
Tests are regression-aware with frozen baselines.
"""
import pytest
import numpy as np
from agents.hpo_agent import (
    HPOAgent,
    HPOResult,
    optimize_hyperparameters,
)


# ── Frozen Test Data for Regression Testing ──────────────────────

np.random.seed(42)
FROZEN_N_SAMPLES = 500
FROZEN_N_FEATURES = 10

FROZEN_X = np.random.randn(FROZEN_N_SAMPLES, FROZEN_N_FEATURES)
FROZEN_Y = (FROZEN_X[:, 0] + FROZEN_X[:, 1] > 0).astype(int)


class TestHPOAgentInit:
    """Test HPOAgent initialization."""

    def test_default_initialization(self):
        """Test default initialization parameters."""
        hpo = HPOAgent()
        
        assert hpo.n_trials == 100
        assert hpo.timeout_minutes is None
        assert hpo.pruner_type == "hyperband"
        assert hpo.sampler_type == "tpe"
        assert hpo.n_folds == 5

    def test_custom_initialization(self):
        """Test custom initialization parameters."""
        hpo = HPOAgent(
            n_trials=50,
            timeout_minutes=10,
            pruner="median",
            sampler="random",
            n_folds=3,
        )
        
        assert hpo.n_trials == 50
        assert hpo.timeout_minutes == 10
        assert hpo.pruner_type == "median"
        assert hpo.sampler_type == "random"


class TestPrunerCreation:
    """Test pruner creation."""

    def test_create_hyperband_pruner(self):
        """Test Hyperband pruner creation."""
        hpo = HPOAgent(pruner="hyperband")
        
        assert hpo.pruner is not None
        assert hpo.pruner_type == "hyperband"

    def test_create_successive_halving_pruner(self):
        """Test SuccessiveHalving pruner creation."""
        hpo = HPOAgent(pruner="successive_halving")
        
        assert hpo.pruner is not None
        assert hpo.pruner_type == "successive_halving"

    def test_create_median_pruner(self):
        """Test Median pruner creation."""
        hpo = HPOAgent(pruner="median")
        
        assert hpo.pruner is not None
        assert hpo.pruner_type == "median"

    def test_create_no_pruner(self):
        """Test no pruner creation."""
        hpo = HPOAgent(pruner="none")
        
        assert hpo.pruner is None


class TestSamplerCreation:
    """Test sampler creation."""

    def test_create_tpe_sampler(self):
        """Test TPE sampler creation."""
        hpo = HPOAgent(sampler="tpe")
        
        assert hpo.sampler is not None
        assert hpo.sampler_type == "tpe"

    def test_create_random_sampler(self):
        """Test Random sampler creation."""
        hpo = HPOAgent(sampler="random")
        
        assert hpo.sampler is not None
        assert hpo.sampler_type == "random"


class TestHPOOptimization:
    """Test HPO optimization."""

    def test_optimize_basic(self):
        """Test basic HPO optimization."""
        hpo = HPOAgent(n_trials=5, n_folds=3, pruner="none")
        
        result = hpo.optimize(
            X=FROZEN_X,
            y=FROZEN_Y,
            model_type="lightgbm",
            metric="auc",
        )
        
        assert isinstance(result, HPOResult)
        assert result.best_score > 0.5  # Better than random
        assert result.n_trials == 5
        assert len(result.best_params) > 0

    def test_optimize_with_pruning(self):
        """Test HPO with pruning enabled."""
        hpo = HPOAgent(n_trials=5, n_folds=3, pruner="hyperband")
        
        result = hpo.optimize(
            X=FROZEN_X,
            y=FROZEN_Y,
            model_type="lightgbm",
            metric="auc",
        )
        
        assert isinstance(result, HPOResult)
        # Some trials may be pruned
        assert result.pruned_trials >= 0

    def test_optimize_different_models(self):
        """Test optimization for different model types."""
        hpo = HPOAgent(n_trials=3, n_folds=3, pruner="none")
        
        for model_type in ["lightgbm", "xgboost"]:
            result = hpo.optimize(
                X=FROZEN_X,
                y=FROZEN_Y,
                model_type=model_type,
                metric="auc",
            )
            
            assert isinstance(result, HPOResult)
            assert result.best_score > 0.5

    def test_optimize_different_metrics(self):
        """Test optimization for different metrics."""
        hpo = HPOAgent(n_trials=3, n_folds=3, pruner="none")
        
        for metric in ["auc", "logloss"]:
            result = hpo.optimize(
                X=FROZEN_X,
                y=FROZEN_Y,
                model_type="lightgbm",
                metric=metric,
            )
            
            assert isinstance(result, HPOResult)


class TestSearchSpace:
    """Test custom search space."""

    def test_optimize_with_custom_search_space(self):
        """Test optimization with custom search space."""
        hpo = HPOAgent(n_trials=3, n_folds=3, pruner="none")
        
        search_space = {
            "n_estimators": {"type": "int", "low": 50, "high": 100, "step": 50},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.1, "log": True},
        }
        
        result = hpo.optimize(
            X=FROZEN_X,
            y=FROZEN_Y,
            model_type="lightgbm",
            metric="auc",
            search_space=search_space,
        )
        
        assert isinstance(result, HPOResult)
        assert "n_estimators" in result.best_params
        assert "learning_rate" in result.best_params


class TestResultStructure:
    """Test result structure."""

    def test_result_has_required_fields(self):
        """Test result has all required fields."""
        hpo = HPOAgent(n_trials=3, n_folds=3, pruner="none")
        
        result = hpo.optimize(
            X=FROZEN_X,
            y=FROZEN_Y,
            model_type="lightgbm",
            metric="auc",
        )
        
        # Check all required fields
        assert hasattr(result, "best_params")
        assert hasattr(result, "best_score")
        assert hasattr(result, "n_trials")
        assert hasattr(result, "pruned_trials")
        assert hasattr(result, "completed_trials")
        assert hasattr(result, "optimization_time")
        assert hasattr(result, "pruner_type")
        assert hasattr(result, "sampler_type")
        assert hasattr(result, "study_direction")
        assert hasattr(result, "trial_history")

    def test_result_to_dict(self):
        """Test result serialization."""
        result = HPOResult(
            best_params={"n_estimators": 100, "learning_rate": 0.1},
            best_score=0.85,
            n_trials=10,
            pruned_trials=2,
            completed_trials=8,
            optimization_time=60.0,
            pruner_type="hyperband",
            sampler_type="tpe",
            study_direction="maximize",
            trial_history=[],
        )
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict["best_score"] == 0.85
        assert result_dict["n_trials"] == 10
        assert result_dict["pruner_type"] == "hyperband"


class TestHistoryTracking:
    """Test history tracking."""

    def test_history_tracking(self):
        """Test optimization history is tracked."""
        hpo = HPOAgent(n_trials=3, n_folds=3, pruner="none")
        
        # Run multiple optimizations
        hpo.optimize(X=FROZEN_X, y=FROZEN_Y, model_type="lightgbm")
        hpo.optimize(X=FROZEN_X, y=FROZEN_Y, model_type="lightgbm")
        
        history = hpo.get_optimization_history()
        
        assert len(history) == 2
        assert all(isinstance(h, dict) for h in history)


class TestBaselineTracking:
    """Test baseline tracking."""

    def test_set_baseline(self):
        """Test setting baseline score."""
        hpo = HPOAgent()
        
        hpo.set_baseline("default_params", 0.75)
        
        assert "default_params" in hpo.baseline_scores
        assert hpo.baseline_scores["default_params"] == 0.75

    def test_compare_to_baseline(self):
        """Test comparing result to baseline."""
        hpo = HPOAgent(n_trials=3, n_folds=3, pruner="none")
        
        hpo.set_baseline("default_params", 0.75)
        
        result = hpo.optimize(X=FROZEN_X, y=FROZEN_Y, model_type="lightgbm")
        
        comparison = hpo.compare_to_baseline(result, "default_params")
        
        assert "baseline_score" in comparison
        assert "optimized_score" in comparison
        assert "improvement" in comparison


class TestConvenienceFunction:
    """Test convenience function."""

    def test_optimize_hyperparameters_function(self):
        """Test optimize_hyperparameters function."""
        result = optimize_hyperparameters(
            X=FROZEN_X,
            y=FROZEN_Y,
            model_type="lightgbm",
            metric="auc",
            n_trials=3,
            n_folds=3,
            pruner="none",
        )
        
        assert isinstance(result, HPOResult)
        assert result.best_score > 0.5


class TestRegressionBaselines:
    """Test regression-aware baselines."""

    def test_frozen_data_shape(self):
        """Test frozen data has expected shape."""
        assert FROZEN_X.shape == (FROZEN_N_SAMPLES, FROZEN_N_FEATURES)
        assert len(FROZEN_Y) == FROZEN_N_SAMPLES

    def test_optimization_improves_over_random(self):
        """Test optimization improves over random guessing."""
        hpo = HPOAgent(n_trials=5, n_folds=3, pruner="none")
        
        result = hpo.optimize(
            X=FROZEN_X,
            y=FROZEN_Y,
            model_type="lightgbm",
            metric="auc",
        )
        
        # Should be better than random (0.5)
        assert result.best_score > 0.5

    def test_pruning_reduces_trials(self):
        """Test pruning can reduce completed trials."""
        hpo_no_prune = HPOAgent(n_trials=5, n_folds=3, pruner="none")
        hpo_with_prune = HPOAgent(n_trials=5, n_folds=3, pruner="hyperband")
        
        result_no_prune = hpo_no_prune.optimize(
            X=FROZEN_X, y=FROZEN_Y, model_type="lightgbm",
        )
        result_with_prune = hpo_with_prune.optimize(
            X=FROZEN_X, y=FROZEN_Y, model_type="lightgbm",
        )
        
        # With pruning, some trials may be pruned
        assert result_with_prune.pruned_trials >= 0
        assert result_with_prune.completed_trials <= result_no_prune.completed_trials
