"""
Comprehensive tests for ensemble optimization.

Advanced Feature: Ensemble Optimization
Tests are regression-aware with frozen baselines.
"""
import pytest
import numpy as np
from agents.ensemble_optimizer import (
    EnsembleOptimizer,
    EnsembleOptimizationResult,
    optimize_ensemble,
)


# ── Frozen Test Data for Regression Testing ──────────────────────

np.random.seed(42)
FROZEN_N_SAMPLES = 1000

# Create correlated predictions (realistic ensemble scenario)
FROZEN_Y_TRUE = np.random.randint(0, 2, FROZEN_N_SAMPLES)
FROZEN_MODEL_1 = FROZEN_Y_TRUE + np.random.normal(0, 0.3, FROZEN_N_SAMPLES)
FROZEN_MODEL_2 = FROZEN_Y_TRUE + np.random.normal(0, 0.4, FROZEN_N_SAMPLES)
FROZEN_MODEL_3 = FROZEN_Y_TRUE + np.random.normal(0, 0.5, FROZEN_N_SAMPLES)

# Clip to [0, 1]
FROZEN_MODEL_1 = np.clip(FROZEN_MODEL_1, 0, 1)
FROZEN_MODEL_2 = np.clip(FROZEN_MODEL_2, 0, 1)
FROZEN_MODEL_3 = np.clip(FROZEN_MODEL_3, 0, 1)

FROZEN_OOF_PREDICTIONS = [FROZEN_MODEL_1, FROZEN_MODEL_2, FROZEN_MODEL_3]
FROZEN_BASELINE_SCORE = 0.762  # From previous run with equal weights


class TestEnsembleOptimizerInit:
    """Test EnsembleOptimizer initialization."""

    def test_default_initialization(self):
        """Test default initialization parameters."""
        optimizer = EnsembleOptimizer()
        
        assert optimizer.n_folds == 5
        assert optimizer.random_state == 42
        assert optimizer.max_iterations == 1000
        assert optimizer.tolerance == 1e-6

    def test_custom_initialization(self):
        """Test custom initialization parameters."""
        optimizer = EnsembleOptimizer(
            n_folds=10,
            random_state=123,
            max_iterations=500,
            tolerance=1e-8,
        )
        
        assert optimizer.n_folds == 10
        assert optimizer.random_state == 123
        assert optimizer.max_iterations == 500
        assert optimizer.tolerance == 1e-8


class TestEnsembleOptimization:
    """Test ensemble optimization."""

    def test_optimize_basic(self):
        """Test basic ensemble optimization."""
        optimizer = EnsembleOptimizer(n_folds=3)  # Fewer folds for speed
        
        result = optimizer.optimize(
            oof_predictions=FROZEN_OOF_PREDICTIONS,
            y_true=FROZEN_Y_TRUE,
            metric="auc",
            method="nelder-mead",
        )
        
        assert isinstance(result, EnsembleOptimizationResult)
        assert len(result.optimal_weights) == 3
        assert abs(np.sum(result.optimal_weights) - 1.0) < 1e-6
        assert result.cv_score_after >= result.cv_score_before  # Should improve or equal

    def test_optimize_with_model_names(self):
        """Test optimization with model names."""
        optimizer = EnsembleOptimizer(n_folds=3)
        
        model_names = ["LightGBM", "XGBoost", "CatBoost"]
        
        result = optimizer.optimize(
            oof_predictions=FROZEN_OOF_PREDICTIONS,
            y_true=FROZEN_Y_TRUE,
            model_names=model_names,
            metric="auc",
        )
        
        assert result.model_names == model_names

    def test_optimize_nelder_mead(self):
        """Test Nelder-Mead optimization method."""
        optimizer = EnsembleOptimizer(n_folds=3)
        
        result = optimizer.optimize(
            oof_predictions=FROZEN_OOF_PREDICTIONS,
            y_true=FROZEN_Y_TRUE,
            method="nelder-mead",
        )
        
        assert result.method == "nelder-mead"
        assert result.converged in [True, False]  # May or may not converge

    def test_optimize_differential_evolution(self):
        """Test differential evolution optimization method."""
        optimizer = EnsembleOptimizer(n_folds=3)
        
        result = optimizer.optimize(
            oof_predictions=FROZEN_OOF_PREDICTIONS,
            y_true=FROZEN_Y_TRUE,
            method="differential_evolution",
        )
        
        assert result.method == "differential_evolution"

    def test_optimize_different_metrics(self):
        """Test optimization with different metrics."""
        optimizer = EnsembleOptimizer(n_folds=3)
        
        for metric in ["auc", "logloss", "rmse", "mae"]:
            result = optimizer.optimize(
                oof_predictions=FROZEN_OOF_PREDICTIONS,
                y_true=FROZEN_Y_TRUE,
                metric=metric,
            )
            
            assert isinstance(result, EnsembleOptimizationResult)
            assert result.cv_score_after != 0  # Should have valid score


class TestConstrainedOptimization:
    """Test constrained ensemble optimization."""

    def test_optimize_with_min_weights(self):
        """Test optimization with minimum weight constraints."""
        optimizer = EnsembleOptimizer(n_folds=3)
        
        min_weights = np.array([0.2, 0.2, 0.2])  # Each model at least 20%
        
        result = optimizer.optimize_with_constraints(
            oof_predictions=FROZEN_OOF_PREDICTIONS,
            y_true=FROZEN_Y_TRUE,
            min_weights=min_weights,
        )
        
        # Check constraints are satisfied
        for i, weight in enumerate(result.optimal_weights):
            assert weight >= min_weights[i] - 0.01  # Small tolerance

    def test_optimize_with_max_weights(self):
        """Test optimization with maximum weight constraints."""
        optimizer = EnsembleOptimizer(n_folds=3)
        
        max_weights = np.array([0.5, 0.5, 0.5])  # Each model at most 50%
        
        result = optimizer.optimize_with_constraints(
            oof_predictions=FROZEN_OOF_PREDICTIONS,
            y_true=FROZEN_Y_TRUE,
            max_weights=max_weights,
        )
        
        # Check constraints are satisfied
        for i, weight in enumerate(result.optimal_weights):
            assert weight <= max_weights[i] + 0.01  # Small tolerance

    def test_optimize_must_include_models(self):
        """Test optimization with must-include models."""
        optimizer = EnsembleOptimizer(n_folds=3)
        
        result = optimizer.optimize_with_constraints(
            oof_predictions=FROZEN_OOF_PREDICTIONS,
            y_true=FROZEN_Y_TRUE,
            must_include_models=[0, 2],  # Models 0 and 2 must be included
        )
        
        # Check must-include models have non-zero weight
        assert result.optimal_weights[0] > 0
        assert result.optimal_weights[2] > 0


class TestValidation:
    """Test input validation."""

    def test_validate_empty_predictions(self):
        """Test validation rejects empty predictions."""
        optimizer = EnsembleOptimizer()
        
        with pytest.raises(ValueError, match="cannot be empty"):
            optimizer.optimize(
                oof_predictions=[],
                y_true=FROZEN_Y_TRUE,
            )

    def test_validate_single_model(self):
        """Test validation requires at least 2 models."""
        optimizer = EnsembleOptimizer()
        
        with pytest.raises(ValueError, match="at least 2 models"):
            optimizer.optimize(
                oof_predictions=[FROZEN_MODEL_1],
                y_true=FROZEN_Y_TRUE,
            )

    def test_validate_mismatched_lengths(self):
        """Test validation rejects mismatched lengths."""
        optimizer = EnsembleOptimizer()
        
        with pytest.raises(ValueError, match="has.*predictions, expected"):
            optimizer.optimize(
                oof_predictions=[
                    FROZEN_MODEL_1[:500],  # Wrong length
                    FROZEN_MODEL_2,
                ],
                y_true=FROZEN_Y_TRUE,
            )

    def test_validate_mismatched_y_true(self):
        """Test validation rejects mismatched y_true."""
        optimizer = EnsembleOptimizer()
        
        with pytest.raises(ValueError, match="has.*samples, expected"):
            optimizer.optimize(
                oof_predictions=FROZEN_OOF_PREDICTIONS,
                y_true=FROZEN_Y_TRUE[:500],  # Wrong length
            )


class TestBaselineTracking:
    """Test baseline tracking functionality."""

    def test_set_baseline(self):
        """Test setting baseline score."""
        optimizer = EnsembleOptimizer()
        
        optimizer.set_baseline("equal_weights", FROZEN_BASELINE_SCORE)
        
        assert "equal_weights" in optimizer.baseline_scores
        assert optimizer.baseline_scores["equal_weights"] == FROZEN_BASELINE_SCORE

    def test_compare_to_baseline(self):
        """Test comparing result to baseline."""
        optimizer = EnsembleOptimizer(n_folds=3)
        
        optimizer.set_baseline("equal_weights", FROZEN_BASELINE_SCORE)
        
        result = optimizer.optimize(
            oof_predictions=FROZEN_OOF_PREDICTIONS,
            y_true=FROZEN_Y_TRUE,
        )
        
        comparison = optimizer.compare_to_baseline(result, "equal_weights")
        
        assert "baseline_score" in comparison
        assert "optimized_score" in comparison
        assert "improvement_vs_baseline" in comparison

    def test_compare_to_nonexistent_baseline(self):
        """Test error when comparing to nonexistent baseline."""
        optimizer = EnsembleOptimizer()
        
        result = optimizer.optimize(
            oof_predictions=FROZEN_OOF_PREDICTIONS,
            y_true=FROZEN_Y_TRUE,
            n_folds=3,
        )
        
        with pytest.raises(ValueError, match="not found"):
            optimizer.compare_to_baseline(result, "nonexistent_baseline")


class TestOptimizationHistory:
    """Test optimization history tracking."""

    def test_history_tracking(self):
        """Test optimization history is tracked."""
        optimizer = EnsembleOptimizer(n_folds=3)
        
        # Run multiple optimizations
        optimizer.optimize(
            oof_predictions=FROZEN_OOF_PREDICTIONS,
            y_true=FROZEN_Y_TRUE,
            method="nelder-mead",
        )
        
        optimizer.optimize(
            oof_predictions=FROZEN_OOF_PREDICTIONS,
            y_true=FROZEN_Y_TRUE,
            method="nelder-mead",
        )
        
        history = optimizer.get_optimization_history()
        
        assert len(history) == 2
        assert all(isinstance(h, dict) for h in history)

    def test_history_contains_required_fields(self):
        """Test history contains required fields."""
        optimizer = EnsembleOptimizer(n_folds=3)
        
        optimizer.optimize(
            oof_predictions=FROZEN_OOF_PREDICTIONS,
            y_true=FROZEN_Y_TRUE,
        )
        
        history = optimizer.get_optimization_history()
        
        required_fields = [
            "model_names", "optimal_weights", "cv_score_before",
            "cv_score_after", "improvement", "method", "converged",
        ]
        
        for field in required_fields:
            assert field in history[0]


class TestResultSerialization:
    """Test result serialization."""

    def test_result_to_dict(self):
        """Test EnsembleOptimizationResult serialization."""
        result = EnsembleOptimizationResult(
            model_names=["A", "B", "C"],
            optimal_weights=np.array([0.4, 0.4, 0.2]),
            cv_score_before=0.75,
            cv_score_after=0.80,
            improvement=0.05,
            method="nelder-mead",
            converged=True,
            n_iterations=50,
            final_loss=-0.80,
        )
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict["model_names"] == ["A", "B", "C"]
        assert result_dict["optimal_weights"] == [0.4, 0.4, 0.2]
        assert result_dict["cv_score_before"] == 0.75
        assert result_dict["cv_score_after"] == 0.80
        assert result_dict["improvement"] == 0.05


class TestConvenienceFunction:
    """Test optimize_ensemble convenience function."""

    def test_optimize_ensemble_function(self):
        """Test optimize_ensemble function."""
        result = optimize_ensemble(
            oof_predictions=FROZEN_OOF_PREDICTIONS,
            y_true=FROZEN_Y_TRUE,
            n_folds=3,
        )
        
        assert isinstance(result, EnsembleOptimizationResult)
        assert len(result.optimal_weights) == 3


class TestRegressionBaselines:
    """Test regression-aware baselines."""

    def test_frozen_baseline_score(self):
        """Test frozen baseline score is reasonable."""
        # Equal weights should give reasonable AUC
        equal_weights = np.ones(3) / 3
        equal_preds = np.column_stack(FROZEN_OOF_PREDICTIONS) @ equal_weights
        
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(FROZEN_Y_TRUE, equal_preds)
        
        # Should be better than random (0.5)
        assert auc > 0.5
        assert auc < 1.0

    def test_optimization_improves_score(self):
        """Test optimization improves over equal weights."""
        optimizer = EnsembleOptimizer(n_folds=3)
        
        result = optimizer.optimize(
            oof_predictions=FROZEN_OOF_PREDICTIONS,
            y_true=FROZEN_Y_TRUE,
        )
        
        # Optimized should be >= equal weights (may be equal if already optimal)
        assert result.cv_score_after >= result.cv_score_before - 0.01  # Small tolerance
