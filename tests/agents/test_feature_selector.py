"""
Comprehensive tests for feature selection.

Advanced Feature: Feature Selection Automation
Tests are regression-aware with frozen baselines.
"""
import pytest
import numpy as np
from agents.feature_selector import (
    FeatureSelector,
    FeatureSelectionResult,
    select_features,
)


# ── Frozen Test Data for Regression Testing ──────────────────────

np.random.seed(42)
FROZEN_N_SAMPLES = 500
FROZEN_N_FEATURES = 20

# Create synthetic data with some informative features
FROZEN_X = np.random.randn(FROZEN_N_SAMPLES, FROZEN_N_FEATURES)
# Make first 5 features informative
FROZEN_Y = (
    FROZEN_X[:, 0] * 2 +
    FROZEN_X[:, 1] * 1.5 +
    FROZEN_X[:, 2] * 1.0 +
    FROZEN_X[:, 3] * 0.5 +
    FROZEN_X[:, 4] * 0.3 +
    np.random.randn(FROZEN_N_SAMPLES) * 0.5
)
FROZEN_Y = (FROZEN_Y > np.median(FROZEN_Y)).astype(int)

FROZEN_FEATURE_NAMES = [f"feature_{i}" for i in range(FROZEN_N_FEATURES)]


class TestFeatureSelectorInit:
    """Test FeatureSelector initialization."""

    def test_default_initialization(self):
        """Test default initialization parameters."""
        selector = FeatureSelector()
        
        assert selector.n_folds == 5
        assert selector.random_state == 42
        assert selector.n_jobs == -1

    def test_custom_initialization(self):
        """Test custom initialization parameters."""
        selector = FeatureSelector(
            n_folds=3,
            random_state=123,
            n_jobs=1,
        )
        
        assert selector.n_folds == 3
        assert selector.random_state == 123
        assert selector.n_jobs == 1


class TestNullImportance:
    """Test null importance feature selection."""

    def test_null_importance_basic(self):
        """Test basic null importance selection."""
        selector = FeatureSelector(n_folds=3)
        
        result = selector.select_features(
            X=FROZEN_X,
            y=FROZEN_Y,
            feature_names=FROZEN_FEATURE_NAMES,
            method="null_importance",
            threshold=0.1,
            n_shuffles=5,  # Fewer for speed
        )
        
        assert isinstance(result, FeatureSelectionResult)
        assert result.method == "null_importance"
        assert len(result.selected_features) > 0
        assert len(result.selected_features) <= FROZEN_N_FEATURES

    def test_null_importance_selects_informative(self):
        """Test null importance selects informative features."""
        selector = FeatureSelector(n_folds=3)
        
        result = selector.select_features(
            X=FROZEN_X,
            y=FROZEN_Y,
            feature_names=FROZEN_FEATURE_NAMES,
            method="null_importance",
            threshold=0.5,
            n_shuffles=5,
        )
        
        # Should select at least some of the first 5 informative features
        selected_indices = [
            int(name.split("_")[1])
            for name in result.selected_features
        ]
        
        n_informative_selected = sum(1 for i in selected_indices if i < 5)
        assert n_informative_selected >= 1


class TestPermutationImportance:
    """Test permutation importance feature selection."""

    def test_permutation_basic(self):
        """Test basic permutation importance selection."""
        selector = FeatureSelector(n_folds=3)
        
        result = selector.select_features(
            X=FROZEN_X,
            y=FROZEN_Y,
            feature_names=FROZEN_FEATURE_NAMES,
            method="permutation",
            threshold=0.01,
        )
        
        assert isinstance(result, FeatureSelectionResult)
        assert result.method == "permutation_importance"
        assert len(result.selected_features) > 0


class TestRecursiveElimination:
    """Test recursive feature elimination."""

    def test_rfe_basic(self):
        """Test basic RFE selection."""
        selector = FeatureSelector(n_folds=3)
        
        result = selector.select_features(
            X=FROZEN_X,
            y=FROZEN_Y,
            feature_names=FROZEN_FEATURE_NAMES,
            method="rfe",
            n_features_to_select=10,
        )
        
        assert isinstance(result, FeatureSelectionResult)
        assert result.method == "recursive_elimination"
        assert len(result.selected_features) == 10

    def test_rfe_custom_step(self):
        """Test RFE with custom step size."""
        selector = FeatureSelector(n_folds=3)
        
        result = selector.select_features(
            X=FROZEN_X,
            y=FROZEN_Y,
            feature_names=FROZEN_FEATURE_NAMES,
            method="rfe",
            n_features_to_select=5,
            step=5,
        )
        
        assert len(result.selected_features) == 5


class TestStabilitySelection:
    """Test stability selection."""

    def test_stability_basic(self):
        """Test basic stability selection."""
        selector = FeatureSelector(n_folds=3)
        
        result = selector.select_features(
            X=FROZEN_X,
            y=FROZEN_Y,
            feature_names=FROZEN_FEATURE_NAMES,
            method="stability",
            threshold=0.6,
            n_iterations=20,  # Fewer for speed
        )
        
        assert isinstance(result, FeatureSelectionResult)
        assert result.method == "stability_selection"
        assert len(result.selected_features) > 0


class TestConsensusSelection:
    """Test consensus selection."""

    def test_consensus_basic(self):
        """Test basic consensus selection."""
        selector = FeatureSelector(n_folds=3)
        
        result = selector.select_features(
            X=FROZEN_X,
            y=FROZEN_Y,
            feature_names=FROZEN_FEATURE_NAMES,
            method="consensus",
            min_methods=2,
        )
        
        assert isinstance(result, FeatureSelectionResult)
        assert result.method == "consensus"
        # Consensus should be more conservative
        assert len(result.selected_features) <= FROZEN_N_FEATURES


class TestValidation:
    """Test input validation."""

    def test_validate_unknown_method(self):
        """Test validation rejects unknown method."""
        selector = FeatureSelector()
        
        with pytest.raises(ValueError, match="Unknown method"):
            selector.select_features(
                X=FROZEN_X,
                y=FROZEN_Y,
                feature_names=FROZEN_FEATURE_NAMES,
                method="unknown_method",
            )


class TestResultStructure:
    """Test result structure."""

    def test_result_has_required_fields(self):
        """Test result has all required fields."""
        selector = FeatureSelector(n_folds=3)
        
        result = selector.select_features(
            X=FROZEN_X,
            y=FROZEN_Y,
            feature_names=FROZEN_FEATURE_NAMES,
            method="null_importance",
            n_shuffles=3,
        )
        
        # Check all required fields
        assert hasattr(result, "method")
        assert hasattr(result, "n_features_before")
        assert hasattr(result, "n_features_after")
        assert hasattr(result, "n_features_removed")
        assert hasattr(result, "selected_features")
        assert hasattr(result, "removed_features")
        assert hasattr(result, "feature_importances")
        assert hasattr(result, "cv_score_before")
        assert hasattr(result, "cv_score_after")
        assert hasattr(result, "improvement")

    def test_result_to_dict(self):
        """Test result serialization."""
        result = FeatureSelectionResult(
            method="test",
            n_features_before=20,
            n_features_after=10,
            n_features_removed=10,
            selected_features=["f1", "f2"],
            removed_features=["f3", "f4"],
            feature_importances={"f1": 0.5, "f2": 0.3},
            cv_score_before=0.75,
            cv_score_after=0.80,
            improvement=0.05,
        )
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict["method"] == "test"
        assert result_dict["n_features_before"] == 20
        assert result_dict["n_features_after"] == 10
        assert len(result_dict["selected_features"]) == 2


class TestHistoryTracking:
    """Test history tracking."""

    def test_history_tracking(self):
        """Test selection history is tracked."""
        selector = FeatureSelector(n_folds=3)
        
        # Run multiple selections
        selector.select_features(
            X=FROZEN_X,
            y=FROZEN_Y,
            feature_names=FROZEN_FEATURE_NAMES,
            method="null_importance",
            n_shuffles=3,
        )
        
        selector.select_features(
            X=FROZEN_X,
            y=FROZEN_Y,
            feature_names=FROZEN_FEATURE_NAMES,
            method="permutation",
        )
        
        history = selector.get_selection_history()
        
        assert len(history) == 2
        assert all(isinstance(h, dict) for h in history)


class TestBaselineTracking:
    """Test baseline tracking."""

    def test_set_baseline(self):
        """Test setting baseline score."""
        selector = FeatureSelector()
        
        selector.set_baseline("all_features", 0.85)
        
        assert "all_features" in selector.baseline_scores
        assert selector.baseline_scores["all_features"] == 0.85


class TestConvenienceFunction:
    """Test convenience function."""

    def test_select_features_function(self):
        """Test select_features convenience function."""
        result = select_features(
            X=FROZEN_X,
            y=FROZEN_Y,
            feature_names=FROZEN_FEATURE_NAMES,
            method="null_importance",
            n_shuffles=3,
        )
        
        assert isinstance(result, FeatureSelectionResult)
        assert result.method == "null_importance"


class TestRegressionBaselines:
    """Test regression-aware baselines."""

    def test_frozen_data_shape(self):
        """Test frozen data has expected shape."""
        assert FROZEN_X.shape == (FROZEN_N_SAMPLES, FROZEN_N_FEATURES)
        assert len(FROZEN_Y) == FROZEN_N_SAMPLES

    def test_frozen_feature_names(self):
        """Test frozen feature names."""
        assert len(FROZEN_FEATURE_NAMES) == FROZEN_N_FEATURES
        assert FROZEN_FEATURE_NAMES[0] == "feature_0"
        assert FROZEN_FEATURE_NAMES[-1] == f"feature_{FROZEN_N_FEATURES - 1}"

    def test_selection_improves_or_maintains_score(self):
        """Test selection doesn't significantly degrade score."""
        selector = FeatureSelector(n_folds=3)
        
        result = selector.select_features(
            X=FROZEN_X,
            y=FROZEN_Y,
            feature_names=FROZEN_FEATURE_NAMES,
            method="null_importance",
            n_shuffles=3,
        )
        
        # Should not degrade by more than 0.05
        assert result.improvement > -0.05
