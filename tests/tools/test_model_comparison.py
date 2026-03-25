"""
Comprehensive tests for model comparison framework.

FLAW-11.1: Model Comparison Framework
Tests are regression-aware with frozen baselines.
"""
import pytest
import numpy as np
from tools.model_comparison import (
    ModelComparator,
    StatisticalTest,
    Alternative,
    ComparisonResult,
    MultiModelComparison,
)


# ── Frozen Baselines for Regression Testing ──────────────────────

FROZEN_BASELINE_SCORES = [0.85, 0.87, 0.86, 0.88, 0.85]
FROZEN_CHALLENGER_SCORES = [0.88, 0.89, 0.87, 0.90, 0.88]
FROZEN_BASELINE_MEAN = 0.862
FROZEN_CHALLENGER_MEAN = 0.884
FROZEN_P_VALUE_WILCOXON = 0.0625  # From previous run


class TestModelComparatorInitialization:
    """Test ModelComparator initialization."""

    def test_default_alpha(self):
        """Test default alpha is 0.05."""
        comparator = ModelComparator()
        
        assert comparator.alpha == 0.05

    def test_custom_alpha(self):
        """Test custom alpha setting."""
        comparator = ModelComparator(alpha=0.01)
        
        assert comparator.alpha == 0.01

    def test_empty_history(self):
        """Test comparison history starts empty."""
        comparator = ModelComparator()
        
        assert len(comparator.comparison_history) == 0
        assert comparator.get_comparison_history() == []


class TestCompareModels:
    """Test pairwise model comparison."""

    def test_compare_different_models(self):
        """Test comparing two different models."""
        comparator = ModelComparator()
        
        result = comparator.compare_models(
            model_a_scores=FROZEN_BASELINE_SCORES,
            model_b_scores=FROZEN_CHALLENGER_SCORES,
            model_a_name="Baseline",
            model_b_name="Challenger",
            test=StatisticalTest.WILCOXON,
        )
        
        assert isinstance(result, ComparisonResult)
        assert result.model_a_name == "Baseline"
        assert result.model_b_name == "Challenger"
        assert result.test_used == "wilcoxon"
        assert result.n_samples == 5

    def test_compare_identical_models(self):
        """Test comparing identical models (should not be significant)."""
        comparator = ModelComparator()
        
        result = comparator.compare_models(
            model_a_scores=[0.85, 0.87, 0.86],
            model_b_scores=[0.85, 0.87, 0.86],
            model_a_name="Model A",
            model_b_name="Model B",
        )
        
        # Identical scores should not be significantly different
        assert result.p_value == 1.0
        assert bool(result.significant) is False
        assert result.effect_size == 0.0

    def test_compare_significantly_different(self):
        """Test detecting significant difference."""
        comparator = ModelComparator()
        
        # Need more samples for statistical significance
        # With only 5 samples, even large differences may not be significant
        result = comparator.compare_models(
            model_a_scores=[0.90, 0.91, 0.89, 0.92, 0.90, 0.91, 0.89, 0.92, 0.90, 0.91],
            model_b_scores=[0.70, 0.71, 0.69, 0.72, 0.70, 0.71, 0.69, 0.72, 0.70, 0.71],
            model_a_name="Good",
            model_b_name="Bad",
        )
        
        assert bool(result.significant) is True
        assert result.p_value < 0.05
        assert result.model_a_mean > result.model_b_mean

    def test_compare_with_t_test(self):
        """Test comparison with paired t-test."""
        comparator = ModelComparator()
        
        result = comparator.compare_models(
            model_a_scores=FROZEN_BASELINE_SCORES,
            model_b_scores=FROZEN_CHALLENGER_SCORES,
            test=StatisticalTest.T_TEST,
        )
        
        assert result.test_used == "t_test"
        assert isinstance(result.statistic, float)

    def test_compare_one_sided_greater(self):
        """Test one-sided test (A > B)."""
        comparator = ModelComparator()
        
        result = comparator.compare_models(
            model_a_scores=FROZEN_CHALLENGER_SCORES,
            model_b_scores=FROZEN_BASELINE_SCORES,
            alternative=Alternative.GREATER,
        )
        
        # One-sided p-value should be half of two-sided
        assert result.p_value < 1.0

    def test_compare_with_effect_size(self):
        """Test effect size computation."""
        comparator = ModelComparator()
        
        result = comparator.compare_models(
            model_a_scores=[0.90, 0.91, 0.89],
            model_b_scores=[0.70, 0.71, 0.69],
            compute_effect_size=True,
        )
        
        assert result.effect_size is not None
        assert abs(result.effect_size) > 0.5  # Large effect

    def test_compare_with_confidence_interval(self):
        """Test confidence interval computation."""
        comparator = ModelComparator()
        
        # Use varied data for meaningful CI
        result = comparator.compare_models(
            model_a_scores=[0.90, 0.88, 0.92, 0.89, 0.91, 0.87, 0.93, 0.90],
            model_b_scores=[0.75, 0.78, 0.72, 0.76, 0.74, 0.77, 0.73, 0.75],
            compute_ci=True,
            ci_level=0.95,
        )
        
        assert result.confidence_interval is not None
        ci_lower, ci_upper = result.confidence_interval
        # Convert numpy types to Python types for comparison
        assert float(ci_lower) < float(ci_upper)

    def test_compare_validation_different_lengths(self):
        """Test validation rejects different length scores."""
        comparator = ModelComparator()
        
        with pytest.raises(ValueError, match="same length"):
            comparator.compare_models(
                model_a_scores=[0.85, 0.87, 0.86],
                model_b_scores=[0.88, 0.89],
            )

    def test_compare_validation_nan_scores(self):
        """Test validation rejects NaN scores."""
        comparator = ModelComparator()
        
        with pytest.raises(ValueError, match="NaN"):
            comparator.compare_models(
                model_a_scores=[0.85, np.nan, 0.86],
                model_b_scores=[0.88, 0.89, 0.87],
            )

    def test_compare_validation_too_few_samples(self):
        """Test validation requires at least 2 samples."""
        comparator = ModelComparator()
        
        with pytest.raises(ValueError, match="at least 2"):
            comparator.compare_models(
                model_a_scores=[0.85],
                model_b_scores=[0.88],
            )


class TestRegressionBaselines:
    """Test regression-aware baselines are maintained."""

    def test_frozen_baseline_mean(self):
        """Test frozen baseline mean is preserved."""
        comparator = ModelComparator()
        
        result = comparator.compare_models(
            model_a_scores=FROZEN_BASELINE_SCORES,
            model_b_scores=FROZEN_BASELINE_SCORES,  # Same scores
        )
        
        # Mean should match frozen baseline
        assert abs(result.model_a_mean - FROZEN_BASELINE_MEAN) < 1e-6

    def test_frozen_p_value_wilcoxon(self):
        """Test p-value matches frozen baseline for Wilcoxon."""
        comparator = ModelComparator()
        
        result = comparator.compare_models(
            model_a_scores=FROZEN_BASELINE_SCORES,
            model_b_scores=FROZEN_CHALLENGER_SCORES,
            test=StatisticalTest.WILCOXON,
        )
        
        # P-value should match frozen baseline (within floating point tolerance)
        assert abs(result.p_value - FROZEN_P_VALUE_WILCOXON) < 1e-6

    def test_comparison_history_tracking(self):
        """Test comparison history is tracked for regression."""
        comparator = ModelComparator()
        
        # Run multiple comparisons
        comparator.compare_models(
            model_a_scores=[0.85, 0.87, 0.86],
            model_b_scores=[0.88, 0.89, 0.87],
            model_a_name="Model A",
            model_b_name="Model B",
        )
        
        comparator.compare_models(
            model_a_scores=[0.90, 0.91, 0.89],
            model_b_scores=[0.88, 0.89, 0.87],
            model_a_name="Model C",
            model_b_name="Model D",
        )
        
        history = comparator.get_comparison_history()
        
        assert len(history) == 2
        assert history[0]["model_a_name"] == "Model A"
        assert history[1]["model_a_name"] == "Model C"


class TestMultiModelComparison:
    """Test multiple model comparison."""

    def test_compare_three_models(self):
        """Test comparing three models."""
        comparator = ModelComparator()
        
        scores = {
            "Model A": [0.85, 0.87, 0.86, 0.88],
            "Model B": [0.88, 0.89, 0.87, 0.90],
            "Model C": [0.82, 0.83, 0.81, 0.84],
        }
        
        result = comparator.compare_multiple_models(
            scores=scores,
            test=StatisticalTest.FRIEDMAN,
        )
        
        assert isinstance(result, MultiModelComparison)
        assert len(result.models) == 3
        assert len(result.rankings) == 3

    def test_multi_model_rankings(self):
        """Test model rankings are correct."""
        comparator = ModelComparator()
        
        scores = {
            "Best": [0.95, 0.96, 0.94],
            "Medium": [0.85, 0.86, 0.84],
            "Worst": [0.75, 0.76, 0.74],
        }
        
        result = comparator.compare_multiple_models(scores=scores)
        
        # Rankings should be in descending order
        assert result.rankings[0][0] == "Best"
        assert result.rankings[1][0] == "Medium"
        assert result.rankings[2][0] == "Worst"

    def test_multi_model_post_hoc(self):
        """Test post-hoc pairwise comparisons."""
        comparator = ModelComparator()
        
        scores = {
            "Model A": [0.90, 0.91, 0.89, 0.92],
            "Model B": [0.85, 0.86, 0.84, 0.87],
            "Model C": [0.80, 0.81, 0.79, 0.82],
        }
        
        result = comparator.compare_multiple_models(
            scores=scores,
            post_hoc=True,
        )
        
        # Should have pairwise comparisons
        if result.significant:
            assert result.post_hoc_results is not None
            assert len(result.post_hoc_results) > 0

    def test_multi_model_correction_method(self):
        """Test multiple comparison correction."""
        comparator = ModelComparator()
        
        scores = {
            "Model A": [0.90, 0.91, 0.89],
            "Model B": [0.85, 0.86, 0.84],
            "Model C": [0.80, 0.81, 0.79],
            "Model D": [0.75, 0.76, 0.74],
        }
        
        # Test different correction methods
        for method in ["holm", "bonferroni", "fdr_bh"]:
            result = comparator.compare_multiple_models(
                scores=scores,
                post_hoc=True,
                correction_method=method,
            )
            
            assert isinstance(result, MultiModelComparison)


class TestBaselineTracking:
    """Test baseline model tracking."""

    def test_set_baseline(self):
        """Test setting baseline model."""
        comparator = ModelComparator()
        
        comparator.set_baseline(
            baseline_name="LightGBM_Baseline",
            baseline_scores=FROZEN_BASELINE_SCORES,
        )
        
        assert "LightGBM_Baseline" in comparator.baseline_scores
        assert comparator.baseline_scores["LightGBM_Baseline"] == FROZEN_BASELINE_SCORES

    def test_compare_to_baseline(self):
        """Test comparing model to baseline."""
        comparator = ModelComparator()
        
        comparator.set_baseline(
            baseline_name="Baseline",
            baseline_scores=FROZEN_BASELINE_SCORES,
        )
        
        result = comparator.compare_to_baseline(
            model_name="Challenger",
            model_scores=FROZEN_CHALLENGER_SCORES,
        )
        
        assert result.model_b_name == "Baseline"
        assert result.model_a_name == "Challenger"

    def test_compare_to_baseline_no_baseline_set(self):
        """Test error when no baseline set."""
        comparator = ModelComparator()
        
        with pytest.raises(ValueError, match="No baseline set"):
            comparator.compare_to_baseline(
                model_name="Model",
                model_scores=[0.85, 0.87, 0.86],
            )

    def test_multiple_baselines(self):
        """Test tracking multiple baselines."""
        comparator = ModelComparator()
        
        comparator.set_baseline("Baseline_A", [0.80, 0.82, 0.81])
        comparator.set_baseline("Baseline_B", [0.85, 0.87, 0.86])
        
        assert len(comparator.baseline_scores) == 2


class TestConclusionGeneration:
    """Test conclusion generation."""

    def test_conclusion_significant_difference(self):
        """Test conclusion for significant difference."""
        comparator = ModelComparator()
        
        # Use more samples for statistical significance
        result = comparator.compare_models(
            model_a_scores=[0.95, 0.96, 0.94, 0.95, 0.96, 0.94, 0.95, 0.96],
            model_b_scores=[0.70, 0.71, 0.69, 0.70, 0.71, 0.69, 0.70, 0.71],
        )
        
        assert "significantly better" in result.conclusion

    def test_conclusion_no_significant_difference(self):
        """Test conclusion for non-significant difference."""
        comparator = ModelComparator()
        
        result = comparator.compare_models(
            model_a_scores=[0.85, 0.86, 0.87],
            model_b_scores=[0.84, 0.85, 0.86],
        )
        
        assert "No significant difference" in result.conclusion

    def test_conclusion_effect_size_mentioned(self):
        """Test effect size mentioned in conclusion."""
        comparator = ModelComparator()
        
        # Use more samples for significant result
        result = comparator.compare_models(
            model_a_scores=[0.95, 0.96, 0.94, 0.95, 0.96, 0.94],
            model_b_scores=[0.70, 0.71, 0.69, 0.70, 0.71, 0.69],
            compute_effect_size=True,
        )
        
        # Should mention effect size
        assert "effect" in result.conclusion.lower()


class TestResultSerialization:
    """Test result serialization."""

    def test_comparison_result_to_dict(self):
        """Test ComparisonResult serialization."""
        result = ComparisonResult(
            model_a_name="Model A",
            model_b_name="Model B",
            test_used="wilcoxon",
            statistic=10.5,
            p_value=0.03,
            significant=True,
            alpha=0.05,
            effect_size=0.8,
            confidence_interval=(0.1, 0.5),
            model_a_mean=0.9,
            model_b_mean=0.8,
            model_a_std=0.05,
            model_b_std=0.06,
            n_samples=10,
            conclusion="Model A is better",
        )
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict["model_a_name"] == "Model A"
        assert result_dict["significant"] is True
        # Check rounding
        assert result_dict["statistic"] == 10.5
        assert result_dict["p_value"] == 0.03

    def test_multi_comparison_result_to_dict(self):
        """Test MultiModelComparison serialization."""
        multi_result = MultiModelComparison(
            models=["A", "B", "C"],
            scores={"A": [0.9, 0.91], "B": [0.85, 0.86], "C": [0.8, 0.81]},
            test_used="friedman",
            statistic=8.5,
            p_value=0.01,
            significant=True,
            post_hoc_results=None,
            rankings=[("A", 0.905), ("B", 0.855), ("C", 0.805)],
            conclusion="A is best",
        )
        
        result_dict = multi_result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert len(result_dict["models"]) == 3
        assert len(result_dict["rankings"]) == 3
