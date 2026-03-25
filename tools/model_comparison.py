# tools/model_comparison.py

"""
Statistically rigorous model comparison framework.

FLAW-11.1 FIX: Model Comparison Framework
- Multiple statistical tests (Wilcoxon, t-test, McNemar)
- Effect size calculations
- Confidence intervals
- Multiple comparison correction
- Regression-aware baseline tracking
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Literal
from dataclasses import dataclass
from enum import Enum
from scipy import stats
from scipy.stats import ttest_rel, wilcoxon

# Optional import for multiple comparison correction
try:
    from statsmodels.stats.multitest import multipletests
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    logger = logging.getLogger(__name__)
    logger.warning("statsmodels not installed. Multiple comparison correction disabled.")

logger = logging.getLogger(__name__)


class StatisticalTest(Enum):
    """Available statistical tests."""
    WILCOXON = "wilcoxon"  # Non-parametric, paired
    T_TEST = "t_test"      # Parametric, paired
    MCNEMAR = "mcnemar"    # For classification
    ANOVA = "anova"        # Multiple models
    FRIEDMAN = "friedman"  # Multiple models, non-parametric


class Alternative(Enum):
    """Alternative hypothesis types."""
    TWO_SIDED = "two-sided"
    GREATER = "greater"  # Model A > Model B
    LESS = "less"        # Model A < Model B


@dataclass
class ComparisonResult:
    """Result of a model comparison."""
    
    model_a_name: str
    model_b_name: str
    test_used: str
    statistic: float
    p_value: float
    significant: bool
    alpha: float
    effect_size: Optional[float]
    confidence_interval: Optional[Tuple[float, float]]
    model_a_mean: float
    model_b_mean: float
    model_a_std: float
    model_b_std: float
    n_samples: int
    conclusion: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict."""
        return {
            "model_a_name": self.model_a_name,
            "model_b_name": self.model_b_name,
            "test_used": self.test_used,
            "statistic": round(self.statistic, 6),
            "p_value": round(self.p_value, 6),
            "significant": self.significant,
            "alpha": self.alpha,
            "effect_size": round(self.effect_size, 6) if self.effect_size else None,
            "confidence_interval": (
                (round(self.confidence_interval[0], 6), round(self.confidence_interval[1], 6))
                if self.confidence_interval else None
            ),
            "model_a_mean": round(self.model_a_mean, 6),
            "model_b_mean": round(self.model_b_mean, 6),
            "model_a_std": round(self.model_a_std, 6),
            "model_b_std": round(self.model_b_std, 6),
            "n_samples": self.n_samples,
            "conclusion": self.conclusion,
        }


@dataclass
class MultiModelComparison:
    """Result of comparing multiple models."""
    
    models: List[str]
    scores: Dict[str, List[float]]
    test_used: str
    statistic: float
    p_value: float
    significant: bool
    post_hoc_results: Optional[Dict[str, ComparisonResult]]
    rankings: List[Tuple[str, float]]
    conclusion: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict."""
        return {
            "models": self.models,
            "scores": {k: [round(s, 6) for s in v] for k, v in self.scores.items()},
            "test_used": self.test_used,
            "statistic": round(self.statistic, 6),
            "p_value": round(self.p_value, 6),
            "significant": self.significant,
            "post_hoc_results": (
                {k: v.to_dict() for k, v in self.post_hoc_results.items()}
                if self.post_hoc_results else None
            ),
            "rankings": [(m, round(s, 6)) for m, s in self.rankings],
            "conclusion": self.conclusion,
        }


class ModelComparator:
    """
    Statistically rigorous model comparison.
    
    Features:
    - Multiple statistical tests
    - Effect size calculations
    - Confidence intervals
    - Multiple comparison correction
    - Regression-aware baseline tracking
    
    Usage:
        comparator = ModelComparator(alpha=0.05)
        
        # Compare two models
        result = comparator.compare_models(
            model_a_scores=[0.85, 0.87, 0.86],
            model_b_scores=[0.82, 0.84, 0.83],
            model_a_name="LightGBM",
            model_b_name="XGBoost",
            test="wilcoxon",
        )
        
        if result.significant:
            print(f"{result.model_a_name} is significantly better!")
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize model comparator.
        
        Args:
            alpha: Significance level (default: 0.05)
        """
        self.alpha = alpha
        self.baseline_scores: Dict[str, List[float]] = {}
        self.comparison_history: List[ComparisonResult] = []
        
        logger.info(f"[ModelComparator] Initialized with alpha={alpha}")
    
    def compare_models(
        self,
        model_a_scores: List[float],
        model_b_scores: List[float],
        model_a_name: str = "Model A",
        model_b_name: str = "Model B",
        test: StatisticalTest = StatisticalTest.WILCOXON,
        alternative: Alternative = Alternative.TWO_SIDED,
        compute_effect_size: bool = True,
        compute_ci: bool = True,
        ci_level: float = 0.95,
    ) -> ComparisonResult:
        """
        Compare two models using statistical tests.
        
        Args:
            model_a_scores: Scores for model A (e.g., CV scores)
            model_b_scores: Scores for model B
            model_a_name: Name for model A
            model_b_name: Name for model B
            test: Statistical test to use
            alternative: Alternative hypothesis
            compute_effect_size: Whether to compute effect size
            compute_ci: Whether to compute confidence interval
            ci_level: Confidence level (0.95 = 95%)
        
        Returns:
            ComparisonResult with full statistical analysis
        
        Raises:
            ValueError: If scores have different lengths or invalid test
        """
        # Validate inputs
        self._validate_scores(model_a_scores, model_b_scores)
        
        n = len(model_a_scores)
        scores_a = np.array(model_a_scores)
        scores_b = np.array(model_b_scores)
        
        # Compute basic statistics
        mean_a = np.mean(scores_a)
        mean_b = np.mean(scores_b)
        std_a = np.std(scores_a, ddof=1)
        std_b = np.std(scores_b, ddof=1)
        
        # Run statistical test
        if test == StatisticalTest.WILCOXON:
            statistic, p_value = self._wilcoxon_test(scores_a, scores_b, alternative)
        elif test == StatisticalTest.T_TEST:
            statistic, p_value = self._t_test(scores_a, scores_b, alternative)
        elif test == StatisticalTest.MCNEMAR:
            raise NotImplementedError("McNemar test requires contingency tables")
        else:
            raise ValueError(f"Unsupported test: {test}")
        
        # Determine significance
        significant = p_value < self.alpha
        
        # Compute effect size
        effect_size = None
        if compute_effect_size:
            effect_size = self._compute_effect_size(scores_a, scores_b, test)
        
        # Compute confidence interval
        confidence_interval = None
        if compute_ci:
            confidence_interval = self._compute_ci(
                scores_a, scores_b, ci_level, test
            )
        
        # Generate conclusion
        conclusion = self._generate_conclusion(
            model_a_name, model_b_name, significant, p_value,
            mean_a, mean_b, effect_size
        )
        
        # Create result
        result = ComparisonResult(
            model_a_name=model_a_name,
            model_b_name=model_b_name,
            test_used=test.value,
            statistic=float(statistic),
            p_value=float(p_value),
            significant=significant,
            alpha=self.alpha,
            effect_size=effect_size,
            confidence_interval=confidence_interval,
            model_a_mean=float(mean_a),
            model_b_mean=float(mean_b),
            model_a_std=float(std_a),
            model_b_std=float(std_b),
            n_samples=n,
            conclusion=conclusion,
        )
        
        # Store in history
        self.comparison_history.append(result)
        
        logger.info(
            f"[ModelComparator] {model_a_name} vs {model_b_name}: "
            f"p={p_value:.4f}, significant={significant}"
        )
        
        return result
    
    def compare_multiple_models(
        self,
        scores: Dict[str, List[float]],
        test: StatisticalTest = StatisticalTest.FRIEDMAN,
        post_hoc: bool = True,
        correction_method: str = "holm",
    ) -> MultiModelComparison:
        """
        Compare multiple models simultaneously.
        
        Args:
            scores: Dict of {model_name: [scores]}
            test: Statistical test (Friedman or ANOVA)
            post_hoc: Whether to run post-hoc pairwise tests
            correction_method: Multiple comparison correction method
        
        Returns:
            MultiModelComparison with rankings and pairwise comparisons
        """
        # Validate inputs
        model_names = list(scores.keys())
        score_lists = list(scores.values())
        
        if len(score_lists) < 2:
            raise ValueError("Need at least 2 models to compare")
        
        n_folds = len(score_lists[0])
        for name, score_list in scores.items():
            if len(score_list) != n_folds:
                raise ValueError(f"All models must have same number of scores")
        
        # Run omnibus test
        if test == StatisticalTest.FRIEDMAN:
            statistic, p_value = stats.friedmanchisquare(*score_lists)
        elif test == StatisticalTest.ANOVA:
            statistic, p_value = stats.fstats_oneway(*score_lists)
        else:
            raise ValueError(f"Unsupported test for multiple models: {test}")
        
        significant = p_value < self.alpha
        
        # Compute rankings
        rankings = self._compute_rankings(scores)
        
        # Post-hoc pairwise comparisons
        post_hoc_results = None
        if post_hoc and significant:
            post_hoc_results = self._post_hoc_pairwise(
                scores, correction_method
            )
        
        # Generate conclusion
        conclusion = self._generate_multi_conclusion(
            rankings, significant, p_value, test.value
        )
        
        return MultiModelComparison(
            models=model_names,
            scores=scores,
            test_used=test.value,
            statistic=float(statistic),
            p_value=float(p_value),
            significant=significant,
            post_hoc_results=post_hoc_results,
            rankings=rankings,
            conclusion=conclusion,
        )
    
    def set_baseline(
        self,
        baseline_name: str,
        baseline_scores: List[float],
    ) -> None:
        """
        Set baseline model for future comparisons.
        
        Args:
            baseline_name: Name of baseline model
            baseline_scores: Baseline model scores
        """
        self.baseline_scores[baseline_name] = baseline_scores
        logger.info(f"[ModelComparator] Set baseline: {baseline_name}")
    
    def compare_to_baseline(
        self,
        model_name: str,
        model_scores: List[float],
        baseline_name: Optional[str] = None,
        **kwargs,
    ) -> ComparisonResult:
        """
        Compare a model to the baseline.
        
        Args:
            model_name: Name of model to compare
            model_scores: Model scores
            baseline_name: Baseline to use (default: first baseline)
            **kwargs: Passed to compare_models()
        
        Returns:
            ComparisonResult
        """
        if baseline_name is None:
            if not self.baseline_scores:
                raise ValueError("No baseline set")
            baseline_name = list(self.baseline_scores.keys())[0]
        
        baseline_scores = self.baseline_scores[baseline_name]
        
        return self.compare_models(
            model_a_scores=model_scores,
            model_b_scores=baseline_scores,
            model_a_name=model_name,
            model_b_name=baseline_name,
            **kwargs,
        )
    
    def get_comparison_history(self) -> List[Dict[str, Any]]:
        """Get all comparison results as dicts."""
        return [r.to_dict() for r in self.comparison_history]
    
    def clear_history(self) -> None:
        """Clear comparison history."""
        self.comparison_history.clear()
    
    # ── Private Methods ─────────────────────────────────────────────
    
    def _validate_scores(
        self,
        scores_a: List[float],
        scores_b: List[float],
    ) -> None:
        """Validate score lists for comparison."""
        if len(scores_a) != len(scores_b):
            raise ValueError(
                f"Score lists must have same length: "
                f"{len(scores_a)} vs {len(scores_b)}"
            )
        
        if len(scores_a) < 2:
            raise ValueError("Need at least 2 scores for statistical test")
        
        if any(np.isnan(scores_a)) or any(np.isnan(scores_b)):
            raise ValueError("Scores cannot contain NaN")
    
    def _wilcoxon_test(
        self,
        scores_a: np.ndarray,
        scores_b: np.ndarray,
        alternative: Alternative,
    ) -> Tuple[float, float]:
        """Run Wilcoxon signed-rank test."""
        statistic, p_value = wilcoxon(
            scores_a, scores_b,
            alternative=alternative.value,
        )
        return statistic, p_value
    
    def _t_test(
        self,
        scores_a: np.ndarray,
        scores_b: np.ndarray,
        alternative: Alternative,
    ) -> Tuple[float, float]:
        """Run paired t-test."""
        statistic, p_value = ttest_rel(
            scores_a, scores_b,
            alternative=alternative.value,
        )
        return statistic, p_value
    
    def _compute_effect_size(
        self,
        scores_a: np.ndarray,
        scores_b: np.ndarray,
        test: StatisticalTest,
    ) -> float:
        """
        Compute effect size.
        
        For Wilcoxon: Rank-biserial correlation
        For t-test: Cohen's d
        """
        if test == StatisticalTest.WILCOXON:
            # Rank-biserial correlation
            diff = scores_a - scores_b
            n_positive = np.sum(diff > 0)
            n_negative = np.sum(diff < 0)
            n = n_positive + n_negative
            if n == 0:
                return 0.0
            return (n_positive - n_negative) / n
        
        elif test == StatisticalTest.T_TEST:
            # Cohen's d
            pooled_std = np.sqrt(
                (np.var(scores_a) + np.var(scores_b)) / 2
            )
            if pooled_std == 0:
                return 0.0
            return (np.mean(scores_a) - np.mean(scores_b)) / pooled_std
        
        return 0.0
    
    def _compute_ci(
        self,
        scores_a: np.ndarray,
        scores_b: np.ndarray,
        ci_level: float,
        test: StatisticalTest,
    ) -> Tuple[float, float]:
        """Compute confidence interval for difference in means."""
        diff = scores_a - scores_b
        mean_diff = np.mean(diff)
        se_diff = np.std(diff, ddof=1) / np.sqrt(len(diff))
        
        alpha = 1 - ci_level
        t_crit = stats.t.ppf(1 - alpha/2, len(diff) - 1)
        
        ci_lower = mean_diff - t_crit * se_diff
        ci_upper = mean_diff + t_crit * se_diff
        
        return (ci_lower, ci_upper)
    
    def _compute_rankings(
        self,
        scores: Dict[str, List[float]],
    ) -> List[Tuple[str, float]]:
        """Compute model rankings by mean score."""
        means = {name: np.mean(score_list) for name, score_list in scores.items()}
        rankings = sorted(means.items(), key=lambda x: x[1], reverse=True)
        return rankings
    
    def _post_hoc_pairwise(
        self,
        scores: Dict[str, List[float]],
        correction_method: str,
    ) -> Dict[str, ComparisonResult]:
        """Run post-hoc pairwise comparisons with correction."""
        if not HAS_STATSMODELS:
            logger.warning(
                "statsmodels not installed. Running uncorrected pairwise tests."
            )
            correction_method = None
        
        model_names = list(scores.keys())
        results = {}
        p_values = []
        comparisons = []
        
        # Run all pairwise tests
        for i, name_a in enumerate(model_names):
            for name_b in model_names[i+1:]:
                result = self.compare_models(
                    model_a_scores=scores[name_a],
                    model_b_scores=scores[name_b],
                    model_a_name=name_a,
                    model_b_name=name_b,
                    test=StatisticalTest.WILCOXON,
                    compute_effect_size=False,
                    compute_ci=False,
                )
                p_values.append(result.p_value)
                comparisons.append((name_a, name_b, result))
        
        # Apply multiple comparison correction
        if p_values and correction_method:
            reject, pvals_corrected, _, _ = multipletests(
                p_values,
                alpha=self.alpha,
                method=correction_method,
            )
            
            # Update results with corrected p-values
            for idx, (name_a, name_b, result) in enumerate(comparisons):
                result.p_value = float(pvals_corrected[idx])
                result.significant = bool(reject[idx])
                key = f"{name_a}_vs_{name_b}"
                results[key] = result
        elif p_values:
            # No correction, just store results
            for name_a, name_b, result in comparisons:
                key = f"{name_a}_vs_{name_b}"
                results[key] = result
        
        return results
    
    def _generate_conclusion(
        self,
        model_a_name: str,
        model_b_name: str,
        significant: bool,
        p_value: float,
        mean_a: float,
        mean_b: float,
        effect_size: Optional[float],
    ) -> str:
        """Generate human-readable conclusion."""
        if significant:
            winner = model_a_name if mean_a > mean_b else model_b_name
            effect_str = ""
            if effect_size is not None:
                if abs(effect_size) < 0.2:
                    effect_str = " (small effect)"
                elif abs(effect_size) < 0.5:
                    effect_str = " (medium effect)"
                else:
                    effect_str = " (large effect)"
            
            return (
                f"{winner} is significantly better (p={p_value:.4f})"
                f"{effect_str}"
            )
        else:
            return (
                f"No significant difference between {model_a_name} "
                f"and {model_b_name} (p={p_value:.4f})"
            )
    
    def _generate_multi_conclusion(
        self,
        rankings: List[Tuple[str, float]],
        significant: bool,
        p_value: float,
        test_name: str,
    ) -> str:
        """Generate conclusion for multi-model comparison."""
        if significant:
            best_model = rankings[0][0]
            return (
                f"Significant difference found ({test_name}, p={p_value:.4f}). "
                f"Best model: {best_model}"
            )
        else:
            return (
                f"No significant difference found between models "
                f"({test_name}, p={p_value:.4f})"
            )
