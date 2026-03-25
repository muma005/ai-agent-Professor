# agents/ensemble_optimizer.py

"""
Ensemble weight optimization using Nelder-Mead and other methods.

Advanced Feature: Ensemble Optimization
- Nelder-Mead simplex optimization
- Cross-validation based weight optimization
- Constraint handling (weights sum to 1)
- Multiple optimization methods
- Regression-aware baseline tracking
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from scipy.optimize import minimize, differential_evolution
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)


@dataclass
class EnsembleOptimizationResult:
    """Result of ensemble optimization."""
    
    model_names: List[str]
    optimal_weights: np.ndarray
    cv_score_before: float
    cv_score_after: float
    improvement: float
    method: str
    converged: bool
    n_iterations: int
    final_loss: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict."""
        return {
            "model_names": self.model_names,
            "optimal_weights": [round(w, 6) for w in self.optimal_weights],
            "cv_score_before": round(self.cv_score_before, 6),
            "cv_score_after": round(self.cv_score_after, 6),
            "improvement": round(self.improvement, 6),
            "method": self.method,
            "converged": self.converged,
            "n_iterations": self.n_iterations,
            "final_loss": round(self.final_loss, 6),
        }


class EnsembleOptimizer:
    """
    Optimize ensemble weights for maximum performance.
    
    Features:
    - Nelder-Mead simplex optimization
    - Differential evolution (global optimization)
    - Cross-validation based optimization
    - Constraint handling (weights sum to 1, non-negative)
    - Multiple scoring metrics
    - Regression-aware baseline tracking
    
    Usage:
        optimizer = EnsembleOptimizer()
        
        result = optimizer.optimize(
            oof_predictions=[model1_oof, model2_oof, model3_oof],
            y_true=y_train,
            metric="auc",
            n_folds=5,
        )
        
        print(f"Optimal weights: {result.optimal_weights}")
        print(f"Improvement: {result.improvement:.4f}")
    """
    
    def __init__(
        self,
        n_folds: int = 5,
        random_state: int = 42,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
    ):
        """
        Initialize ensemble optimizer.
        
        Args:
            n_folds: Number of CV folds for optimization
            random_state: Random seed for reproducibility
            max_iterations: Maximum optimization iterations
            tolerance: Convergence tolerance
        """
        self.n_folds = n_folds
        self.random_state = random_state
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        
        self.baseline_scores: Dict[str, float] = {}
        self.optimization_history: List[EnsembleOptimizationResult] = []
        
        logger.info(
            f"[EnsembleOptimizer] Initialized -- n_folds: {n_folds}, "
            f"max_iter: {max_iterations}"
        )
    
    def optimize(
        self,
        oof_predictions: List[np.ndarray],
        y_true: np.ndarray,
        model_names: Optional[List[str]] = None,
        metric: str = "auc",
        method: str = "nelder-mead",
        n_folds: Optional[int] = None,
    ) -> EnsembleOptimizationResult:
        """
        Optimize ensemble weights.
        
        Args:
            oof_predictions: List of OOF prediction arrays (n_models, n_samples)
            y_true: True target values
            model_names: Names of models (default: Model_0, Model_1, ...)
            metric: Scoring metric ("auc", "logloss", "rmse", "mae")
            method: Optimization method ("nelder-mead", "differential_evolution")
            n_folds: Override default n_folds
        
        Returns:
            EnsembleOptimizationResult with optimal weights
        
        Raises:
            ValueError: If inputs are invalid
        """
        # Validate inputs
        self._validate_inputs(oof_predictions, y_true)
        
        n_models = len(oof_predictions)
        n_samples = len(y_true)
        
        if model_names is None:
            model_names = [f"Model_{i}" for i in range(n_models)]
        
        if n_folds is None:
            n_folds = self.n_folds
        
        # Stack predictions: (n_samples, n_models)
        oof_stack = np.column_stack(oof_predictions)
        
        # Calculate baseline score (equal weights)
        equal_weights = np.ones(n_models) / n_models
        equal_preds = oof_stack @ equal_weights
        cv_score_before = self._cross_validate_score(
            equal_preds, y_true, metric, n_folds
        )
        
        logger.info(
            f"[EnsembleOptimizer] Baseline CV {metric}: {cv_score_before:.6f} "
            f"(equal weights)"
        )
        
        # Define loss function
        def loss_function(weights: np.ndarray) -> float:
            """Calculate negative CV score (to minimize)."""
            # Ensure weights sum to 1
            weights = weights / np.sum(weights)
            
            # Calculate ensemble predictions
            ensemble_preds = oof_stack @ weights
            
            # Calculate CV score
            cv_score = self._cross_validate_score(
                ensemble_preds, y_true, metric, n_folds
            )
            
            # Return negative (we want to maximize)
            return -cv_score
        
        # Run optimization
        if method == "nelder-mead":
            optimal_weights, converged, n_iter, final_loss = self._optimize_nelder_mead(
                loss_function, n_models
            )
        elif method == "differential_evolution":
            optimal_weights, converged, n_iter, final_loss = self._optimize_differential_evolution(
                loss_function, n_models
            )
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        # Normalize weights to sum to 1
        optimal_weights = optimal_weights / np.sum(optimal_weights)
        
        # Calculate final score
        final_preds = oof_stack @ optimal_weights
        cv_score_after = self._cross_validate_score(
            final_preds, y_true, metric, n_folds
        )
        
        improvement = cv_score_after - cv_score_before
        
        logger.info(
            f"[EnsembleOptimizer] Optimized CV {metric}: {cv_score_after:.6f} "
            f"(improvement: {improvement:+.6f})"
        )
        
        # Create result
        result = EnsembleOptimizationResult(
            model_names=model_names,
            optimal_weights=optimal_weights,
            cv_score_before=cv_score_before,
            cv_score_after=cv_score_after,
            improvement=improvement,
            method=method,
            converged=converged,
            n_iterations=n_iter,
            final_loss=-final_loss,
        )
        
        # Store in history
        self.optimization_history.append(result)
        
        return result
    
    def optimize_with_constraints(
        self,
        oof_predictions: List[np.ndarray],
        y_true: np.ndarray,
        model_names: Optional[List[str]] = None,
        metric: str = "auc",
        min_weights: Optional[np.ndarray] = None,
        max_weights: Optional[np.ndarray] = None,
        must_include_models: Optional[List[int]] = None,
    ) -> EnsembleOptimizationResult:
        """
        Optimize ensemble weights with constraints.
        
        Args:
            oof_predictions: List of OOF prediction arrays
            y_true: True target values
            model_names: Names of models
            metric: Scoring metric
            min_weights: Minimum weight for each model
            max_weights: Maximum weight for each model
            must_include_models: Indices of models that must have non-zero weight
        
        Returns:
            EnsembleOptimizationResult with constrained optimal weights
        """
        n_models = len(oof_predictions)
        
        # Set default constraints
        if min_weights is None:
            min_weights = np.zeros(n_models)
        if max_weights is None:
            max_weights = np.ones(n_models)
        if must_include_models is not None:
            for idx in must_include_models:
                min_weights[idx] = max(min_weights[idx], 0.01)  # At least 1%
        
        # Stack predictions
        oof_stack = np.column_stack(oof_predictions)
        
        # Define constrained loss function
        def constrained_loss(weights: np.ndarray) -> float:
            """Calculate loss with penalty for constraint violations."""
            # Normalize
            weights = weights / np.sum(weights)
            
            # Calculate penalty for constraint violations
            penalty = 0.0
            
            # Min weight penalty
            violation_min = np.sum(np.maximum(0, min_weights - weights))
            penalty += violation_min * 10.0
            
            # Max weight penalty
            violation_max = np.sum(np.maximum(0, weights - max_weights))
            penalty += violation_max * 10.0
            
            # Calculate ensemble predictions
            ensemble_preds = oof_stack @ weights
            
            # Calculate CV score
            cv_score = self._cross_validate_score(
                ensemble_preds, y_true, metric, self.n_folds
            )
            
            # Return negative score + penalty
            return -cv_score + penalty
        
        # Run optimization
        optimal_weights, converged, n_iter, final_loss = self._optimize_nelder_mead(
            constrained_loss, n_models
        )
        
        # Normalize and apply constraints
        optimal_weights = optimal_weights / np.sum(optimal_weights)
        optimal_weights = np.clip(optimal_weights, min_weights, max_weights)
        optimal_weights = optimal_weights / np.sum(optimal_weights)
        
        # Calculate scores
        n_models_base = len(oof_predictions)
        equal_weights = np.ones(n_models_base) / n_models_base
        equal_preds = oof_stack @ equal_weights
        cv_score_before = self._cross_validate_score(
            equal_preds, y_true, metric, self.n_folds
        )
        
        final_preds = oof_stack @ optimal_weights
        cv_score_after = self._cross_validate_score(
            final_preds, y_true, metric, self.n_folds
        )
        
        improvement = cv_score_after - cv_score_before
        
        if model_names is None:
            model_names = [f"Model_{i}" for i in range(n_models)]
        
        result = EnsembleOptimizationResult(
            model_names=model_names,
            optimal_weights=optimal_weights,
            cv_score_before=cv_score_before,
            cv_score_after=cv_score_after,
            improvement=improvement,
            method="nelder-mead-constrained",
            converged=converged,
            n_iterations=n_iter,
            final_loss=-final_loss,
        )
        
        self.optimization_history.append(result)
        
        return result
    
    def _optimize_nelder_mead(
        self,
        loss_function: Callable,
        n_models: int,
    ) -> Tuple[np.ndarray, bool, int, float]:
        """
        Run Nelder-Mead optimization.
        
        Args:
            loss_function: Function to minimize
            n_models: Number of models
        
        Returns:
            (optimal_weights, converged, n_iterations, final_loss)
        """
        # Initial guess: equal weights
        x0 = np.ones(n_models) / n_models
        
        # Run optimization
        result = minimize(
            loss_function,
            x0,
            method="Nelder-Mead",
            options={
                "maxiter": self.max_iterations,
                "xatol": self.tolerance,
                "fatol": self.tolerance,
            },
        )
        
        return (
            result.x,
            result.success,
            result.nit,
            result.fun,
        )
    
    def _optimize_differential_evolution(
        self,
        loss_function: Callable,
        n_models: int,
    ) -> Tuple[np.ndarray, bool, int, float]:
        """
        Run differential evolution (global optimization).
        
        Args:
            loss_function: Function to minimize
            n_models: Number of models
        
        Returns:
            (optimal_weights, converged, n_iterations, final_loss)
        """
        # Bounds: [0, 1] for each weight
        bounds = [(0, 1) for _ in range(n_models)]
        
        # Run optimization
        result = differential_evolution(
            loss_function,
            bounds,
            maxiter=self.max_iterations,
            tol=self.tolerance,
            seed=self.random_state,
            polish=True,  # Final polish with Nelder-Mead
        )
        
        return (
            result.x,
            result.success,
            result.nit,
            result.fun,
        )
    
    def _cross_validate_score(
        self,
        predictions: np.ndarray,
        y_true: np.ndarray,
        metric: str,
        n_folds: int,
    ) -> float:
        """
        Calculate cross-validated score.
        
        Args:
            predictions: Ensemble predictions
            y_true: True target values
            metric: Scoring metric
            n_folds: Number of folds
        
        Returns:
            Mean CV score
        """
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        
        scores = []
        
        for train_idx, val_idx in kf.split(predictions):
            train_pred = predictions[train_idx]
            train_true = y_true[train_idx]
            val_pred = predictions[val_idx]
            val_true = y_true[val_idx]
            
            score = self._calculate_metric(val_true, val_pred, metric)
            scores.append(score)
        
        return np.mean(scores)
    
    def _calculate_metric(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metric: str,
    ) -> float:
        """
        Calculate scoring metric.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            metric: Metric name
        
        Returns:
            Metric score
        """
        if metric == "auc":
            from sklearn.metrics import roc_auc_score
            return roc_auc_score(y_true, y_pred)
        
        elif metric == "logloss":
            from sklearn.metrics import log_loss
            # Clip to avoid log(0)
            y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
            return -log_loss(y_true, y_pred)  # Negative because we maximize
        
        elif metric == "rmse":
            from sklearn.metrics import mean_squared_error
            return -np.sqrt(mean_squared_error(y_true, y_pred))
        
        elif metric == "mae":
            from sklearn.metrics import mean_absolute_error
            return -mean_absolute_error(y_true, y_pred)
        
        elif metric == "r2":
            from sklearn.metrics import r2_score
            return r2_score(y_true, y_pred)
        
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def _validate_inputs(
        self,
        oof_predictions: List[np.ndarray],
        y_true: np.ndarray,
    ) -> None:
        """Validate input arrays."""
        if not oof_predictions:
            raise ValueError("oof_predictions cannot be empty")
        
        if len(oof_predictions) < 2:
            raise ValueError("Need at least 2 models for ensemble optimization")
        
        # Check all arrays have same length
        n_samples = len(oof_predictions[0])
        for i, oof in enumerate(oof_predictions):
            if len(oof) != n_samples:
                raise ValueError(
                    f"Model {i} has {len(oof)} predictions, expected {n_samples}"
                )
        
        if len(y_true) != n_samples:
            raise ValueError(
                f"y_true has {len(y_true)} samples, expected {n_samples}"
            )
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get optimization history as list of dicts."""
        return [result.to_dict() for result in self.optimization_history]
    
    def set_baseline(
        self,
        baseline_name: str,
        baseline_score: float,
    ) -> None:
        """
        Set baseline score for comparison.
        
        Args:
            baseline_name: Name of baseline
            baseline_score: Baseline score
        """
        self.baseline_scores[baseline_name] = baseline_score
        logger.info(f"[EnsembleOptimizer] Set baseline '{baseline_name}': {baseline_score}")
    
    def compare_to_baseline(
        self,
        result: EnsembleOptimizationResult,
        baseline_name: str,
    ) -> Dict[str, Any]:
        """
        Compare optimization result to baseline.
        
        Args:
            result: Optimization result
            baseline_name: Baseline name
        
        Returns:
            Comparison dict
        """
        if baseline_name not in self.baseline_scores:
            raise ValueError(f"Baseline '{baseline_name}' not found")
        
        baseline_score = self.baseline_scores[baseline_name]
        
        return {
            "baseline_score": baseline_score,
            "optimized_score": result.cv_score_after,
            "improvement_vs_baseline": result.cv_score_after - baseline_score,
            "improvement_vs_equal": result.improvement,
        }


def optimize_ensemble(
    oof_predictions: List[np.ndarray],
    y_true: np.ndarray,
    model_names: Optional[List[str]] = None,
    metric: str = "auc",
    method: str = "nelder-mead",
    **kwargs,
) -> EnsembleOptimizationResult:
    """
    Convenience function for ensemble optimization.
    
    Args:
        oof_predictions: List of OOF prediction arrays
        y_true: True target values
        model_names: Names of models
        metric: Scoring metric
        method: Optimization method
        **kwargs: Passed to EnsembleOptimizer
    
    Returns:
        EnsembleOptimizationResult
    """
    optimizer = EnsembleOptimizer(**kwargs)
    return optimizer.optimize(
        oof_predictions=oof_predictions,
        y_true=y_true,
        model_names=model_names,
        metric=metric,
        method=method,
    )
