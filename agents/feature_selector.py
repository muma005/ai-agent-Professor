# agents/feature_selector.py

"""
Automated feature selection with multiple methods.

Advanced Feature: Feature Selection Automation
- Null importance filtering
- Permutation importance
- Recursive feature elimination
- Stability selection
- Multi-method consensus
- Regression-aware baseline tracking
"""

import logging
import numpy as np
import polars as pl
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from sklearn.model_selection import KFold
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFE
import lightgbm as lgb

logger = logging.getLogger(__name__)


@dataclass
class FeatureSelectionResult:
    """Result of feature selection."""
    
    method: str
    n_features_before: int
    n_features_after: int
    n_features_removed: int
    selected_features: List[str]
    removed_features: List[str]
    feature_importances: Dict[str, float]
    cv_score_before: float
    cv_score_after: float
    improvement: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict."""
        return {
            "method": self.method,
            "n_features_before": self.n_features_before,
            "n_features_after": self.n_features_after,
            "n_features_removed": self.n_features_removed,
            "selected_features": self.selected_features,
            "removed_features": self.removed_features,
            "feature_importances": {k: round(v, 6) for k, v in self.feature_importances.items()},
            "cv_score_before": round(self.cv_score_before, 6),
            "cv_score_after": round(self.cv_score_after, 6),
            "improvement": round(self.improvement, 6),
        }


class FeatureSelector:
    """
    Automated feature selection with multiple methods.
    
    Features:
    - Null importance filtering
    - Permutation importance
    - Recursive feature elimination
    - Stability selection
    - Multi-method consensus
    - Cross-validation based selection
    
    Usage:
        selector = FeatureSelector()
        
        result = selector.select_features(
            X=X_train,
            y=y_train,
            feature_names=feature_names,
            method="null_importance",
            threshold=0.1,
        )
        
        print(f"Selected {len(result.selected_features)} features")
        print(f"Removed {len(result.removed_features)} features")
    """
    
    def __init__(
        self,
        n_folds: int = 5,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        """
        Initialize feature selector.
        
        Args:
            n_folds: Number of CV folds
            random_state: Random seed
            n_jobs: Number of parallel jobs
        """
        self.n_folds = n_folds
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        self.baseline_scores: Dict[str, float] = {}
        self.selection_history: List[FeatureSelectionResult] = []
        
        logger.info(
            f"[FeatureSelector] Initialized -- n_folds: {n_folds}, "
            f"n_jobs: {n_jobs}"
        )
    
    def select_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        method: str = "null_importance",
        threshold: float = 0.1,
        **kwargs,
    ) -> FeatureSelectionResult:
        """
        Select features using specified method.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector
            feature_names: List of feature names
            method: Selection method
            threshold: Method-specific threshold
            **kwargs: Passed to specific method
        
        Returns:
            FeatureSelectionResult with selected features
        
        Raises:
            ValueError: If method is unknown
        """
        if method == "null_importance":
            return self._null_importance(X, y, feature_names, threshold, **kwargs)
        elif method == "permutation":
            return self._permutation_importance(X, y, feature_names, threshold, **kwargs)
        elif method == "rfe":
            return self._recursive_elimination(X, y, feature_names, threshold, **kwargs)
        elif method == "stability":
            return self._stability_selection(X, y, feature_names, threshold, **kwargs)
        elif method == "consensus":
            return self._consensus_selection(X, y, feature_names, threshold, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _null_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        threshold: float = 0.1,
        n_shuffles: int = 10,
        **kwargs,
    ) -> FeatureSelectionResult:
        """
        Select features using null importance filtering.
        
        Args:
            X: Feature matrix
            y: Target vector
            feature_names: Feature names
            threshold: Importance ratio threshold
            n_shuffles: Number of shuffles for null distribution
            **kwargs: Additional parameters
        
        Returns:
            FeatureSelectionResult
        """
        n_samples, n_features = X.shape
        
        logger.info(
            f"[FeatureSelector] Null importance -- {n_features} features, "
            f"{n_shuffles} shuffles"
        )
        
        # Train on real target
        model = self._train_model(X, y)
        real_importance = model.feature_importances_
        
        # Train on shuffled targets
        null_importances = []
        
        for i in range(n_shuffles):
            y_shuffled = np.random.permutation(y)
            model_null = self._train_model(X, y_shuffled)
            null_importances.append(model_null.feature_importances_)
        
        null_importances = np.array(null_importances)
        null_mean = np.mean(null_importances, axis=0)
        null_std = np.std(null_importances, axis=0) + 1e-10
        
        # Calculate importance ratio
        importance_ratio = real_importance / (null_mean + null_std)
        
        # Select features with ratio > threshold
        selected_mask = importance_ratio > threshold
        
        selected_features = [f for f, m in zip(feature_names, selected_mask) if m]
        removed_features = [f for f, m in zip(feature_names, selected_mask) if not m]
        
        # Calculate CV scores
        cv_score_before = self._cv_score(X, y)
        cv_score_after = self._cv_score(X[:, selected_mask], y)
        
        improvement = cv_score_after - cv_score_before
        
        logger.info(
            f"[FeatureSelector] Null importance: {len(selected_features)} selected, "
            f"{len(removed_features)} removed, improvement: {improvement:+.6f}"
        )
        
        # Create result
        feature_importances = {
            name: float(imp) for name, imp in zip(feature_names, importance_ratio)
        }
        
        result = FeatureSelectionResult(
            method="null_importance",
            n_features_before=n_features,
            n_features_after=len(selected_features),
            n_features_removed=len(removed_features),
            selected_features=selected_features,
            removed_features=removed_features,
            feature_importances=feature_importances,
            cv_score_before=cv_score_before,
            cv_score_after=cv_score_after,
            improvement=improvement,
        )
        
        self.selection_history.append(result)
        
        return result
    
    def _permutation_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        threshold: float = 0.01,
        **kwargs,
    ) -> FeatureSelectionResult:
        """
        Select features using permutation importance.
        
        Args:
            X: Feature matrix
            y: Target vector
            feature_names: Feature names
            threshold: Minimum importance threshold
            **kwargs: Passed to permutation_importance
        
        Returns:
            FeatureSelectionResult
        """
        n_samples, n_features = X.shape
        
        logger.info(
            f"[FeatureSelector] Permutation importance -- {n_features} features"
        )
        
        # Train model
        model = self._train_model(X, y)
        
        # Calculate permutation importance
        perm_result = permutation_importance(
            model, X, y,
            n_repeats=10,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        
        importance = perm_result.importances_mean
        
        # Select features with importance > threshold
        selected_mask = importance > threshold
        
        selected_features = [f for f, m in zip(feature_names, selected_mask) if m]
        removed_features = [f for f, m in zip(feature_names, selected_mask) if not m]
        
        # Calculate CV scores
        cv_score_before = self._cv_score(X, y)
        cv_score_after = self._cv_score(X[:, selected_mask], y)
        
        improvement = cv_score_after - cv_score_before
        
        logger.info(
            f"[FeatureSelector] Permutation: {len(selected_features)} selected, "
            f"improvement: {improvement:+.6f}"
        )
        
        feature_importances = {
            name: float(imp) for name, imp in zip(feature_names, importance)
        }
        
        result = FeatureSelectionResult(
            method="permutation_importance",
            n_features_before=n_features,
            n_features_after=len(selected_features),
            n_features_removed=len(removed_features),
            selected_features=selected_features,
            removed_features=removed_features,
            feature_importances=feature_importances,
            cv_score_before=cv_score_before,
            cv_score_after=cv_score_after,
            improvement=improvement,
        )
        
        self.selection_history.append(result)
        
        return result
    
    def _recursive_elimination(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        threshold: float = 0.5,  # Ignored for RFE, using n_features_to_select instead
        n_features_to_select: Optional[int] = None,
        step: int = 1,
        **kwargs,
    ) -> FeatureSelectionResult:
        """
        Select features using recursive feature elimination.
        
        Args:
            X: Feature matrix
            y: Target vector
            feature_names: Feature names
            n_features_to_select: Number of features to select
            step: Number of features to remove at each step
            **kwargs: Passed to RFE
        
        Returns:
            FeatureSelectionResult
        """
        n_samples, n_features = X.shape
        
        if n_features_to_select is None:
            n_features_to_select = max(1, n_features // 2)
        
        logger.info(
            f"[FeatureSelector] RFE -- {n_features} features, "
            f"select {n_features_to_select}"
        )
        
        # Base model for RFE
        base_model = lgb.LGBMClassifier(
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=-1,
        )
        
        # Run RFE
        rfe = RFE(
            estimator=base_model,
            n_features_to_select=n_features_to_select,
            step=step,
        )
        
        rfe.fit(X, y)
        
        selected_mask = rfe.support_
        
        selected_features = [f for f, m in zip(feature_names, selected_mask) if m]
        removed_features = [f for f, m in zip(feature_names, selected_mask) if not m]
        
        # Calculate CV scores
        cv_score_before = self._cv_score(X, y)
        cv_score_after = self._cv_score(X[:, selected_mask], y)
        
        improvement = cv_score_after - cv_score_before
        
        logger.info(
            f"[FeatureSelector] RFE: {len(selected_features)} selected, "
            f"improvement: {improvement:+.6f}"
        )
        
        feature_importances = {
            name: float(imp) if m else 0.0
            for name, imp, m in zip(feature_names, rfe.ranking_, selected_mask)
        }
        
        result = FeatureSelectionResult(
            method="recursive_elimination",
            n_features_before=n_features,
            n_features_after=len(selected_features),
            n_features_removed=len(removed_features),
            selected_features=selected_features,
            removed_features=removed_features,
            feature_importances=feature_importances,
            cv_score_before=cv_score_before,
            cv_score_after=cv_score_after,
            improvement=improvement,
        )
        
        self.selection_history.append(result)
        
        return result
    
    def _stability_selection(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        threshold: float = 0.6,
        n_iterations: int = 100,
        sample_fraction: float = 0.8,
        **kwargs,
    ) -> FeatureSelectionResult:
        """
        Select features using stability selection.
        
        Args:
            X: Feature matrix
            y: Target vector
            feature_names: Feature names
            threshold: Selection probability threshold
            n_iterations: Number of iterations
            sample_fraction: Fraction of samples per iteration
            **kwargs: Additional parameters
        
        Returns:
            FeatureSelectionResult
        """
        n_samples, n_features = X.shape
        
        logger.info(
            f"[FeatureSelector] Stability selection -- {n_features} features, "
            f"{n_iterations} iterations"
        )
        
        selection_counts = np.zeros(n_features)
        
        for i in range(n_iterations):
            # Sample subset of data
            n_sample = int(sample_fraction * n_samples)
            indices = np.random.choice(n_samples, n_sample, replace=False)
            
            X_sample = X[indices]
            y_sample = y[indices]
            
            # Train model and get importance
            model = self._train_model(X_sample, y_sample)
            importance = model.feature_importances_
            
            # Count selected features (top 50%)
            threshold_idx = np.argsort(importance)[len(importance) // 2]
            selection_counts[importance >= importance[threshold_idx]] += 1
        
        # Calculate selection probabilities
        selection_probs = selection_counts / n_iterations
        
        # Select features with probability > threshold
        selected_mask = selection_probs > threshold
        
        selected_features = [f for f, m in zip(feature_names, selected_mask) if m]
        removed_features = [f for f, m in zip(feature_names, selected_mask) if not m]
        
        # Calculate CV scores
        cv_score_before = self._cv_score(X, y)
        cv_score_after = self._cv_score(X[:, selected_mask], y)
        
        improvement = cv_score_after - cv_score_before
        
        logger.info(
            f"[FeatureSelector] Stability: {len(selected_features)} selected, "
            f"improvement: {improvement:+.6f}"
        )
        
        feature_importances = {
            name: float(prob) for name, prob in zip(feature_names, selection_probs)
        }
        
        result = FeatureSelectionResult(
            method="stability_selection",
            n_features_before=n_features,
            n_features_after=len(selected_features),
            n_features_removed=len(removed_features),
            selected_features=selected_features,
            removed_features=removed_features,
            feature_importances=feature_importances,
            cv_score_before=cv_score_before,
            cv_score_after=cv_score_after,
            improvement=improvement,
        )
        
        self.selection_history.append(result)
        
        return result
    
    def _consensus_selection(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        threshold: float = 0.5,  # Ignored for consensus, using min_methods instead
        min_methods: int = 2,
        **kwargs,
    ) -> FeatureSelectionResult:
        """
        Select features using consensus of multiple methods.
        
        Args:
            X: Feature matrix
            y: Target vector
            feature_names: Feature names
            min_methods: Minimum number of methods that must select feature
            **kwargs: Additional parameters
        
        Returns:
            FeatureSelectionResult
        """
        n_features = X.shape[1]
        
        logger.info(
            f"[FeatureSelector] Consensus selection -- {n_features} features, "
            f"min {min_methods} methods"
        )
        
        # Run multiple methods
        methods = ["null_importance", "permutation", "stability"]
        selection_votes = np.zeros(n_features)
        
        for method in methods:
            try:
                # Use method-specific kwargs
                method_kwargs = {"n_shuffles": 5} if method == "null_importance" else {}
                method_kwargs.update(kwargs)
                
                result = self.select_features(
                    X, y, feature_names,
                    method=method,
                    **method_kwargs,
                )
                
                # Add votes for selected features
                for name in result.selected_features:
                    idx = feature_names.index(name)
                    selection_votes[idx] += 1
                
            except Exception as e:
                logger.warning(f"[FeatureSelector] {method} failed: {e}")
        
        # Select features with votes >= min_methods
        selected_mask = selection_votes >= min_methods
        
        selected_features = [f for f, m in zip(feature_names, selected_mask) if m]
        removed_features = [f for f, m in zip(feature_names, selected_mask) if not m]
        
        # Calculate CV scores
        cv_score_before = self._cv_score(X, y)
        cv_score_after = self._cv_score(X[:, selected_mask], y)
        
        improvement = cv_score_after - cv_score_before
        
        logger.info(
            f"[FeatureSelector] Consensus: {len(selected_features)} selected, "
            f"improvement: {improvement:+.6f}"
        )
        
        feature_importances = {
            name: float(votes) for name, votes in zip(feature_names, selection_votes)
        }
        
        result = FeatureSelectionResult(
            method="consensus",
            n_features_before=n_features,
            n_features_after=len(selected_features),
            n_features_removed=len(removed_features),
            selected_features=selected_features,
            removed_features=removed_features,
            feature_importances=feature_importances,
            cv_score_before=cv_score_before,
            cv_score_after=cv_score_after,
            improvement=improvement,
        )
        
        self.selection_history.append(result)
        
        return result
    
    def _train_model(self, X: np.ndarray, y: np.ndarray) -> Any:
        """Train a LightGBM model."""
        model = lgb.LGBMClassifier(
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=-1,
            n_estimators=100,
        )
        model.fit(X, y)
        return model
    
    def _cv_score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate cross-validated AUC score."""
        from sklearn.model_selection import cross_val_score
        
        model = lgb.LGBMClassifier(
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=-1,
            n_estimators=100,
        )
        
        scores = cross_val_score(
            model, X, y,
            cv=self.n_folds,
            scoring="roc_auc",
            n_jobs=self.n_jobs,
        )
        
        return np.mean(scores)
    
    def get_selection_history(self) -> List[Dict[str, Any]]:
        """Get selection history as list of dicts."""
        return [result.to_dict() for result in self.selection_history]
    
    def set_baseline(
        self,
        baseline_name: str,
        baseline_score: float,
    ) -> None:
        """Set baseline score for comparison."""
        self.baseline_scores[baseline_name] = baseline_score
        logger.info(f"[FeatureSelector] Set baseline '{baseline_name}': {baseline_score}")


def select_features(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    method: str = "null_importance",
    n_folds: int = 5,
    **kwargs,
) -> FeatureSelectionResult:
    """
    Convenience function for feature selection.
    
    Args:
        X: Feature matrix
        y: Target vector
        feature_names: Feature names
        method: Selection method
        n_folds: Number of CV folds
        **kwargs: Passed to FeatureSelector
    
    Returns:
        FeatureSelectionResult
    """
    selector = FeatureSelector(n_folds=n_folds)
    return selector.select_features(X, y, feature_names, method, **kwargs)
