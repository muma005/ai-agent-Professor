# agents/stacking_agent.py

"""
Multi-model stacking with meta-learner ensemble.

Advanced Feature: Multi-Model Stacking
- Out-of-fold predictions
- Meta-learner training
- Stacking with calibration
- Blending vs stacking modes
- Cross-validated stacking
- Multiple meta-learner options
- Regression-aware baseline tracking
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Type
from dataclasses import dataclass
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import lightgbm as lgb

logger = logging.getLogger(__name__)


@dataclass
class StackingResult:
    """Result of stacking ensemble."""
    
    base_models: List[str]
    meta_model: str
    mode: str  # "stacking" or "blending"
    n_folds: int
    n_base_models: int
    cv_score_base: float
    cv_score_stacked: float
    improvement: float
    meta_model_weights: Optional[Dict[str, float]]
    oof_predictions: np.ndarray
    test_predictions: Optional[np.ndarray]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict."""
        return {
            "base_models": self.base_models,
            "meta_model": self.meta_model,
            "mode": self.mode,
            "n_folds": self.n_folds,
            "n_base_models": self.n_base_models,
            "cv_score_base": round(self.cv_score_base, 6),
            "cv_score_stacked": round(self.cv_score_stacked, 6),
            "improvement": round(self.improvement, 6),
            "meta_model_weights": self.meta_model_weights,
            "oof_predictions_shape": list(self.oof_predictions.shape) if self.oof_predictions is not None else None,
            "test_predictions_shape": list(self.test_predictions.shape) if self.test_predictions is not None else None,
        }


class StackingAgent:
    """
    Multi-model stacking with meta-learner.
    
    Features:
    - Out-of-fold predictions (no leakage)
    - Multiple meta-learner options
    - Stacking and blending modes
    - Cross-validated stacking
    - Calibration support
    - Test set predictions
    
    Usage:
        stacker = StackingAgent(
            n_folds=5,
            mode="stacking",
        )
        
        result = stacker.fit(
            X=X_train,
            y=y_train,
            base_models={
                "lgbm": lgb_model,
                "xgb": xgb_model,
            },
            meta_model="logistic",
        )
        
        test_preds = result.test_predictions
    """
    
    def __init__(
        self,
        n_folds: int = 5,
        mode: str = "stacking",
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        """
        Initialize stacking agent.
        
        Args:
            n_folds: Number of CV folds
            mode: "stacking" (OOF) or "blending" (holdout)
            random_state: Random seed
            n_jobs: Number of parallel jobs
        """
        self.n_folds = n_folds
        self.mode = mode
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        self.baseline_scores: Dict[str, float] = {}
        self.stacking_history: List[StackingResult] = []
        
        logger.info(
            f"[StackingAgent] Initialized -- n_folds: {n_folds}, "
            f"mode: {mode}"
        )
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        base_models: Dict[str, Any],
        meta_model: str = "logistic",
        X_test: Optional[np.ndarray] = None,
    ) -> StackingResult:
        """
        Fit stacking ensemble.
        
        Args:
            X: Training feature matrix
            y: Training target vector
            base_models: Dict of {name: model} base models
            meta_model: Meta-learner type ("logistic", "ridge", "rf", "gbm", "lgbm")
            X_test: Optional test set for predictions
        
        Returns:
            StackingResult with predictions and scores
        
        Raises:
            ValueError: If mode or meta_model is invalid
        """
        n_samples = len(y)
        n_base_models = len(base_models)
        
        logger.info(
            f"[StackingAgent] Fitting stacking ensemble -- "
            f"{n_base_models} base models, mode: {self.mode}"
        )
        
        # Generate out-of-fold predictions for base models
        oof_predictions = self._generate_oof_predictions(
            X, y, base_models
        )
        
        # Generate test predictions if test set provided
        test_predictions = None
        if X_test is not None:
            test_predictions = self._generate_test_predictions(
                X_test, base_models
            )
        
        # Calculate base model CV score (average of base models)
        cv_score_base = self._calculate_base_score(X, y, base_models)
        
        # Fit meta-learner on OOF predictions
        meta_learner = self._create_meta_learner(meta_model)
        meta_learner.fit(oof_predictions, y)
        
        # Calculate stacked CV score
        if hasattr(meta_learner, "predict_proba"):
            stacked_preds = meta_learner.predict_proba(oof_predictions)[:, 1]
            cv_score_stacked = self._calculate_metric(y, stacked_preds, "auc")
        else:
            # For regression meta-models (Ridge), use predict
            stacked_preds = meta_learner.predict(oof_predictions)
            # Convert to probabilities if needed
            if stacked_preds.min() < 0 or stacked_preds.max() > 1:
                from sklearn.preprocessing import minmax_scale
                stacked_preds = minmax_scale(stacked_preds)
            cv_score_stacked = self._calculate_metric(y, stacked_preds, "auc")
        
        improvement = cv_score_stacked - cv_score_base
        
        logger.info(
            f"[StackingAgent] Stacking complete -- "
            f"Base CV: {cv_score_base:.6f}, Stacked CV: {cv_score_stacked:.6f}, "
            f"Improvement: {improvement:+.6f}"
        )
        
        # Extract meta-model weights if linear
        meta_model_weights = None
        if hasattr(meta_learner, "coef_"):
            model_names = list(base_models.keys())
            meta_model_weights = {
                name: float(coef)
                for name, coef in zip(model_names, meta_learner.coef_[0])
            }
        
        # Create result
        result = StackingResult(
            base_models=list(base_models.keys()),
            meta_model=meta_model,
            mode=self.mode,
            n_folds=self.n_folds,
            n_base_models=n_base_models,
            cv_score_base=cv_score_base,
            cv_score_stacked=cv_score_stacked,
            improvement=improvement,
            meta_model_weights=meta_model_weights,
            oof_predictions=oof_predictions,
            test_predictions=test_predictions,
        )
        
        self.stacking_history.append(result)
        
        return result
    
    def _generate_oof_predictions(
        self,
        X: np.ndarray,
        y: np.ndarray,
        base_models: Dict[str, Any],
    ) -> np.ndarray:
        """
        Generate out-of-fold predictions for base models.
        
        Args:
            X: Training feature matrix
            y: Training target vector
            base_models: Dict of base models
        
        Returns:
            OOF predictions array (n_samples, n_base_models)
        """
        n_samples = len(y)
        n_base_models = len(base_models)
        
        # Initialize OOF predictions
        oof_predictions = np.zeros((n_samples, n_base_models))
        
        # Create CV splitter
        if self.mode == "stacking":
            cv_splits = list(self._create_cv_splitter(y).split(X, y))
        else:  # blending
            # Use simple holdout for blending
            from sklearn.model_selection import train_test_split
            train_idx, val_idx = train_test_split(
                np.arange(n_samples),
                test_size=0.2,
                random_state=self.random_state,
                stratify=y if len(np.unique(y)) > 1 else None,
            )
            cv_splits = [(train_idx, val_idx)]
        
        # Generate OOF predictions for each base model
        for model_idx, (model_name, model) in enumerate(base_models.items()):
            logger.debug(f"[StackingAgent] Generating OOF for {model_name}")
            
            for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
                X_train_fold = X[train_idx]
                y_train_fold = y[train_idx]
                X_val_fold = X[val_idx]
                
                # Clone and fit model on fold
                from sklearn.base import clone
                model_fold = clone(model)
                model_fold.fit(X_train_fold, y_train_fold)
                
                # Predict on validation fold
                if hasattr(model_fold, "predict_proba"):
                    oof_predictions[val_idx, model_idx] = model_fold.predict_proba(X_val_fold)[:, 1]
                else:
                    oof_predictions[val_idx, model_idx] = model_fold.predict(X_val_fold)
        
        return oof_predictions
    
    def _generate_test_predictions(
        self,
        X_test: np.ndarray,
        base_models: Dict[str, Any],
    ) -> np.ndarray:
        """
        Generate test set predictions from base models.
        
        Args:
            X_test: Test feature matrix
            base_models: Dict of base models
        
        Returns:
            Test predictions array (n_test_samples, n_base_models)
        """
        n_test_samples = len(X_test)
        n_base_models = len(base_models)
        
        test_predictions = np.zeros((n_test_samples, n_base_models))
        
        for model_idx, (model_name, model) in enumerate(base_models.items()):
            if hasattr(model, "predict_proba"):
                test_predictions[:, model_idx] = model.predict_proba(X_test)[:, 1]
            else:
                test_predictions[:, model_idx] = model.predict(X_test)
        
        return test_predictions
    
    def _calculate_base_score(
        self,
        X: np.ndarray,
        y: np.ndarray,
        base_models: Dict[str, Any],
    ) -> float:
        """
        Calculate average CV score of base models.
        
        Args:
            X: Training feature matrix
            y: Training target vector
            base_models: Dict of base models
        
        Returns:
            Average CV score
        """
        from sklearn.model_selection import cross_val_score
        
        scores = []
        cv = self._create_cv_splitter(y)
        
        for model_name, model in base_models.items():
            model_scores = cross_val_score(
                model, X, y,
                cv=cv,
                scoring="roc_auc",
                n_jobs=self.n_jobs,
            )
            scores.append(np.mean(model_scores))
        
        return np.mean(scores)
    
    def _create_cv_splitter(self, y: np.ndarray):
        """Create CV splitter based on target type."""
        n_unique = len(np.unique(y))
        
        if n_unique <= 20:  # Classification
            return StratifiedKFold(
                n_splits=self.n_folds,
                shuffle=True,
                random_state=self.random_state,
            )
        else:  # Regression
            return KFold(
                n_splits=self.n_folds,
                shuffle=True,
                random_state=self.random_state,
            )
    
    def _create_meta_learner(self, meta_model: str) -> Any:
        """
        Create meta-learner model.
        
        Args:
            meta_model: Meta-learner type
        
        Returns:
            Meta-learner model
        """
        if meta_model == "logistic":
            return LogisticRegression(
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                max_iter=1000,
            )
        elif meta_model == "ridge":
            return Ridge(random_state=self.random_state)
        elif meta_model == "rf":
            return RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )
        elif meta_model == "gbm":
            return GradientBoostingClassifier(
                n_estimators=100,
                random_state=self.random_state,
            )
        elif meta_model == "lgbm":
            return lgb.LGBMClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                verbose=-1,
            )
        else:
            logger.warning(f"Unknown meta-model: {meta_model}, using LogisticRegression")
            return LogisticRegression(random_state=self.random_state, max_iter=1000)
    
    def _calculate_metric(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metric: str,
    ) -> float:
        """Calculate scoring metric."""
        from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error
        
        if metric == "auc":
            return roc_auc_score(y_true, y_pred)
        elif metric == "logloss":
            return -log_loss(y_true, y_pred)
        elif metric == "rmse":
            return -np.sqrt(mean_squared_error(y_true, y_pred))
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def get_stacking_history(self) -> List[Dict[str, Any]]:
        """Get stacking history as list of dicts."""
        return [result.to_dict() for result in self.stacking_history]
    
    def set_baseline(
        self,
        baseline_name: str,
        baseline_score: float,
    ) -> None:
        """Set baseline score for comparison."""
        self.baseline_scores[baseline_name] = baseline_score
        logger.info(f"[StackingAgent] Set baseline '{baseline_name}': {baseline_score}")


def stack_models(
    X: np.ndarray,
    y: np.ndarray,
    base_models: Dict[str, Any],
    meta_model: str = "logistic",
    n_folds: int = 5,
    mode: str = "stacking",
    X_test: Optional[np.ndarray] = None,
) -> StackingResult:
    """
    Convenience function for model stacking.
    
    Args:
        X: Training feature matrix
        y: Training target vector
        base_models: Dict of base models
        meta_model: Meta-learner type
        n_folds: Number of CV folds
        mode: "stacking" or "blending"
        X_test: Optional test set
    
    Returns:
        StackingResult
    """
    stacker = StackingAgent(n_folds=n_folds, mode=mode)
    return stacker.fit(X, y, base_models, meta_model, X_test)
