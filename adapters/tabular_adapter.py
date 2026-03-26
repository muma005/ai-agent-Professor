# adapters/tabular_adapter.py

"""
Tabular competition adapter.

Default adapter for standard tabular data competitions.
"""

import logging
import numpy as np
import polars as pl
from typing import Dict, Any, Optional, List, Tuple
from sklearn.model_selection import StratifiedKFold, KFold

from adapters.base import BaseAdapter, AdapterResult, CompetitionType

logger = logging.getLogger(__name__)


class TabularAdapter(BaseAdapter):
    """
    Adapter for tabular data competitions.
    
    Features:
    - Standard preprocessing
    - Stratified CV for classification
    - Automatic metric detection
    - Feature type detection
    """
    
    def detect(
        self,
        X: Any,
        y: Optional[Any] = None,
    ) -> Tuple[CompetitionType, float]:
        """
        Detect if data is tabular.
        
        Args:
            X: Feature matrix
            y: Target vector
        
        Returns:
            (CompetitionType.TABULAR, confidence)
        """
        confidence = 0.5  # Base confidence
        
        # Check if 2D array-like
        if hasattr(X, "shape") and len(X.shape) == 2:
            confidence += 0.3
        
        # Check if numeric or mixed types
        if hasattr(X, "dtypes"):
            # Polars DataFrame
            numeric_ratio = sum(
                1 for dtype in X.dtypes
                if dtype in [pl.Int32, pl.Int64, pl.Float32, pl.Float64]
            ) / len(X.dtypes)
            
            if numeric_ratio > 0.5:
                confidence += 0.2
        elif hasattr(X, "dtype"):
            # NumPy array
            if np.issubdtype(X.dtype, np.number):
                confidence += 0.2
        
        return CompetitionType.TABULAR, min(confidence, 1.0)
    
    def fit_transform(
        self,
        X: Any,
        y: Optional[Any] = None,
        X_test: Optional[Any] = None,
    ) -> AdapterResult:
        """
        Fit adapter and transform tabular data.
        
        Args:
            X: Training feature matrix
            y: Training target
            X_test: Test feature matrix
        
        Returns:
            AdapterResult with processed data
        """
        logger.info("[TabularAdapter] Processing tabular data")
        
        # Convert to numpy if needed
        if hasattr(X, "to_numpy"):
            X_processed = X.to_numpy()
        else:
            X_processed = np.array(X)
        
        # Process test set if provided
        X_test_processed = None
        if X_test is not None:
            if hasattr(X_test, "to_numpy"):
                X_test_processed = X_test.to_numpy()
            else:
                X_test_processed = np.array(X_test)
        
        # Get feature names
        if hasattr(X, "columns"):
            feature_names = list(X.columns)
        else:
            feature_names = [f"feature_{i}" for i in range(X_processed.shape[1])]
        
        # Get CV splitter
        cv_splitter = self.get_default_cv_splitter(y)
        
        # Get metric
        metric = self.get_default_metric(y)
        
        # Create result
        result = AdapterResult(
            competition_type=CompetitionType.TABULAR,
            confidence=0.9,
            X_processed=X_processed,
            y_processed=y,
            X_test_processed=X_test_processed,
            feature_names=feature_names,
            cv_splitter=cv_splitter,
            metric=metric,
            metadata={
                "n_samples": X_processed.shape[0],
                "n_features": X_processed.shape[1],
                "adapter": "tabular",
            },
        )
        
        self.adapter_history.append(result)
        
        logger.info(
            f"[TabularAdapter] Complete -- {X_processed.shape[0]} samples, "
            f"{X_processed.shape[1]} features, metric: {metric}"
        )
        
        return result
    
    def get_default_metric(self, y: Optional[Any] = None) -> str:
        """Get default metric based on target type."""
        if y is None:
            return "auc"
        
        # Check if classification or regression
        if hasattr(y, "dtype"):
            if np.issubdtype(y.dtype, np.integer) and len(np.unique(y)) <= 20:
                return "auc"  # Classification
            else:
                return "rmse"  # Regression
        else:
            return "auc"
    
    def get_default_cv_splitter(self, y: Optional[Any] = None):
        """Get default CV splitter based on target type."""
        if y is None:
            return KFold(
                n_splits=self.n_folds,
                shuffle=True,
                random_state=self.random_state,
            )
        
        # Check if classification or regression
        if hasattr(y, "dtype"):
            if np.issubdtype(y.dtype, np.integer) and len(np.unique(y)) <= 20:
                return StratifiedKFold(
                    n_splits=self.n_folds,
                    shuffle=True,
                    random_state=self.random_state,
                )
            else:
                return KFold(
                    n_splits=self.n_folds,
                    shuffle=True,
                    random_state=self.random_state,
                )
        else:
            return StratifiedKFold(
                n_splits=self.n_folds,
                shuffle=True,
                random_state=self.random_state,
            )
