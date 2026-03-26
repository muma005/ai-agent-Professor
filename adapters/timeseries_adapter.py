# adapters/timeseries_adapter.py

"""
Time series competition adapter.

Adapter for time series data with time-based splits and lag features.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from sklearn.model_selection import TimeSeriesSplit

from adapters.base import BaseAdapter, AdapterResult, CompetitionType

logger = logging.getLogger(__name__)


class TimeSeriesAdapter(BaseAdapter):
    """
    Adapter for time series competitions.
    
    Features:
    - Time-based CV splits
    - Lag feature generation
    - Rolling statistics
    - Date feature extraction
    """
    
    def __init__(
        self,
        random_state: int = 42,
        n_folds: int = 5,
        n_lags: int = 7,
        rolling_windows: Optional[List[int]] = None,
    ):
        """
        Initialize time series adapter.
        
        Args:
            random_state: Random seed
            n_folds: Number of CV folds
            n_lags: Number of lag features
            rolling_windows: List of rolling window sizes
        """
        super().__init__(random_state, n_folds)
        
        self.n_lags = n_lags
        self.rolling_windows = rolling_windows or [7, 14, 30]
        
        self.date_column: Optional[str] = None
        self.target_column: Optional[str] = None
    
    def detect(
        self,
        X: Any,
        y: Optional[Any] = None,
    ) -> Tuple[CompetitionType, float]:
        """
        Detect if data is time series.
        
        Args:
            X: Feature matrix
            y: Target vector
        
        Returns:
            (CompetitionType.TIMESERIES, confidence)
        """
        confidence = 0.3  # Base confidence
        
        # Check for date/time columns
        if hasattr(X, "columns"):
            for col in X.columns:
                col_lower = str(col).lower()
                if "date" in col_lower or "time" in col_lower or "timestamp" in col_lower:
                    confidence += 0.5
                    self.date_column = col
                    break
        
        # Check if index is datetime
        if hasattr(X, "index") and hasattr(X.index, "dtype"):
            if "datetime" in str(X.index.dtype):
                confidence += 0.4
        
        # Check for sequential patterns in data
        if hasattr(X, "iloc") and len(X) > 10:
            # Check if first column looks like dates
            try:
                first_col = X.iloc[:, 0]
                if hasattr(first_col, "dtype") and "datetime" in str(first_col.dtype):
                    confidence += 0.4
            except:
                pass
        
        return CompetitionType.TIMESERIES, min(confidence, 1.0)
    
    def fit_transform(
        self,
        X: Any,
        y: Optional[Any] = None,
        X_test: Optional[Any] = None,
    ) -> AdapterResult:
        """
        Fit adapter and transform time series data.
        
        Args:
            X: Training feature matrix
            y: Training target
            X_test: Test feature matrix
        
        Returns:
            AdapterResult with processed data
        """
        logger.info("[TimeSeriesAdapter] Processing time series data")
        
        # Convert to DataFrame if needed
        if not hasattr(X, "columns"):
            X = pd.DataFrame(X)
        
        # Extract date features if date column exists
        if self.date_column and self.date_column in X.columns:
            X = self._extract_date_features(X, self.date_column)
        
        # Generate lag features
        X = self._generate_lag_features(X, y)
        
        # Generate rolling features
        X = self._generate_rolling_features(X, y)
        
        # Convert to numpy
        X_processed = X.values
        
        # Process test set
        X_test_processed = None
        if X_test is not None:
            if not hasattr(X_test, "columns"):
                X_test = pd.DataFrame(X_test)
            X_test_processed = X_test.values
        
        # Get feature names
        feature_names = list(X.columns)
        
        # Get CV splitter (time-based)
        cv_splitter = self.get_default_cv_splitter(y)
        
        # Get metric
        metric = self.get_default_metric(y)
        
        # Create result
        result = AdapterResult(
            competition_type=CompetitionType.TIMESERIES,
            confidence=0.85,
            X_processed=X_processed,
            y_processed=y,
            X_test_processed=X_test_processed,
            feature_names=feature_names,
            cv_splitter=cv_splitter,
            metric=metric,
            metadata={
                "n_samples": X_processed.shape[0],
                "n_features": X_processed.shape[1],
                "adapter": "timeseries",
                "n_lags": self.n_lags,
                "rolling_windows": self.rolling_windows,
                "date_column": self.date_column,
            },
        )
        
        self.adapter_history.append(result)
        
        logger.info(
            f"[TimeSeriesAdapter] Complete -- {X_processed.shape[0]} samples, "
            f"{X_processed.shape[1]} features (with lags/rolling), metric: {metric}"
        )
        
        return result
    
    def _extract_date_features(
        self,
        df: pd.DataFrame,
        date_column: str,
    ) -> pd.DataFrame:
        """Extract features from date column."""
        df = df.copy()
        
        try:
            df[date_column] = pd.to_datetime(df[date_column])
            
            # Extract date components
            df["year"] = df[date_column].dt.year
            df["month"] = df[date_column].dt.month
            df["day"] = df[date_column].dt.day
            df["dayofweek"] = df[date_column].dt.dayofweek
            df["dayofyear"] = df[date_column].dt.dayofyear
            df["week"] = df[date_column].dt.isocalendar().week.astype(int)
            df["quarter"] = df[date_column].dt.quarter
            df["is_month_start"] = df[date_column].dt.is_month_start.astype(int)
            df["is_month_end"] = df[date_column].dt.is_month_end.astype(int)
            
            logger.debug(f"[TimeSeriesAdapter] Extracted date features from {date_column}")
            
        except Exception as e:
            logger.warning(f"[TimeSeriesAdapter] Failed to extract date features: {e}")
        
        return df
    
    def _generate_lag_features(
        self,
        df: pd.DataFrame,
        y: Optional[Any],
    ) -> pd.DataFrame:
        """Generate lag features from target."""
        df = df.copy()
        
        if y is None:
            return df
        
        # Convert y to Series if needed
        if not hasattr(y, "index"):
            y = pd.Series(y)
        
        # Generate lags
        for lag in range(1, self.n_lags + 1):
            df[f"target_lag_{lag}"] = y.shift(lag)
        
        logger.debug(f"[TimeSeriesAdapter] Generated {self.n_lags} lag features")
        
        return df
    
    def _generate_rolling_features(
        self,
        df: pd.DataFrame,
        y: Optional[Any],
    ) -> pd.DataFrame:
        """Generate rolling statistics from target."""
        df = df.copy()
        
        if y is None:
            return df
        
        # Convert y to Series if needed
        if not hasattr(y, "index"):
            y = pd.Series(y)
        
        # Generate rolling stats
        for window in self.rolling_windows:
            df[f"target_rolling_mean_{window}"] = y.shift(1).rolling(window).mean()
            df[f"target_rolling_std_{window}"] = y.shift(1).rolling(window).std()
        
        logger.debug(f"[TimeSeriesAdapter] Generated rolling features for windows {self.rolling_windows}")
        
        return df
    
    def get_default_metric(self, y: Optional[Any] = None) -> str:
        """Get default metric for time series."""
        return "rmse"  # Time series typically uses RMSE
    
    def get_default_cv_splitter(self, y: Optional[Any] = None):
        """Get time-based CV splitter."""
        return TimeSeriesSplit(
            n_splits=self.n_folds,
            gap=0,
            max_train_size=None,
            test_size=None,
        )
