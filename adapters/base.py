# adapters/base.py

"""
Base adapter for competition-specific adaptations.

Advanced Feature: Competition Adapters
- Base adapter class
- Auto-detection of competition type
- Standardized interface
- Regression-aware baseline tracking
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class CompetitionType(Enum):
    """Types of competitions."""
    TABULAR = "tabular"
    TIMESERIES = "timeseries"
    NLP = "nlp"
    IMAGE = "image"
    UNKNOWN = "unknown"


@dataclass
class AdapterResult:
    """Result of adapter processing."""
    
    competition_type: CompetitionType
    confidence: float
    X_processed: Any
    y_processed: Optional[Any]
    X_test_processed: Optional[Any]
    feature_names: List[str]
    cv_splitter: Any
    metric: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict."""
        return {
            "competition_type": self.competition_type.value,
            "confidence": round(self.confidence, 4),
            "n_features": len(self.feature_names),
            "cv_splitter_type": type(self.cv_splitter).__name__,
            "metric": self.metric,
            "metadata": self.metadata,
        }


class BaseAdapter(ABC):
    """
    Abstract base class for competition adapters.
    
    Features:
    - Standardized interface
    - Auto-detection support
    - Preprocessing pipeline
    - CV splitter selection
    - Metric selection
    
    Usage:
        adapter = TabularAdapter()
        
        result = adapter.fit_transform(
            X=X_train,
            y=y_train,
            X_test=X_test,
        )
    """
    
    def __init__(
        self,
        random_state: int = 42,
        n_folds: int = 5,
    ):
        """
        Initialize adapter.
        
        Args:
            random_state: Random seed
            n_folds: Number of CV folds
        """
        self.random_state = random_state
        self.n_folds = n_folds
        
        self.baseline_scores: Dict[str, float] = {}
        self.adapter_history: List[AdapterResult] = []
        
        logger.info(f"[{self.__class__.__name__}] Initialized")
    
    @abstractmethod
    def detect(self, X: Any, y: Optional[Any] = None) -> Tuple[CompetitionType, float]:
        """
        Detect competition type from data.
        
        Args:
            X: Feature matrix
            y: Target vector (optional)
        
        Returns:
            (CompetitionType, confidence)
        """
        pass
    
    @abstractmethod
    def fit_transform(
        self,
        X: Any,
        y: Optional[Any] = None,
        X_test: Optional[Any] = None,
    ) -> AdapterResult:
        """
        Fit adapter and transform data.
        
        Args:
            X: Training feature matrix
            y: Training target (optional)
            X_test: Test feature matrix (optional)
        
        Returns:
            AdapterResult with processed data
        """
        pass
    
    @abstractmethod
    def get_default_metric(self) -> str:
        """Get default metric for this competition type."""
        pass
    
    @abstractmethod
    def get_default_cv_splitter(self, y: Optional[Any] = None):
        """Get default CV splitter for this competition type."""
        pass
    
    def set_baseline(
        self,
        baseline_name: str,
        baseline_score: float,
    ) -> None:
        """Set baseline score for comparison."""
        self.baseline_scores[baseline_name] = baseline_score
        logger.info(f"[{self.__class__.__name__}] Set baseline '{baseline_name}': {baseline_score}")
    
    def get_adapter_history(self) -> List[Dict[str, Any]]:
        """Get adapter history as list of dicts."""
        return [result.to_dict() for result in self.adapter_history]


def detect_competition_type(
    X: Any,
    y: Optional[Any] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[CompetitionType, float, BaseAdapter]:
    """
    Auto-detect competition type and return appropriate adapter.
    
    Args:
        X: Feature matrix
        y: Target vector (optional)
        metadata: Optional metadata (column names, etc.)
    
    Returns:
        (CompetitionType, confidence, adapter)
    """
    from adapters.tabular_adapter import TabularAdapter
    from adapters.timeseries_adapter import TimeSeriesAdapter
    from adapters.nlp_adapter import NLPAdapter
    
    # Try each adapter's detection
    adapters = [
        TabularAdapter(),
        TimeSeriesAdapter(),
        NLPAdapter(),
    ]
    
    best_type = CompetitionType.UNKNOWN
    best_confidence = 0.0
    best_adapter = adapters[0]  # Default to tabular
    
    for adapter in adapters:
        comp_type, confidence = adapter.detect(X, y)
        
        if confidence > best_confidence:
            best_type = comp_type
            best_confidence = confidence
            best_adapter = adapter
    
    logger.info(
        f"[AutoDetect] Detected {best_type.value} with {best_confidence:.2%} confidence"
    )
    
    return best_type, best_confidence, best_adapter
