# adapters/nlp_adapter.py

"""
NLP competition adapter.

Adapter for text data with text feature extraction.
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from sklearn.model_selection import StratifiedKFold, KFold

from adapters.base import BaseAdapter, AdapterResult, CompetitionType

logger = logging.getLogger(__name__)


class NLPAdapter(BaseAdapter):
    """
    Adapter for NLP/text competitions.
    
    Features:
    - Text length features
    - Word count features
    - Sentiment features (optional)
    - TF-IDF features
    - Text statistics
    """
    
    def __init__(
        self,
        random_state: int = 42,
        n_folds: int = 5,
        max_features: int = 1000,
        text_column: Optional[str] = None,
    ):
        """
        Initialize NLP adapter.
        
        Args:
            random_state: Random seed
            n_folds: Number of CV folds
            max_features: Max TF-IDF features
            text_column: Name of text column
        """
        super().__init__(random_state, n_folds)
        
        self.max_features = max_features
        self.text_column = text_column
        self.tfidf_vectorizer = None
    
    def detect(
        self,
        X: Any,
        y: Optional[Any] = None,
    ) -> Tuple[CompetitionType, float]:
        """
        Detect if data is NLP/text data.
        
        Args:
            X: Feature matrix
            y: Target vector
        
        Returns:
            (CompetitionType.NLP, confidence)
        """
        confidence = 0.3  # Base confidence
        
        # Check for text columns
        if hasattr(X, "columns"):
            for col in X.columns:
                col_lower = str(col).lower()
                
                # Check column name for text indicators
                if any(keyword in col_lower for keyword in ["text", "comment", "review", "description", "title"]):
                    confidence += 0.5
                    self.text_column = col
                    break
                
                # Check if column contains text data
                try:
                    if hasattr(X[col], "dtype") and X[col].dtype == "object":
                        # Sample some values to check if text
                        sample = X[col].dropna().head(10)
                        if sample.apply(lambda x: isinstance(x, str) and len(x) > 20).any():
                            confidence += 0.4
                            self.text_column = col
                            break
                except:
                    pass
        
        return CompetitionType.NLP, min(confidence, 1.0)
    
    def fit_transform(
        self,
        X: Any,
        y: Optional[Any] = None,
        X_test: Optional[Any] = None,
    ) -> AdapterResult:
        """
        Fit adapter and transform NLP data.
        
        Args:
            X: Training feature matrix
            y: Training target
            X_test: Test feature matrix
        
        Returns:
            AdapterResult with processed data
        """
        logger.info("[NLPAdapter] Processing NLP data")
        
        # Convert to DataFrame if needed
        if not hasattr(X, "columns"):
            X = X.copy()
            if hasattr(X, "toarray"):
                X = X.toarray()
            X = self._array_to_dataframe(X)
        
        # Extract text features
        X_text_features = self._extract_text_features(X)
        
        # Generate TF-IDF features
        X_tfidf = self._generate_tfidf_features(X, X_test)
        
        # Combine features
        X_processed = np.hstack([X_text_features, X_tfidf])
        
        # Process test set
        X_test_processed = None
        if X_test is not None:
            if not hasattr(X_test, "columns"):
                X_test = self._array_to_dataframe(X_test)
            X_test_text = self._extract_text_features(X_test)
            X_test_tfidf = self.tfidf_vectorizer.transform(X_test[self.text_column])
            X_test_processed = np.hstack([X_test_text, X_test_tfidf.toarray()])
        
        # Get feature names
        text_feature_names = [f"text_{i}" for i in range(X_text_features.shape[1])]
        tfidf_feature_names = [f"tfidf_{i}" for i in range(X_tfidf.shape[1])]
        feature_names = text_feature_names + tfidf_feature_names
        
        # Get CV splitter
        cv_splitter = self.get_default_cv_splitter(y)
        
        # Get metric
        metric = self.get_default_metric(y)
        
        # Create result
        result = AdapterResult(
            competition_type=CompetitionType.NLP,
            confidence=0.8,
            X_processed=X_processed,
            y_processed=y,
            X_test_processed=X_test_processed,
            feature_names=feature_names,
            cv_splitter=cv_splitter,
            metric=metric,
            metadata={
                "n_samples": X_processed.shape[0],
                "n_features": X_processed.shape[1],
                "adapter": "nlp",
                "text_column": self.text_column,
                "max_tfidf_features": self.max_features,
            },
        )
        
        self.adapter_history.append(result)
        
        logger.info(
            f"[NLPAdapter] Complete -- {X_processed.shape[0]} samples, "
            f"{X_processed.shape[1]} features (text + TF-IDF), metric: {metric}"
        )
        
        return result
    
    def _array_to_dataframe(self, X: Any) -> Any:
        """Convert array to DataFrame."""
        import pandas as pd
        return pd.DataFrame(X)
    
    def _extract_text_features(self, df: Any) -> np.ndarray:
        """Extract statistical features from text."""
        import pandas as pd
        
        if self.text_column is None or self.text_column not in df.columns:
            return np.zeros((len(df), 5))
        
        text = df[self.text_column].fillna("")
        
        features = []
        
        # Text length
        features.append(text.str.len())
        
        # Word count
        features.append(text.str.split().str.len())
        
        # Average word length
        features.append(text.str.len() / (text.str.split().str.len() + 1))
        
        # Sentence count (rough estimate)
        features.append(text.str.count(r'[.!?]') + 1)
        
        # Uppercase ratio
        features.append(text.str.count(r'[A-Z]') / (text.str.len() + 1))
        
        X_features = pd.DataFrame(features).T.values
        
        logger.debug(f"[NLPAdapter] Extracted {X_features.shape[1]} text statistics features")
        
        return X_features
    
    def _generate_tfidf_features(
        self,
        df: Any,
        df_test: Optional[Any] = None,
    ) -> np.ndarray:
        """Generate TF-IDF features from text."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        if self.text_column is None or self.text_column not in df.columns:
            return np.zeros((len(df), 1))
        
        # Create TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
        )
        
        # Fit on training data
        X_tfidf = self.tfidf_vectorizer.fit_transform(df[self.text_column])
        
        logger.debug(f"[NLPAdapter] Generated {X_tfidf.shape[1]} TF-IDF features")
        
        return X_tfidf.toarray()
    
    def get_default_metric(self, y: Optional[Any] = None) -> str:
        """Get default metric for NLP."""
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
        """Get default CV splitter for NLP."""
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
