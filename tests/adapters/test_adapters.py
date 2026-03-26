"""
Comprehensive tests for competition adapters.

Advanced Feature: Competition Adapters
Tests are regression-aware with frozen baselines.
"""
import pytest
import numpy as np
import pandas as pd
from adapters import (
    TabularAdapter,
    TimeSeriesAdapter,
    NLPAdapter,
    CompetitionType,
    detect_competition_type,
)


# ── Frozen Test Data for Regression Testing ──────────────────────

np.random.seed(42)
FROZEN_N_SAMPLES = 500
FROZEN_N_FEATURES = 10

# Tabular data
FROZEN_TABULAR_X = np.random.randn(FROZEN_N_SAMPLES, FROZEN_N_FEATURES)
FROZEN_TABULAR_Y = (FROZEN_TABULAR_X[:, 0] + FROZEN_TABULAR_X[:, 1] > 0).astype(int)

# Time series data
FROZEN_TS_DATES = pd.date_range("2020-01-01", periods=FROZEN_N_SAMPLES, freq="D")
FROZEN_TS_X = pd.DataFrame({
    "date": FROZEN_TS_DATES,
    "value": np.random.randn(FROZEN_N_SAMPLES),
})
FROZEN_TS_Y = np.random.randn(FROZEN_N_SAMPLES)

# NLP data
FROZEN_NLP_TEXT = [
    "This is a sample text document",
    "Another example of text data",
    "Natural language processing is fun",
] * (FROZEN_N_SAMPLES // 3 + 1)
FROZEN_NLP_X = pd.DataFrame({
    "text": FROZEN_NLP_TEXT[:FROZEN_N_SAMPLES],
    "numeric": np.random.randn(FROZEN_N_SAMPLES),
})
FROZEN_NLP_Y = np.random.randint(0, 2, FROZEN_N_SAMPLES)


class TestTabularAdapter:
    """Test TabularAdapter."""

    def test_detect_tabular(self):
        """Test tabular data detection."""
        adapter = TabularAdapter()
        
        comp_type, confidence = adapter.detect(FROZEN_TABULAR_X, FROZEN_TABULAR_Y)
        
        assert comp_type == CompetitionType.TABULAR
        assert confidence >= 0.5

    def test_fit_transform_tabular(self):
        """Test tabular data transformation."""
        adapter = TabularAdapter(n_folds=3)
        
        result = adapter.fit_transform(
            X=FROZEN_TABULAR_X,
            y=FROZEN_TABULAR_Y,
        )
        
        assert result.competition_type == CompetitionType.TABULAR
        assert result.X_processed.shape[0] == FROZEN_N_SAMPLES
        assert len(result.feature_names) == FROZEN_N_FEATURES

    def test_get_default_metric_classification(self):
        """Test metric detection for classification."""
        adapter = TabularAdapter()
        y_class = np.array([0, 1, 0, 1, 1, 0])
        
        metric = adapter.get_default_metric(y_class)
        
        assert metric == "auc"

    def test_get_default_metric_regression(self):
        """Test metric detection for regression."""
        adapter = TabularAdapter()
        y_reg = np.random.randn(100)
        
        metric = adapter.get_default_metric(y_reg)
        
        assert metric == "rmse"

    def test_get_default_cv_splitter_stratified(self):
        """Test CV splitter for classification."""
        adapter = TabularAdapter(n_folds=3)
        y_class = np.array([0, 1, 0, 1, 1, 0] * 10)
        
        cv = adapter.get_default_cv_splitter(y_class)
        
        assert cv.n_splits == 3


class TestTimeSeriesAdapter:
    """Test TimeSeriesAdapter."""

    def test_detect_timeseries(self):
        """Test time series detection."""
        adapter = TimeSeriesAdapter()
        
        comp_type, confidence = adapter.detect(FROZEN_TS_X, FROZEN_TS_Y)
        
        assert comp_type == CompetitionType.TIMESERIES
        assert confidence >= 0.3

    def test_fit_transform_timeseries(self):
        """Test time series transformation."""
        adapter = TimeSeriesAdapter(n_folds=3, n_lags=3)
        
        result = adapter.fit_transform(
            X=FROZEN_TS_X,
            y=FROZEN_TS_Y,
        )
        
        assert result.competition_type == CompetitionType.TIMESERIES
        # Should have original + lag + rolling features
        assert result.X_processed.shape[1] > FROZEN_TS_X.shape[1]

    def test_extract_date_features(self):
        """Test date feature extraction."""
        adapter = TimeSeriesAdapter()
        adapter.date_column = "date"
        
        df = adapter._extract_date_features(FROZEN_TS_X.copy(), "date")
        
        # Should have extracted date features
        assert "year" in df.columns
        assert "month" in df.columns
        assert "dayofweek" in df.columns

    def test_generate_lag_features(self):
        """Test lag feature generation."""
        adapter = TimeSeriesAdapter(n_lags=5)
        
        df = adapter._generate_lag_features(
            FROZEN_TS_X.copy(),
            FROZEN_TS_Y,
        )
        
        # Should have lag features
        assert "target_lag_1" in df.columns
        assert "target_lag_5" in df.columns

    def test_get_default_cv_splitter_timeseries(self):
        """Test time-based CV splitter."""
        adapter = TimeSeriesAdapter(n_folds=3)
        
        cv = adapter.get_default_cv_splitter(FROZEN_TS_Y)
        
        assert cv.n_splits == 3


class TestNLPAdapter:
    """Test NLPAdapter."""

    def test_detect_nlp(self):
        """Test NLP data detection."""
        adapter = NLPAdapter()
        
        comp_type, confidence = adapter.detect(FROZEN_NLP_X, FROZEN_NLP_Y)
        
        assert comp_type == CompetitionType.NLP
        assert confidence >= 0.3

    def test_fit_transform_nlp(self):
        """Test NLP data transformation."""
        adapter = NLPAdapter(n_folds=3, max_features=100)
        
        result = adapter.fit_transform(
            X=FROZEN_NLP_X,
            y=FROZEN_NLP_Y,
        )
        
        assert result.competition_type == CompetitionType.NLP
        # Should have text stats + TF-IDF features
        assert result.X_processed.shape[1] > 5

    def test_extract_text_features(self):
        """Test text feature extraction."""
        adapter = NLPAdapter()
        adapter.text_column = "text"
        
        X_features = adapter._extract_text_features(FROZEN_NLP_X)
        
        # Should have 5 text statistics features
        assert X_features.shape[1] == 5

    def test_generate_tfidf_features(self):
        """Test TF-IDF feature generation."""
        adapter = NLPAdapter(max_features=50)
        adapter.text_column = "text"
        
        X_tfidf = adapter._generate_tfidf_features(FROZEN_NLP_X)
        
        # Should have TF-IDF features
        assert X_tfidf.shape[1] <= 50


class TestAutoDetection:
    """Test auto-detection of competition type."""

    def test_detect_tabular_auto(self):
        """Test auto-detection for tabular data."""
        comp_type, confidence, adapter = detect_competition_type(
            FROZEN_TABULAR_X,
            FROZEN_TABULAR_Y,
        )
        
        assert comp_type == CompetitionType.TABULAR
        assert isinstance(adapter, TabularAdapter)

    def test_detect_timeseries_auto(self):
        """Test auto-detection for time series data."""
        comp_type, confidence, adapter = detect_competition_type(
            FROZEN_TS_X,
            FROZEN_TS_Y,
        )
        
        assert comp_type == CompetitionType.TIMESERIES
        assert isinstance(adapter, TimeSeriesAdapter)

    def test_detect_nlp_auto(self):
        """Test auto-detection for NLP data."""
        comp_type, confidence, adapter = detect_competition_type(
            FROZEN_NLP_X,
            FROZEN_NLP_Y,
        )
        
        assert comp_type == CompetitionType.NLP
        assert isinstance(adapter, NLPAdapter)


class TestAdapterResult:
    """Test AdapterResult structure."""

    def test_result_to_dict(self):
        """Test result serialization."""
        from adapters.base import AdapterResult
        from sklearn.model_selection import KFold
        
        result = AdapterResult(
            competition_type=CompetitionType.TABULAR,
            confidence=0.9,
            X_processed=np.random.randn(100, 10),
            y_processed=np.random.randint(0, 2, 100),
            X_test_processed=None,
            feature_names=[f"f{i}" for i in range(10)],
            cv_splitter=KFold(n_splits=5),
            metric="auc",
            metadata={"test": "value"},
        )
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict["competition_type"] == "tabular"
        assert result_dict["confidence"] == 0.9
        assert "n_features" in result_dict


class TestRegressionBaselines:
    """Test regression-aware baselines."""

    def test_frozen_tabular_data_shape(self):
        """Test frozen tabular data has expected shape."""
        assert FROZEN_TABULAR_X.shape == (FROZEN_N_SAMPLES, FROZEN_N_FEATURES)
        assert len(FROZEN_TABULAR_Y) == FROZEN_N_SAMPLES

    def test_frozen_timeseries_data_shape(self):
        """Test frozen time series data has expected shape."""
        assert len(FROZEN_TS_X) == FROZEN_N_SAMPLES
        assert len(FROZEN_TS_Y) == FROZEN_N_SAMPLES

    def test_frozen_nlp_data_shape(self):
        """Test frozen NLP data has expected shape."""
        assert len(FROZEN_NLP_X) == FROZEN_N_SAMPLES
        assert len(FROZEN_NLP_Y) == FROZEN_N_SAMPLES
