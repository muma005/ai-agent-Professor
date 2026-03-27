"""
Contract Tests for Professor Simulator

These tests verify the core contracts of the simulator:
1. Data Splitter: Determinism, isolation, distribution preservation
2. Simulated Leaderboard: Public/private separation, submission limits
3. Scorers: Accuracy against sklearn reference implementations

File: simulator/tests/test_simulator_contract.py (IMMUTABLE)

These tests must pass for the simulator to be considered valid.
"""

import pytest
import numpy as np
import polars as pl
from pathlib import Path
from tempfile import TemporaryDirectory
from datetime import date
from sklearn.metrics import (
    accuracy_score, roc_auc_score, mean_squared_error,
    log_loss, f1_score, mean_absolute_error
)

from simulator.competition_registry import CompetitionEntry
from simulator.data_splitter import split_competition_data
from simulator.leaderboard import SimulatedLeaderboard
from simulator.scorers import (
    get_scorer, score_predictions,
    _balanced_log_loss, _rmsle, _quadratic_weighted_kappa
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def binary_classification_entry():
    """Binary classification competition entry for testing."""
    return CompetitionEntry(
        slug="test-binary",
        title="Test Binary Classification",
        kaggle_id="test-binary",
        task_type="binary",
        target_column="target",
        id_column="id",
        metric="accuracy",
        metric_direction="maximize",
        lb_percentiles={10: 0.85, 25: 0.82, 50: 0.80, 75: 0.78},
        gold_threshold=0.85,
        silver_threshold=0.82,
        bronze_threshold=0.80,
        total_teams=1000,
        split_strategy="stratified",
        test_ratio=0.40,
        public_ratio=0.30,
        random_seed=42,
    )


@pytest.fixture
def regression_entry():
    """Regression competition entry for testing."""
    return CompetitionEntry(
        slug="test-regression",
        title="Test Regression",
        kaggle_id="test-regression",
        task_type="regression",
        target_column="target",
        id_column="id",
        metric="rmsle",
        metric_direction="minimize",
        lb_percentiles={10: 0.12, 25: 0.14, 50: 0.16, 75: 0.18},
        gold_threshold=0.12,
        silver_threshold=0.14,
        bronze_threshold=0.16,
        total_teams=500,
        split_strategy="stratified",
        test_ratio=0.40,
        public_ratio=0.30,
        random_seed=42,
    )


@pytest.fixture
def sample_binary_data():
    """Generate sample binary classification data."""
    np.random.seed(42)
    n_samples = 1000
    
    # Create balanced binary data
    n_positive = n_samples // 2
    data = {
        "id": list(range(n_samples)),
        "feature1": np.random.randn(n_samples),
        "feature2": np.random.randn(n_samples),
        "target": [1] * n_positive + [0] * (n_samples - n_positive),
    }
    
    # Shuffle
    indices = np.random.permutation(n_samples)
    data["id"] = [data["id"][i] for i in indices]
    data["feature1"] = data["feature1"][indices]
    data["feature2"] = data["feature2"][indices]
    data["target"] = [data["target"][i] for i in indices]
    
    return pl.DataFrame(data)


@pytest.fixture
def sample_regression_data():
    """Generate sample regression data."""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        "id": list(range(n_samples)),
        "feature1": np.random.randn(n_samples),
        "feature2": np.random.randn(n_samples),
        "target": np.abs(np.random.randn(n_samples) * 100 + 500),  # Positive values
    }
    
    return pl.DataFrame(data)


# =============================================================================
# Data Splitter Contract Tests
# =============================================================================

class TestDataSplitter:
    """Tests for data_splitter.py"""
    
    def test_split_is_deterministic(self, binary_classification_entry, sample_binary_data):
        """Same data + same seed = exact same split."""
        with TemporaryDirectory() as tmpdir:
            # Save data
            data_path = Path(tmpdir) / "data.csv"
            sample_binary_data.write_csv(data_path)
            
            # Split twice
            entry1 = binary_classification_entry
            split1 = split_competition_data(str(data_path), entry1, force=True)
            
            split2 = split_competition_data(str(data_path), entry1, force=True)
            
            # Verify hashes match
            assert split1.train_hash == split2.train_hash
            assert split1.public_hash == split2.public_hash
            assert split1.private_hash == split2.private_hash
    
    def test_no_row_appears_in_multiple_partitions(
        self, binary_classification_entry, sample_binary_data
    ):
        """Train, public, and private sets must be disjoint."""
        with TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "data.csv"
            sample_binary_data.write_csv(data_path)
            
            split = split_competition_data(str(data_path), binary_classification_entry, force=True)
            
            # Load IDs
            train_ids = set(pl.read_csv(split.train_path)["id"].to_list())
            public_ids = set(pl.read_csv(split.public_labels_path)["id"].to_list())
            private_ids = set(pl.read_csv(split.private_labels_path)["id"].to_list())
            
            # Verify no overlap
            assert len(train_ids & public_ids) == 0
            assert len(train_ids & private_ids) == 0
            assert len(public_ids & private_ids) == 0
            
            # Verify all IDs accounted for
            all_ids = train_ids | public_ids | private_ids
            assert len(all_ids) == len(sample_binary_data)
    
    def test_target_distribution_preserved(
        self, binary_classification_entry, sample_binary_data
    ):
        """Stratified split should preserve target distribution."""
        with TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "data.csv"
            sample_binary_data.write_csv(data_path)
            
            split = split_competition_data(str(data_path), binary_classification_entry, force=True)
            
            # Compute target distributions
            train_df = pl.read_csv(split.train_path)
            public_df = pl.read_csv(split.public_labels_path)
            private_df = pl.read_csv(split.private_labels_path)
            
            train_mean = train_df["target"].mean()
            public_mean = public_df["target"].mean()
            private_mean = private_df["target"].mean()
            
            # Should be within 5% of each other
            assert abs(train_mean - public_mean) < 0.05
            assert abs(train_mean - private_mean) < 0.05
    
    def test_test_file_has_no_target_column(
        self, binary_classification_entry, sample_binary_data
    ):
        """Test file (features only) must not contain target column."""
        with TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "data.csv"
            sample_binary_data.write_csv(data_path)
            
            split = split_competition_data(str(data_path), binary_classification_entry, force=True)
            
            test_df = pl.read_csv(split.test_path)
            assert binary_classification_entry.target_column not in test_df.columns
    
    def test_ratios_correct(self, binary_classification_entry, sample_binary_data):
        """Train ≈ 60%, public ≈ 12%, private ≈ 28%."""
        with TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "data.csv"
            sample_binary_data.write_csv(data_path)
            
            split = split_competition_data(str(data_path), binary_classification_entry, force=True)
            
            n_total = len(sample_binary_data)
            n_train = split.n_train
            n_public = split.n_public
            n_private = split.n_private
            
            # Check ratios (within 5% tolerance)
            assert abs(n_train / n_total - 0.60) < 0.05
            assert abs(n_public / n_total - 0.12) < 0.05
            assert abs(n_private / n_total - 0.28) < 0.05
            
            # Check sum
            assert n_train + n_public + n_private == n_total
    
    def test_temporal_split(self, sample_binary_data):
        """Temporal split: last 40% of sorted data forms test."""
        with TemporaryDirectory() as tmpdir:
            # Add date column
            from datetime import date, timedelta
            dates = [date(2023, 1, 1) + timedelta(days=i) for i in range(len(sample_binary_data))]
            data = sample_binary_data.with_columns(
                pl.Series(dates).alias("date")
            )
            
            data_path = Path(tmpdir) / "data.csv"
            data.write_csv(data_path)
            
            entry = CompetitionEntry(
                slug="test-temporal",
                title="Test Temporal",
                kaggle_id="test-temporal",
                task_type="binary",
                target_column="target",
                id_column="id",
                metric="accuracy",
                metric_direction="maximize",
                lb_percentiles={10: 0.85},
                gold_threshold=0.85,
                silver_threshold=0.82,
                bronze_threshold=0.80,
                total_teams=1000,
                split_strategy="temporal",
                split_column="date",
                test_ratio=0.40,
                public_ratio=0.30,
                random_seed=42,
            )
            
            split = split_competition_data(str(data_path), entry, force=True)
            
            # Verify temporal ordering
            # Read from test.csv (which has all columns except target) and label files
            train_df = pl.read_csv(split.train_path)
            test_df = pl.read_csv(split.test_path)  # Has date column
            public_labels = pl.read_csv(split.public_labels_path)
            private_labels = pl.read_csv(split.private_labels_path)
            
            # Join test with labels to get dates for public/private
            public_df = test_df.join(public_labels, on="id")
            private_df = test_df.join(private_labels, on="id")
            
            train_max_date = max(train_df["date"].to_list())
            public_min_date = min(public_df["date"].to_list())
            private_min_date = min(private_df["date"].to_list())
            test_min_date = min(public_min_date, private_min_date)
            
            # All train dates should be before test dates
            assert train_max_date <= test_min_date


# =============================================================================
# Simulated Leaderboard Contract Tests
# =============================================================================

class TestSimulatedLeaderboard:
    """Tests for leaderboard.py"""
    
    @pytest.fixture
    def setup_leaderboard(self, binary_classification_entry, sample_binary_data):
        """Create a leaderboard with split data."""
        with TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "data.csv"
            sample_binary_data.write_csv(data_path)
            
            split = split_competition_data(str(data_path), binary_classification_entry, force=True)
            lb = SimulatedLeaderboard(binary_classification_entry, split)
            
            yield lb, split, sample_binary_data
    
    def test_submit_returns_public_score_only(self, setup_leaderboard):
        """Submit returns public score, private score is hidden (None)."""
        lb, split, _ = setup_leaderboard
        
        # Create a dummy submission with binary predictions
        test_df = pl.read_csv(split.test_path)
        # For binary classification, predictions should be 0 or 1
        sub_df = test_df.with_columns(pl.lit(1).alias("target"))  # All positive
        sub_path = Path(split.test_path).parent / "submission.csv"
        sub_df.write_csv(sub_path)
        
        result = lb.submit(str(sub_path))
        
        assert result.success is True
        assert result.public_score is not None
        assert result.private_score is None  # HIDDEN
    
    def test_daily_limit_enforced(self, setup_leaderboard):
        """Daily submission limit must be enforced."""
        lb, split, _ = setup_leaderboard
        
        # Create a dummy submission with binary predictions
        test_df = pl.read_csv(split.test_path)
        sub_df = test_df.with_columns(pl.lit(1).alias("target"))
        sub_path = Path(split.test_path).parent / "submission.csv"
        sub_df.write_csv(sub_path)
        
        # Submit 5 times (limit)
        for i in range(5):
            result = lb.submit(str(sub_path))
            assert result.success is True
        
        # 6th submission should fail
        result = lb.submit(str(sub_path))
        assert result.success is False
        assert "limit" in result.error.lower()
    
    def test_advance_day_resets_limit(self, setup_leaderboard):
        """Advancing day should reset submission counter."""
        lb, split, _ = setup_leaderboard
        
        # Create a dummy submission with binary predictions
        test_df = pl.read_csv(split.test_path)
        sub_df = test_df.with_columns(pl.lit(1).alias("target"))
        sub_path = Path(split.test_path).parent / "submission.csv"
        sub_df.write_csv(sub_path)
        
        # Submit 5 times
        for i in range(5):
            lb.submit(str(sub_path))
        
        # Advance day
        lb.advance_day()
        
        # Should be able to submit again
        result = lb.submit(str(sub_path))
        assert result.success is True
    
    def test_competition_end_reveals_private(self, setup_leaderboard):
        """competition_end() reveals private scores and computes medal."""
        lb, split, _ = setup_leaderboard
        
        # Create a dummy submission
        sub_df = pl.read_csv(split.test_path)
        sub_df = sub_df.with_columns(pl.lit(0.5).alias("target"))
        sub_path = Path(split.test_path).parent / "submission.csv"
        sub_df.write_csv(sub_path)
        
        lb.submit(str(sub_path))
        result = lb.competition_end()
        
        assert result.best_private_score is not None
        assert result.medal in ["gold", "silver", "bronze", "none"]
    
    def test_perfect_submission_gets_perfect_score(self, setup_leaderboard):
        """Submitting actual labels should get perfect score."""
        lb, split, _ = setup_leaderboard
        
        # Create perfect submission using ALL test IDs (public + private)
        # We need to join test.csv with both public and private labels
        test_df = pl.read_csv(split.test_path)
        public_labels = pl.read_csv(split.public_labels_path)
        private_labels = pl.read_csv(split.private_labels_path)
        
        # Combine public and private labels
        all_labels = pl.concat([public_labels, private_labels])
        
        # Join to get the correct target values for all test IDs
        sub_df = test_df.join(all_labels, on="id")[["id", "target"]]
        sub_path = Path(split.test_path).parent / "submission_perfect.csv"
        sub_df.write_csv(sub_path)
        
        # For binary classification with accuracy, perfect = 1.0
        result = lb.submit(str(sub_path))
        
        assert result.success is True
        assert result.public_score == 1.0
    
    def test_scorer_matches_sklearn(self, setup_leaderboard):
        """Scorer must produce identical output to sklearn."""
        lb, split, _ = setup_leaderboard
        
        # Create test data
        y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1])
        y_pred_binary = np.array([0, 0, 1, 1, 0, 0, 1, 1, 1, 1])  # Binary predictions
        y_pred_proba = np.array([0.1, 0.3, 0.6, 0.8, 0.9, 0.2, 0.7, 0.4, 0.85, 0.95])
        
        # Test accuracy (with binary predictions)
        sklearn_acc = accuracy_score(y_true, y_pred_binary)
        our_acc = score_predictions(y_true, y_pred_binary, "accuracy", "maximize")
        assert abs(sklearn_acc - our_acc) < 1e-10
        
        # Test AUC (with probabilities)
        sklearn_auc = roc_auc_score(y_true, y_pred_proba)
        our_auc = score_predictions(y_true, y_pred_proba, "auc", "maximize")
        assert abs(sklearn_auc - our_auc) < 1e-10
        
        # Test log loss (with probabilities)
        sklearn_ll = log_loss(y_true, y_pred_proba)
        our_ll = score_predictions(y_true, y_pred_proba, "log_loss", "maximize")
        assert abs(sklearn_ll - our_ll) < 1e-6


# =============================================================================
# Scorers Contract Tests
# =============================================================================

class TestScorers:
    """Tests for scorers.py"""
    
    def test_accuracy_matches_sklearn(self):
        """Accuracy scorer must match sklearn exactly."""
        y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0])
        
        sklearn_score = accuracy_score(y_true, y_pred)
        our_score = score_predictions(y_true, y_pred, "accuracy", "maximize")
        
        assert abs(sklearn_score - our_score) < 1e-10
    
    def test_auc_matches_sklearn(self):
        """AUC scorer must match sklearn exactly."""
        y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1])
        y_pred = np.array([0.1, 0.3, 0.6, 0.8, 0.9, 0.2, 0.7, 0.4, 0.85, 0.95])
        
        sklearn_score = roc_auc_score(y_true, y_pred)
        our_score = score_predictions(y_true, y_pred, "auc", "maximize")
        
        assert abs(sklearn_score - our_score) < 1e-10
    
    def test_f1_matches_sklearn(self):
        """F1 scorer must match sklearn exactly."""
        y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 0, 1, 1, 1, 1])
        
        sklearn_score = f1_score(y_true, y_pred, average="binary")
        our_score = score_predictions(y_true, y_pred, "f1", "maximize")
        
        assert abs(sklearn_score - our_score) < 1e-10
    
    def test_rmse_matches_sklearn(self):
        """RMSE scorer must match sklearn exactly."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.2, 2.9, 4.1, 4.8])
        
        sklearn_score = np.sqrt(mean_squared_error(y_true, y_pred))
        our_score = score_predictions(y_true, y_pred, "rmse", "minimize")
        
        assert abs(sklearn_score - our_score) < 1e-10
    
    def test_rmsle_implementation(self):
        """RMSLE must handle log transform correctly."""
        y_true = np.array([100, 200, 300, 400, 500])
        y_pred = np.array([110, 210, 290, 410, 480])
        
        # Manual RMSLE
        y_true_log = np.log1p(y_true)
        y_pred_log = np.log1p(y_pred)
        expected = np.sqrt(np.mean((y_true_log - y_pred_log) ** 2))
        
        our_score = score_predictions(y_true, y_pred, "rmsle", "minimize")
        
        assert abs(expected - our_score) < 1e-10
    
    def test_mae_matches_sklearn(self):
        """MAE scorer must match sklearn exactly."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.2, 2.9, 4.1, 4.8])
        
        sklearn_score = mean_absolute_error(y_true, y_pred)
        our_score = score_predictions(y_true, y_pred, "mae", "minimize")
        
        assert abs(sklearn_score - our_score) < 1e-10
    
    def test_balanced_log_loss_symmetry(self):
        """Balanced log loss should weight classes equally."""
        # Imbalanced dataset: 90% negative, 10% positive
        y_true = np.array([0] * 90 + [1] * 10)
        y_pred = np.array([0.3] * 90 + [0.7] * 10)
        
        score = _balanced_log_loss(y_true, y_pred)
        
        # Score should be finite and positive
        assert 0 < score < 10
        
        # Perfect predictions should give low loss
        y_pred_perfect = np.array([0.01] * 90 + [0.99] * 10)
        score_perfect = _balanced_log_loss(y_true, y_pred_perfect)
        
        assert score_perfect < score
    
    def test_qwk_perfect_agreement(self):
        """QWK should be 1.0 for perfect agreement."""
        y_true = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
        y_pred = y_true.copy()
        
        score = _quadratic_weighted_kappa(y_true, y_pred)
        
        assert abs(score - 1.0) < 1e-10
    
    def test_qwk_complete_disagreement(self):
        """QWK should be negative for complete disagreement."""
        y_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        y_pred = np.array([2, 2, 2, 0, 0, 0, 1, 1, 1])  # Completely wrong
        
        score = _quadratic_weighted_kappa(y_true, y_pred)
        
        assert score < 0


# =============================================================================
# Benchmark Report Contract Tests
# =============================================================================

class TestBenchmarkReport:
    """Tests for report_generator.py"""
    
    def test_version_comparison_detects_regression(self):
        """Regression detection should flag score drops > 1 percentile."""
        from simulator.report_generator import _compare_versions, BenchmarkReport
        
        # Create mock previous report
        previous = BenchmarkReport(
            run_id="test_v1",
            professor_version="1.0.0",
            timestamp="2024-01-01T00:00:00",
            n_competitions=1,
            aggregate_metrics={"median_percentile": 20.0},
            per_competition=[
                {
                    "slug": "test-comp",
                    "private_percentile": 20.0,  # Better (lower percentile = better rank)
                    "medal": "bronze",
                }
            ],
        )
        
        # Save and compare
        with TemporaryDirectory() as tmpdir:
            prev_path = Path(tmpdir) / "prev.json"
            previous.save(str(prev_path))
            
            # Current results (regression: worse by 5 percentiles)
            # Higher percentile = worse rank (e.g., 25% means you're behind more people)
            current_results = [
                {
                    "slug": "test-comp",
                    "private_percentile": 25.0,  # Worse (higher = worse rank)
                    "medal": "none",
                }
            ]
            
            comparison = _compare_versions(current_results, str(prev_path))
            
            # Delta = previous - current = 20 - 25 = -5 (negative = degraded)
            # Lower percentile = better rank, so 25% is worse than 20%
            assert "test-comp" in comparison.degraded_competitions
    
    def test_component_attribution_populated(self):
        """Component attribution should be populated when provided."""
        from simulator.report_generator import generate_benchmark_report
        
        results = [
            {
                "slug": "test-comp",
                "task_type": "binary",
                "domain": "test",
                "metric": "accuracy",
                "cv_score": 0.8,
                "public_score": 0.79,
                "private_score": 0.78,
                "cv_public_gap": 0.01,
                "cv_private_gap": 0.02,
                "public_percentile": 20.0,
                "private_percentile": 25.0,
                "shakeup": 5.0,
                "medal": "bronze",
                "total_submissions": 3,
                "runtime_seconds": 100.0,
                "winning_model": "lgbm",
                "n_features_final": 20,
                "domain_features_generated": 5,
                "domain_features_kept": 3,
            }
        ]
        
        component_stats = {
            "shift_detector_triggered": 2,
            "shift_detector_helped": 1,
            "domain_features_generated": 15,
            "domain_features_kept": 8,
            "creative_features_generated": 23,
            "creative_features_kept": 6,
            "postprocess_improved": 7,
            "postprocess_delta_mean": 0.004,
            "pseudo_label_applied": 3,
            "pseudo_label_helped": 2,
            "pseudo_label_reverted": 1,
            "critic_critical_count": 1,
            "critic_replan_count": 1,
        }
        
        report = generate_benchmark_report(
            results=results,
            professor_version="2.0.0",
            component_stats=component_stats,
        )
        
        assert report.component_attribution is not None
        assert report.component_attribution["domain_features_generated"] == 15


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
