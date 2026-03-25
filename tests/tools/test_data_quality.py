"""
Comprehensive data quality tests.

FLAW-5.6: Data Quality Tests
Tests are regression-aware with frozen baselines.
"""
import pytest
import polars as pl
import numpy as np
from tools.data_quality import (
    DataQualityValidator,
    DataQualityIssue,
    DataQualityReport,
    validate_data_quality,
)


# ── Frozen Test Data for Regression Testing ──────────────────────

FROZEN_CLEAN_DATA = pl.DataFrame({
    "feature1": [1.0, 2.0, 3.0, 4.0, 5.0] * 20,
    "feature2": [2.0, 4.0, 6.0, 8.0, 10.0] * 20,
    "feature3": ["A", "B", "C", "D", "E"] * 20,
    "target": [0, 1, 0, 1, 0] * 20,
})

FROZEN_LEAKAGE_DATA = pl.DataFrame({
    "feature1": [1.0, 2.0, 3.0, 4.0, 5.0] * 20,
    "leak_feature": [0, 1, 0, 1, 0] * 20,  # Perfect correlation with target
    "target": [0, 1, 0, 1, 0] * 20,
})

FROZEN_ID_DATA = pl.DataFrame({
    "id": list(range(100)),  # 100% unique
    "feature1": np.random.randn(100),
    "target": np.random.randint(0, 2, 100),
})

FROZEN_CONSTANT_DATA = pl.DataFrame({
    "constant": [5.0] * 100,
    "feature1": np.random.randn(100),
    "target": np.random.randint(0, 2, 100),
})

FROZEN_MISSING_DATA = pl.DataFrame({
    "feature1": [1.0, 2.0, None, 4.0, 5.0] * 20,
    "feature2": [None] * 60 + list(range(40)),  # 60% missing
    "target": [0, 1, 0, 1, 0] * 20,
})


class TestDataQualityValidatorInit:
    """Test DataQualityValidator initialization."""

    def test_default_thresholds(self):
        """Test default threshold values."""
        validator = DataQualityValidator()
        
        assert validator.leakage_threshold == 0.9
        assert validator.uniqueness_threshold == 0.95
        assert validator.missing_threshold == 0.5
        assert validator.outlier_std_threshold == 3.0

    def test_custom_thresholds(self):
        """Test custom threshold settings."""
        validator = DataQualityValidator(
            leakage_threshold=0.8,
            uniqueness_threshold=0.9,
            missing_threshold=0.3,
            outlier_std_threshold=2.5,
        )
        
        assert validator.leakage_threshold == 0.8
        assert validator.uniqueness_threshold == 0.9


class TestTargetLeakageDetection:
    """Test target leakage detection."""

    def test_detect_perfect_leakage(self):
        """Test detection of perfect target leakage."""
        validator = DataQualityValidator(leakage_threshold=0.8)
        
        report = validator.validate(
            df=FROZEN_LEAKAGE_DATA,
            target_col="target",
        )
        
        leakage_issues = [i for i in report.issues if i.issue_type == "target_leakage"]
        
        assert len(leakage_issues) > 0
        assert "leak_feature" in leakage_issues[0].affected_columns
        assert leakage_issues[0].severity == "critical"

    def test_no_leakage_clean_data(self):
        """Test no false positives on clean data."""
        validator = DataQualityValidator()
        
        report = validator.validate(
            df=FROZEN_CLEAN_DATA,
            target_col="target",
        )
        
        leakage_issues = [i for i in report.issues if i.issue_type == "target_leakage"]
        
        assert len(leakage_issues) == 0

    def test_no_target_col(self):
        """Test validation without target column."""
        validator = DataQualityValidator()
        
        report = validator.validate(
            df=FROZEN_CLEAN_DATA,
            target_col=None,
        )
        
        leakage_issues = [i for i in report.issues if i.issue_type == "target_leakage"]
        
        assert len(leakage_issues) == 0


class TestIDColumnDetection:
    """Test ID column detection."""

    def test_detect_known_id_columns(self):
        """Test detection of known ID columns."""
        validator = DataQualityValidator()
        
        report = validator.validate(
            df=FROZEN_ID_DATA,
            id_cols=["id"],
            target_col="target",
        )
        
        id_issues = [i for i in report.issues if i.issue_type == "id_as_feature"]
        
        assert len(id_issues) > 0
        assert "id" in id_issues[0].affected_columns
        assert id_issues[0].severity == "critical"

    def test_detect_id_by_name(self):
        """Test detection of ID columns by name pattern."""
        df = pl.DataFrame({
            "user_id": list(range(100)),
            "feature1": np.random.randn(100),
            "target": np.random.randint(0, 2, 100),
        })
        
        validator = DataQualityValidator()
        
        report = validator.validate(df=df, target_col="target")
        
        # Should detect potential ID by name
        id_issues = [i for i in report.issues if "id" in i.issue_type.lower()]
        
        assert len(id_issues) > 0


class TestConstantFeatureDetection:
    """Test constant feature detection."""

    def test_detect_constant_features(self):
        """Test detection of constant features."""
        validator = DataQualityValidator()
        
        report = validator.validate(df=FROZEN_CONSTANT_DATA)
        
        constant_issues = [i for i in report.issues if "constant" in i.issue_type]
        
        assert len(constant_issues) > 0
        assert "constant" in constant_issues[0].affected_columns[0]
        assert constant_issues[0].severity == "medium"

    def test_no_constant_features(self):
        """Test no false positives on varied data."""
        validator = DataQualityValidator()
        
        report = validator.validate(df=FROZEN_CLEAN_DATA)
        
        constant_issues = [i for i in report.issues if "constant" in i.issue_type]
        
        assert len(constant_issues) == 0


class TestDuplicateDetection:
    """Test duplicate row detection."""

    def test_detect_duplicates(self):
        """Test detection of duplicate rows."""
        df = pl.DataFrame({
            "feature1": [1.0, 1.0, 2.0, 2.0, 3.0],
            "feature2": [5.0, 5.0, 6.0, 6.0, 7.0],
        })
        
        validator = DataQualityValidator()
        
        report = validator.validate(df=df)
        
        duplicate_issues = [i for i in report.issues if i.issue_type == "duplicate_rows"]
        
        assert len(duplicate_issues) > 0
        assert duplicate_issues[0].affected_rows == 2  # 2 duplicates

    def test_no_duplicates(self):
        """Test no false positives on unique data."""
        # Create truly unique data
        np.random.seed(42)
        df = pl.DataFrame({
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
            "feature3": np.random.choice(["A", "B", "C", "D", "E"], 100),
            "target": np.random.randint(0, 2, 100),
        })
        
        validator = DataQualityValidator()
        
        report = validator.validate(df=df)
        
        duplicate_issues = [i for i in report.issues if i.issue_type == "duplicate_rows"]
        
        # With random data, duplicates are unlikely but possible
        # Just check severity is not critical
        for issue in duplicate_issues:
            assert issue.severity != "critical"


class TestMissingValueDetection:
    """Test missing value detection."""

    def test_detect_missing_values(self):
        """Test detection of missing values."""
        validator = DataQualityValidator()
        
        report = validator.validate(df=FROZEN_MISSING_DATA)
        
        missing_issues = [i for i in report.issues if i.issue_type == "missing_values"]
        
        assert len(missing_issues) > 0
        assert "feature2" in [i.affected_columns[0] for i in missing_issues]

    def test_high_missing_severity(self):
        """Test high missing rate gets high severity."""
        validator = DataQualityValidator(missing_threshold=0.5)
        
        report = validator.validate(df=FROZEN_MISSING_DATA)
        
        missing_issues = [i for i in report.issues if i.issue_type == "missing_values"]
        
        # feature2 has 60% missing - should be high severity
        feature2_issue = next(
            (i for i in missing_issues if "feature2" in i.affected_columns),
            None,
        )
        
        assert feature2_issue is not None
        assert feature2_issue.severity == "high"


class TestOutlierDetection:
    """Test outlier detection."""

    def test_detect_outliers(self):
        """Test detection of outliers."""
        # Create data with clear outliers
        values = list(range(100)) + [1000, 2000]  # 2 extreme outliers
        df = pl.DataFrame({
            "feature1": values,
            "target": [0, 1] * 51,
        })
        
        validator = DataQualityValidator(outlier_std_threshold=3.0)
        
        report = validator.validate(df=df, target_col="target")
        
        outlier_issues = [i for i in report.issues if i.issue_type == "outliers"]
        
        assert len(outlier_issues) > 0
        assert outlier_issues[0].affected_rows >= 2

    def test_no_outliers_normal_data(self):
        """Test no false positives on normal data."""
        np.random.seed(42)
        df = pl.DataFrame({
            "feature1": np.random.randn(1000),
            "target": np.random.randint(0, 2, 1000),
        })
        
        validator = DataQualityValidator()
        
        report = validator.validate(df=df, target_col="target")
        
        outlier_issues = [i for i in report.issues if i.issue_type == "outliers"]
        
        # May have some outliers in normal data, but should be low severity
        for issue in outlier_issues:
            assert issue.severity in ["low", "medium"]


class TestDataQualityReport:
    """Test DataQualityReport structure."""

    def test_report_structure(self):
        """Test report has all required fields."""
        validator = DataQualityValidator()
        
        report = validator.validate(
            df=FROZEN_CLEAN_DATA,
            target_col="target",
            dataset_name="test_data",
        )
        
        assert report.dataset_name == "test_data"
        assert report.n_rows == 100
        assert report.n_columns == 4
        assert isinstance(report.passed, bool)
        assert isinstance(report.issues, list)
        assert isinstance(report.summary, dict)

    def test_report_passed_logic(self):
        """Test report.passed is False for critical/high issues."""
        # Create data with critical issue
        df = pl.DataFrame({
            "id": list(range(100)),
            "feature1": np.random.randn(100),
            "target": np.random.randint(0, 2, 100),
        })
        
        validator = DataQualityValidator()
        
        report = validator.validate(
            df=df,
            id_cols=["id"],
            target_col="target",
        )
        
        # Should fail due to ID column (critical)
        assert report.passed is False
        assert report.critical_issues > 0

    def test_report_to_dict(self):
        """Test report serialization."""
        validator = DataQualityValidator()
        
        report = validator.validate(df=FROZEN_CLEAN_DATA)
        
        report_dict = report.to_dict()
        
        assert isinstance(report_dict, dict)
        assert "dataset_name" in report_dict
        assert "issues" in report_dict
        assert "summary" in report_dict


class TestConvenienceFunction:
    """Test validate_data_quality convenience function."""

    def test_validate_data_quality_function(self):
        """Test convenience function works."""
        report = validate_data_quality(
            df=FROZEN_CLEAN_DATA,
            target_col="target",
        )
        
        assert isinstance(report, DataQualityReport)
        assert report.n_rows == 100

    def test_validate_with_custom_thresholds(self):
        """Test convenience function with custom thresholds."""
        report = validate_data_quality(
            df=FROZEN_CLEAN_DATA,
            target_col="target",
            leakage_threshold=0.8,
        )
        
        assert isinstance(report, DataQualityReport)


class TestRegressionBaselines:
    """Test regression-aware baselines are maintained."""

    def test_frozen_clean_data_rows(self):
        """Test frozen clean data has expected rows."""
        assert len(FROZEN_CLEAN_DATA) == 100

    def test_frozen_leakage_data_structure(self):
        """Test frozen leakage data structure."""
        assert "leak_feature" in FROZEN_LEAKAGE_DATA.columns
        assert "target" in FROZEN_LEAKAGE_DATA.columns
        assert len(FROZEN_LEAKAGE_DATA) == 100

    def test_frozen_id_data_uniqueness(self):
        """Test frozen ID data has unique IDs."""
        n_unique = FROZEN_ID_DATA["id"].n_unique()
        n_total = len(FROZEN_ID_DATA)
        
        assert n_unique == n_total  # 100% unique

    def test_frozen_constant_data_constant(self):
        """Test frozen constant data has constant column."""
        n_unique = FROZEN_CONSTANT_DATA["constant"].n_unique()
        
        assert n_unique == 1  # Only 1 unique value

    def test_frozen_missing_data_missing_rate(self):
        """Test frozen missing data has expected missing rate."""
        null_count = FROZEN_MISSING_DATA["feature2"].null_count()
        total = len(FROZEN_MISSING_DATA)
        
        assert null_count == 60  # 60 missing
        assert null_count / total == 0.6  # 60% missing


class TestIssueRecommendations:
    """Test issue recommendations are provided."""

    def test_leakage_recommendation(self):
        """Test leakage issues have recommendations."""
        validator = DataQualityValidator(leakage_threshold=0.8)
        
        report = validator.validate(
            df=FROZEN_LEAKAGE_DATA,
            target_col="target",
        )
        
        leakage_issues = [i for i in report.issues if i.issue_type == "target_leakage"]
        
        assert len(leakage_issues) > 0
        assert leakage_issues[0].recommendation != ""
        assert "Remove" in leakage_issues[0].recommendation

    def test_id_recommendation(self):
        """Test ID issues have recommendations."""
        validator = DataQualityValidator()
        
        report = validator.validate(
            df=FROZEN_ID_DATA,
            id_cols=["id"],
            target_col="target",
        )
        
        id_issues = [i for i in report.issues if i.issue_type == "id_as_feature"]
        
        assert len(id_issues) > 0
        assert id_issues[0].recommendation != ""
        assert "Remove" in id_issues[0].recommendation
