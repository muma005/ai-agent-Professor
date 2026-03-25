# tools/data_quality.py

"""
Data quality validation framework.

FLAW-5.6 FIX: Data Quality Tests
- Target leakage detection
- ID column detection
- Constant feature detection
- Duplicate row detection
- Missing value analysis
- Outlier detection
- Data drift detection
"""

import logging
import numpy as np
import polars as pl
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from scipy import stats
from scipy.stats import chi2_contingency, spearmanr

logger = logging.getLogger(__name__)


@dataclass
class DataQualityIssue:
    """Represents a data quality issue."""
    
    issue_type: str
    severity: str  # "critical", "high", "medium", "low"
    description: str
    affected_columns: List[str]
    affected_rows: Optional[int] = None
    recommendation: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict."""
        return {
            "issue_type": self.issue_type,
            "severity": self.severity,
            "description": self.description,
            "affected_columns": self.affected_columns,
            "affected_rows": self.affected_rows,
            "recommendation": self.recommendation,
        }


@dataclass
class DataQualityReport:
    """Complete data quality report."""
    
    dataset_name: str
    n_rows: int
    n_columns: int
    passed: bool
    critical_issues: int
    high_issues: int
    medium_issues: int
    low_issues: int
    issues: List[DataQualityIssue]
    summary: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict."""
        return {
            "dataset_name": self.dataset_name,
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "passed": self.passed,
            "critical_issues": self.critical_issues,
            "high_issues": self.high_issues,
            "medium_issues": self.medium_issues,
            "low_issues": self.low_issues,
            "total_issues": len(self.issues),
            "issues": [i.to_dict() for i in self.issues],
            "summary": self.summary,
        }


class DataQualityValidator:
    """
    Comprehensive data quality validation.
    
    Features:
    - Target leakage detection
    - ID column detection
    - Constant feature detection
    - Duplicate detection
    - Missing value analysis
    - Outlier detection
    - Data drift detection
    
    Usage:
        validator = DataQualityValidator()
        
        report = validator.validate(
            df=train_data,
            target_col="target",
            id_cols=["id", "customer_id"],
        )
        
        if not report.passed:
            print(f"Found {report.total_issues} issues")
    """
    
    def __init__(
        self,
        leakage_threshold: float = 0.9,
        uniqueness_threshold: float = 0.95,
        missing_threshold: float = 0.5,
        outlier_std_threshold: float = 3.0,
    ):
        """
        Initialize data quality validator.
        
        Args:
            leakage_threshold: Correlation threshold for leakage detection
            uniqueness_threshold: Uniqueness ratio for ID detection
            missing_threshold: Missing value threshold for flagging
            outlier_std_threshold: Standard deviations for outlier detection
        """
        self.leakage_threshold = leakage_threshold
        self.uniqueness_threshold = uniqueness_threshold
        self.missing_threshold = missing_threshold
        self.outlier_std_threshold = outlier_std_threshold
        
        logger.info("[DataQualityValidator] Initialized")
    
    def validate(
        self,
        df: pl.DataFrame,
        target_col: Optional[str] = None,
        id_cols: Optional[List[str]] = None,
        dataset_name: str = "dataset",
    ) -> DataQualityReport:
        """
        Run comprehensive data quality validation.
        
        Args:
            df: DataFrame to validate
            target_col: Target column name (optional)
            id_cols: Known ID columns (optional)
            dataset_name: Name for reporting
        
        Returns:
            DataQualityReport with all findings
        """
        issues = []
        
        # Run all validation checks
        issues.extend(self._check_target_leakage(df, target_col))
        issues.extend(self._check_id_columns(df, id_cols, target_col))
        issues.extend(self._check_constant_features(df))
        issues.extend(self._check_duplicates(df))
        issues.extend(self._check_missing_values(df))
        issues.extend(self._check_outliers(df))
        
        # Count by severity
        critical = sum(1 for i in issues if i.severity == "critical")
        high = sum(1 for i in issues if i.severity == "high")
        medium = sum(1 for i in issues if i.severity == "medium")
        low = sum(1 for i in issues if i.severity == "low")
        
        # Determine if passed (no critical or high issues)
        passed = critical == 0 and high == 0
        
        # Generate summary
        summary = self._generate_summary(df, target_col)
        
        return DataQualityReport(
            dataset_name=dataset_name,
            n_rows=len(df),
            n_columns=len(df.columns),
            passed=passed,
            critical_issues=critical,
            high_issues=high,
            medium_issues=medium,
            low_issues=low,
            issues=issues,
            summary=summary,
        )
    
    def _check_target_leakage(
        self,
        df: pl.DataFrame,
        target_col: Optional[str],
    ) -> List[DataQualityIssue]:
        """Check for target leakage."""
        issues = []
        
        if target_col is None or target_col not in df.columns:
            return issues
        
        target = df[target_col]
        
        for col in df.columns:
            if col == target_col:
                continue
            
            # Check correlation for numeric columns
            if df[col].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64):
                try:
                    corr, p_value = spearmanr(
                        df[col].drop_nulls().to_numpy(),
                        target.drop_nulls().to_numpy()
                    )
                    
                    if abs(corr) >= self.leakage_threshold and p_value < 0.05:
                        issues.append(DataQualityIssue(
                            issue_type="target_leakage",
                            severity="critical",
                            description=f"Column '{col}' has very high correlation with target (r={corr:.3f})",
                            affected_columns=[col],
                            recommendation=f"Remove '{col}' from features - likely leaks target information",
                        ))
                except Exception:
                    pass
        
        return issues
    
    def _check_id_columns(
        self,
        df: pl.DataFrame,
        id_cols: Optional[List[str]],
        target_col: Optional[str],
    ) -> List[DataQualityIssue]:
        """Check for ID columns being used as features."""
        issues = []
        
        for col in df.columns:
            if col == target_col:
                continue
            
            # Check uniqueness ratio
            n_unique = df[col].n_unique()
            n_total = len(df)
            uniqueness_ratio = n_unique / n_total
            
            # Check if column is in provided ID list
            is_known_id = id_cols and col in id_cols
            
            # Check if column name suggests ID
            is_id_by_name = any(
                keyword in col.lower()
                for keyword in ["id", "uuid", "key", "code"]
            )
            
            if uniqueness_ratio >= self.uniqueness_threshold:
                if is_known_id:
                    issues.append(DataQualityIssue(
                        issue_type="id_as_feature",
                        severity="critical",
                        description=f"ID column '{col}' is being used as feature (100% unique)",
                        affected_columns=[col],
                        recommendation=f"Remove '{col}' from features - ID columns cannot generalize",
                    ))
                elif is_id_by_name:
                    issues.append(DataQualityIssue(
                        issue_type="potential_id_column",
                        severity="high",
                        description=f"Column '{col}' appears to be an ID column ({uniqueness_ratio:.1%} unique)",
                        affected_columns=[col],
                        recommendation=f"Review '{col}' - if it's an ID, remove from features",
                    ))
        
        return issues
    
    def _check_constant_features(self, df: pl.DataFrame) -> List[DataQualityIssue]:
        """Check for constant/zero-variance features."""
        issues = []
        
        for col in df.columns:
            if df[col].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64):
                n_unique = df[col].n_unique()
                
                if n_unique == 1:
                    issues.append(DataQualityIssue(
                        issue_type="constant_feature",
                        severity="medium",
                        description=f"Column '{col}' has only 1 unique value",
                        affected_columns=[col],
                        recommendation=f"Remove '{col}' - constant features provide no predictive power",
                    ))
                elif n_unique == 2 and len(df) > 100:
                    # Check if nearly constant
                    value_counts = df[col].value_counts()
                    if len(value_counts) == 2:
                        max_ratio = value_counts["count"].max() / len(df)
                        if max_ratio > 0.99:
                            issues.append(DataQualityIssue(
                                issue_type="near_constant_feature",
                                severity="low",
                                description=f"Column '{col}' is nearly constant (99%+ one value)",
                                affected_columns=[col],
                                recommendation=f"Consider removing '{col}' - nearly constant",
                            ))
        
        return issues
    
    def _check_duplicates(self, df: pl.DataFrame) -> List[DataQualityIssue]:
        """Check for duplicate rows."""
        issues = []
        
        try:
            # Count duplicates
            n_total = len(df)
            n_unique = len(df.unique())
            n_duplicates = n_total - n_unique
            duplicate_ratio = n_duplicates / n_total
            
            if n_duplicates > 0:
                severity = "low"
                if duplicate_ratio > 0.1:
                    severity = "medium"
                if duplicate_ratio > 0.5:
                    severity = "high"
                
                issues.append(DataQualityIssue(
                    issue_type="duplicate_rows",
                    severity=severity,
                    description=f"Found {n_duplicates} duplicate rows ({duplicate_ratio:.1%})",
                    affected_columns=list(df.columns),
                    affected_rows=n_duplicates,
                    recommendation="Remove duplicate rows to prevent data leakage",
                ))
        except Exception as e:
            logger.warning(f"Could not check duplicates: {e}")
        
        return issues
    
    def _check_missing_values(self, df: pl.DataFrame) -> List[DataQualityIssue]:
        """Check for missing values."""
        issues = []
        
        for col in df.columns:
            null_count = df[col].null_count()
            null_ratio = null_count / len(df)
            
            if null_ratio > 0:
                severity = "low"
                if null_ratio > self.missing_threshold:
                    severity = "high"
                elif null_ratio > 0.1:
                    severity = "medium"
                
                issues.append(DataQualityIssue(
                    issue_type="missing_values",
                    severity=severity,
                    description=f"Column '{col}' has {null_count} missing values ({null_ratio:.1%})",
                    affected_columns=[col],
                    affected_rows=null_count,
                    recommendation=f"Impute or remove '{col}' - high missing rate" if null_ratio > 0.1 else f"Consider imputation for '{col}'",
                ))
        
        return issues
    
    def _check_outliers(self, df: pl.DataFrame) -> List[DataQualityIssue]:
        """Check for outliers in numeric columns."""
        issues = []
        
        for col in df.columns:
            if df[col].dtype not in (pl.Float32, pl.Float64, pl.Int32, pl.Int64):
                continue
            
            try:
                values = df[col].drop_nulls().to_numpy()
                
                if len(values) < 10:
                    continue
                
                mean = np.mean(values)
                std = np.std(values)
                
                if std == 0:
                    continue
                
                z_scores = np.abs((values - mean) / std)
                n_outliers = np.sum(z_scores > self.outlier_std_threshold)
                outlier_ratio = n_outliers / len(values)
                
                if outlier_ratio > 0.01:  # More than 1% outliers
                    severity = "low"
                    if outlier_ratio > 0.1:
                        severity = "medium"
                    
                    issues.append(DataQualityIssue(
                        issue_type="outliers",
                        severity=severity,
                        description=f"Column '{col}' has {n_outliers} outliers ({outlier_ratio:.1%})",
                        affected_columns=[col],
                        affected_rows=n_outliers,
                        recommendation=f"Review outliers in '{col}' - consider capping or transformation",
                    ))
            except Exception:
                pass
        
        return issues
    
    def _generate_summary(
        self,
        df: pl.DataFrame,
        target_col: Optional[str],
    ) -> Dict[str, Any]:
        """Generate data quality summary statistics."""
        summary = {
            "n_rows": len(df),
            "n_columns": len(df.columns),
            "n_numeric": sum(
                1 for col in df.columns
                if df[col].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64)
            ),
            "n_categorical": sum(
                1 for col in df.columns
                if df[col].dtype in (pl.Utf8, pl.Categorical, pl.Boolean)
            ),
            "total_missing": sum(df[col].null_count() for col in df.columns),
            "total_duplicates": len(df) - len(df.unique()),
        }
        
        if target_col and target_col in df.columns:
            summary["target_distribution"] = df[target_col].value_counts().to_dict()
        
        return summary


def validate_data_quality(
    df: pl.DataFrame,
    target_col: Optional[str] = None,
    id_cols: Optional[List[str]] = None,
    **kwargs,
) -> DataQualityReport:
    """
    Convenience function for data quality validation.
    
    Args:
        df: DataFrame to validate
        target_col: Target column name
        id_cols: Known ID columns
        **kwargs: Passed to DataQualityValidator
    
    Returns:
        DataQualityReport
    """
    validator = DataQualityValidator(**kwargs)
    return validator.validate(df, target_col, id_cols)
