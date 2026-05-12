import pytest
import numpy as np
from agents.feature_factory import _hackathon_gate, _get_condition_values, _extract_feature_categories
from core.state import ProfessorState
import polars as pl

# ── Tests ───────────────────────────────────────────────────────────────────

class TestHypothesisFeatureFactory:

    def test_significant_effect_passes(self):
        """Feature with significant effect and |d| > 0.2 passes the gate."""
        np.random.seed(42)
        # Group A: mean=5, Group B: mean=3 — clear difference
        feature = np.concatenate([
            np.random.normal(5, 1, 500),  # condition=1
            np.random.normal(3, 1, 500),  # condition=0
        ])
        condition = np.array([1]*500 + [0]*500)
        
        passed, p_value, effect_size, details = _hackathon_gate(
            feature, condition, {"wilcoxon_p": 0.05}
        )
        
        assert passed == True
        assert p_value < 0.05
        assert abs(effect_size) > 0.2
        assert details["effect_magnitude"] in ("medium", "large")
    
    def test_no_effect_fails(self):
        """Feature with no real difference between groups fails the gate."""
        np.random.seed(42)
        # Both groups: same distribution
        feature = np.random.normal(5, 1, 1000)
        condition = np.array([1]*500 + [0]*500)
        
        passed, p_value, effect_size, details = _hackathon_gate(
            feature, condition, {"wilcoxon_p": 0.05}
        )
        
        assert passed == False
        assert abs(effect_size) < 0.2  # Negligible effect
    
    def test_significant_but_tiny_effect_fails(self):
        """Effect size gate catches small effects even with large N."""
        np.random.seed(42)
        # Groups with very small difference but large N
        feature = np.concatenate([
            np.random.normal(5.01, 1, 5000),  # condition=1
            np.random.normal(5.00, 1, 5000),  # condition=0
        ])
        condition = np.array([1]*5000 + [0]*5000)
        
        passed, p_value, effect_size, details = _hackathon_gate(
            feature, condition, {"wilcoxon_p": 0.05}
        )
        
        # p-value might be significant due to large N, but effect size is negligible
        assert abs(effect_size) < 0.2
        assert passed == False  # Effect size gate prevents false positives
    
    def test_uses_adaptive_threshold(self):
        """Gate uses the p-value threshold from gate_config, not hardcoded."""
        np.random.seed(42)
        feature = np.concatenate([
            np.random.normal(5, 1, 200),
            np.random.normal(4.2, 1, 200),
        ])
        condition = np.array([1]*200 + [0]*200)
        
        # With strict threshold
        passed_strict, p_strict, _, _ = _hackathon_gate(feature, condition, {"wilcoxon_p": 0.01})
        
        # With relaxed threshold
        passed_relaxed, p_relaxed, _, _ = _hackathon_gate(feature, condition, {"wilcoxon_p": 0.10})
        
        assert p_strict == p_relaxed  # Same test, same p-value
    
    def test_effect_details_complete(self):
        """Effect details dict has all required fields."""
        np.random.seed(42)
        feature = np.concatenate([
            np.random.normal(5, 1, 500),
            np.random.normal(3, 1, 500),
        ])
        condition = np.array([1]*500 + [0]*500)
        
        _, _, _, details = _hackathon_gate(feature, condition, {"wilcoxon_p": 0.05})
        
        required_keys = [
            "p_value", "effect_size", "effect_magnitude", 
            "group_a_mean", "group_b_mean", "group_a_std", "group_b_std",
            "group_a_n", "group_b_n", "group_a_median", "group_b_median",
            "direction", "mann_whitney_U"
        ]
        for key in required_keys:
            assert key in details, f"Missing key: {key}"
    
    def test_small_groups_rejected(self):
        """Groups with fewer than 30 observations are rejected."""
        feature = np.random.normal(5, 1, 50)
        condition = np.array([1]*10 + [0]*40)  # Only 10 in group A
        
        passed, p_value, effect_size, details = _hackathon_gate(
            feature, condition, {"wilcoxon_p": 0.05}
        )
        
        assert passed == False
        assert "Insufficient" in details.get("reason", "")
    
    def test_nan_handling(self):
        """NaN values are dropped before testing."""
        np.random.seed(42)
        feature = np.concatenate([
            np.random.normal(5, 1, 500),
            np.random.normal(3, 1, 500),
        ])
        # Inject NaNs
        feature[0:20] = np.nan
        feature[500:520] = np.nan
        condition = np.array([1]*500 + [0]*500)
        
        passed, p_value, effect_size, details = _hackathon_gate(
            feature, condition, {"wilcoxon_p": 0.05}
        )
        
        # Should still work — NaNs dropped, enough observations remain
        assert details["group_a_n"] == 480  # 500 - 20 NaNs
        assert details["group_b_n"] == 480
    
    def test_effect_magnitude_classification(self):
        """Effect magnitude labels match Cohen's conventions."""
        _, _, _, details_large = _hackathon_gate(
            np.concatenate([np.random.normal(5, 1, 500), np.random.normal(2, 1, 500)]),
            np.array([1]*500 + [0]*500), {"wilcoxon_p": 0.05}
        )
        assert details_large["effect_magnitude"] == "large"  # d > 0.8

class TestConditionClassification:
    
    def test_uses_existing_condition_feature(self):
        """When a condition feature exists in manifest, use it."""
        df = pl.DataFrame({
            "is_elderly": [1, 0, 1, 0, 1, 0] * 100,
            "target": [3, 2, 4, 1, 3, 2] * 100,
        })
        
        manifest = [{"name": "is_elderly", "category": "condition"}]
        thesis = {"condition_variable": "age_group"}
        
        condition = _get_condition_values(df, thesis, manifest)
        assert len(condition) == 600
        assert set(np.unique(condition)) == {0, 1}
        assert sum(condition) == 300  # Half are elderly
    
    def test_derives_from_thesis_variable(self):
        """When no condition feature exists, derive from thesis condition_variable."""
        df = pl.DataFrame({
            "age": [25, 70, 35, 80, 45, 90] * 100,
            "target": [1, 2, 1, 3, 1, 2] * 100,
        })
        
        manifest = []  # No condition features yet (round 1)
        thesis = {"condition_variable": "age"}
        
        condition = _get_condition_values(df, thesis, manifest)
        assert len(condition) == 600
        assert set(np.unique(condition)) == {0, 1}
    
    def test_fallback_when_no_match(self):
        """When nothing matches, return a random split (degradation mode)."""
        df = pl.DataFrame({
            "feature_a": [1, 2, 3, 4, 5] * 100,
            "target": [0, 1, 0, 1, 0] * 100,
        })
        
        manifest = []
        thesis = {"condition_variable": "nonexistent_variable_xyz"}
        
        condition = _get_condition_values(df, thesis, manifest)
        assert len(condition) == 500
        assert set(np.unique(condition)) == {0, 1}

class TestFeatureClassification:

    def test_categories_categorized_correctly(self):
        """Features are sorted into the 4 thesis categories."""
        categories = _extract_feature_categories(
            code="""
# CATEGORY: condition
df = df.with_columns((pl.col("age") > 65).cast(pl.Int8).alias("is_elderly"))

# CATEGORY: delta
df = df.with_columns((pl.col("acc_elderly") - pl.col("acc_young")).alias("age_accuracy_gap"))

# CATEGORY: moderator
df = df.with_columns((pl.col("age_accuracy_gap") * pl.col("comorbidity_count")).alias("gap_x_comorbidity"))
""",
            feature_names=["is_elderly", "age_accuracy_gap", "gap_x_comorbidity"]
        )
        
        assert categories["is_elderly"] == "condition"
        assert categories["age_accuracy_gap"] == "delta"
        assert categories["gap_x_comorbidity"] == "moderator"
    
    def test_category_inferred_from_naming(self):
        """When no CATEGORY comment exists, infer from feature name."""
        categories = _extract_feature_categories(
            code="df = df.with_columns(pl.col('age').alias('is_senior_flag'))",
            feature_names=["is_senior_flag"]
        )
        assert categories["is_senior_flag"] == "condition"
    
    def test_delta_inferred_from_name(self):
        categories = _extract_feature_categories(
            code="df = df.with_columns(pl.lit(0.1).alias('severity_gap'))",
            feature_names=["severity_gap"]
        )
        assert categories["severity_gap"] == "delta"
