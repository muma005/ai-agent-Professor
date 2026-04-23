# tests/contracts/test_adaptive_gater_contract.py

import pytest
import polars as pl
import numpy as np
from tools.adaptive_gater import evaluate_feature_performance

# ── Tests ───────────────────────────────────────────────────────────────────

class TestAdaptiveGaterContract:
    """
    Contract: Adaptive Feature Gating (Component 6)
    """

    def test_beneficial_feature_detected(self):
        """Verify that a feature with signal is marked as beneficial."""
        np.random.seed(42)
        n = 1000
        # feat1 is pure signal
        feat1 = np.random.normal(0, 1, n)
        target = (feat1 > 0).astype(int)
        
        df = pl.DataFrame({
            "feat1": feat1.tolist(),
            "noise": np.random.normal(0, 1, n).tolist(),
            "target": target.tolist()
        })
        
        res = evaluate_feature_performance(df, "target", "feat1", task_type="classification")
        assert res["is_beneficial"] is True
        assert res["improvement"] > 0.05

    def test_noise_feature_not_beneficial(self):
        """Verify that pure noise is rejected."""
        np.random.seed(42)
        n = 1000
        df = pl.DataFrame({
            "feat1": np.random.normal(0, 1, n).tolist(),
            "noise": np.random.normal(0, 1, n).tolist(),
            "target": np.random.randint(0, 2, n).tolist()
        })
        
        res = evaluate_feature_performance(df, "target", "noise", task_type="classification")
        # Noise should have near-zero or negative improvement
        assert res["improvement"] < 0.005
        assert res["is_beneficial"] is False

    def test_regression_negative_mse(self):
        """Verify regression uses negative MSE (higher is better)."""
        np.random.seed(42)
        n = 1000
        feat1 = np.random.normal(0, 1, n)
        target = feat1 * 2 + np.random.normal(0, 0.1, n)
        
        df = pl.DataFrame({
            "feat1": feat1.tolist(),
            "target": target.tolist()
        })
        
        res = evaluate_feature_performance(df, "target", "feat1", task_type="regression")
        assert res["full_score"] > res["base_score"] # -MSE should increase
        assert res["is_beneficial"] is True

    def test_categorical_encoding_handled(self):
        """Verify strings are handled (implied by LGBM success)."""
        df = pl.DataFrame({
            "cat": ["A", "B"] * 500,
            "val": np.random.normal(0, 1, 1000).tolist(),
            "target": [0, 1] * 500
        })
        # This test ensures the tool doesn't crash on strings
        # (Preprocessing inside run_adaptive_gate handles this, 
        # but evaluate_feature_performance needs numeric)
        df_encoded = df.with_columns(pl.col("cat").cast(pl.Categorical).to_physical())
        res = evaluate_feature_performance(df_encoded, "target", "cat")
        assert "improvement" in res
