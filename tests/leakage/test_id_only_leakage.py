"""
ID-Only Model Test for Data Leakage Detection.

PRINCIPLE: If model uses ONLY ID columns, it should achieve AUC ≈ 0.5.
If AUC > 0.65, data ordering leaks target information.

NOTE: Uses minimal Optuna trials (1 trial) for fast execution.
"""
import pytest
import numpy as np
import polars as pl
import tempfile
import os

# Set minimal config BEFORE importing professor
os.environ["N_OPTUNA_TRIALS"] = "1"

from core.state import initial_state
from core.professor import run_professor


def test_id_only_leakage():
    """
    Train model using ONLY ID columns.
    Uses 1 Optuna trial for fast execution (< 2 min).

    PASS: AUC < 0.65 (no ordering leakage)
    FAIL: AUC >= 0.65 (ordering leakage detected)
    """
    np.random.seed(42)
    n_rows = 100

    # Create data where target is NOT correlated with row order
    df = pl.DataFrame({
        "id": range(n_rows),
        "feature_1": np.random.randn(n_rows),
        "target": np.random.randint(0, 2, n_rows)  # Random, not ordered
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = os.path.join(tmpdir, "train.csv")
        df.write_csv(data_path)

        # Create sample_submission.csv (needed for target detection)
        sample_df = pl.DataFrame({
            "id": list(range(20)),
            "target": [0] * 20
        })
        sample_path = os.path.join(tmpdir, "sample_submission.csv")
        sample_df.write_csv(sample_path)

        state = initial_state(
            competition="leakage_test_id_only",
            data_path=data_path,
            budget_usd=0.10
        )

        # Drop all real features, keep only ID
        state["dropped_features"] = ["feature_1"]

        result = run_professor(state)

        cv_mean = result.get("cv_mean", 0.5)

        # ASSERT: Should be ~0.5 (random)
        assert cv_mean < 0.65, (
            f"LEAKAGE DETECTED: ID-only AUC={cv_mean:.4f}. "
            f"Data ordering may encode target information."
        )
