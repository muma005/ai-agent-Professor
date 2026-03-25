"""
Shuffle Test for Data Leakage Detection.

PRINCIPLE: If target is shuffled, model should achieve AUC ≈ 0.5 (random).
If AUC > 0.55, leakage is present.

NOTE: Uses minimal Optuna trials (1 trial) for fast execution.
Full pipeline runs but ML optimization is minimized.
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


def test_shuffle_leakage_minimal():
    """
    Minimal shuffle test with synthetic data.
    Uses 1 Optuna trial for fast execution (< 2 min).

    PASS: AUC < 0.55 (no leakage)
    FAIL: AUC >= 0.55 (leakage detected)
    """
    np.random.seed(42)
    n_rows = 100
    n_features = 5

    # Create synthetic data
    X = np.random.randn(n_rows, n_features)
    y_true = (X[:, 0] + X[:, 1] > 0).astype(int)

    # SHUFFLE target (breaks any real signal)
    y_shuffled = np.random.permutation(y_true)

    # Create DataFrame
    df = pl.DataFrame({
        f"feature_{i}": X[:, i] for i in range(n_features)
    })
    df = df.with_columns(pl.Series("target", y_shuffled))

    # Save to temp file
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
            competition="leakage_test_shuffle",
            data_path=data_path,
            budget_usd=0.10
        )

        result = run_professor(state)

        cv_mean = result.get("cv_mean", 0.5)

        # ASSERT: Should be ~0.5 (random)
        assert cv_mean < 0.55, (
            f"LEAKAGE DETECTED: Shuffled target AUC={cv_mean:.4f}. "
            f"Expected ~0.50. Check preprocessor and feature factory."
        )


def test_shuffle_leakage_full():
    """
    Full shuffle test with realistic data size.
    Uses 1 Optuna trial for fast execution (< 3 min).

    PASS: AUC < 0.55 (no leakage)
    FAIL: AUC >= 0.55 (leakage detected)
    """
    np.random.seed(42)
    n_rows = 500
    n_features = 10

    # Create synthetic data with some structure
    X = np.random.randn(n_rows, n_features)
    # True signal: only first 3 features matter
    y_true = (X[:, 0] + X[:, 1] - X[:, 2] > 0).astype(int)

    # SHUFFLE target
    y_shuffled = np.random.permutation(y_true)

    # Create DataFrame
    df = pl.DataFrame({
        f"feature_{i}": X[:, i] for i in range(n_features)
    })
    df = df.with_columns(pl.Series("target", y_shuffled))

    # Save to temp file
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = os.path.join(tmpdir, "train.csv")
        df.write_csv(data_path)

        # Create sample_submission.csv (needed for target detection)
        sample_df = pl.DataFrame({
            "id": list(range(50)),
            "target": [0] * 50
        })
        sample_path = os.path.join(tmpdir, "sample_submission.csv")
        sample_df.write_csv(sample_path)

        state = initial_state(
            competition="leakage_test_shuffle_full",
            data_path=data_path,
            budget_usd=0.50
        )

        result = run_professor(state)

        cv_mean = result.get("cv_mean", 0.5)

        # ASSERT: Should be ~0.5 (random)
        assert cv_mean < 0.55, (
            f"LEAKAGE DETECTED: Shuffled target AUC={cv_mean:.4f}. "
            f"Expected ~0.50."
        )
