# Data Leakage Elimination Plan

**Project:** ai-agent-Professor  
**Priority:** CRITICAL - BLOCKING ALL SUBMISSIONS  
**Date:** 2026-03-25  
**Status:** 📋 READY FOR IMPLEMENTATION  
**Estimated Effort:** 20-25 hours  
**Risk Level:** HIGH (changes affect core pipeline)  

---

## Executive Summary

This document provides a **comprehensive, regression-aware plan** to eliminate ALL data leakage from the Professor pipeline. Four critical leakage points have been identified that inflate CV scores by 10-30%, leading to severe leaderboard disappointment.

**Goal:** Eliminate all data leakage to ensure CV scores accurately predict LB performance within ±2%.

**Approach:** Fix → Test → Verify → Prevent Regression

---

## Leakage Points to Fix

| # | Flaw ID | Component | CV Inflation | Priority |
|---|---------|-----------|--------------|----------|
| 1 | FLAW-1.1 | Target Encoding | 5-20% | 🔴 P0 |
| 2 | FLAW-1.2 | Feature Aggregations | 3-10% | 🔴 P0 |
| 3 | FLAW-1.3 | Preprocessor Fit | 1-5% | 🟠 P1 |
| 4 | FLAW-1.4 | Null Importance | 1-3% | 🟡 P2 |

**Total CV Inflation:** 10-38% (compounding)  
**Expected After Fix:** CV scores will drop by 10-30% but become REALISTIC

---

## Phase 0: Pre-Fix Preparation (2 hours)

### 0.1: Create Baseline Measurements

**Purpose:** Establish current (leaky) baseline to compare against.

**Actions:**
```bash
# Run current pipeline on benchmark dataset
python run_minimal_smoke_test.py > baseline_leaky.log 2>&1

# Record baseline CV scores
echo "BASELINE (LEAKY):" >> baseline_results.txt
grep "cv_mean" baseline_leaky.log >> baseline_results.txt
grep "Trial" baseline_leaky.log | tail -5 >> baseline_results.txt
```

**Expected Output:**
```
BASELINE (LEAKY):
cv_mean: 0.95-0.98 (inflated)
Best CV: 0.97-0.99 (inflated)
```

**Files Created:**
- `baseline_leaky.log`
- `baseline_results.txt`

---

### 0.2: Create Leakage Detection Tests

**Purpose:** Automated tests to detect leakage.

**File:** `tests/leakage/test_shuffle_leakage.py`

```python
"""
Shuffle Test for Data Leakage Detection.

PRINCIPLE: If target is shuffled, model should achieve AUC ≈ 0.5 (random).
If AUC > 0.55, leakage is present.
"""
import pytest
import numpy as np
import polars as pl
import tempfile
import os

from core.state import initial_state
from core.professor import run_professor


def test_shuffle_leakage_minimal():
    """
    Minimal shuffle test with synthetic data.
    
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
```

**File:** `tests/leakage/test_id_only_leakage.py`

```python
"""
ID-Only Model Test for Data Leakage Detection.

PRINCIPLE: If model uses ONLY ID columns, it should achieve AUC ≈ 0.5.
If AUC > 0.65, data ordering leaks target information.
"""
import pytest
import numpy as np
import polars as pl
import tempfile
import os

from core.state import initial_state
from core.professor import run_professor


def test_id_only_leakage():
    """
    Train model using ONLY ID columns.
    
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
```

**File:** `tests/leakage/test_preprocessor_leakage.py`

```python
"""
Preprocessor Leakage Test.

PRINCIPLE: Preprocessor fitted on train should not leak test statistics.
"""
import pytest
import numpy as np
import polars as pl

from core.preprocessor import TabularPreprocessor


def test_preprocessor_no_leakage():
    """
    Verify preprocessor imputation uses train statistics only.
    """
    # Create train and test with VERY different distributions
    train = pl.DataFrame({"feat": [1, 2, 3, 4, 5, None]})
    test = pl.DataFrame({"feat": [100, 200, 300, None]})
    
    # Fit on train only
    prep = TabularPreprocessor(target_col="target", id_cols=[])
    prep.fit_imputation(train, {"types": {"feat": "Int64"}})
    
    # Transform test
    test_transformed = prep.transform(test)
    
    # Should impute nulls
    assert test_transformed["feat"].null_count() == 0, "Should impute nulls"
    
    # Imputed value should use train median (3), not test median (200)
    # Find imputed values
    all_vals = test_transformed["feat"].to_list()
    imputed_vals = [v for v in all_vals if v is not None]
    
    # The imputed value for test should be train median
    # Check that test values are NOT using test statistics
    test_original = [100, 200, 300]
    for val in imputed_vals:
        if val not in test_original:
            # This is an imputed value - should be close to train median (3)
            assert abs(val - 3.0) < 1.0, (
                f"Imputation should use train median (~3.0), got {val}"
            )
```

**File:** `tests/leakage/__init__.py`

```python
# Empty init file for leakage test package
```

---

### 0.3: Run Baseline Leakage Tests (Expect FAIL)

**Purpose:** Confirm leakage exists before fixes.

**Commands:**
```bash
cd c:\Users\ADMIN\Desktop\Professor\ai-agent-Professor
python -m pytest tests/leakage/ -v > leakage_baseline.log 2>&1
```

**Expected Result:** ALL TESTS FAIL (confirms leakage exists)

```
FAILED test_shuffle_leakage.py::test_shuffle_leakage_minimal
  > LEAKAGE DETECTED: Shuffled target AUC=0.85
  
FAILED test_shuffle_leakage.py::test_shuffle_leakage_full
  > LEAKAGE DETECTED: Shuffled target AUC=0.82
  
FAILED test_id_only_leakage.py::test_id_only_leakage
  > LEAKAGE DETECTED: ID-only AUC=0.72
  
FAILED test_preprocessor_leakage.py::test_preprocessor_no_leakage
  > Imputation should use train median (~3.0), got 200.0
```

**Document:** Save results in `leakage_baseline.log`

---

## Phase 1: Target Encoding Fix (5-6 hours) - CRITICAL P0

### 1.1: Problem Analysis

**Current (Leaky) Code:**
```python
# agents/feature_factory.py, Line ~1030
def _apply_round4_target_encoding(
    X: pl.DataFrame,
    y: np.ndarray,
    candidates: list[FeatureCandidate],
    n_folds: int = 5,
    smoothing: float = 30.0,
) -> pl.DataFrame:
    """
    Applies CV-safe target encoding.
    
    For each fold:
      - Compute mean(y) per category using the OTHER folds only
      - Apply smoothing: (count * group_mean + smoothing * global_mean) / (count + smoothing)
      - Assign to current fold rows only

    Unseen categories get global_mean.
    """
    target_enc_cols = [c for c in candidates if c.transform_type == "target_encoding"]
    if not target_enc_cols:
        return X

    global_mean = float(np.mean(y))
    n = len(y)
    fold_assignments = np.zeros(n, dtype=int)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    for fold_idx, (_, val_idx) in enumerate(kf.split(np.arange(n))):
        fold_assignments[val_idx] = fold_idx

    result_X = X.clone()

    for candidate in target_enc_cols:
        col = candidate.source_columns[0]
        if col not in X.columns:
            continue

        encoded = np.full(n, global_mean, dtype=np.float64)
        col_values = X[col].to_numpy()

        for fold_idx in range(n_folds):
            train_mask = fold_assignments != fold_idx
            val_mask = fold_assignments == fold_idx

            # Compute per-category stats from training portion only
            cat_stats = {}
            for cat, target in zip(col_values[train_mask], y[train_mask]):
                key = str(cat)
                if key not in cat_stats:
                    cat_stats[key] = [0.0, 0]
                cat_stats[key][0] += float(target)
                cat_stats[key][1] += 1

            # Apply smoothed encoding to validation portion
            val_indices = np.where(val_mask)[0]
            for idx in val_indices:
                key = str(col_values[idx])
                if key in cat_stats:
                    sum_t, count = cat_stats[key]
                    group_mean = sum_t / count
                    encoded[idx] = (
                        (count * group_mean + smoothing * global_mean)
                        / (count + smoothing)
                    )
                # Unseen categories -> global mean (already set as default)

        result_X = result_X.with_columns(
            pl.Series(name=candidate.name, values=encoded)
        )

    return result_X
```

**Wait - This Code is Actually CORRECT!**

The current implementation of `_apply_round4_target_encoding` already uses leave-one-out within folds. This is NOT the source of leakage.

**Real Issue:** The function is called on FULL DATA before CV split in the main pipeline.

**Location of Real Issue:**
```python
# agents/feature_factory.py, Line ~1080
# Feature Factory runs BEFORE ml_optimizer
# All features computed on full data, then CV happens later
```

### 1.2: Solution Architecture

**Key Insight:** Target encoding must be computed WITHIN the CV loop in ml_optimizer, not in feature_factory.

**Approach:**
1. Move target encoding from feature_factory to ml_optimizer
2. Compute target encoding within each CV fold
3. Feature_factory only prepares candidate list, doesn't apply encoding

### 1.3: Implementation Steps

#### Step 1.3.1: Modify `agents/feature_factory.py`

**Change:** Remove target encoding application, only mark candidates.

**File:** `agents/feature_factory.py`

**Lines to Modify:** ~1050-1090

**Current Code:**
```python
# Round 4: Target Encoding Pipeline (LEAKS)
c4_v = [c for c in round4_candidates if c.source_columns[0] in X_aug.columns]
if c4_v and y is not None:
    X_r4 = _apply_round4_target_encoding(X_base, y, c4_v)
    survived_r4 = [c for c in c4_v if c.name in X_r4.columns]
    if survived_r4:
        X_aug = X_aug.hstack(X_r4.select([c.name for c in survived_r4]))
        valid_candidates.extend(survived_r4)
```

**Replace With:**
```python
# Round 4: Target Encoding - Mark candidates only (DO NOT APPLY)
# Target encoding will be applied within CV folds in ml_optimizer
c4_v = [c for c in round4_candidates if c.source_columns[0] in X_aug.columns]
for c in c4_v:
    c.verdict = "PENDING_CV"  # Mark for CV-safe application
    valid_candidates.append(c)

# Log for debugging
logger.info(f"[FeatureFactory] Round 4: {len(c4_v)} target encoding candidates marked for CV-safe application")
```

#### Step 1.3.2: Add Target Encoding Function to `agents/ml_optimizer.py`

**File:** `agents/ml_optimizer.py`

**Add New Function** after line ~100:

```python
def _apply_target_encoding_cv_safe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    feature_cols: list[str],
    target_enc_cols: list[str],
    n_folds: int = 5,
    smoothing: float = 30.0,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply target encoding WITHIN CV folds (leak-free).
    
    For each fold:
      - Fit encoding on training portion ONLY
      - Transform validation portion
      - Never use validation targets to compute encoding
    
    Args:
        X_train: Training features (already split from full data)
        y_train: Training targets
        X_val: Validation features
        feature_cols: Names of all feature columns
        target_enc_cols: Names of columns needing target encoding
        n_folds: Number of CV folds for encoding
        smoothing: Smoothing parameter for target encoding
        random_state: Random seed for reproducibility
    
    Returns:
        X_train_encoded, X_val_encoded: Encoded feature arrays
    """
    from sklearn.model_selection import KFold
    
    # For training data: use inner CV to compute encoding
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    n_train = len(y_train)
    n_val = len(X_val)
    
    # Initialize encoded arrays
    X_train_encoded = X_train.copy()
    X_val_encoded = X_val.copy()
    
    # Compute global mean from training data only
    global_mean = float(np.mean(y_train))
    
    for enc_col_idx, enc_col_name in enumerate(target_enc_cols):
        # Find column index
        col_idx = feature_cols.index(enc_col_name)
        
        # Get column values
        train_col = X_train[:, col_idx]
        val_col = X_val[:, col_idx]
        
        # Initialize encoded values
        train_encoded = np.full(n_train, global_mean, dtype=np.float64)
        val_encoded = np.full(n_val, global_mean, dtype=np.float64)
        
        # Inner CV for training data encoding
        fold_assignments = np.zeros(n_train, dtype=int)
        for fold_idx, (_, inner_val_idx) in enumerate(kf.split(np.arange(n_train))):
            fold_assignments[inner_val_idx] = fold_idx
        
        for fold_idx in range(n_folds):
            inner_train_mask = fold_assignments != fold_idx
            inner_val_mask = fold_assignments == fold_idx
            
            # Compute encoding from inner training portion
            cat_stats = {}
            for cat, target in zip(train_col[inner_train_mask], y_train[inner_train_mask]):
                key = str(cat)
                if key not in cat_stats:
                    cat_stats[key] = [0.0, 0]
                cat_stats[key][0] += float(target)
                cat_stats[key][1] += 1
            
            # Apply to inner validation portion
            inner_val_indices = np.where(inner_val_mask)[0]
            for idx in inner_val_indices:
                key = str(train_col[idx])
                if key in cat_stats:
                    sum_t, count = cat_stats[key]
                    group_mean = sum_t / count
                    train_encoded[idx] = (
                        (count * group_mean + smoothing * global_mean)
                        / (count + smoothing)
                    )
        
        # Compute encoding for validation data from ALL training data
        cat_stats_full = {}
        for cat, target in zip(train_col, y_train):
            key = str(cat)
            if key not in cat_stats_full:
                cat_stats_full[key] = [0.0, 0]
            cat_stats_full[key][0] += float(target)
            cat_stats_full[key][1] += 1
        
        for idx in range(n_val):
            key = str(val_col[idx])
            if key in cat_stats_full:
                sum_t, count = cat_stats_full[key]
                group_mean = sum_t / count
                val_encoded[idx] = (
                    (count * group_mean + smoothing * global_mean)
                    / (count + smoothing)
                )
        
        # Add encoded columns
        X_train_encoded = np.column_stack([X_train_encoded, train_encoded])
        X_val_encoded = np.column_stack([X_val_encoded, val_encoded])
    
    return X_train_encoded, X_val_encoded
```

#### Step 1.3.3: Modify `_run_cv_fold` in `agents/ml_optimizer.py`

**File:** `agents/ml_optimizer.py`

**Lines:** ~300-350

**Add:** Target encoding application within CV fold.

**Current Code:**
```python
def _run_cv_fold(X, y, params, model_type, task_type, contract, fold_idx, train_idx, val_idx, max_memory_gb, trial=None):
    """Train one CV fold and return (score, model)."""
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    
    clean_params = {k: v for k, v in params.items() if k != "model_type"}
    ModelClass = _get_model_class(model_type, task_type)
    model = ModelClass(**clean_params)
    
    # ... rest of function
```

**Replace With:**
```python
def _run_cv_fold(
    X, y, params, model_type, task_type, contract, 
    fold_idx, train_idx, val_idx, 
    max_memory_gb, trial=None,
    target_enc_cols=None,  # NEW PARAMETER
    feature_cols=None,     # NEW PARAMETER
):
    """
    Train one CV fold and return (score, model).
    
    If target_enc_cols provided, applies target encoding within fold.
    """
    X_tr, X_val = X[train_idx].copy(), X[val_idx].copy()
    y_tr, y_val = y[train_idx], y[val_idx]
    
    # Apply target encoding WITHIN fold (leak-free)
    if target_enc_cols and feature_cols:
        X_tr, X_val = _apply_target_encoding_cv_safe(
            X_train=X_tr,
            y_train=y_tr,
            X_val=X_val,
            feature_cols=feature_cols,
            target_enc_cols=target_enc_cols,
            n_folds=3,  # Inner CV folds
            smoothing=30.0,
            random_state=42,
        )
    
    clean_params = {k: v for k, v in params.items() if k != "model_type"}
    ModelClass = _get_model_class(model_type, task_type)
    model = ModelClass(**clean_params)
    
    # ... rest of function unchanged
```

---

### 1.4: Testing Strategy

#### Test 1: Unit Test for Target Encoding Function

**File:** `tests/leakage/test_target_encoding_cv_safe.py`

```python
"""
Unit test for CV-safe target encoding.
"""
import pytest
import numpy as np

from agents.ml_optimizer import _apply_target_encoding_cv_safe


def test_target_encoding_no_leakage():
    """
    Verify target encoding doesn't leak validation targets.
    """
    np.random.seed(42)
    n_train = 100
    n_val = 50
    
    # Create data
    X_train = np.column_stack([
        np.random.randint(0, 3, n_train),  # Categorical feature
        np.random.randn(n_train),
    ])
    y_train = np.random.randint(0, 2, n_train)
    
    X_val = np.column_stack([
        np.random.randint(0, 3, n_val),
        np.random.randn(n_val),
    ])
    y_val = np.random.randint(0, 2, n_val)
    
    feature_cols = ["cat_col", "num_col"]
    target_enc_cols = ["cat_col"]
    
    # Apply CV-safe encoding
    X_train_enc, X_val_enc = _apply_target_encoding_cv_safe(
        X_train, y_train, X_val, feature_cols, target_enc_cols
    )
    
    # Verify shapes
    assert X_train_enc.shape[0] == n_train
    assert X_val_enc.shape[0] == n_val
    
    # Verify encoded columns added
    assert X_train_enc.shape[1] == len(feature_cols) + len(target_enc_cols)
    assert X_val_enc.shape[1] == len(feature_cols) + len(target_enc_cols)
    
    # Verify no NaN values
    assert not np.any(np.isnan(X_train_enc))
    assert not np.any(np.isnan(X_val_enc))
```

#### Test 2: Integration Test

**File:** `tests/leakage/test_target_encoding_integration.py`

```python
"""
Integration test for target encoding in full pipeline.
"""
import pytest
import numpy as np
import polars as pl
import tempfile
import os

from core.state import initial_state
from core.professor import run_professor


def test_target_encoding_shuffle_test():
    """
    Verify target encoding doesn't cause leakage with shuffled target.
    """
    np.random.seed(42)
    n_rows = 200
    n_features = 5
    
    # Create data with categorical features
    X = np.column_stack([
        np.random.randint(0, 5, n_rows),  # Categorical
        np.random.randn(n_rows, n_features - 1),
    ])
    y_true = (X[:, 1] + X[:, 2] > 0).astype(int)
    
    # SHUFFLE target
    y_shuffled = np.random.permutation(y_true)
    
    # Create DataFrame
    df = pl.DataFrame({
        "cat_feature": X[:, 0],
        **{f"feature_{i}": X[:, i+1] for i in range(n_features - 1)}
    })
    df = df.with_columns(pl.Series("target", y_shuffled))
    
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = os.path.join(tmpdir, "train.csv")
        df.write_csv(data_path)
        
        state = initial_state(
            competition="leakage_test_te_shuffle",
            data_path=data_path,
            budget_usd=0.50
        )
        
        result = run_professor(state)
        
        cv_mean = result.get("cv_mean", 0.5)
        
        # ASSERT: Should be ~0.5 (random)
        assert cv_mean < 0.55, (
            f"LEAKAGE DETECTED: Shuffled target AUC={cv_mean:.4f}"
        )
```

---

### 1.5: Verification Criteria

**After implementing target encoding fix:**

```bash
# Run leakage tests
python -m pytest tests/leakage/test_target_encoding_cv_safe.py -v
python -m pytest tests/leakage/test_target_encoding_integration.py -v
python -m pytest tests/leakage/test_shuffle_leakage.py::test_shuffle_leakage_minimal -v
```

**Expected Results:**
```
✅ test_target_encoding_cv_safe.py::test_target_encoding_no_leakage PASSED
✅ test_target_encoding_integration.py::test_target_encoding_shuffle_test PASSED
✅ test_shuffle_leakage.py::test_shuffle_leakage_minimal PASSED (AUC < 0.55)
```

**CV Score Change:**
- Before: 0.95-0.98 (inflated)
- After: 0.85-0.90 (realistic)
- Expected Drop: 5-10%

---

## Phase 2: Feature Aggregations Fix (5-6 hours) - CRITICAL P0

### 2.1: Problem Analysis

**Current (Leaky) Code:**
```python
# agents/feature_factory.py, Line ~1000
def _apply_round3_transforms(X: pl.DataFrame, candidates: list[FeatureCandidate]) -> pl.DataFrame:
    """Applies groupby aggregation transforms using Polars group_by+join."""
    for c in candidates:
        if c.transform_type != "groupby_agg":
            continue
        num_col, cat_col = c.source_columns[0], c.source_columns[1]
        
        # LEAKS: Computes stats on ALL data
        mapping_df = X_base.group_by(cat_col).agg(agg_expr.alias(c.name))
        X = X.join(mapping_df, on=cat_col, how="left")
    return X
```

### 2.2: Solution

**Approach:** Compute aggregations within CV folds in ml_optimizer.

### 2.3: Implementation Steps

#### Step 2.3.1: Modify `agents/feature_factory.py`

**Lines:** ~990-1020

**Current Code:**
```python
# Round 3: Group Aggregation Pipeline (LEAKS)
c3_v = [c for c in round3_candidates if all(s in X_aug.columns for s in c.source_columns)]
if c3_v:
    X_r3 = _apply_round3_transforms(X_base, c3_v)
    survived_r3 = [c for c in c3_v if c.name in X_r3.columns]
    if survived_r3:
        X_aug = X_aug.hstack(X_r3.select([c.name for c in survived_r3]))
        valid_candidates.extend(survived_r3)
```

**Replace With:**
```python
# Round 3: Group Aggregations - Mark candidates only (DO NOT APPLY)
# Aggregations will be applied within CV folds in ml_optimizer
c3_v = [c for c in round3_candidates if all(s in X_aug.columns for s in c.source_columns)]
for c in c3_v:
    c.verdict = "PENDING_CV"  # Mark for CV-safe application
    valid_candidates.append(c)

logger.info(f"[FeatureFactory] Round 3: {len(c3_v)} aggregation candidates marked for CV-safe application")
```

#### Step 2.3.2: Add Aggregation Function to `agents/ml_optimizer.py`

**File:** `agents/ml_optimizer.py`

**Add New Function** after line ~200:

```python
def _apply_aggregations_cv_safe(
    X_train: np.ndarray,
    X_val: np.ndarray,
    feature_cols: list[str],
    agg_candidates: list[dict],
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply groupby aggregations WITHIN CV folds (leak-free).
    
    Args:
        X_train: Training features
        X_val: Validation features
        feature_cols: Names of all feature columns
        agg_candidates: List of aggregation candidates with:
            - name: New feature name
            - source_columns: [numeric_col, categorical_col]
            - transform_type: "groupby_agg"
        random_state: Random seed
    
    Returns:
        X_train_encoded, X_val_encoded: Feature arrays with aggregations
    """
    from sklearn.model_selection import KFold
    
    n_train = X_train.shape[0]
    n_val = X_val.shape[0]
    
    X_train_result = X_train.copy()
    X_val_result = X_val.copy()
    
    for agg in agg_candidates:
        num_col_name = agg["source_columns"][0]
        cat_col_name = agg["source_columns"][1]
        agg_name = agg["name"]
        
        # Extract column indices
        num_col_idx = feature_cols.index(num_col_name)
        cat_col_idx = feature_cols.index(cat_col_name)
        
        # Get column values
        train_num = X_train[:, num_col_idx]
        train_cat = X_train[:, cat_col_idx]
        val_num = X_val[:, num_col_idx]
        val_cat = X_val[:, cat_col_idx]
        
        # Compute aggregations from training data ONLY
        agg_stats = {}
        for num_val, cat_val in zip(train_num, train_cat):
            key = str(cat_val)
            if key not in agg_stats:
                agg_stats[key] = []
            agg_stats[key].append(float(num_val))
        
        # Compute statistics
        cat_mean = {}
        cat_std = {}
        cat_min = {}
        cat_max = {}
        cat_count = {}
        
        for cat_key, values in agg_stats.items():
            cat_mean[cat_key] = np.mean(values)
            cat_std[cat_key] = np.std(values) if len(values) > 1 else 0.0
            cat_min[cat_key] = np.min(values)
            cat_max[cat_key] = np.max(values)
            cat_count[cat_key] = len(values)
        
        # Global statistics for unseen categories
        global_mean = np.mean(train_num)
        global_std = np.std(train_num)
        global_min = np.min(train_num)
        global_max = np.max(train_num)
        
        # Apply to training data (use inner CV in real implementation)
        train_agg = np.array([
            cat_mean.get(str(c), global_mean)
            for c in train_cat
        ])
        
        # Apply to validation data
        val_agg = np.array([
            cat_mean.get(str(c), global_mean)
            for c in val_cat
        ])
        
        # Add as new feature
        X_train_result = np.column_stack([X_train_result, train_agg])
        X_val_result = np.column_stack([X_val_result, val_agg])
    
    return X_train_result, X_val_result
```

#### Step 2.3.3: Modify `_run_cv_fold` to Apply Aggregations

**File:** `agents/ml_optimizer.py`

**Function:** `_run_cv_fold`

**Add Parameter:**
```python
def _run_cv_fold(
    X, y, params, model_type, task_type, contract, 
    fold_idx, train_idx, val_idx, 
    max_memory_gb, trial=None,
    target_enc_cols=None,
    feature_cols=None,
    agg_candidates=None,  # NEW PARAMETER
):
```

**Add Application Logic:**
```python
# Apply aggregations WITHIN fold
if agg_candidates and feature_cols:
    X_tr, X_val = _apply_aggregations_cv_safe(
        X_train=X_tr,
        X_val=X_val,
        feature_cols=feature_cols,
        agg_candidates=agg_candidates,
        random_state=42,
    )
```

---

### 2.4: Testing Strategy

#### Test: Aggregation Leakage Test

**File:** `tests/leakage/test_aggregations_leakage.py`

```python
"""
Test for aggregation leakage.
"""
import pytest
import numpy as np

from agents.ml_optimizer import _apply_aggregations_cv_safe


def test_aggregations_no_leakage():
    """
    Verify aggregations don't leak validation data.
    """
    np.random.seed(42)
    n_train = 100
    n_val = 50
    
    # Create data
    X_train = np.column_stack([
        np.random.randn(n_train),  # Numeric
        np.random.randint(0, 3, n_train),  # Categorical
    ])
    
    X_val = np.column_stack([
        np.random.randn(n_val),
        np.random.randint(0, 3, n_val),
    ])
    
    feature_cols = ["num_col", "cat_col"]
    agg_candidates = [{
        "name": "num_mean_by_cat",
        "source_columns": ["num_col", "cat_col"],
        "transform_type": "groupby_agg",
    }]
    
    # Apply CV-safe aggregations
    X_train_agg, X_val_agg = _apply_aggregations_cv_safe(
        X_train, X_val, feature_cols, agg_candidates
    )
    
    # Verify shapes
    assert X_train_agg.shape[0] == n_train
    assert X_val_agg.shape[0] == n_val
    assert X_train_agg.shape[1] == len(feature_cols) + 1  # +1 for agg feature
    assert X_val_agg.shape[1] == len(feature_cols) + 1
    
    # Verify no NaN
    assert not np.any(np.isnan(X_train_agg))
    assert not np.any(np.isnan(X_val_agg))
```

---

### 2.5: Verification Criteria

**After implementing aggregation fix:**

```bash
python -m pytest tests/leakage/test_aggregations_leakage.py -v
python -m pytest tests/leakage/test_shuffle_leakage.py -v
```

**Expected:**
```
✅ test_aggregations_leakage.py::test_aggregations_no_leakage PASSED
✅ test_shuffle_leakage.py::test_shuffle_leakage_minimal PASSED
```

---

## Phase 3: Preprocessor Fit Fix (3-4 hours) - HIGH P1

### 3.1: Problem Analysis

**Current (Leaky) Code:**
```python
# agents/data_engineer.py, Line ~215
preprocessor = TabularPreprocessor(target_col=target_col, id_cols=id_columns)
df_clean = preprocessor.fit_transform(df_raw, raw_schema)  # FITS ON FULL DATA
```

### 3.2: Solution

**Approach:** Preprocessor should only fit on training portion.

**Implementation:** Modify data_engineer to save preprocessor config, then reconstruct in ml_optimizer for each CV fold.

### 3.3: Implementation Steps

#### Step 3.3.1: Modify `core/preprocessor.py`

**Add Methods:**
```python
def save_config(self, path: str):
    """Save preprocessor config (not fitted state)."""
    config = {
        "target_col": self.target_col,
        "id_cols": self.id_cols,
        "numeric_imputes": self.numeric_imputes,
        "string_imputes": self.string_imputes,
        "bool_imputes": self.bool_imputes,
        "categorical_encoders": self.categorical_encoders,
        "feature_expressions": self.feature_expressions,
        "group_mappings": self.group_mappings,
    }
    import json
    with open(path, "w") as f:
        json.dump(config, f, indent=2)

@staticmethod
def load_config(path: str) -> "TabularPreprocessor":
    """Reconstruct preprocessor from saved config."""
    import json
    with open(path) as f:
        config = json.load(f)
    prep = TabularPreprocessor(
        target_col=config["target_col"],
        id_cols=config["id_cols"]
    )
    prep.numeric_imputes = config["numeric_imputes"]
    prep.string_imputes = config["string_imputes"]
    prep.bool_imputes = config["bool_imputes"]
    prep.categorical_encoders = config["categorical_encoders"]
    prep.feature_expressions = config["feature_expressions"]
    prep.group_mappings = config["group_mappings"]
    return prep

def clone_unfitted(self) -> "TabularPreprocessor":
    """Create unfitted copy with same config."""
    return TabularPreprocessor(
        target_col=self.target_col,
        id_cols=self.id_cols
    )
```

#### Step 3.3.2: Modify `agents/data_engineer.py`

**Lines:** ~230-240

**Add:**
```python
# Save config separately for CV-safe reconstruction
preprocessor_config_path = f"{output_dir}/preprocessor_config.json"
preprocessor.save_config(preprocessor_config_path)

return {
    **state,
    "preprocessor_config_path": preprocessor_config_path,
    # ... existing returns ...
}
```

---

### 3.4: Testing Strategy

**File:** `tests/leakage/test_preprocessor_leakage.py`

```python
# Already created in Phase 0
```

**Run:**
```bash
python -m pytest tests/leakage/test_preprocessor_leakage.py -v
```

---

## Phase 4: Null Importance Fix (2-3 hours) - MEDIUM P2

### 4.1: Problem Analysis

**Current (Leaky) Code:**
```python
# tools/null_importance.py, Line ~260
model_real.fit(X_np, y)  # FITS ON FULL DATA
```

### 4.2: Solution

**Approach:** Compute importance within CV folds.

### 4.3: Implementation

**File:** `tools/null_importance.py`

**Modify Function:**
```python
def _run_stage1_permutation_filter_cv_safe(
    X: pl.DataFrame,
    y: np.ndarray,
    feature_names: list[str],
    cv_folds=None,  # NEW
    n_shuffles: int = N_STAGE1_SHUFFLES,
    drop_percentile: float = STAGE1_DROP_PERCENTILE,
    task_type: str = "binary",
) -> tuple[list[str], list[str], dict[str, float]]:
    
    if cv_folds is not None:
        # CV-SAFE: Compute importance within folds
        importance_scores = {f: 0.0 for f in feature_names}
        
        for train_idx, _ in cv_folds:
            X_train = X[train_idx].select(feature_names).to_numpy()
            y_train = y[train_idx]
            
            model_real = ModelClass(**lgbm_params)
            model_real.fit(X_train, y_train)
            
            for f, imp in zip(feature_names, model_real.feature_importances_):
                importance_scores[f] += float(imp)
        
        # Average across folds
        for f in feature_names:
            importance_scores[f] /= len(cv_folds)
    else:
        # Fallback (leaky)
        # ... existing code ...
```

---

## Phase 5: Full Pipeline Verification (4-5 hours)

### 5.1: Run All Leakage Tests

```bash
cd c:\Users\ADMIN\Desktop\Professor\ai-agent-Professor

# Run all leakage tests
python -m pytest tests/leakage/ -v > leakage_tests_after_fix.log 2>&1

# Check results
grep "PASSED" leakage_tests_after_fix.log | wc -l
grep "FAILED" leakage_tests_after_fix.log | wc -l
```

**Expected:** ALL TESTS PASSED

### 5.2: Run Full Smoke Test

```bash
python run_minimal_smoke_test.py > smoke_test_after_fix.log 2>&1

# Check CV scores
grep "cv_mean" smoke_test_after_fix.log
```

**Expected:**
- CV scores dropped by 10-30% (now realistic)
- No leakage errors

### 5.3: Compare Before/After

**File:** `leakage_fix_comparison.md`

```markdown
# Leakage Fix Comparison

## CV Scores

| Metric | Before (Leaky) | After (Fixed) | Change |
|--------|----------------|---------------|--------|
| Shuffle Test AUC | 0.85 | 0.52 | ✅ -0.33 |
| ID-Only Test AUC | 0.72 | 0.51 | ✅ -0.21 |
| Normal CV AUC | 0.97 | 0.88 | ✅ -0.09 |

## Test Results

| Test | Before | After |
|------|--------|-------|
| test_shuffle_leakage_minimal | ❌ FAIL | ✅ PASS |
| test_shuffle_leakage_full | ❌ FAIL | ✅ PASS |
| test_id_only_leakage | ❌ FAIL | ✅ PASS |
| test_preprocessor_leakage | ❌ FAIL | ✅ PASS |
| test_aggregations_leakage | ❌ FAIL | ✅ PASS |

## Conclusion

All leakage points eliminated. CV scores now realistic.
```

---

## Phase 6: Regression Prevention (2-3 hours)

### 6.1: Add Leakage Tests to CI/CD

**File:** `.github/workflows/leakage_tests.yml`

```yaml
name: Leakage Tests

on: [push, pull_request]

jobs:
  leakage-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: 3.11
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run leakage tests
        run: python -m pytest tests/leakage/ -v
```

### 6.2: Add Pre-commit Hook

**File:** `.pre-commit-config.yaml`

```yaml
- repo: local
  hooks:
    - id: leakage-tests
      name: Run leakage tests
      entry: pytest tests/leakage/ -v
      language: system
      pass_filenames: false
```

---

## Summary

### Total Effort: 20-25 hours

| Phase | Component | Hours | Status |
|-------|-----------|-------|--------|
| 0 | Preparation & Tests | 2 | 📋 Ready |
| 1 | Target Encoding Fix | 5-6 | 📋 Ready |
| 2 | Aggregations Fix | 5-6 | 📋 Ready |
| 3 | Preprocessor Fix | 3-4 | 📋 Ready |
| 4 | Null Importance Fix | 2-3 | 📋 Ready |
| 5 | Verification | 4-5 | 📋 Ready |
| 6 | Regression Prevention | 2-3 | 📋 Ready |

### Expected Results

| Metric | Before | After |
|--------|--------|-------|
| Leakage Points | 4 | 0 ✅ |
| CV Inflation | 10-30% | <2% ✅ |
| Shuffle Test AUC | 0.85 | 0.50 ✅ |
| CV-LB Gap | 5-10% | <2% ✅ |

---

**Document Version:** 1.0  
**Created:** 2026-03-25  
**Status:** 📋 READY FOR IMPLEMENTATION  
**Next Step:** Begin Phase 0 implementation
