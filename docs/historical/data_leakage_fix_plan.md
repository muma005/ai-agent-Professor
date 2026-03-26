# Data Leakage Fix Implementation Plan

**Date:** 2026-03-25  
**Priority:** CRITICAL  
**Status:** 📋 READY TO IMPLEMENT  
**Estimated Effort:** 15-20 hours  

---

## Executive Summary

This document provides a **step-by-step implementation plan** to eliminate all data leakage from the Professor pipeline. Each fix includes:

- ✅ Code changes required
- ✅ Files to modify
- ✅ Testing strategy
- ✅ Rollback plan

---

## Architecture Change Overview

### Current (Leaky) Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FULL DATA                             │
│              (train.csv + test.csv combined)             │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │  Preprocessor.FIT()   │  ← LEAKS: Fits on test data
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │  Feature Factory      │  ← LEAKS: Aggregations on all data
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │  Target Encoding      │  ← LEAKS: Uses all targets
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │  CV Split             │  ← TOO LATE: Already leaked!
         └───────────┬───────────┘
                     │
                     ▼
              Model Training
```

### Fixed (Leak-Free) Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FULL DATA                             │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │  CV SPLIT FIRST       │  ← Split BEFORE any processing
         └───────────┬───────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
┌───────────────┐         ┌───────────────┐
│  TRAIN FOLD   │         │  VAL FOLD     │
│  (80%)        │         │  (20%)        │
└───────┬───────┘         └───────┬───────┘
        │                         │
        ▼                         │
┌───────────────┐                 │
│ Preprocessor  │                 │
│ .FIT()        │                 │
└───────┬───────┘                 │
        │                         │
        ▼                         │
┌───────────────┐                 │
│ Preprocessor  │                 │
│ .TRANSFORM()  │────────────────►│
└───────┬───────┘                 │
        │                         │
        ▼                         │
┌───────────────┐                 │
│ Feature Eng.  │                 │
│ (within fold) │                 │
└───────┬───────┘                 │
        │                         │
        ▼                         │
┌───────────────┐                 │
│ Target Encode │                 │
│ (leave-one-out)                │
└───────┬───────┘                 │
        │                         │
        ▼                         ▼
         ┌───────────────────────┐
         │    Model Training     │
         │    (NO LEAKAGE)       │
         └───────────────────────┘
```

---

## Phase 1: Preprocessor Leakage Fix (3-4 hours)

### Problem

Current code fits preprocessor on full dataset before CV split.

### Solution

**Option A: Fit within CV loop (Recommended)**

### Implementation Steps

#### Step 1.1: Modify `core/preprocessor.py`

**File:** `core/preprocessor.py`

**Changes:** Add method to clone preprocessor with same config but unfitted state.

```python
# ADD THIS METHOD to TabularPreprocessor class

def clone_unfitted(self) -> "TabularPreprocessor":
    """
    Create a new preprocessor with same config but unfitted state.
    Used for CV where we need fresh preprocessor per fold.
    """
    return TabularPreprocessor(
        target_col=self.target_col,
        id_cols=self.id_cols
    )


def fit_only(self, df: pl.DataFrame, schema: Dict[str, Any]) -> "TabularPreprocessor":
    """
    Fit preprocessor WITHOUT transforming.
    Returns self for method chaining.
    """
    self.fit_imputation(df, schema)
    self.expected_columns = df.columns
    return self
```

---

#### Step 1.2: Modify `agents/data_engineer.py`

**File:** `agents/data_engineer.py`

**Changes:** Save preprocessor config separately, don't fit on full data.

```python
# CURRENT (LINE ~215) - LEAKY
preprocessor = TabularPreprocessor(target_col=target_col, id_cols=id_columns)
df_clean = preprocessor.fit_transform(df_raw, raw_schema)  # ← LEAKS

# REPLACE WITH
preprocessor = TabularPreprocessor(target_col=target_col, id_cols=id_columns)
preprocessor.fit_imputation(df_raw, raw_schema)  # Fit only on train
df_clean = preprocessor.transform(df_raw)  # Transform only
```

**Actually, this is correct for data_engineer** - it only processes train.csv.

The real issue is in feature_factory. Let me update:

---

#### Step 1.3: Modify `agents/feature_factory.py`

**File:** `agents/feature_factory.py`

**Current (Leaky):**
```python
# Line ~950
df = pl.read_parquet(clean_path)
X_base = preprocessor.transform(df)  # All rows together
```

**Fixed:**
```python
# We need to split BEFORE feature factory
# This requires passing train/test split from ml_optimizer
```

**Actually, the correct fix is:**

1. **data_engineer** should save train/test separately
2. **feature_factory** should only process train
3. **ml_optimizer** should apply same transforms to test

Let me create the proper fix:

---

### Fix Implementation: Preprocessor

#### File 1: `core/preprocessor.py`

Add these methods:

```python
def save_config(self, path: str):
    """Save preprocessor config (not fitted state) for later reconstruction."""
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
```

---

#### File 2: `agents/data_engineer.py`

**Line ~215-235:** Modify to save preprocessor config:

```python
# AFTER fitting preprocessor
preprocessor.save(preprocessor_path)  # Already done

# ADD: Save config separately for reconstruction
preprocessor_config_path = f"{output_dir}/preprocessor_config.json"
preprocessor.save_config(preprocessor_config_path)

# Update state
return {
    **state,
    "preprocessor_config_path": preprocessor_config_path,  # NEW
    # ... existing returns ...
}
```

---

#### File 3: `agents/feature_factory.py`

**Line ~950-970:** Modify to load preprocessor config and apply ONLY to train:

```python
# CURRENT
df = pl.read_parquet(clean_path)
preprocessor = TabularPreprocessor.load(preprocessor_path)
X_base = preprocessor.transform(df)

# REPLACE WITH
# Load full data
df = pl.read_parquet(clean_path)

# Load preprocessor config (fitted on train only)
preprocessor = TabularPreprocessor.load(preprocessor_path)

# Transform all data (this is OK - preprocessor already fitted)
X_base = preprocessor.transform(df)

# BUT: For target encoding and aggregations, we need to split
# This is handled in Round 3/4 fixes below
```

---

## Phase 2: Target Encoding Fix (4-5 hours) - CRITICAL

### Problem

Current code computes target encoding on full dataset, leaking validation targets.

### Solution

Implement **leave-one-out target encoding within CV folds**.

---

#### File 4: `agents/feature_factory.py`

**Lines 1026-1050:** Replace Round 4 target encoding

**CURRENT (LEAKY):**
```python
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
    # ... this code is actually CORRECT!
```

**Wait - the Round 4 encoding IS correct!** It uses leave-one-out within folds.

The issue is in the **Round 3 aggregations**. Let me check:

---

#### File 4 (Updated): `agents/feature_factory.py`

**Lines 990-1020:** Round 3 Aggregations

**CURRENT (LEAKY):**
```python
def _apply_round3_transforms(X: pl.DataFrame, candidates: list[FeatureCandidate]) -> pl.DataFrame:
    """Applies groupby aggregation transforms using Polars group_by+join."""
    for c in candidates:
        if c.transform_type != "groupby_agg":
            continue
        num_col, cat_col = c.source_columns[0], c.source_columns[1]
        
        # LEAKS: Computes stats on ALL data
        group_stats = X.group_by(cat_col).agg(agg_fn.alias(c.name))
        X = X.join(group_stats, on=cat_col, how="left")
    return X
```

**FIXED:**
```python
def _apply_round3_transforms_cv_safe(
    X: pl.DataFrame,
    y: np.ndarray,
    candidates: list[FeatureCandidate],
    n_folds: int = 5,
) -> pl.DataFrame:
    """
    Applies groupby aggregation transforms WITHIN CV FOLDS ONLY.
    
    For each fold:
      - Compute aggregations using training folds ONLY
      - Apply to validation fold
      - Never use validation data to compute training statistics
    """
    from sklearn.model_selection import KFold
    
    result_X = X.clone()
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Initialize new columns with nulls
    for c in candidates:
        if c.transform_type == "groupby_agg":
            result_X = result_X.with_columns(pl.lit(None).alias(c.name))
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train = X[train_idx]
        X_val = X[val_idx]
        
        for c in candidates:
            if c.transform_type != "groupby_agg":
                continue
            
            num_col, cat_col = c.source_columns[0], c.source_columns[1]
            
            # Compute aggregations on TRAIN ONLY
            fn_name = extract_fn_name(c.name)  # Extract 'mean', 'std', etc from name
            agg_fn = get_agg_function(fn_name, num_col)
            
            group_stats = X_train.group_by(cat_col).agg(agg_fn.alias(c.name))
            
            # Join to validation
            X_val_with_feat = X_val.join(group_stats, on=cat_col, how="left")
            
            # Update result for validation rows
            for i, row_idx in enumerate(val_idx):
                result_X = result_X.with_columns(
                    pl.lit(X_val_with_feat[c.name][i]).set_at_idx([row_idx])
                )
    
    # Fill any remaining nulls with global stats
    for c in candidates:
        if c.transform_type == "groupby_agg":
            global_stat = X.select(get_agg_function(extract_fn_name(c.name), c.source_columns[0])).item()
            result_X = result_X.with_columns(
                pl.col(c.name).fill_null(global_stat)
            )
    
    return result_X
```

---

## Phase 3: Null Importance Fix (2-3 hours)

### Problem

Null importance computed on full dataset leaks test information.

### Solution

Compute importance on training folds only.

---

#### File 5: `tools/null_importance.py`

**Lines 250-280:** Modify Stage 1 to accept CV folds

**CURRENT (LEAKY):**
```python
def _run_stage1_permutation_filter(
    X: pl.DataFrame,
    y: np.ndarray,
    feature_names: list[str],
    ...
) -> tuple[list[str], list[str], dict[str, float]]:
    
    # LEAKS: Fits on full data
    model_real = ModelClass(**lgbm_params)
    model_real.fit(X_np, y)  # ← FULL DATA
```

**FIXED:**
```python
def _run_stage1_permutation_filter_cv_safe(
    X: pl.DataFrame,
    y: np.ndarray,
    feature_names: list[str],
    cv_folds=None,  # NEW PARAMETER
    ...
) -> tuple[list[str], list[str], dict[str, float]]:
    
    if cv_folds is not None:
        # CV-SAFE: Compute importance within folds
        importance_scores = {f: 0.0 for f in feature_names}
        
        for train_idx, _ in cv_folds:
            X_train = X[train_idx]
            y_train = y[train_idx]
            
            model = ModelClass(**lgbm_params)
            model.fit(X_train.to_numpy(), y_train)
            
            for f, imp in zip(feature_names, model.feature_importances_):
                importance_scores[f] += imp
        
        # Average across folds
        for f in feature_names:
            importance_scores[f] /= len(cv_folds)
    else:
        # Fallback to current (leaky) behavior
        model_real.fit(X_np, y)
        importance_scores = dict(zip(feature_names, model_real.feature_importances_))
```

---

## Phase 4: Testing Strategy (4-5 hours)

### Test 1: Shuffle Test

**File:** `tests/leakage/test_shuffle.py`

```python
import pytest
import numpy as np
from core.state import initial_state
from core.professor import run_professor

def test_shuffle_leakage():
    """
    Shuffle target column. If model achieves AUC > 0.55, leakage is present.
    """
    # Create synthetic data
    np.random.seed(42)
    n_rows = 100
    X = np.random.randn(n_rows, 5)
    y_true = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    # SHUFFLE target
    y_shuffled = np.random.permutation(y_true)
    
    # Run pipeline with shuffled target
    state = initial_state(
        competition="leakage_test_shuffle",
        data_path="path_to_shuffled_data.csv",
        budget_usd=0.10
    )
    
    result = run_professor(state)
    
    cv_mean = result.get("cv_mean", 0.5)
    
    # Should be ~0.5 (random)
    assert cv_mean < 0.55, f"LEAKAGE DETECTED: Shuffled target AUC={cv_mean:.4f}"
```

---

### Test 2: ID-Only Model Test

**File:** `tests/leakage/test_id_only.py`

```python
def test_id_only_leakage():
    """
    Train model using ONLY ID columns. If AUC > 0.65, data ordering leaks target.
    """
    # Create data with obvious ID column
    df = pl.DataFrame({
        "id": range(100),
        "feature_1": np.random.randn(100),
        "target": (np.arange(100) > 50).astype(int)  # Correlated with row order
    })
    
    # Run pipeline with only ID as feature
    state = initial_state(...)
    state["dropped_features"] = ["feature_1"]  # Drop all real features
    
    result = run_professor(state)
    
    cv_mean = result.get("cv_mean", 0.5)
    
    assert cv_mean < 0.65, f"LEAKAGE DETECTED: ID-only AUC={cv_mean:.4f}"
```

---

### Test 3: Preprocessor Leakage Test

**File:** `tests/leakage/test_preprocessor.py`

```python
def test_preprocessor_no_leakage():
    """
    Verify preprocessor doesn't leak test statistics.
    """
    from core.preprocessor import TabularPreprocessor
    
    # Create train and test with different distributions
    train = pl.DataFrame({"feat": [1, 2, 3, 4, 5, None]})
    test = pl.DataFrame({"feat": [100, 200, 300, None]})
    
    # Fit on train only
    prep = TabularPreprocessor(target_col="target", id_cols=[])
    prep.fit_imputation(train, {"types": {"feat": "Int64"}})
    
    # Transform test
    test_transformed = prep.transform(test)
    
    # Imputation should use train median (3), not test median
    assert test_transformed["feat"].null_count() == 0, "Should impute nulls"
    
    # Check imputed value
    imputed_val = test_transformed.filter(pl.col("feat").is_not_null())["feat"][0]
    assert abs(imputed_val - 3.0) < 0.1, f"Should use train median, got {imputed_val}"
```

---

## Phase 5: Integration & Validation (2-3 hours)

### Step 5.1: Run All Leakage Tests

```bash
cd c:\Users\ADMIN\Desktop\Professor\ai-agent-Professor
python -m pytest tests/leakage/ -v
```

**Expected Results:**
- test_shuffle_leakage: PASS (AUC < 0.55)
- test_id_only_leakage: PASS (AUC < 0.65)
- test_preprocessor_no_leakage: PASS

### Step 5.2: Run Minimal Smoke Test

```bash
python run_minimal_smoke_test.py
```

**Expected:** Pipeline completes with slightly lower (but more realistic) CV scores.

### Step 5.3: Compare CV Scores Before/After

| Metric | Before (Leaky) | After (Fixed) | Expected Change |
|--------|----------------|---------------|-----------------|
| CV AUC | 0.98 | 0.90-0.93 | -5 to -8% |
| LB AUC | ? | 0.88-0.92 | Should match CV |
| CV-LB Gap | 5-10% | <2% | ✅ FIXED |

---

## Rollback Plan

If fixes cause issues:

### Rollback Step 1: Revert Code Changes

```bash
cd c:\Users\ADMIN\Desktop\Professor\ai-agent-Professor
git stash  # Stash all changes
git checkout main  # Return to main branch
```

### Rollback Step 2: Use Cached Results

```bash
# If smoke test was running, results are in:
outputs/smoke_te_*/
```

### Rollback Step 3: Document Issues

Create `rollback_reason.md` with:
- What failed
- Error messages
- Suggested fixes

---

## Implementation Checklist

### Phase 1: Preprocessor (3-4 hours)
- [ ] Add `save_config()` and `load_config()` to `TabularPreprocessor`
- [ ] Update `data_engineer.py` to save config
- [ ] Test preprocessor reconstruction

### Phase 2: Target Encoding (4-5 hours)
- [ ] Implement `_apply_round3_transforms_cv_safe()`
- [ ] Update feature_factory to use CV-safe version
- [ ] Test with known leakage case

### Phase 3: Null Importance (2-3 hours)
- [ ] Add `cv_folds` parameter to null importance
- [ ] Implement CV-safe importance computation
- [ ] Test importance stability

### Phase 4: Testing (4-5 hours)
- [ ] Create `tests/leakage/` directory
- [ ] Implement shuffle test
- [ ] Implement ID-only test
- [ ] Implement preprocessor test
- [ ] Run all tests

### Phase 5: Integration (2-3 hours)
- [ ] Run minimal smoke test
- [ ] Compare CV scores before/after
- [ ] Document results
- [ ] Update BUG_TRACKER.md

---

## Summary

### Total Effort: 15-20 hours

| Phase | Hours | Priority |
|-------|-------|----------|
| Preprocessor Fix | 3-4 | 🔴 CRITICAL |
| Target Encoding Fix | 4-5 | 🔴 CRITICAL |
| Null Importance Fix | 2-3 | 🟡 HIGH |
| Testing | 4-5 | 🔴 CRITICAL |
| Integration | 2-3 | 🟡 HIGH |

### Expected Impact

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Leakage Points | 4 | 0 | ✅ 100% |
| CV Inflation | 10-30% | <2% | ✅ REALISTIC |
| CV-LB Gap | 5-10% | <2% | ✅ RELIABLE |
| Submission Confidence | LOW | HIGH | ✅ TRUSTWORTHY |

---

**Document Version:** 1.0  
**Created:** 2026-03-25  
**Ready to Implement:** YES  
**Approval Required:** YES (before implementation)
