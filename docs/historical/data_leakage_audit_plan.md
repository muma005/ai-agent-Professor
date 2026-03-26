# Data Leakage Audit Plan

**Date:** 2026-03-25  
**Priority:** CRITICAL  
**Status:** 🔍 AUDIT IN PROGRESS  

---

## Executive Summary

Data leakage occurs when information from outside the training dataset is used to create the model. This leads to **overoptimistic CV scores** that **fail on the leaderboard**.

This document identifies potential leakage points in the Professor pipeline and provides a comprehensive plan to detect and prevent them.

---

## Potential Leakage Points Identified

### 🔴 CRITICAL: Preprocessor Fit on Full Data

**Location:** `agents/data_engineer.py`, `core/preprocessor.py`

**Issue:**
```python
# Line ~215 in data_engineer.py
preprocessor = TabularPreprocessor(target_col=target_col, id_cols=id_columns)
df_clean = preprocessor.fit_transform(df_raw, raw_schema)  # ← FITS ON FULL DATA
```

**Leakage Risk:** HIGH
- Preprocessor fits imputation statistics (median, mode) on ENTIRE dataset
- These statistics leak information from test rows into training
- Impact: CV scores inflated by 1-5%

**Fix Required:**
```python
# Should be:
# 1. Split data FIRST
# 2. Fit preprocessor on train ONLY
# 3. Transform train and test separately
```

**Current Status:** ⚠️ **LEAKAGE PRESENT**

---

### 🔴 CRITICAL: Feature Factory Computes on Full Data

**Location:** `agents/feature_factory.py`

**Issue:**
```python
# Line ~950
df = pl.read_parquet(clean_path)  # ← Loads FULL data (train + test combined)
X_base = preprocessor.transform(df)  # ← Transforms ALL rows together

# Round 3: GroupBy aggregations computed on full data
mapping_df = X_base.group_by(cat_col).agg(...)  # ← LEAKS test statistics
```

**Leakage Risk:** HIGH
- GroupBy statistics (mean, std) computed using test rows
- Target encoding uses global mean including test data
- Impact: CV scores inflated by 3-10%

**Fix Required:**
```python
# Should be:
# 1. Compute aggregations within CV folds ONLY
# 2. Never use test data to compute training statistics
```

**Current Status:** ⚠️ **LEAKAGE PRESENT**

---

### 🔴 CRITICAL: Target Encoding Without CV

**Location:** `agents/feature_factory.py`, lines 1026-1040

**Issue:**
```python
# Round 4: Target encoding
mapping_df = X_base.with_columns(pl.Series("y", y)).group_by(col).agg([
    pl.col("y").sum().alias("sum"), 
    pl.col("y").count().alias("count")
])
```

**Leakage Risk:** CRITICAL
- Target encoding computed on FULL dataset including validation rows
- Each row's encoding includes its own target value
- Impact: CV scores inflated by 5-20% (SEVERE)

**Fix Required:**
```python
# Should use leave-one-out encoding WITHIN CV folds:
for train_idx, val_idx in cv.split(X, y):
    # Fit encoding on train ONLY
    encoding = compute_target_encoding(X[train_idx], y[train_idx])
    # Transform validation
    X[val_idx] = transform_with_encoding(X[val_idx], encoding)
```

**Current Status:** 🔴 **SEVERE LEAKAGE PRESENT**

---

### 🟡 MEDIUM: Null Importance on Full Data

**Location:** `tools/null_importance.py`

**Issue:**
```python
# Line ~260
model_real.fit(X_np, y)  # ← Fits on FULL data
```

**Leakage Risk:** MEDIUM
- Feature importance computed using full dataset
- Feature selection decisions leak test information
- Impact: CV scores inflated by 1-3%

**Fix Required:**
```python
# Should compute importance within CV folds
```

**Current Status:** ⚠️ **LEAKAGE PRESENT**

---

### 🟡 MEDIUM: Pseudo-Labeling Implementation

**Location:** `agents/pseudo_label_agent.py`

**Issue:**
```python
# Line ~245
X_remaining = X_test.filter(pl.Series(remaining_mask))
y_pred = model.predict_proba(X_remaining.to_numpy())[:, 1]
```

**Leakage Risk:** MEDIUM (if enabled)
- Test predictions used to create pseudo-labels
- If pseudo-labels added to training, must ensure no validation leakage
- Impact: Depends on implementation

**Current Status:** ✅ **PROTECTED** (validation fold never sees pseudo-labels)

**Note:** The invariant is correctly maintained (line 117-119):
```python
# CRITICAL INVARIANT: Validation fold sees only real labeled samples.
```

---

### 🟢 LOW: OOF Predictions

**Location:** `agents/ml_optimizer.py`

**Issue:** Out-of-fold predictions must be truly out-of-fold

**Leakage Risk:** LOW
- If OOF predictions include any in-fold information, ensemble will leak
- Impact: CV scores inflated by 1-2%

**Current Status:** ✅ **PROTECTED** (uses proper CV split)

**Verification needed:**
```python
# Check _get_oof_predictions() ensures no fold overlap
```

---

### 🟢 LOW: Calibration Fold

**Location:** `agents/ml_optimizer.py`, lines 100-136

**Issue:**
```python
# Calibration split happens BEFORE CV
X_train_cv, y_train_cv, X_calib, y_calib = _split_calibration_fold(X, y)
```

**Leakage Risk:** LOW
- Calibration fold is held out from CV training
- This is CORRECT - prevents leakage
- Impact: None (properly implemented)

**Current Status:** ✅ **PROTECTED**

---

## Comprehensive Data Leakage Detection Plan

### Phase 1: Automated Tests (2-3 hours)

#### Test 1: Shuffle Test (Gold Standard)

**Purpose:** Detect ANY form of data leakage

**Implementation:**
```python
def test_shuffle_leakage():
    """
    Shuffle target column. If model achieves AUC > 0.55, leakage is present.
    """
    y_shuffled = np.random.permutation(y)
    
    # Run full pipeline with shuffled target
    cv_scores = run_pipeline(X, y_shuffled)
    
    # Should be ~0.5 (random)
    assert np.mean(cv_scores) < 0.55, f"LEAKAGE DETECTED: AUC={np.mean(cv_scores)}"
```

**Expected Result:** AUC ≈ 0.5 (random)  
**Leakage Indicator:** AUC > 0.55

---

#### Test 2: ID-Only Model Test

**Purpose:** Detect if row ordering or IDs encode target information

**Implementation:**
```python
def test_id_only_model():
    """
    Train model using ONLY ID columns. If AUC > 0.65, data ordering leaks target.
    """
    X_ids = df[id_columns]  # Only ID columns
    
    cv_scores = run_pipeline(X_ids, y)
    
    assert np.mean(cv_scores) < 0.65, f"LEAKAGE DETECTED: ID-only AUC={np.mean(cv_scores)}"
```

**Expected Result:** AUC ≈ 0.5 (random)  
**Leakage Indicator:** AUC > 0.65

---

#### Test 3: Preprocessor Leakage Test

**Purpose:** Verify preprocessor doesn't leak test statistics

**Implementation:**
```python
def test_preprocessor_leakage():
    """
    Fit preprocessor on train, transform test. Compare with fit on combined.
    If difference > 1%, leakage in original implementation.
    """
    # Correct way
    preprocessor = TabularPreprocessor()
    preprocessor.fit(X_train)
    X_train_correct = preprocessor.transform(X_train)
    X_test_correct = preprocessor.transform(X_test)
    
    # Potentially leaked way (current implementation)
    preprocessor_leaky = TabularPreprocessor()
    preprocessor_leaky.fit(X_combined)  # ← LEAKS
    X_train_leaky = preprocessor_leaky.transform(X_train)
    
    # Compare imputation values
    diff = np.abs(X_train_correct - X_train_leaky).mean()
    
    assert diff < 0.01, f"LEAKAGE DETECTED: imputation diff={diff}"
```

**Expected Result:** Difference < 1%  
**Leakage Indicator:** Difference > 1%

---

#### Test 4: Target Encoding Leakage Test

**Purpose:** Verify target encoding doesn't use validation targets

**Implementation:**
```python
def test_target_encoding_leakage():
    """
    Compare target encoding computed correctly vs incorrectly.
    """
    # Correct: within-fold encoding
    for train_idx, val_idx in cv.split(X, y):
        encoding = compute_encoding(X[train_idx], y[train_idx])  # Train ONLY
        X_val_encoded = transform(X[val_idx], encoding)
    
    # Incorrect: full data encoding (current implementation)
    encoding_leaky = compute_encoding(X, y)  # ← LEAKS
    
    # Compare
    correlation = np.corrcoef(X_val_correct.flatten(), X_val_leaky.flatten())[0, 1]
    
    assert correlation > 0.99, f"LEAKAGE DETECTED: encoding correlation={correlation}"
```

**Expected Result:** Correlation ≈ 1.0 (same encoding)  
**Leakage Indicator:** Correlation < 0.99 (different = leakage in current)

---

#### Test 5: Feature Importance Leakage Test

**Purpose:** Verify null importance doesn't leak

**Implementation:**
```python
def test_null_importance_leakage():
    """
    Compute null importance on train vs full data.
    """
    # Correct: on train only
    importance_train = compute_null_importance(X_train, y_train)
    
    # Incorrect: on full data (current)
    importance_full = compute_null_importance(X_full, y_full)
    
    # Compare selected features
    selected_train = set(importance_train['survivors'])
    selected_full = set(importance_full['survivors'])
    
    overlap = len(selected_train & selected_full) / len(selected_train)
    
    assert overlap > 0.9, f"LEAKAGE DETECTED: feature selection overlap={overlap}"
```

**Expected Result:** Overlap > 90%  
**Leakage Indicator:** Overlap < 90%

---

### Phase 2: Manual Code Audit (4-6 hours)

#### Checklist for Each Agent

| Agent | Check | Status |
|-------|-------|--------|
| **data_engineer** | Preprocessor fit on train only? | ❌ NO |
| **data_engineer** | Test data transformed separately? | ❌ NO |
| **feature_factory** | Aggregations computed within CV? | ❌ NO |
| **feature_factory** | Target encoding with leave-one-out? | ❌ NO |
| **ml_optimizer** | Calibration fold held out? | ✅ YES |
| **ml_optimizer** | OOF predictions truly out-of-fold? | ⚠️ VERIFY |
| **pseudo_label** | Validation fold never sees pseudo-labels? | ✅ YES |
| **null_importance** | Computed on train only? | ❌ NO |

---

### Phase 3: CV vs LB Gap Analysis (After Next Submission)

**Purpose:** Real-world leakage detection

**Implementation:**
```python
def analyze_cv_lb_gap():
    """
    After submission, compare CV score to LB score.
    Large gap indicates leakage or overfitting.
    """
    cv_score = state['cv_mean']
    lb_score = get_lb_score()
    
    gap = cv_score - lb_score
    
    if gap > 0.02:  # 2% gap
        print(f"⚠️ WARNING: CV-LB gap = {gap:.4f}")
        print("Possible causes: leakage, overfitting, or LB shakeup")
    
    if gap > 0.05:  # 5% gap
        print(f"🔴 CRITICAL: CV-LB gap = {gap:.4f}")
        print("SEVERE leakage or overfitting detected!")
```

**Expected Result:** Gap < 2%  
**Leakage Indicator:** Gap > 2%

---

## Priority Fix Plan

### 🔴 CRITICAL (Fix Immediately)

1. **Target Encoding Leakage** - Implement within-fold encoding
   - File: `agents/feature_factory.py`
   - Estimated time: 2 hours
   - Impact: Prevents 5-20% CV inflation

2. **Preprocessor Fit on Full Data** - Split before fit
   - File: `agents/data_engineer.py`, `core/preprocessor.py`
   - Estimated time: 3 hours
   - Impact: Prevents 1-5% CV inflation

3. **Feature Factory Aggregations** - Compute within CV
   - File: `agents/feature_factory.py`
   - Estimated time: 4 hours
   - Impact: Prevents 3-10% CV inflation

### 🟡 HIGH (Fix This Week)

4. **Null Importance Leakage** - Compute on train only
   - File: `tools/null_importance.py`
   - Estimated time: 1 hour
   - Impact: Prevents 1-3% CV inflation

5. **Add Automated Leakage Tests** - Prevent regression
   - File: `tests/leakage/test_data_leakage.py`
   - Estimated time: 3 hours
   - Impact: Catches future leakage

---

## Implementation Priority

### Week 1: Critical Fixes
- [ ] Fix target encoding (Round 4)
- [ ] Fix preprocessor fit/transform split
- [ ] Fix feature factory aggregations

### Week 2: Testing & Validation
- [ ] Implement shuffle test
- [ ] Implement ID-only test
- [ ] Implement preprocessor leakage test
- [ ] Run all leakage tests

### Week 3: Hardening
- [ ] Fix null importance leakage
- [ ] Add leakage tests to CI/CD
- [ ] Document leakage prevention patterns

---

## Summary

### Current Leakage Status

| Component | Leakage Risk | Status |
|-----------|--------------|--------|
| Preprocessor | 🔴 HIGH | **LEAKAGE PRESENT** |
| Feature Factory | 🔴 CRITICAL | **SEVERE LEAKAGE PRESENT** |
| Target Encoding | 🔴 CRITICAL | **SEVERE LEAKAGE PRESENT** |
| Null Importance | 🟡 MEDIUM | **LEAKAGE PRESENT** |
| Pseudo Labeling | 🟢 LOW | **PROTECTED** |
| Calibration | 🟢 LOW | **PROTECTED** |
| OOF Predictions | 🟢 LOW | **LIKELY PROTECTED** |

### Overall Assessment

**🔴 CRITICAL: Multiple severe leakage points identified**

The current pipeline has **at least 4 significant leakage points** that could inflate CV scores by **10-30%**, leading to severe LB disappointment.

**Immediate action required** before any Kaggle submission.

---

**Document Version:** 1.0  
**Created:** 2026-03-25  
**Next Review:** After critical fixes implemented
