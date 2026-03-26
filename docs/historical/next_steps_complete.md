# Next Steps Completion Report

**Date:** 2026-03-24  
**Status:** ✅ PHASE 1 COMPLETE  

---

## Summary

Successfully completed the "Next Steps" from the pseudo_label_agent fix plan:

### ✅ 1. Updated Upstream Agents

#### feature_factory.py
- **Added:** `feature_order` to return state
- **Line:** 1154
- **Change:** `"feature_order": list(X_current.columns)`

#### ml_optimizer.py
- **Added:** Test data processing and `feature_data_path_test` write
- **Lines:** 927-953 (test data processing), 1031 (state return)
- **Changes:**
  - Processes test data through same preprocessor
  - Applies feature_order column selection
  - Saves to `X_test.parquet`
  - Returns `feature_data_path_test` in state

#### core/state.py
- **Added:** State schema declarations for new keys
- **Lines:** 85-86 (TypedDict), 255-256 (initial_state)
- **Changes:**
  - `feature_data_path: Optional[str]`
  - `feature_data_path_test: Optional[str]`

---

## Smoke Test Results

### Before Fixes
```
[MLOptimizer] Attempt 1/3 failed. Error: feature_data_path not in state
[MLOptimizer] Attempt 2/3 failed. Error: feature_data_path not in state
[MLOptimizer] Attempt 3/3 failed. Error: feature_data_path not in state
```

### After Fixes
```
[MLOptimizer] Starting — session: smoke_te_b4591a4e
[MLOptimizer] Data loaded: (100, 6)  ← ✅ Data loads successfully!
[MLOptimizer] Loaded existing contract: auc
[MLOptimizer] Target column: target
[MLOptimizer] Features: 5 | Rows: 100
[MLOptimizer] Running Optuna...  ← ✅ Optuna starts!
```

**Progress:** Pipeline now reaches Optuna optimization phase!

---

## Remaining Issues (Pre-existing Bugs)

### Issue #1: XGBRegressor for Classification Task

**Error:**
```
AttributeError: 'XGBRegressor' object has no attribute 'predict_proba'
```

**Location:** `agents/ml_optimizer.py`, line 328

**Root Cause:** The `_get_model_class` function selects XGBRegressor for a binary classification task when Optuna suggests 'xgb' model type.

**Fix Required:** Update `_get_model_class` to properly select XGBClassifier for classification tasks.

---

### Issue #2: Null Importance Stage 2 Sandbox Failure

**Error:**
```
Import of 'sys' is not allowed in sandbox. Blocked modules: ..., sys, ...
```

**Location:** `tools/null_importance.py`, STAGE2_SCRIPT_TEMPLATE

**Root Cause:** The sandbox script template imports `sys` for error handling, but `sys` is in BLOCKED_MODULES.

**Impact:** Stage 2 null importance filtering always fails, returns all survivors.

**Fix Required:** Remove `sys` import from STAGE2_SCRIPT_TEMPLATE or remove `sys` from BLOCKED_MODULES.

---

### Issue #3: Round 2 LLM Feature Generation

**Warning:**
```
[feature_factory] Round 2 LLM call failed: Expecting ',' delimiter
```

**Root Cause:** LLM generates invalid JSON for feature expressions.

**Impact:** Round 2 domain features are not generated.

**Note:** This is a known limitation - Round 2 is a "nice to have" feature.

---

## Test Results

### pseudo_label_agent Tests
```
================ 12 passed, 34 warnings in 20.05s ================
```
✅ All 12 tests passing

### Smoke Test Pipeline Progress
| Agent | Status |
|-------|--------|
| semantic_router | ✅ Pass |
| competition_intel | ✅ Pass |
| data_engineer | ✅ Pass |
| eda_agent | ✅ Pass |
| validation_architect | ✅ Pass |
| feature_factory | ✅ Pass (with Round 2 fallback) |
| ml_optimizer | ⚠️ Partial (data loads, Optuna fails on XGB bug) |
| red_team_critic | ⏹️ Not reached |
| submit | ⏹️ Not reached |

---

## Files Modified

1. **agents/feature_factory.py** - Added `feature_order` to return state
2. **agents/ml_optimizer.py** - Added test data processing and `feature_data_path_test`
3. **core/state.py** - Added state schema for new keys
4. **agents/pseudo_label_agent.py** - All 20 bugs fixed (previous phase)
5. **tests/agents/test_pseudo_label_agent_fix.py** - 12 tests created

---

## Next Actions Required

### Priority 1: Fix XGB Classifier/Regressor Selection
**File:** `agents/ml_optimizer.py`  
**Function:** `_get_model_class`  
**Estimated Time:** 15 minutes

### Priority 2: Fix Null Importance Sandbox Import
**File:** `tools/null_importance.py`  
**Constant:** STAGE2_SCRIPT_TEMPLATE  
**Estimated Time:** 10 minutes

### Priority 3: Run Full Smoke Test
After Priority 1-2 fixes, run:
```bash
python run_smoke_test.py
```
**Expected:** Pipeline completes to submission

---

## Verification Checklist

- [x] feature_factory writes `feature_data_path`
- [x] feature_factory writes `feature_order`
- [x] ml_optimizer writes `feature_data_path_test`
- [x] State schema declares new keys
- [x] State initialization includes new keys
- [x] pseudo_label_agent tests pass (12/12)
- [x] Smoke test reaches ml_optimizer
- [ ] Smoke test completes to submission (blocked by XGB bug)

---

**Document Version:** 1.0  
**Status:** ✅ PHASE 1 COMPLETE, PHASE 2 READY TO START
