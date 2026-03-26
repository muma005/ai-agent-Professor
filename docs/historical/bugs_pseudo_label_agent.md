# Bug Report: agents/pseudo_label_agent.py

**File:** `c:\Users\ADMIN\Desktop\Professor\ai-agent-Professor\agents\pseudo_label_agent.py`

**Date:** 2026-03-24

**Status:** 🔴 CRITICAL — Agent is completely non-functional

**Summary:** The pseudo-labeling agent contains 20 documented bugs, including 5 critical undefined variable errors that cause immediate `NameError` crashes on invocation. The agent cannot execute any pseudo-labeling logic in its current state.

---

## Executive Summary

| Category | Count |
|----------|-------|
| **CRITICAL** | 9 |
| **HIGH** | 5 |
| **MEDIUM** | 5 |
| **LOW** | 2 |
| **TOTAL** | **20** |

### Root Causes

1. **Undefined variables** — `X_train`, `X_test`, `y_train`, `metric` are used but never loaded from disk or state
2. **Missing imports** — `is_significantly_better()` called but not imported
3. **State contract violations** — Agent expects `feature_data_path`, `feature_data_path_test`, `selected_models` that upstream agents never write
4. **Logic errors** — Soft labels used instead of hard labels, stale baseline comparisons
5. **Data integrity gaps** — No target column extraction, no feature alignment validation

---

## Critical Bugs (Agent Crashes)

### Bug #1: Undefined variable `X_train`

**Location:** Line 217

**Code:**
```python
X_pseudo_accumulated = pl.DataFrame(schema=X_train.schema)
```

**Problem:** `X_train` is never defined. The agent reads `feature_data_path` from state at line 161 but never calls `read_parquet()` to load the data.

**Error:** `NameError: name 'X_train' is not defined`

**Fix:**
```python
# Add after line 162
X_train = read_parquet(feature_data_path)
target_col = state.get("target_col")
y_train = X_train[target_col].to_numpy()
X_train = X_train.drop(target_col)
```

---

### Bug #2: Undefined variable `y_train`

**Location:** Line 218

**Code:**
```python
y_pseudo_accumulated = np.array([], dtype=y_train.dtype)
```

**Problem:** `y_train` is never extracted from the training data. No code separates the target column from features.

**Error:** `NameError: name 'y_train' is not defined`

**Fix:** See Bug #1 fix — extract target column after loading.

---

### Bug #3: Undefined variable `X_test`

**Location:** Line 220

**Code:**
```python
current_test_mask = np.zeros(len(X_test), dtype=bool)
```

**Problem:** `X_test` is never loaded. The agent reads `feature_data_path_test` at line 162 but never loads the parquet file.

**Error:** `NameError: name 'X_test' is not defined`

**Fix:**
```python
# Add after line 162
X_test = read_parquet(feature_data_path_test)
```

---

### Bug #4: Undefined variable `metric`

**Location:** Line 237

**Code:**
```python
is_cls = metric in ("auc", "logloss", "binary")
```

**Problem:** `metric` is never read from the metric contract or state. The agent needs to know which metric to optimize for confidence computation.

**Error:** `NameError: name 'metric' is not defined`

**Fix:**
```python
# Add after line 162
from tools.data_tools import read_json
metric_contract = read_json(state["metric_contract_path"])
metric = metric_contract.get("scorer_name", "auc")
```

---

### Bug #5: Missing import for `is_significantly_better`

**Location:** Line 329

**Code:**
```python
gate_passed = is_significantly_better(cv_with, baseline_cv)
```

**Problem:** The Wilcoxon gate function is called but never imported. It exists in `tools/wilcoxon_gate.py`.

**Error:** `NameError: name 'is_significantly_better' is not defined`

**Fix:**
```python
# Add to imports at top of file (line 18)
from tools.wilcoxon_gate import is_significantly_better
```

---

### Bug #6: `feature_data_path` not set by upstream agent

**Location:** Lines 161-162

**Code:**
```python
feature_data_path = state.get("feature_data_path")
feature_data_path_test = state.get("feature_data_path_test")
```

**Problem:** No upstream agent writes these keys to state:
- `data_engineer` writes `clean_data_path` (parquet)
- `ml_optimizer` writes model artifacts but not feature matrix paths
- `feature_factory` writes `feature_manifest` but not data paths

**Impact:** Agent always skips at lines 164-170 because paths are `None`.

**Fix Options:**
1. Have `ml_optimizer` write these paths after feature preparation
2. Have pseudo_label_agent reconstruct from `clean_data_path`:
```python
session_id = state["session_id"]
feature_data_path = f"outputs/{session_id}/X_train.parquet"
feature_data_path_test = f"outputs/{session_id}/X_test.parquet"
```

---

### Bug #7: `selected_models` not set (ensemble_architect not in pipeline)

**Location:** Line 174

**Code:**
```python
selected = state.get("selected_models", [])
```

**Problem:** `selected_models` is set by `ensemble_architect.blend_models()`, but `ensemble_architect` is **never added as a node** in the LangGraph graph (`professor.py` lines 131-141).

**Impact:** `selected` is always empty, agent skips at lines 175-182.

**Fix Options:**
1. Add `ensemble_architect` node to `professor.py` graph
2. Have `ml_optimizer` set `selected_models` directly after training

---

### Bug #8: Target column not extracted (data leakage)

**Location:** Throughout main function (lines 217-365)

**Code:** No target extraction anywhere

**Problem:** Even if `X_train` were loaded, there's no code to separate features from target. The target column name is available via `state.get("target_col")` but never used.

**Impact:** If agent ran, it would train with target as a feature — severe data leakage.

**Fix:**
```python
# Add after loading X_train
target_col = state.get("target_col")
y_train = X_train[target_col].to_numpy()
X_train = X_train.drop(target_col)

# Also drop from X_test
if target_col in X_test.columns:
    X_test = X_test.drop(target_col)
```

---

### Bug #9: No feature alignment between train and test

**Location:** Lines 217, 220

**Problem:** No code ensures `X_train` and `X_test` have the same columns in the same order. The `ml_optimizer` should save `feature_order` to state but doesn't.

**Impact:** Model predicts on misaligned features, producing garbage predictions.

**Fix:**
```python
# After loading both DataFrames
feature_order = state.get("feature_order")
if feature_order:
    X_test = X_test.select(feature_order)
    X_train = X_train.select(feature_order)
```

---

## High Severity Bugs (Logic Errors)

### Bug #10: Soft labels used instead of hard labels

**Location:** Lines 280, 283-284

**Code:**
```python
# Line 280 - uses soft labels
y_new_pseudo = y_pred[conf_mask]

# Lines 283-284 - converts to hard labels (too late!)
if is_cls:
    y_new_pseudo = (y_new_pseudo >= 0.5).astype(y_train.dtype)
```

**Problem:** For classification, `y_pred` contains probabilities from `predict_proba()[:, 1]`. The conversion to hard labels (0/1) happens AFTER `y_new_pseudo` is already used at line 280.

**Impact:** Model trains on probability values (e.g., 0.73) as if they were class labels, corrupting the learning signal.

**Fix:**
```python
# Move conversion BEFORE using y_new_pseudo
y_new_pseudo = y_pred[conf_mask]
if is_cls:
    y_new_pseudo = (y_new_pseudo >= 0.5).astype(y_train.dtype)  # Move up
```

---

### Bug #11: Wilcoxon gate receives wrong data structure

**Location:** Line 329

**Code:**
```python
gate_passed = is_significantly_better(cv_with, baseline_cv)
```

**Problem:** 
- `is_significantly_better()` expects two lists of **fold scores** (e.g., `[0.81, 0.79, 0.82, 0.80, 0.83]`)
- `cv_with` is a list of fold scores from one CV run
- `baseline_cv` is loaded from registry at line 204 as fold scores

But the comparison is inconsistent — `baseline_cv` is the original model's fold scores, while `cv_with` is from the current iteration with pseudo-labels. The gate should compare iteration N vs iteration N-1, not iteration N vs original model.

**Impact:** Wilcoxon test compares against increasingly outdated baseline.

**Fix:**
```python
# Track previous iteration's fold scores
prev_fold_scores = baseline_cv  # Initialize from original model

# Inside loop, after cv_with is computed:
gate_passed = is_significantly_better(cv_with, prev_fold_scores)
prev_fold_scores = cv_with  # Update for next iteration
```

---

### Bug #12: Test data may contain target column

**Location:** Line 245

**Code:**
```python
X_remaining = X_test.filter(pl.Series(remaining_mask))
```

**Problem:** If `X_test` contains the target column (common when Kaggle provides test labels separately for validation), it would leak into predictions.

**Impact:** Data leakage — model "predicts" using the actual target.

**Fix:**
```python
target_col = state.get("target_col")
if target_col in X_test.columns:
    X_test = X_test.drop(target_col)
```

---

### Bug #13: `model_registry` schema may not match

**Location:** Lines 185-195

**Code:**
```python
if isinstance(registry, list):
    for entry in registry:
        if isinstance(entry, dict) and entry.get("model_type") == best_model_name:
```

**Problem:** The code assumes registry entries have `model_type` key. The `ProfessorState` schema (state.py line 109) shows `model_registry: Annotated[Optional[list], _replace]` but doesn't define the entry schema. If `ml_optimizer` uses a different key (e.g., `name`, `model_name`), lookup fails.

**Impact:** `best_entry` remains `{}`, agent uses default params.

**Fix:** Verify `ml_optimizer` writes `model_type` key, or add fallback lookups:
```python
best_entry = None
for entry in registry:
    model_name = entry.get("model_type") or entry.get("name") or entry.get("model_name")
    if model_name == best_model_name:
        best_entry = entry
        break
```

---

### Bug #14: Baseline CV comparison is stale

**Location:** Line 329

**Code:**
```python
gate_passed = is_significantly_better(cv_with, baseline_cv)
```

**Problem:** `baseline_cv` is set once at line 204 from the model registry. It should be updated after each successful iteration to compare against the previous iteration's results.

**Impact:** Later iterations compare against the original model, not the previous pseudo-labeled model.

**Fix:** Already partially done at line 344 (`baseline_cv = cv_with`), but the gate at line 329 uses stale baseline. Move the update before the gate check.

---

## Medium Severity Bugs

### Bug #15: Inconsistent fold score vs mean treatment

**Location:** Lines 157, 308

**Code:**
```python
# Line 157 in _run_cv_with_pseudo_labels
fold_scores.append(float(score))
# ...
return fold_scores  # Returns list of fold scores

# Line 308 in main function
cv_mean_with = float(np.mean(cv_with))  # Treats as list, takes mean
```

**Problem:** The function returns a list of individual fold scores, but callers inconsistently treat it as either means or fold lists.

**Impact:** Confusing API, potential bugs in statistical tests.

**Fix:** Document clearly that `_run_cv_with_pseudo_labels` returns fold scores (list), and all callers must call `np.mean()` themselves.

---

### Bug #16: Type mismatch in pseudo-label concatenation

**Location:** Line 280

**Code:**
```python
y_new_pseudo = y_pred[conf_mask]
```

**Problem:** `y_pred` is `float64` (probabilities), but `y_train.dtype` is likely `int32` or `int64`. Concatenating at line 287 may fail or produce unexpected dtype.

**Impact:** `TypeError` or silent dtype coercion.

**Fix:**
```python
y_new_pseudo = y_pred[conf_mask]
if is_cls:
    y_new_pseudo = (y_new_pseudo >= 0.5).astype(y_train.dtype)
else:
    y_new_pseudo = y_new_pseudo.astype(y_train.dtype)
```

---

### Bug #17: No try/except around model training

**Location:** Lines 238-240

**Code:**
```python
ModelClass = lgb.LGBMClassifier if is_cls else lgb.LGBMRegressor
model = ModelClass(**lgbm_params)
model.fit(X_all.to_numpy(), y_all)
```

**Problem:** If model training fails (OOM, convergence failure, invalid params), the exception propagates without logging which iteration failed.

**Impact:** Difficult to debug iteration-specific failures.

**Fix:**
```python
try:
    model = ModelClass(**lgbm_params)
    model.fit(X_all.to_numpy(), y_all)
except Exception as e:
    logger.error(f"[pseudo_label] Iteration {iteration}: model training failed: {e}")
    result.halt_reason = f"model_training_failed: {e}"
    result.halted_early = True
    break
```

---

### Bug #18: No validation of loaded data

**Location:** Lines 161-170

**Code:**
```python
if not feature_data_path or not os.path.exists(feature_data_path):
    logger.warning("[pseudo_label] feature_data_path missing. Skipping.")
    return {**state, ...}
```

**Problem:** Files are checked for existence but not validated (empty files, wrong schema, missing columns, zero rows).

**Impact:** Agent proceeds with invalid data, crashes later with confusing error.

**Fix:**
```python
X_train = read_parquet(feature_data_path)
if X_train.is_empty():
    raise ValueError("Training data is empty")
if target_col not in X_train.columns:
    raise ValueError(f"Target '{target_col}' not in training data")
```

---

## Low Severity Bugs

### Bug #19: Mutable defaults in dataclass

**Location:** Lines 30-39

**Code:**
```python
@dataclass
class PseudoLabelResult:
    pseudo_labels_added: list[int] = []
    cv_scores_with_pl: list[float] = []
    cv_improvements: list[float] = []
```

**Problem:** Using mutable defaults (`[]`) in dataclasses is an anti-pattern that can cause shared state bugs across instances.

**Impact:** Potential cross-invocation state pollution if default is ever used.

**Fix:**
```python
from dataclasses import dataclass, field

@dataclass
class PseudoLabelResult:
    pseudo_labels_added: list[int] = field(default_factory=list)
    cv_scores_with_pl: list[float] = field(default_factory=list)
    cv_improvements: list[float] = field(default_factory=list)
```

---

### Bug #20: Incomplete memory cleanup in early returns

**Location:** Lines 256-259, 271-273

**Code:**
```python
if X_remaining.is_empty():
    result.halt_reason = "no_confident_samples"
    result.halted_early = True
    del model; gc.collect()  # Only deletes model
```

**Problem:** Large arrays (`X_all`, `y_all`, `y_pred`, `confidence`, `X_remaining`) are not explicitly deleted before early returns.

**Impact:** Memory accumulation across iterations, potential OOM on large datasets.

**Fix:**
```python
def _cleanup():
    """Delete large arrays and run GC."""
    nonlocal X_all, y_all, y_pred, confidence, X_remaining, model
    for obj in [X_all, y_all, y_pred, confidence, X_remaining, model]:
        if obj is not None:
            del obj
    gc.collect()

# Call in early returns
if X_remaining.is_empty():
    result.halt_reason = "no_confident_samples"
    result.halted_early = True
    _cleanup()
    break
```

---

## Required Imports (Missing)

Add these imports at the top of the file (after line 18):

```python
from tools.data_tools import read_parquet, read_json
from tools.wilcoxon_gate import is_significantly_better
```

---

## State Contract Requirements

For this agent to function, the following state keys **must** be set by upstream agents:

| Key | Expected Type | Should Be Set By |
|-----|---------------|------------------|
| `feature_data_path` | `str` (path to parquet) | `ml_optimizer` or `feature_factory` |
| `feature_data_path_test` | `str` (path to parquet) | `ml_optimizer` or `feature_factory` |
| `target_col` | `str` | `data_engineer` ✓ (already set) |
| `selected_models` | `list[str]` | `ensemble_architect` or `ml_optimizer` |
| `model_registry` | `list[dict]` with `model_type`, `params`, `fold_scores` | `ml_optimizer` ✓ (already set) |
| `metric_contract_path` | `str` (path to JSON) | `validation_architect` ✓ (already set) |

---

## Recommended Fix Priority

### Phase 1: Make agent runnable (fix crashes)
1. Bug #1-5: Add missing variable loads and imports
2. Bug #8: Extract target column from training data
3. Bug #13: Drop target from test data

### Phase 2: Fix state contracts
4. Bug #6: Have upstream agent write `feature_data_path` keys
5. Bug #7: Add `ensemble_architect` to pipeline or set `selected_models` in `ml_optimizer`
6. Bug #9: Add feature alignment using `feature_order`

### Phase 3: Fix logic errors
7. Bug #10: Move hard label conversion before use
8. Bug #11: Fix Wilcoxon gate baseline comparison
9. Bug #14: Update baseline after each iteration

### Phase 4: Hardening
10. Bug #16-20: Add validation, error handling, memory cleanup

---

## Complete Fixed Code Skeleton

```python
def run_pseudo_label_agent(state: ProfessorState) -> ProfessorState:
    from tools.data_tools import read_parquet, read_json
    from tools.wilcoxon_gate import is_significantly_better
    import os

    # ── Load paths from state ────────────────────────────────────
    feature_data_path = state.get("feature_data_path")
    feature_data_path_test = state.get("feature_data_path_test")

    # Fallback: reconstruct from session_id if not set
    if not feature_data_path:
        session_id = state["session_id"]
        feature_data_path = f"outputs/{session_id}/X_train.parquet"
    if not feature_data_path_test:
        session_id = state["session_id"]
        feature_data_path_test = f"outputs/{session_id}/X_test.parquet"

    if not os.path.exists(feature_data_path):
        logger.warning(f"[pseudo_label] Training data not found: {feature_data_path}. Skipping.")
        return {**state, "pseudo_labels_applied": False, "pseudo_label_cv_improvement": 0.0}

    if not os.path.exists(feature_data_path_test):
        logger.warning(f"[pseudo_label] Test data not found: {feature_data_path_test}. Skipping.")
        return {**state, "pseudo_labels_applied": False, "pseudo_label_cv_improvement": 0.0}

    # ── Load data from disk ──────────────────────────────────────
    X_train = read_parquet(feature_data_path)
    X_test = read_parquet(feature_data_path_test)

    # ── Extract target column ────────────────────────────────────
    target_col = state.get("target_col")
    if not target_col:
        raise ValueError("[pseudo_label] target_col not set in state")

    y_train = X_train[target_col].to_numpy()
    X_train = X_train.drop(target_col)

    # Drop target from test if present
    if target_col in X_test.columns:
        X_test = X_test.drop(target_col)

    # ── Enforce feature order ────────────────────────────────────
    feature_order = state.get("feature_order")
    if feature_order:
        X_train = X_train.select(feature_order)
        X_test = X_test.select(feature_order)

    # ── Load metric ──────────────────────────────────────────────
    metric_contract_path = state.get("metric_contract_path")
    if metric_contract_path and os.path.exists(metric_contract_path):
        metric_contract = read_json(metric_contract_path)
        metric = metric_contract.get("scorer_name", "auc")
    else:
        metric = "auc"

    # ── Rest of function continues with selected_models lookup... ──
    # [Continue with existing logic, now with all variables defined]
```

---

## Verification Checklist

After fixes, verify:

- [ ] Agent loads `X_train`, `X_test`, `y_train` from disk
- [ ] Agent reads `metric` from metric contract
- [ ] Agent imports `is_significantly_better`
- [ ] Target column is dropped from features before training
- [ ] Feature order is enforced between train and test
- [ ] Hard labels are used for classification pseudo-labels
- [ ] Wilcoxon gate compares iteration N vs N-1
- [ ] Memory is cleaned up in all early return paths
- [ ] Upstream agent writes `feature_data_path` and `feature_data_path_test`
- [ ] `selected_models` is set before pseudo_label_agent runs

---

**Document Version:** 1.0

**Author:** AI Code Review

**Next Review:** After Phase 1 fixes are implemented
