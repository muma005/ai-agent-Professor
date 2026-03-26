# Pseudo-Label Agent — Comprehensive Bug Fix Plan

**Document Type:** Implementation Plan  
**Priority:** CRITICAL (Blocks pipeline execution)  
**Risk Level:** HIGH (Must prevent regression)  
**Estimated Effort:** 4-6 hours  

---

## Executive Summary

The `pseudo_label_agent.py` contains **20 documented bugs** that make it completely non-functional. This plan provides a water-tight, regression-proof approach to fixing all bugs while maintaining compatibility with the rest of the pipeline.

### Key Principles

1. **Minimal changes to existing interfaces** — Don't change state contracts that other agents depend on
2. **Defensive programming** — Validate all inputs, handle missing data gracefully
3. **Test-driven** — Write tests BEFORE implementing fixes
4. **Backward compatible** — Support both list and dict formats for model_registry
5. **Memory safe** — Explicit cleanup in all code paths

---

## Phase 0: Pre-Fix Preparation (30 minutes)

### Step 0.1: Create Backup

```bash
cd c:\Users\ADMIN\Desktop\Professor\ai-agent-Professor
git status
git add agents/pseudo_label_agent.py
git commit -m "backup: pseudo_label_agent before bug fix"
```

### Step 0.2: Create Test Fixture

Create `tests/agents/test_pseudo_label_agent_fix.py`:

```python
"""
Test fixture for pseudo_label_agent bug fixes.
Run BEFORE making any changes to establish baseline.
Run AFTER each fix to verify no regression.
"""
import pytest
import numpy as np
import polars as pl
from unittest.mock import patch, MagicMock
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from agents.pseudo_label_agent import (
    run_pseudo_label_agent,
    _compute_confidence,
    _select_confident_samples,
    _run_cv_with_pseudo_labels,
    PseudoLabelResult,
)
from core.state import initial_state


class TestPseudoLabelAgentFixtures:
    """Test fixtures that MUST pass after fixes."""
    
    @pytest.fixture
    def minimal_state(self, tmp_path):
        """Create minimal valid state for pseudo_label_agent."""
        session_id = "test_pl_fix"
        output_dir = tmp_path / "outputs" / session_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create synthetic train data
        n_rows, n_features = 100, 5
        X = np.random.randn(n_rows, n_features)
        y = (X[:, 0] + X[:, 1] + np.random.randn(n_rows) * 0.5 > 0).astype(int)
        
        feature_cols = [f"feature_{i}" for i in range(n_features)]
        train_df = pl.DataFrame({
            **{col: X[:, i] for i, col in enumerate(feature_cols)},
            "target": y
        })
        test_df = pl.DataFrame({
            col: X[:50, i] for i, col in enumerate(feature_cols)
        })
        
        train_path = output_dir / "X_train.parquet"
        test_path = output_dir / "X_test.parquet"
        train_df.write_parquet(train_path)
        test_df.write_parquet(test_path)
        
        # Create metric contract
        metric_contract = {
            "scorer_name": "auc",
            "direction": "maximize",
            "requires_proba": True,
            "task_type": "classification"
        }
        import json
        with open(output_dir / "metric_contract.json", "w") as f:
            json.dump(metric_contract, f)
        
        # Create model registry entry
        model_registry = [{
            "model_type": "lgbm",
            "model_path": str(output_dir / "model.pkl"),
            "params": {"n_estimators": 10, "verbosity": -1},
            "fold_scores": [0.75, 0.78, 0.76, 0.77, 0.74],
            "cv_mean": 0.76
        }]
        
        # Create dummy model file
        import pickle
        with open(output_dir / "model.pkl", "wb") as f:
            pickle.dump(MagicMock(), f)
        
        state = initial_state(
            competition="test_pl",
            data_path=str(train_path),
            budget_usd=0.10
        )
        
        # Set required state keys
        state["session_id"] = session_id
        state["feature_data_path"] = str(train_path)
        state["feature_data_path_test"] = str(test_path)
        state["target_col"] = "target"
        state["metric_contract_path"] = str(output_dir / "metric_contract.json")
        state["model_registry"] = model_registry
        state["selected_models"] = ["lgbm"]
        
        return state
    
    def test_state_has_required_keys(self, minimal_state):
        """Verify test fixture has all required state keys."""
        required = [
            "feature_data_path",
            "feature_data_path_test", 
            "target_col",
            "metric_contract_path",
            "model_registry",
            "selected_models"
        ]
        for key in required:
            assert key in minimal_state, f"Missing required key: {key}"
            assert minimal_state[key] is not None, f"Key is None: {key}"
    
    def test_data_files_exist(self, minimal_state):
        """Verify data files exist on disk."""
        assert os.path.exists(minimal_state["feature_data_path"])
        assert os.path.exists(minimal_state["feature_data_path_test"])
```

### Step 0.3: Run Baseline Test

```bash
cd c:\Users\ADMIN\Desktop\Professor\ai-agent-Professor
python -m pytest tests/agents/test_pseudo_label_agent_fix.py -v
```

**Expected:** Tests fail with NameError (confirms bugs exist)

---

## Phase 1: Critical Bug Fixes — Undefined Variables (60 minutes)

### Bug #1-4: Load Data and Metric from Disk

**File:** `agents/pseudo_label_agent.py`  
**Lines to modify:** 156-240

**Fix:**

```python
def run_pseudo_label_agent(state: ProfessorState) -> ProfessorState:
    """
    GM-CAP 6: Pseudo-labeling with confidence gating.
    """
    from tools.data_tools import read_parquet, read_json
    from tools.wilcoxon_gate import is_significantly_better  # Bug #5 fix
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

    # Validate paths
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
    
    # Validate target exists in training data
    if target_col not in X_train.columns:
        raise ValueError(f"[pseudo_label] Target '{target_col}' not in training data columns: {X_train.columns}")
    
    y_train = X_train[target_col].to_numpy()
    X_train = X_train.drop(target_col)

    # Drop target from test if present (prevent leakage)
    if target_col in X_test.columns:
        logger.warning(f"[pseudo_label] Dropping target column from test data")
        X_test = X_test.drop(target_col)

    # ── Enforce feature order ────────────────────────────────────
    feature_order = state.get("feature_order")
    if feature_order:
        try:
            X_train = X_train.select(feature_order)
            X_test = X_test.select(feature_order)
        except pl.exceptions.ColumnNotFoundError as e:
            logger.error(f"[pseudo_label] Feature order mismatch: {e}")
            raise ValueError(f"Test data columns don't match feature_order: {e}")

    # ── Load metric ──────────────────────────────────────────────
    metric_contract_path = state.get("metric_contract_path")
    if metric_contract_path and os.path.exists(metric_contract_path):
        metric_contract = read_json(metric_contract_path)
        metric = metric_contract.get("scorer_name", "auc")
    else:
        metric = "auc"
        logger.warning("[pseudo_label] metric_contract not found, defaulting to 'auc'")

    # ── Validate selected_models ─────────────────────────────────
    selected = state.get("selected_models", [])
    if not selected:
        logger.warning("[pseudo_label] No selected_models. Skipping.")
        return {**state, "pseudo_labels_applied": False, "pseudo_label_cv_improvement": 0.0}

    best_model_name = selected[0]

    # Look up best model entry from registry (support both list and dict formats)
    registry = state.get("model_registry", [])
    best_entry = None
    
    if isinstance(registry, list):
        for entry in registry:
            if isinstance(entry, dict):
                model_name = entry.get("model_type") or entry.get("name") or entry.get("model_name")
                if model_name == best_model_name:
                    best_entry = entry
                    break
    elif isinstance(registry, dict):
        best_entry = registry.get(best_model_name)
    
    if best_entry is None:
        logger.warning(f"[pseudo_label] Model '{best_model_name}' not found in registry. Skipping.")
        return {**state, "pseudo_labels_applied": False, "pseudo_label_cv_improvement": 0.0}

    # ── Extract model params ─────────────────────────────────────
    lgbm_params = best_entry.get("params", {"n_estimators": 500, "learning_rate": 0.05, "verbosity": -1})
    if "verbosity" not in lgbm_params:
        lgbm_params["verbosity"] = -1

    # ── Get baseline CV scores ───────────────────────────────────
    baseline_cv = best_entry.get("fold_scores", [])
    if not baseline_cv:
        logger.warning("[pseudo_label] No baseline fold_scores. Skipping.")
        return {**state, "pseudo_labels_applied": False, "pseudo_label_cv_improvement": 0.0}

    # ── Initialize result ────────────────────────────────────────
    result = PseudoLabelResult(
        iterations_completed=0,
        pseudo_labels_added=[],
        cv_scores_with_pl=[],
        cv_scores_without_pl=[float(np.mean(baseline_cv))],
        cv_improvements=[],
        halted_early=False,
        halt_reason="",
        final_pseudo_label_mask=[],
        confidence_thresholds=[],
    )

    # ── Initialize working variables ─────────────────────────────
    # Use actual loaded data schema
    X_pseudo_accumulated = X_train.slice(0, 0)  # Empty DataFrame with same schema
    y_pseudo_accumulated = np.array([], dtype=y_train.dtype)
    current_test_mask = np.zeros(len(X_test), dtype=bool)
```

**Verification:**
- ✅ Bug #1 fixed: `X_train` loaded from disk
- ✅ Bug #2 fixed: `y_train` extracted from training data
- ✅ Bug #3 fixed: `X_test` loaded from disk
- ✅ Bug #4 fixed: `metric` loaded from metric_contract
- ✅ Bug #5 fixed: `is_significantly_better` imported
- ✅ Bug #8 fixed: Target column extracted and dropped
- ✅ Bug #9 fixed: Feature order enforced
- ✅ Bug #14 fixed: Target dropped from test data

---

## Phase 2: Logic Bug Fixes (45 minutes)

### Bug #10: Soft Labels → Hard Labels

**Location:** Main iteration loop, around line 280

**Fix:**

```python
# Select high-confidence samples
confidence = _compute_confidence(y_pred, metric)
conf_mask, threshold = _select_confident_samples(confidence, y_pred)

n_selected = int(conf_mask.sum())
if n_selected == 0:
    result.halt_reason = "no_confident_samples"
    result.halted_early = True
    del model, y_pred, confidence
    gc.collect()
    break

result.confidence_thresholds.append(threshold)

X_new_pseudo = X_remaining.filter(pl.Series(conf_mask))
y_new_pseudo = y_pred[conf_mask]

# ⚠️ CRITICAL FIX: Convert to hard labels BEFORE using
if is_cls:
    y_new_pseudo = (y_new_pseudo >= 0.5).astype(y_train.dtype)

# CV with pseudo-labels — validation fold ONLY sees real labels
if len(y_pseudo_accumulated) > 0:
    X_pseudo_for_cv = pl.concat([X_pseudo_accumulated, X_new_pseudo])
    y_pseudo_for_cv = np.concatenate([y_pseudo_accumulated, y_new_pseudo])
else:
    X_pseudo_for_cv = X_new_pseudo
    y_pseudo_for_cv = y_new_pseudo
```

**Verification:**
- ✅ Bug #10 fixed: Hard labels converted before use

---

### Bug #11-12: Wilcoxon Gate Baseline Update

**Location:** Line 329 and surrounding

**Fix:**

```python
# Track previous iteration's fold scores for fair comparison
prev_fold_scores = baseline_cv.copy()  # Initialize from original model

for iteration in range(1, MAX_PL_ITERATIONS + 1):
    # ... iteration logic ...
    
    cv_with = _run_cv_with_pseudo_labels(
        X_train=X_train,
        y_train=y_train,
        X_pseudo=X_pseudo_for_cv,
        y_pseudo=y_pseudo_for_cv,
        lgbm_params=lgbm_params,
        metric=metric,
    )

    cv_mean_with = float(np.mean(cv_with))
    cv_mean_without = float(np.mean(prev_fold_scores))
    improvement = cv_mean_with - cv_mean_without

    result.cv_scores_with_pl.append(cv_mean_with)
    result.cv_improvements.append(round(improvement, 6))
    result.pseudo_labels_added.append(n_selected)

    logger.info(
        f"[pseudo_label] Iteration {iteration}: "
        f"n_added={n_selected}, threshold={threshold:.4f}, "
        f"cv_before={cv_mean_without:.5f}, cv_after={cv_mean_with:.5f}, "
        f"improvement={improvement:+.5f}"
    )

    # Wilcoxon gate: compare against PREVIOUS iteration, not original model
    gate_passed = is_significantly_better(cv_with, prev_fold_scores)

    if not gate_passed and improvement < MIN_CV_IMPROVEMENT:
        result.halt_reason = "cv_did_not_improve"
        result.halted_early = True
        break

    # Accept iteration — update baseline for next comparison
    prev_fold_scores = cv_with.copy()
    baseline_cv = cv_with  # Also update for final state
    
    # ... rest of acceptance logic ...
```

**Verification:**
- ✅ Bug #11 fixed: Wilcoxon compares iteration N vs N-1
- ✅ Bug #12 fixed: Consistent fold score treatment

---

### Bug #16: Type Safety in Concatenation

**Location:** After hard label conversion

**Fix:**

```python
# Ensure dtype matches before concatenation
if y_new_pseudo.dtype != y_train.dtype:
    y_new_pseudo = y_new_pseudo.astype(y_train.dtype)
```

**Verification:**
- ✅ Bug #16 fixed: Explicit dtype cast

---

## Phase 3: Error Handling & Validation (30 minutes)

### Bug #17-19: Add Try/Except and Validation

**Location:** Model training section

**Fix:**

```python
for iteration in range(1, MAX_PL_ITERATIONS + 1):
    logger.info(f"[pseudo_label] Iteration {iteration}/{MAX_PL_ITERATIONS}")
    
    try:
        # Train on labelled + accumulated pseudo-labels
        if len(y_pseudo_accumulated) > 0:
            X_all = pl.concat([X_train, X_pseudo_accumulated])
            y_all = np.concatenate([y_train, y_pseudo_accumulated])
        else:
            X_all = X_train
            y_all = y_train
        
        # Validate training data
        if X_all.is_empty():
            raise ValueError(f"Iteration {iteration}: Training data is empty")
        if len(y_all) == 0:
            raise ValueError(f"Iteration {iteration}: Training labels are empty")
        
        is_cls = metric in ("auc", "logloss", "binary")
        ModelClass = lgb.LGBMClassifier if is_cls else lgb.LGBMRegressor
        model = ModelClass(**lgbm_params)
        
        try:
            model.fit(X_all.to_numpy(), y_all)
        except Exception as fit_error:
            logger.error(f"[pseudo_label] Iteration {iteration}: model training failed: {fit_error}")
            result.halt_reason = f"model_training_failed: {fit_error}"
            result.halted_early = True
            break
        
        # Predict test set — exclude already pseudo-labeled samples
        remaining_mask = ~current_test_mask
        X_remaining = X_test.filter(pl.Series(remaining_mask))

        if X_remaining.is_empty():
            result.halt_reason = "no_confident_samples"
            result.halted_early = True
            del model
            gc.collect()
            break

        y_pred = model.predict_proba(X_remaining.to_numpy())[:, 1] if is_cls \
                 else model.predict(X_remaining.to_numpy())

        del model  # Free memory early
        gc.collect()
        
        # ... rest of iteration logic ...
        
    except Exception as e:
        logger.error(f"[pseudo_label] Iteration {iteration} failed: {e}")
        result.halt_reason = f"iteration_failed: {e}"
        result.halted_early = True
        break
```

**Verification:**
- ✅ Bug #17 fixed: Try/except around training
- ✅ Bug #18 fixed: Data validation
- ✅ Bug #19 fixed: Memory cleanup

---

## Phase 4: Memory Management (15 minutes)

### Bug #20: Complete Memory Cleanup

**Add cleanup function:**

```python
def _cleanup_pl_iteration(X_all=None, y_all=None, y_pred=None, 
                          confidence=None, X_remaining=None, model=None):
    """Delete large arrays and run GC."""
    for obj in [X_all, y_all, y_pred, confidence, X_remaining, model]:
        if obj is not None:
            del obj
    gc.collect()
```

**Call in all early returns:**

```python
if X_remaining.is_empty():
    result.halt_reason = "no_confident_samples"
    result.halted_early = True
    _cleanup_pl_iteration(X_all=X_all, y_all=y_all, y_pred=y_pred, 
                          confidence=confidence, X_remaining=X_remaining)
    break
```

**Verification:**
- ✅ Bug #20 fixed: Complete memory cleanup

---

## Phase 5: Dataclass Fix (5 minutes)

### Bug #19: Mutable Defaults

**Location:** Lines 30-39

**Fix:**

```python
from dataclasses import dataclass, field

@dataclass
class PseudoLabelResult:
    iterations_completed: int
    pseudo_labels_added: list[int] = field(default_factory=list)
    cv_scores_with_pl: list[float] = field(default_factory=list)
    cv_improvements: list[float] = field(default_factory=list)
    cv_scores_without_pl: list[float] = field(default_factory=list)
    halted_early: bool = False
    halt_reason: str = ""
    final_pseudo_label_mask: list[int] = field(default_factory=list)
    confidence_thresholds: list[float] = field(default_factory=list)
```

**Verification:**
- ✅ Bug #19 fixed: Mutable defaults replaced with field(default_factory=list)

---

## Phase 6: State Contract Updates (15 minutes)

### Bug #6-7: Upstream Agent Fixes

The pseudo_label_agent now handles missing state keys gracefully, but we need to ensure upstream agents write the required keys.

**File:** `agents/feature_factory.py` (to be fixed separately)

Add to return statement:
```python
return {
    **state,
    "feature_data_path": feature_data_path,
    "feature_data_path_test": feature_data_path_test,
    "feature_order": feature_cols,
    # ... existing keys ...
}
```

**File:** `agents/ml_optimizer.py` (to be fixed separately)

Add to return statement:
```python
return {
    **state,
    "feature_data_path": feature_data_path,
    "feature_data_path_test": feature_data_path_test,
    "feature_order": feature_cols,
    "selected_models": [best_model_name],
    # ... existing keys ...
}
```

**Temporary workaround for testing:**

Add fallback paths in pseudo_label_agent (already done in Phase 1).

---

## Phase 7: Testing & Validation (60 minutes)

### Step 7.1: Unit Tests

```bash
cd c:\Users\ADMIN\Desktop\Professor\ai-agent-Professor
python -m pytest tests/agents/test_pseudo_label_agent_fix.py -v
```

**Expected:** All tests pass

### Step 7.2: Integration Test

```bash
cd c:\Users\ADMIN\Desktop\Professor\ai-agent-Professor
python run_smoke_test.py
```

**Expected:** Pipeline reaches pseudo_label_agent without crashing

### Step 7.3: Regression Test

Run existing tests to ensure no regression:

```bash
python -m pytest tests/contracts/test_ml_optimizer_contract.py -v
python -m pytest tests/contracts/test_feature_factory_contract.py -v
```

---

## Phase 8: Documentation Update (15 minutes)

### Update State Contract Documentation

Add to `BUG_TRACKER.md`:

```markdown
## State Contract for Pseudo-Label Agent

### Required State Keys (Input)

| Key | Type | Provided By | Fallback |
|-----|------|-------------|----------|
| `feature_data_path` | str | feature_factory | `outputs/{session}/X_train.parquet` |
| `feature_data_path_test` | str | feature_factory | `outputs/{session}/X_test.parquet` |
| `target_col` | str | data_engineer | None (required) |
| `metric_contract_path` | str | validation_architect | Default to "auc" |
| `model_registry` | list[dict] | ml_optimizer | Skip if missing |
| `selected_models` | list[str] | ensemble_architect or ml_optimizer | Skip if missing |
| `feature_order` | list[str] | ml_optimizer | Use DataFrame column order |

### State Keys Written (Output)

| Key | Type | Description |
|-----|------|-------------|
| `pseudo_label_result` | PseudoLabelResult | Full result object |
| `pseudo_labels_applied` | bool | Whether pseudo-labeling succeeded |
| `pseudo_label_cv_improvement` | float | Total CV improvement |
```

---

## Verification Checklist

After all fixes, verify:

- [ ] **Unit tests pass** — `test_pseudo_label_agent_fix.py`
- [ ] **Smoke test passes** — Pipeline completes without NameError
- [ ] **No regression** — Existing contract tests still pass
- [ ] **Memory safe** — No memory leaks in iteration loop
- [ ] **Type safe** — No dtype mismatches in concatenation
- [ ] **Error handling** — Graceful degradation on missing data
- [ ] **State contract** — All required keys documented
- [ ] **Code reviewed** — Another engineer reviews changes

---

## Rollback Plan

If fixes cause issues:

```bash
cd c:\Users\ADMIN\Desktop\Professor\ai-agent-Professor
git log --oneline -5
git revert HEAD~1  # Revert the fix commit
git checkout agents/pseudo_label_agent.py  # Or restore from backup
```

---

## Success Criteria

The fix is complete when:

1. ✅ No `NameError` on undefined variables
2. ✅ No `ImportError` on missing imports
3. ✅ Agent loads data from disk correctly
4. ✅ Hard labels used for classification
5. ✅ Wilcoxon gate compares correct baselines
6. ✅ Memory cleaned up in all paths
7. ✅ All unit tests pass
8. ✅ Smoke test reaches submit node
9. ✅ No regression in existing tests

---

**Document Version:** 1.0  
**Created:** 2026-03-24  
**Approved By:** [Pending Review]  
**Implementation Status:** [ ] Not Started [ ] In Progress [ ] Complete
