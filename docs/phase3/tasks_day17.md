# Professor Agent — Day 17 Implementation
**Theme: Statistical rigour at the feature level — nothing survives that doesn't earn its place**

Build order: Task 1 → Task 2
```
Task 1  →  Extend tools/wilcoxon_gate.py — feature-level test
Task 2  →  Build tools/null_importance.py — two-stage filter
           commit: "Day 17: Wilcoxon feature gate + null importance two-stage filter"
```

**Context from Day 13:** `tools/wilcoxon_gate.py` already exists with:
- `is_significantly_better(fold_scores_a, fold_scores_b)` — model-level comparison
- `gate_result(a, b, name_a, name_b)` — structured dict for lineage

Day 17 adds feature-level functions to the same file without modifying the existing API.

---

## TASK 1 — Extend `tools/wilcoxon_gate.py`: feature-level test

**Why a separate function and not reuse `is_significantly_better`:**
Model comparison asks "is model A better than model B across these folds?" Feature comparison asks "does adding this feature to the model produce a statistically significant improvement?" The framing is identical but the caller context differs — feature selection calls should be tagged differently in lineage, and the function signature benefits from naming that makes the intent obvious at the call site.

### New function: `is_feature_worth_adding`
```python
def is_feature_worth_adding(
    baseline_fold_scores: list[float],    # CV scores WITHOUT the candidate feature
    augmented_fold_scores: list[float],   # CV scores WITH the candidate feature
    feature_name: str = "candidate",
    p_threshold: float = P_VALUE_THRESHOLD,
) -> bool:
    """
    Returns True iff adding the candidate feature produces a statistically
    significant improvement in CV fold scores (p < p_threshold, Wilcoxon).

    This is a thin wrapper over is_significantly_better() with explicit naming
    for the feature selection context.

    Args:
        baseline_fold_scores:   Per-fold CV scores of model WITHOUT the feature.
                                Train model on (X \ {feature}), score on each fold.
        augmented_fold_scores:  Per-fold CV scores of model WITH the feature.
                                Train model on (X ∪ {feature}), score on each fold.
        feature_name:           Used only in logging — does not affect the result.
        p_threshold:            Significance threshold (default 0.05).

    Returns:
        True  — feature significantly improves CV. Safe to add.
        False — improvement not significant. Drop the feature.

    Never raises. Conservative default (False) on any error.
    """
    result = is_significantly_better(
        fold_scores_a=augmented_fold_scores,
        fold_scores_b=baseline_fold_scores,
        p_threshold=p_threshold,
        alternative="greater",
    )

    logger.info(
        f"[WilcoxonGate] Feature '{feature_name}': "
        f"{'KEEP' if result else 'DROP'} "
        f"(baseline_mean={np.mean(baseline_fold_scores):.5f}, "
        f"augmented_mean={np.mean(augmented_fold_scores):.5f}, "
        f"delta={np.mean(augmented_fold_scores) - np.mean(baseline_fold_scores):+.5f})"
    )

    return result


def feature_gate_result(
    baseline_fold_scores: list[float],
    augmented_fold_scores: list[float],
    feature_name: str,
    p_threshold: float = P_VALUE_THRESHOLD,
) -> dict:
    """
    Returns a structured gate result for lineage logging.
    Extends gate_result() with feature-selection-specific fields.
    """
    base = gate_result(
        fold_scores_a=augmented_fold_scores,
        fold_scores_b=baseline_fold_scores,
        model_name_a=f"{feature_name}_added",
        model_name_b="baseline_without_feature",
        p_threshold=p_threshold,
    )
    return {
        **base,
        "gate_type":    "feature_selection",
        "feature_name": feature_name,
        "decision":     "KEEP" if base["gate_passed"] else "DROP",
    }
```

### Integration point: `agents/feature_factory.py`

The feature factory generates candidate features. After generating each one, it must run a quick CV evaluation and apply the Wilcoxon gate before adding the feature to the final set.
```python
# In feature_factory.py, for each candidate feature:

from tools.wilcoxon_gate import feature_gate_result, is_feature_worth_adding

def _evaluate_candidate_feature(
    state: ProfessorState,
    X_base: pl.DataFrame,         # current approved feature set
    X_with_candidate: pl.DataFrame,  # base + one new feature
    y: np.ndarray,
    feature_name: str,
) -> bool:
    """
    Runs a quick 3-fold CV with and without the candidate feature.
    Returns True if the feature passes the Wilcoxon gate.

    Uses 3 folds (not 5) to keep feature evaluation fast.
    If the feature passes here, the full 5-fold CV runs in ml_optimizer.
    """
    baseline_scores  = _quick_cv(X_base, y, n_folds=3)
    augmented_scores = _quick_cv(X_with_candidate, y, n_folds=3)

    result = feature_gate_result(
        baseline_fold_scores=baseline_scores,
        augmented_fold_scores=augmented_scores,
        feature_name=feature_name,
    )

    log_event(
        state=state,
        action="wilcoxon_feature_gate",
        agent="feature_factory",
        details=result,
    )

    return result["gate_passed"]
```

**Performance note:** `_quick_cv` uses `n_folds=3` and `n_estimators=100` (not the full Optuna-tuned model). This keeps each feature evaluation at ~1–3 seconds rather than 30+ seconds. The Wilcoxon minimum-folds fallback (< 5 folds → mean comparison) already handles the 3-fold case gracefully.

**When to skip the gate:** Features derived from known-good transformations (log of a positive feature, ratio of two existing features) can skip the gate to avoid evaluating hundreds of low-risk candidates. Gate applies to: target encoding variants, lag features, interaction terms, external data joins. Add a `skip_wilcoxon_gate: bool` parameter to `_evaluate_candidate_feature` for this.

### New `ProfessorState` fields
```python
features_gate_passed: list[str]   # features that passed Wilcoxon gate this session
features_gate_dropped: list[str]  # features that failed Wilcoxon gate this session
```

---

## TASK 2 — Build `tools/null_importance.py`

**What null importance is:** Train a model on the real target. Record feature importances. Now shuffle the target randomly and train again — this null model cannot learn real relationships, so any apparent "importance" it assigns is pure noise. Repeat the shuffle N times to build a null distribution for each feature. A feature's actual importance must exceed the 95th percentile of its own null distribution to survive. Features that don't beat their own noise floor are dropped.

**Why two stages:**
- Stage 1 (5 shuffles): Fast pre-filter. Drops features whose actual importance is below the median of just 5 null models. Eliminates ~60-70% of features quickly.
- Stage 2 (50 shuffles, survivors only): Rigorous test. Builds a proper null distribution with 50 samples. Only run on the ~30-40% that survived Stage 1 — saves 35× the compute compared to running 50 shuffles on all features.

**The persistent sandbox requirement:** 50 LightGBM trainings in Stage 2. Without a persistent sandbox: 50 Docker container spin-ups = ~50 × 2 seconds = 100 seconds of overhead alone, before any computation. With persistent sandbox: one container spin-up, 50 trainings inside it, one container teardown.

Implementation: generate a self-contained Python script that runs all 50 shuffles inside a single `execute_code()` call. This is automatically a persistent sandbox — one container runs the entire script.

### Data structures
```python
from dataclasses import dataclass, field

@dataclass
class NullImportanceResult:
    survivors:             list[str]          # features that passed both stages
    dropped_stage1:        list[str]          # dropped by 5-shuffle pre-filter
    dropped_stage2:        list[str]          # dropped by 50-shuffle rigorous test
    stage1_importances:    dict[str, float]   # actual importance per feature (Stage 1)
    null_distributions:    dict[str, list[float]]  # 50 null importance values per survivor
    threshold_percentiles: dict[str, float]   # 95th percentile of null dist per survivor
    actual_vs_threshold:   dict[str, dict]    # {feature: {actual, threshold, ratio, passed}}
    total_features_input:  int
    total_features_output: int
    stage1_drop_count:     int
    stage2_drop_count:     int
    elapsed_seconds:       float
```

### Stage 1: `_run_stage1_permutation_filter`
```python
import time
import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, KFold

N_STAGE1_SHUFFLES = 5
STAGE1_DROP_PERCENTILE = 0.65   # drop bottom 65% by null importance ratio


def _run_stage1_permutation_filter(
    X: pl.DataFrame,
    y: np.ndarray,
    feature_names: list[str],
    n_shuffles: int = N_STAGE1_SHUFFLES,
    drop_percentile: float = STAGE1_DROP_PERCENTILE,
    task_type: str = "binary",
) -> tuple[list[str], list[str], dict[str, float]]:
    """
    Stage 1: 5-shuffle permutation importance pre-filter.

    For each of n_shuffles:
      - Shuffle y (break all real signal)
      - Train a fast LightGBM model
      - Record feature importances (split-based, fast)

    A feature's null importance = mean importance across n_shuffles.
    A feature's actual importance = importance from model trained on real y.

    Importance ratio = actual / (null + epsilon)
    Features in the bottom drop_percentile by ratio are dropped.

    Returns:
      (survivors, dropped, actual_importances_dict)
    """
    X_np = X.select(feature_names).to_numpy()
    n_samples = len(y)

    lgb_params = {
        "objective":    "binary" if task_type == "binary" else "regression",
        "n_estimators": 100,
        "num_leaves":   31,
        "learning_rate": 0.1,
        "verbosity":    -1,
        "n_jobs":       1,
    }

    # Train on real y — get actual importances
    model_real = lgb.LGBMClassifier(**lgb_params) if task_type == "binary" \
                 else lgb.LGBMRegressor(**lgb_params)
    model_real.fit(X_np, y)
    actual_importances = dict(zip(feature_names, model_real.feature_importances_.astype(float)))

    # Train n_shuffles times on shuffled y — build null importance per feature
    null_sums = {f: 0.0 for f in feature_names}
    rng = np.random.default_rng(seed=42)

    for _ in range(n_shuffles):
        y_shuffled = rng.permutation(y)
        model_null = lgb.LGBMClassifier(**lgb_params) if task_type == "binary" \
                     else lgb.LGBMRegressor(**lgb_params)
        model_null.fit(X_np, y_shuffled)
        for f, imp in zip(feature_names, model_null.feature_importances_):
            null_sums[f] += float(imp)

        del model_null

    null_means = {f: null_sums[f] / n_shuffles for f in feature_names}

    # Compute importance ratio: how much better is actual vs null?
    EPSILON = 1e-6
    ratios = {
        f: actual_importances[f] / (null_means[f] + EPSILON)
        for f in feature_names
    }

    # Drop bottom drop_percentile by ratio
    threshold_ratio = np.percentile(list(ratios.values()), drop_percentile * 100)
    survivors = [f for f in feature_names if ratios[f] >= threshold_ratio]
    dropped   = [f for f in feature_names if ratios[f] <  threshold_ratio]

    logger.info(
        f"[NullImportance] Stage 1: {len(feature_names)} features → "
        f"{len(survivors)} survivors, {len(dropped)} dropped "
        f"(threshold ratio={threshold_ratio:.3f})"
    )

    del model_real
    gc.collect()

    return survivors, dropped, actual_importances
```

### Stage 2: `_run_stage2_null_importance_persistent_sandbox`

This is the persistent sandbox function. It generates a complete self-contained Python script and runs it in a single `execute_code()` call — one container, 50 shuffles inside it, JSON result piped back via stdout.
```python
STAGE2_SCRIPT_TEMPLATE = '''
import json
import gc
import numpy as np
import lightgbm as lgb

X_np         = np.array({X_list})
y            = np.array({y_list})
feature_names = {feature_names}
n_shuffles   = {n_shuffles}
task_type    = "{task_type}"
random_seed  = {random_seed}

lgb_params = {{
    "objective":    "binary" if task_type == "binary" else "regression",
    "n_estimators": 200,
    "num_leaves":   31,
    "learning_rate": 0.05,
    "verbosity":    -1,
    "n_jobs":       1,
}}

ModelClass = lgb.LGBMClassifier if task_type == "binary" else lgb.LGBMRegressor

# Actual importances (real y)
model_real = ModelClass(**lgb_params)
model_real.fit(X_np, y)
actual_importances = dict(zip(feature_names, model_real.feature_importances_.tolist()))
del model_real
gc.collect()

# Null distributions (shuffled y, n_shuffles times)
rng = np.random.default_rng(seed=random_seed)
null_records = {{f: [] for f in feature_names}}

for i in range(n_shuffles):
    y_shuffled = rng.permutation(y)
    model_null = ModelClass(**lgb_params)
    model_null.fit(X_np, y_shuffled)
    for f, imp in zip(feature_names, model_null.feature_importances_):
        null_records[f].append(float(imp))
    del model_null
    gc.collect()
    if (i + 1) % 10 == 0:
        import sys
        print(f"Progress: {{i+1}}/{n_shuffles} shuffles complete", file=sys.stderr, flush=True)

result = {{
    "actual_importances": actual_importances,
    "null_distributions": null_records,
}}

print(json.dumps(result))
'''


def _run_stage2_null_importance_persistent_sandbox(
    X_survivors: pl.DataFrame,
    y: np.ndarray,
    survivor_names: list[str],
    n_shuffles: int = 50,
    task_type: str = "binary",
    threshold_percentile: float = 95.0,
) -> tuple[list[str], list[str], dict[str, list[float]], dict[str, float]]:
    """
    Stage 2: 50-shuffle null importance on survivors only.
    Runs all 50 shuffles in a SINGLE execute_code() call — persistent sandbox.

    The entire computation runs inside one Docker container. No per-shuffle
    container spin-up overhead.

    Returns:
      (stage2_survivors, stage2_dropped, null_distributions, threshold_percentiles)
    """
    from tools.e2b_sandbox import execute_code

    X_list = X_survivors.select(survivor_names).to_numpy().tolist()
    y_list  = y.tolist()

    script = STAGE2_SCRIPT_TEMPLATE.format(
        X_list=json.dumps(X_list),
        y_list=json.dumps(y_list),
        feature_names=json.dumps(survivor_names),
        n_shuffles=n_shuffles,
        task_type=task_type,
        random_seed=42,
    )

    logger.info(
        f"[NullImportance] Stage 2: running {n_shuffles} shuffles on "
        f"{len(survivor_names)} survivor features in persistent sandbox..."
    )

    result = execute_code(
        script,
        timeout=600,   # 10 minutes — plenty for 50 × LightGBM on survivors
    )

    if result["returncode"] != 0 or result["timed_out"]:
        logger.warning(
            f"[NullImportance] Stage 2 sandbox failed "
            f"(returncode={result['returncode']}, timed_out={result['timed_out']}). "
            f"stderr: {result['stderr'][:500]}. "
            f"Returning all survivors (no Stage 2 filtering)."
        )
        return survivor_names, [], {f: [] for f in survivor_names}, {}

    try:
        payload = json.loads(result["stdout"].strip())
        actual_importances = payload["actual_importances"]
        null_distributions  = payload["null_distributions"]
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(
            f"[NullImportance] Stage 2 result parse failed: {e}. "
            f"stdout: {result['stdout'][:300]}. Returning all survivors."
        )
        return survivor_names, [], {f: [] for f in survivor_names}, {}

    # For each feature: is actual importance > threshold_percentile of its null distribution?
    stage2_survivors = []
    stage2_dropped   = []
    threshold_pcts   = {}

    for f in survivor_names:
        actual = actual_importances.get(f, 0.0)
        null_dist = null_distributions.get(f, [])

        if not null_dist:
            # No null data — keep the feature conservatively
            stage2_survivors.append(f)
            continue

        threshold = float(np.percentile(null_dist, threshold_percentile))
        threshold_pcts[f] = threshold

        if actual > threshold:
            stage2_survivors.append(f)
        else:
            stage2_dropped.append(f)

    logger.info(
        f"[NullImportance] Stage 2: {len(survivor_names)} survivors → "
        f"{len(stage2_survivors)} final, {len(stage2_dropped)} dropped "
        f"(threshold={threshold_percentile}th percentile of null dist)"
    )

    return stage2_survivors, stage2_dropped, null_distributions, threshold_pcts
```

### Public API: `run_null_importance_filter`
```python
def run_null_importance_filter(
    X: pl.DataFrame,
    y: np.ndarray,
    feature_names: list[str],
    task_type: str = "binary",
    n_stage1_shuffles: int = N_STAGE1_SHUFFLES,
    n_stage2_shuffles: int = 50,
    stage1_drop_percentile: float = STAGE1_DROP_PERCENTILE,
    stage2_threshold_percentile: float = 95.0,
) -> NullImportanceResult:
    """
    Two-stage null importance filter.

    Stage 1: 5-shuffle permutation filter — drops bottom 65% quickly.
    Stage 2: 50-shuffle null importance on survivors — rigorous 95th pct test.

    Stage 2 runs entirely in a single sandbox execution (persistent container).

    Args:
        X:                          Feature DataFrame (all candidate features)
        y:                          Target array
        feature_names:              Which columns from X to evaluate
        task_type:                  "binary" | "regression" | "multiclass"
        n_stage1_shuffles:          Fast pre-filter shuffle count (default 5)
        n_stage2_shuffles:          Rigorous null distribution shuffles (default 50)
        stage1_drop_percentile:     Bottom fraction to drop in Stage 1 (default 0.65)
        stage2_threshold_percentile: Null distribution percentile threshold (default 95.0)

    Returns:
        NullImportanceResult with survivors, dropped, and full null distributions
    """
    t_start = time.time()

    # Stage 1
    s1_survivors, s1_dropped, actual_importances = _run_stage1_permutation_filter(
        X=X,
        y=y,
        feature_names=feature_names,
        n_shuffles=n_stage1_shuffles,
        drop_percentile=stage1_drop_percentile,
        task_type=task_type,
    )

    if not s1_survivors:
        logger.warning(
            "[NullImportance] Stage 1 dropped ALL features. "
            "Returning all features as survivors (safety fallback)."
        )
        return NullImportanceResult(
            survivors=feature_names,
            dropped_stage1=[],
            dropped_stage2=[],
            stage1_importances=actual_importances,
            null_distributions={},
            threshold_percentiles={},
            actual_vs_threshold={},
            total_features_input=len(feature_names),
            total_features_output=len(feature_names),
            stage1_drop_count=0,
            stage2_drop_count=0,
            elapsed_seconds=time.time() - t_start,
        )

    # Stage 2 — persistent sandbox
    X_survivors = X.select(s1_survivors)
    s2_survivors, s2_dropped, null_dists, threshold_pcts = \
        _run_stage2_null_importance_persistent_sandbox(
            X_survivors=X_survivors,
            y=y,
            survivor_names=s1_survivors,
            n_shuffles=n_stage2_shuffles,
            task_type=task_type,
            threshold_percentile=stage2_threshold_percentile,
        )

    # Build actual_vs_threshold comparison dict
    actual_vs_threshold = {}
    for f in s1_survivors:
        threshold = threshold_pcts.get(f)
        actual    = actual_importances.get(f, 0.0)
        actual_vs_threshold[f] = {
            "actual":    actual,
            "threshold": threshold,
            "ratio":     actual / (threshold + 1e-6) if threshold else None,
            "passed":    f in s2_survivors,
        }

    elapsed = time.time() - t_start
    logger.info(
        f"[NullImportance] Complete: {len(feature_names)} → {len(s2_survivors)} "
        f"(stage1 dropped {len(s1_dropped)}, stage2 dropped {len(s2_dropped)}) "
        f"in {elapsed:.1f}s"
    )

    return NullImportanceResult(
        survivors=s2_survivors,
        dropped_stage1=s1_dropped,
        dropped_stage2=s2_dropped,
        stage1_importances=actual_importances,
        null_distributions=null_dists,
        threshold_percentiles=threshold_pcts,
        actual_vs_threshold=actual_vs_threshold,
        total_features_input=len(feature_names),
        total_features_output=len(s2_survivors),
        stage1_drop_count=len(s1_dropped),
        stage2_drop_count=len(s2_dropped),
        elapsed_seconds=elapsed,
    )
```

### Integration point: `agents/feature_factory.py`

After generating all candidate features, before passing to `ml_optimizer`:
```python
from tools.null_importance import run_null_importance_filter

def _apply_null_importance_filter(state: ProfessorState, X: pl.DataFrame, y: np.ndarray) -> tuple[pl.DataFrame, ProfessorState]:
    """
    Applies two-stage null importance filter to candidate feature set.
    Updates state with survivor list and dropped features.
    """
    feature_names = [c for c in X.columns if c not in {state["target_column"], state["id_column"]}]

    if len(feature_names) < 10:
        # Not worth running on tiny feature sets — keep all
        logger.info("[feature_factory] < 10 features — skipping null importance filter.")
        return X, state

    result = run_null_importance_filter(
        X=X, y=y,
        feature_names=feature_names,
        task_type=state.get("task_type", "binary"),
    )

    log_event(
        state=state,
        action="null_importance_filter_complete",
        agent="feature_factory",
        details={
            "total_input":    result.total_features_input,
            "total_output":   result.total_features_output,
            "stage1_dropped": result.stage1_drop_count,
            "stage2_dropped": result.stage2_drop_count,
            "elapsed_s":      round(result.elapsed_seconds, 1),
        }
    )

    state = {
        **state,
        "null_importance_result": result,
        "features_dropped_stage1": result.dropped_stage1,
        "features_dropped_stage2": result.dropped_stage2,
    }

    # Return only survivor columns (plus id and target)
    keep_cols = result.survivors + [state["target_column"], state.get("id_column", "")]
    keep_cols = [c for c in keep_cols if c in X.columns]
    return X.select(keep_cols), state
```

### New `ProfessorState` fields
```python
null_importance_result:    object        # NullImportanceResult dataclass instance
features_dropped_stage1:   list[str]     # [] by default
features_dropped_stage2:   list[str]     # [] by default
features_gate_passed:      list[str]     # Wilcoxon feature gate — passed
features_gate_dropped:     list[str]     # Wilcoxon feature gate — dropped
```

**Note:** `null_importance_result` is a dataclass — exclude from Redis checkpoint (not JSON-serialisable). Serialize the key fields individually instead.

---

## INTEGRATION CHECKLIST

- [ ] `is_feature_worth_adding()` and `feature_gate_result()` added to `tools/wilcoxon_gate.py`
- [ ] Existing Day 13 Wilcoxon functions unchanged — no breaking changes
- [ ] `feature_factory.py` calls `_evaluate_candidate_feature()` for: target encoding variants, lag features, interaction terms
- [ ] `feature_factory.py` skips Wilcoxon gate for: log transforms, ratio features (low-risk)
- [ ] `tools/null_importance.py` has only ONE `execute_code()` call for Stage 2 (not 50)
- [ ] Stage 2 script uses `json.dumps(result)` to stdout — not print statements with mixed text
- [ ] Stage 2 graceful fallback: if `execute_code()` fails, return all Stage 1 survivors (no Stage 2 filtering)
- [ ] Stage 1 safety fallback: if ALL features dropped, return all features
- [ ] `null_importance_result` excluded from Redis checkpoint serialisation
- [ ] `run_null_importance_filter()` skipped when `len(feature_names) < 10`
- [ ] All GC calls (`del model; gc.collect()`) present inside Stage 1 shuffle loop
- [ ] Stage 2 script includes `gc.collect()` inside the null shuffle loop (running inside container)
- [ ] Day 13 Wilcoxon contract tests still pass — zero regressions

## GIT COMMIT MESSAGE
```
Day 17: Wilcoxon feature gate + null importance two-stage filter

- wilcoxon_gate.py: is_feature_worth_adding() + feature_gate_result()
  Feature-level gate — 3-fold quick CV, Wilcoxon p<0.05 required
- null_importance.py: two-stage filter
  Stage 1: 5-shuffle permutation, drops bottom 65%
  Stage 2: 50-shuffle null dist on survivors, 95th pct threshold
  Persistent sandbox: all 50 shuffles in ONE execute_code() call
- feature_factory.py: Wilcoxon gate + null importance filter wired in
- tests/test_day17_quality.py: 44 adversarial tests — all green
```