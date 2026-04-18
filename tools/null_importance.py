# tools/null_importance.py
# -------------------------------------------------------------------------
# Day 17: Two-stage null importance filter
# Stage 1: 5-shuffle permutation pre-filter — drops bottom 65%
# Stage 2: 50-shuffle null distribution on survivors — 95th pct threshold
# Stage 2 runs ALL shuffles in ONE execute_code() call (persistent sandbox)
# -------------------------------------------------------------------------

import gc
import json
import time
import logging
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from tools.e2b_sandbox import execute_code

logger = logging.getLogger(__name__)

N_STAGE1_SHUFFLES = 5
STAGE1_DROP_PERCENTILE = 0.65   # drop bottom 65% by null importance ratio


# ── Data structures ──────────────────────────────────────────────

@dataclass
class NullImportanceResult:
    survivors:             list[str]
    dropped_stage1:        list[str]
    dropped_stage2:        list[str]
    stage1_importances:    dict[str, float]
    null_distributions:    dict[str, list[float]]
    threshold_percentiles: dict[str, float]
    actual_vs_threshold:   dict[str, dict]
    total_features_input:  int
    total_features_output: int
    stage1_drop_count:     int
    stage2_drop_count:     int
    elapsed_seconds:       float


# ── Stage 1: 5-shuffle permutation pre-filter ────────────────────

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

    Importance ratio = actual / (null_mean + epsilon).
    Features in the bottom drop_percentile by ratio are dropped.

    Returns:
      (survivors, dropped, actual_importances_dict)
    """
    import lightgbm as lgb

    X_np = X.select(feature_names).to_numpy()

    lgb_params = {
        "objective":     "multiclass" if task_type == "multiclass" else "binary" if task_type == "binary" else "regression",
        "n_estimators":  100,
        "num_leaves":    31,
        "learning_rate": 0.1,
        "verbosity":     -1,
        "n_jobs":        1,
    }

    # Train on REAL y first — get actual importances
    model_real = (lgb.LGBMClassifier(**lgb_params) if task_type in ("binary", "multiclass")
                  else lgb.LGBMRegressor(**lgb_params))
    model_real.fit(X_np, y)
    actual_importances = dict(
        zip(feature_names, model_real.feature_importances_.astype(float))
    )
    del model_real
    gc.collect()

    # Train n_shuffles times on SHUFFLED y — build null importance per feature
    null_sums = {f: 0.0 for f in feature_names}
    rng = np.random.default_rng(seed=42)

    for _ in range(n_shuffles):
        y_shuffled = rng.permutation(y)
        model_null = (lgb.LGBMClassifier(**lgb_params) if task_type in ("binary", "multiclass")
                      else lgb.LGBMRegressor(**lgb_params))
        model_null.fit(X_np, y_shuffled)
        for f, imp in zip(feature_names, model_null.feature_importances_):
            null_sums[f] += float(imp)
        del model_null
        gc.collect()

    null_means = {f: null_sums[f] / n_shuffles for f in feature_names}

    # Compute importance ratio: actual vs null
    EPSILON = 1e-6
    ratios = {
        f: actual_importances[f] / (null_means[f] + EPSILON)
        for f in feature_names
    }

    # Drop bottom drop_percentile by ratio
    threshold_ratio = float(np.percentile(list(ratios.values()), drop_percentile * 100))
    survivors = [f for f in feature_names if ratios[f] >= threshold_ratio]
    dropped   = [f for f in feature_names if ratios[f] < threshold_ratio]

    logger.info(
        f"[NullImportance] Stage 1: {len(feature_names)} features → "
        f"{len(survivors)} survivors, {len(dropped)} dropped "
        f"(threshold ratio={threshold_ratio:.3f})"
    )

    return survivors, dropped, actual_importances


# ── LEAKAGE FIX: CV-Safe Stage 1 ──────────────────────────────────

def _run_stage1_permutation_filter_cv_safe(
    X: pl.DataFrame,
    y: np.ndarray,
    feature_names: list[str],
    cv_folds=None,  # NEW: Optional CV folds for CV-safe computation
    n_shuffles: int = N_STAGE1_SHUFFLES,
    drop_percentile: float = STAGE1_DROP_PERCENTILE,
    task_type: str = "binary",
) -> tuple[list[str], list[str], dict[str, float]]:
    """
    LEAKAGE FIX: CV-safe version of Stage 1 permutation filter.
    
    If cv_folds provided, computes importance within folds only.
    Otherwise, falls back to original (leaky) behavior.
    
    Returns:
      (survivors, dropped, actual_importances_dict)
    """
    import lightgbm as lgb

    X_np = X.select(feature_names).to_numpy()

    lgb_params = {
        "objective":     "multiclass" if task_type == "multiclass" else "binary" if task_type == "binary" else "regression",
        "n_estimators":  100,
        "num_leaves":    31,
        "learning_rate": 0.1,
        "verbosity":     -1,
        "n_jobs":        1,
    }

    if cv_folds is not None and len(cv_folds) > 0:
        # CV-SAFE: Compute importance within folds
        logger.info(f"[NullImportance] Stage 1 CV-safe: Computing importance within {len(cv_folds)} folds")
        
        importance_scores = {f: 0.0 for f in feature_names}
        
        for train_idx, _ in cv_folds:
            X_train = X[train_idx].select(feature_names).to_numpy()
            y_train = y[train_idx]
            
            # Train on REAL y — get actual importances
            model_real = (lgb.LGBMClassifier(**lgb_params) if task_type in ("binary", "multiclass")
                          else lgb.LGBMRegressor(**lgb_params))
            model_real.fit(X_train, y_train)
            
            for f, imp in zip(feature_names, model_real.feature_importances_):
                importance_scores[f] += float(imp)
            
            del model_real
            gc.collect()
        
        # Average across folds
        n_folds = len(cv_folds)
        actual_importances = {f: importance_scores[f] / n_folds for f in feature_names}
    else:
        # Fallback to original (leaky) behavior
        logger.warning("[NullImportance] Stage 1: No cv_folds provided, using leaky computation")
        
        # Train on REAL y first — get actual importances
        model_real = (lgb.LGBMClassifier(**lgb_params) if task_type in ("binary", "multiclass")
                      else lgb.LGBMRegressor(**lgb_params))
        model_real.fit(X_np, y)
        actual_importances = dict(
            zip(feature_names, model_real.feature_importances_.astype(float))
        )
        del model_real
        gc.collect()

    # Train n_shuffles times on SHUFFLED y — build null importance per feature
    # Note: This still uses full data for null distribution (acceptable tradeoff)
    null_sums = {f: 0.0 for f in feature_names}
    rng = np.random.default_rng(seed=42)

    for _ in range(n_shuffles):
        y_shuffled = rng.permutation(y)
        model_null = (lgb.LGBMClassifier(**lgb_params) if task_type in ("binary", "multiclass")
                      else lgb.LGBMRegressor(**lgb_params))
        model_null.fit(X_np, y_shuffled)
        for f, imp in zip(feature_names, model_null.feature_importances_):
            null_sums[f] += float(imp)
        del model_null
        gc.collect()

    null_means = {f: null_sums[f] / n_shuffles for f in feature_names}

    # Compute importance ratio: actual vs null
    EPSILON = 1e-6
    ratios = {
        f: actual_importances[f] / (null_means[f] + EPSILON)
        for f in feature_names
    }

    # Drop bottom drop_percentile by ratio
    threshold_ratio = float(np.percentile(list(ratios.values()), drop_percentile * 100))
    survivors = [f for f in feature_names if ratios[f] >= threshold_ratio]
    dropped   = [f for f in feature_names if ratios[f] < threshold_ratio]

    logger.info(
        f"[NullImportance] Stage 1: {len(feature_names)} features → "
        f"{len(survivors)} survivors, {len(dropped)} dropped "
        f"(threshold ratio={threshold_ratio:.3f})"
    )

    return survivors, dropped, actual_importances


# ── Stage 2: persistent sandbox script template ──────────────────

STAGE2_SCRIPT_TEMPLATE = '''
import json
import gc
import numpy as np
import lightgbm as lgb
import os

try:
    data = np.load("{data_path}")
    X_np = data["X"]
    y = data["y"]
except Exception as e:
    # Print error to stderr and exit with error code
    import sys
    print(json.dumps({{"error": f"Failed to load data: {{e}}" }}), file=sys.stderr)
    sys.exit(1)

feature_names = {feature_names}
n_shuffles    = {n_shuffles}
task_type     = "{task_type}"
random_seed   = {random_seed}

lgb_params = {{
    "objective":     "multiclass" if task_type == "multiclass" else "binary" if task_type == "binary" else "regression",
    "n_estimators":  200,
    "num_leaves":    31,
    "learning_rate": 0.05,
    "verbosity":     -1,
    "n_jobs":        1,
}}

ModelClass = lgb.LGBMClassifier if task_type in ("binary", "multiclass") else lgb.LGBMRegressor

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
        # Print progress to stderr
        import sys
        print(f"Progress: {{i+1}}/{{n_shuffles}} shuffles complete", file=sys.stderr, flush=True)

result = {{
    "actual_importances": actual_importances,
    "null_distributions": null_records,
}}

print(json.dumps(result))
'''


# ── Stage 2: 50-shuffle null importance in persistent sandbox ────

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
    Runs all shuffles in a SINGLE execute_code() call — persistent sandbox.

    Returns:
      (stage2_survivors, stage2_dropped, null_distributions, threshold_percentiles)
    """
    import tempfile
    import os
    import time
    
    tmp_path = os.path.join(tempfile.gettempdir(), f"null_imp_stage2_{int(time.time())}.npz")
    np.savez(tmp_path, X=X_survivors.select(survivor_names).to_numpy(), y=y)

    script = STAGE2_SCRIPT_TEMPLATE.format(
        data_path=tmp_path.replace("\\", "/"),
        feature_names=json.dumps(survivor_names),
        n_shuffles=n_shuffles,
        task_type=task_type,
        random_seed=42,
    )

    logger.info(
        f"[NullImportance] Stage 2: running {n_shuffles} shuffles on "
        f"{len(survivor_names)} survivor features in persistent sandbox..."
    )

    try:
        result = execute_code(
            script,
            session_id="null_importance",
            max_attempts=1,
            timeout_seconds=600,
        )
    except Exception as e:
        logger.warning(
            f"[NullImportance] Stage 2 sandbox raised: {e}. "
            f"Returning all survivors (no Stage 2 filtering)."
        )
        return survivor_names, [], {f: [] for f in survivor_names}, {}

    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    if result.get("returncode", -1) != 0 or result.get("timed_out", False):
        logger.warning(
            f"[NullImportance] Stage 2 sandbox failed "
            f"(returncode={result.get('returncode')}, "
            f"timed_out={result.get('timed_out')}). "
            f"stderr: {str(result.get('stderr', ''))[:500]}. "
            f"Returning all survivors (no Stage 2 filtering)."
        )
        return survivor_names, [], {f: [] for f in survivor_names}, {}

    try:
        payload = json.loads(result["stdout"].strip())
        if "error" in payload:
            logger.warning(f"[NullImportance] Sandbox script returned error: {payload['error']}")
            return survivor_names, [], {f: [] for f in survivor_names}, {}
        actual_importances = payload["actual_importances"]
        null_distributions = payload["null_distributions"]
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(
            f"[NullImportance] Stage 2 result parse failed: {e}. "
            f"stdout: {str(result.get('stdout', ''))[:300]}. Returning all survivors."
        )
        return survivor_names, [], {f: [] for f in survivor_names}, {}

    # For each feature: is actual importance > threshold_percentile of null dist?
    stage2_survivors = []
    stage2_dropped   = []
    threshold_pcts   = {}

    for f in survivor_names:
        actual = actual_importances.get(f, 0.0)
        null_dist = null_distributions.get(f, [])

        if not null_dist:
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


# ── Public API ───────────────────────────────────────────────────

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

    Skips both stages if len(feature_names) < 10.
    """
    t_start = time.time()

    # Skip on small feature sets
    if len(feature_names) < 10:
        logger.info(
            f"[NullImportance] Only {len(feature_names)} features — "
            f"skipping null importance filter (minimum 10)."
        )
        return NullImportanceResult(
            survivors=feature_names,
            dropped_stage1=[],
            dropped_stage2=[],
            stage1_importances={},
            null_distributions={},
            threshold_percentiles={},
            actual_vs_threshold={},
            total_features_input=len(feature_names),
            total_features_output=len(feature_names),
            stage1_drop_count=0,
            stage2_drop_count=0,
            elapsed_seconds=time.time() - t_start,
        )

    # Stage 1
    s1_survivors, s1_dropped, actual_importances = _run_stage1_permutation_filter(
        X=X,
        y=y,
        feature_names=feature_names,
        n_shuffles=n_stage1_shuffles,
        drop_percentile=stage1_drop_percentile,
        task_type=task_type,
    )

    # Safety fallback: if Stage 1 dropped ALL features (or nearly all),
    # return all features as survivors — never let the filter eliminate everything
    if not s1_survivors or len(s1_survivors) == 0:
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

    # Stage 2 — Lightning offload or persistent sandbox
    X_survivors = X.select(s1_survivors)
    
    # ── Lightning Offload Hook ─────────────────────────────────────────
    import os
    from tools.lightning_runner import is_lightning_configured, run_on_lightning, sync_files_to_lightning
    USE_LIGHTNING_NI = (
        is_lightning_configured() and
        os.getenv("LIGHTNING_OFFLOAD_NULL_IMPORTANCE", "0") == "1"
    )
    
    s2_survivors = None
    s2_dropped = None
    null_dists = {}
    threshold_pcts = {}
    
    if USE_LIGHTNING_NI:
        logger.info("[NullImportance] ⚡ Offloading Stage 2 to Lightning AI...")
        # We need to write survivors list and training data  
        # Derive a session_id-like identifier for file paths
        import tempfile
        tmp_dir = tempfile.mkdtemp(prefix="null_imp_")
        surv_path = os.path.join(tmp_dir, "stage1_survivors.json")
        train_tmp = os.path.join(tmp_dir, "train.csv")
        
        with open(surv_path, "w") as f:
            json.dump(s1_survivors, f)
        
        # Write X + y as CSV for Lightning
        df_for_lightning = X_survivors.clone()
        target_col_name = "__target__"
        df_for_lightning = df_for_lightning.with_columns(pl.Series(target_col_name, y))
        df_for_lightning.write_csv(train_tmp)
        
        session_id = f"null_imp_{int(time.time())}"
        synced = sync_files_to_lightning(
            session_id=session_id,
            files={
                train_tmp: "train.csv",
                surv_path: "stage1_survivors.json",
            }
        )
        if synced:
            machine = os.getenv("LIGHTNING_NULL_IMPORTANCE_MACHINE", "CPU")
            res = run_on_lightning(
                script="tools/lightning_jobs/run_null_importance.py",
                args={"session_id": session_id, "target_col": target_col_name, "task_type": task_type},
                job_name=f"null_imp_{session_id}",
                machine=machine,
                interruptible=True,
                result_path=os.path.join(tmp_dir, "null_importance_stage2_result.json"),
            )
            if res["success"] and res["result"].get("success"):
                lightning_data = res["result"]
                s2_survivors = lightning_data.get("survivors", s1_survivors)
                s2_dropped = lightning_data.get("dropped_stage2", [])
                logger.info(f"[NullImportance] Lightning returned {len(s2_survivors)} Stage 2 survivors.")
            else:
                logger.warning(f"[NullImportance] Lightning failed: {res.get('error')}. Running locally.")
                USE_LIGHTNING_NI = False
        else:
            USE_LIGHTNING_NI = False
    
    if not USE_LIGHTNING_NI:
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
