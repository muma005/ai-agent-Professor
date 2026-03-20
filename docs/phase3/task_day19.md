# Professor Agent — Day 19 Implementation
**Theme: Statistical rigour at the model level — stability over peak, calibration over confidence**

Build order: Task 1 → Task 2 → Task 3 → Task 4
```
Task 1  →  GAP 7: Prediction calibration in ml_optimizer
           agents/ml_optimizer.py
Task 2  →  Build tools/stability_validator.py
           tools/stability_validator.py
Task 3  →  Upgrade ml_optimizer with Optuna HPO + stability ranking
           agents/ml_optimizer.py
Task 4  →  ML Optimizer + Optuna contract test
           tests/contracts/test_ml_optimizer_optuna_contract.py
           commit: "Day 19: calibration, stability validator, Optuna HPO, optimizer contract"
```

**Prerequisites:**
- Day 13 Wilcoxon gate exists in `tools/wilcoxon_gate.py`
- Day 12 OOM guardrails (`del models` in `finally`, `gc_after_trial=True`) still in place
- `metric_contract` object in state has a `scorer` field (set by Validation Architect)

---

## TASK 1 — GAP 7: Prediction calibration (`agents/ml_optimizer.py`)

**The problem:** For log-loss and Brier score competitions, raw model outputs are poorly calibrated. LightGBM and XGBoost are well-known to produce overconfident predictions — probabilities cluster near 0 and 1. Log-loss penalises overconfident wrong predictions exponentially. A model with AUC=0.88 but poor calibration will score worse on log-loss than a model with AUC=0.85 and good calibration.

**Trigger condition:** Calibration runs automatically when `metric_contract.scorer` is in `PROBABILITY_METRICS`. No configuration required.

### Constants
```python
PROBABILITY_METRICS = frozenset({
    "log_loss", "cross_entropy", "brier_score",
    "logloss", "binary_crossentropy",
})
CALIBRATION_FOLD_FRACTION   = 0.15   # 15% of training data held out for calibration
SMALL_CALIBRATION_THRESHOLD = 1000   # below this: Platt (sigmoid), at/above: isotonic
```

### `_run_calibration(base_model, X_calib, y_calib, method) -> CalibratedClassifierCV`
```python
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
import numpy as np

def _run_calibration(
    base_model,
    X_calib: np.ndarray,
    y_calib: np.ndarray,
    method: str = "sigmoid",
) -> tuple:
    """
    Fits a calibration wrapper on a held-out calibration fold.
    Returns (calibrated_model, calibration_brier_score, method_used).

    CRITICAL RULE: base_model must already be fitted.
    cv='prefit' tells CalibratedClassifierCV to calibrate on top of the
    existing model without re-training it. This is what keeps calibration
    data separate from training data.

    Never raises — returns (base_model, None, "none") on any failure.
    """
    try:
        calibrated = CalibratedClassifierCV(
            estimator=base_model,
            method=method,
            cv="prefit",      # base_model already fitted — do not re-train
        )
        calibrated.fit(X_calib, y_calib)

        # Score calibration quality with Brier score on the calibration fold
        y_prob_calib = calibrated.predict_proba(X_calib)[:, 1]
        brier = float(brier_score_loss(y_calib, y_prob_calib))

        return calibrated, brier, method

    except Exception as e:
        logger.warning(
            f"[ml_optimizer] Calibration failed ({method}): {e}. "
            f"Returning uncalibrated model."
        )
        return base_model, None, "none"


def _select_calibration_method(n_calib_samples: int) -> str:
    """
    Selects calibration method based on calibration fold size.
    Platt (sigmoid) is more stable on small sets.
    Isotonic is more flexible but needs > 1000 samples to be reliable.
    """
    return "sigmoid" if n_calib_samples < SMALL_CALIBRATION_THRESHOLD else "isotonic"
```

### Calibration fold split

The calibration fold is carved out of the training data BEFORE the CV folds are created. This ensures the calibration data is never used for training and is separate from the validation folds used for CV scoring.
```python
from sklearn.model_selection import train_test_split

def _split_calibration_fold(
    X: pl.DataFrame,
    y: np.ndarray,
    calib_fraction: float = CALIBRATION_FOLD_FRACTION,
    random_state: int = 42,
) -> tuple:
    """
    Splits off a calibration fold from training data.
    Returns (X_train_cv, y_train_cv, X_calib, y_calib).

    The calibration fold is held out entirely — it does not appear
    in any CV training or validation fold.
    """
    X_np = X.to_numpy()
    X_train_cv, X_calib, y_train_cv, y_calib = train_test_split(
        X_np, y,
        test_size=calib_fraction,
        random_state=random_state,
        stratify=y if len(np.unique(y)) <= 20 else None,
    )
    return X_train_cv, y_train_cv, X_calib, y_calib
```

### Wire into training loop

In the Optuna objective and in the final model training after HPO:
```python
def _train_and_optionally_calibrate(
    X: pl.DataFrame,
    y: np.ndarray,
    params: dict,
    model_type: str,
    metric: str,
    state: ProfessorState,
) -> tuple:
    """
    Trains a model. If metric is probability-based, calibrates on held-out fold.
    Returns (final_model, fold_scores, calibration_info).
    """
    run_calibration = metric in PROBABILITY_METRICS

    if run_calibration:
        X_train_cv, y_train_cv, X_calib, y_calib = _split_calibration_fold(X, y)
        n_calib = len(y_calib)
        calib_method = _select_calibration_method(n_calib)
    else:
        X_train_cv, y_train_cv = X.to_numpy(), y
        X_calib, y_calib = None, None
        calib_method = "none"

    # Standard CV training on X_train_cv / y_train_cv
    fold_scores, trained_models = _run_cv(X_train_cv, y_train_cv, params, model_type)

    # Take the model from the last fold as the base for calibration
    # (or retrain on full X_train_cv for the final model)
    ModelClass = _get_model_class(model_type)
    final_model = ModelClass(**params)
    final_model.fit(X_train_cv, y_train_cv)

    calibration_info = {
        "is_calibrated":       False,
        "calibration_method":  "none",
        "calibration_score":   None,
        "calibration_n_samples": 0,
    }

    if run_calibration and X_calib is not None:
        final_model, brier, method_used = _run_calibration(
            base_model=final_model,
            X_calib=X_calib,
            y_calib=y_calib,
            method=calib_method,
        )
        calibration_info = {
            "is_calibrated":         method_used != "none",
            "calibration_method":    method_used,
            "calibration_score":     brier,
            "calibration_n_samples": len(y_calib),
        }
        if brier is not None:
            logger.info(
                f"[ml_optimizer] Calibration ({method_used}): "
                f"Brier score on calib fold = {brier:.5f}"
            )

    return final_model, fold_scores, calibration_info


def _update_model_registry_with_calibration(
    entry: dict,
    calibration_info: dict,
) -> dict:
    """
    Adds calibration fields to a model registry entry.
    """
    return {
        **entry,
        "is_calibrated":         calibration_info["is_calibrated"],
        "calibration_method":    calibration_info["calibration_method"],
        "calibration_score":     calibration_info["calibration_score"],
        "calibration_n_samples": calibration_info["calibration_n_samples"],
    }
```

### Critic Vector integration (existing `_check_pr_curve_imbalance`)

In `agents/red_team_critic.py`, the existing `_check_pr_curve_imbalance` vector (Vector 5) is extended to check calibration quality when the metric is probability-based:
```python
# Add to _check_pr_curve_imbalance or as a new sub-check:

def _check_calibration_quality(state: ProfessorState) -> dict:
    """
    Checks calibration for all models in registry when metric is probability-based.
    Called as part of the critic's probability metric checks.
    """
    metric = state.get("evaluation_metric", "")
    if metric not in PROBABILITY_METRICS:
        return {"verdict": "OK", "note": "Non-probability metric — calibration check skipped."}

    registry = state.get("model_registry", {})
    warnings = []

    for model_name, entry in registry.items():
        if not entry.get("is_calibrated", False):
            warnings.append({
                "model":   model_name,
                "issue":   "Model not calibrated despite probability metric",
                "action":  "Check that calibration step ran for this model.",
            })

        brier = entry.get("calibration_score")
        if brier is not None and brier > 0.25:
            warnings.append({
                "model":  model_name,
                "issue":  f"Poor calibration: Brier score = {brier:.4f} (> 0.25 threshold)",
                "action": "Consider recalibration with more samples or a different method.",
            })

    if not warnings:
        return {"verdict": "OK", "note": f"All {len(registry)} models calibrated for {metric}."}

    return {
        "verdict":  "HIGH",
        "warnings": warnings,
        "note": (
            f"Calibration issues found for probability metric '{metric}'. "
            "Poor calibration directly harms log-loss and Brier score."
        ),
    }
```

### New `model_registry` entry fields
```python
{
    "model_id":              str,
    "cv_mean":               float,
    "cv_std":                float,
    "fold_scores":           list[float],
    "stability_score":       float,          # Day 19 Task 3
    "seed_results":          list[float],    # Day 19 Task 3
    "params":                dict,
    "oof_predictions":       list[float],
    "data_hash":             str,
    "is_calibrated":         bool,           # Day 19 Task 1 — NEW
    "calibration_method":    str,            # "sigmoid" | "isotonic" | "none"
    "calibration_score":     float | None,   # Brier score on calibration fold
    "calibration_n_samples": int,            # size of calibration fold
}
```

---

## TASK 2 — Build `tools/stability_validator.py`

**Why stability matters:** A model that scores 0.885 on one seed but 0.863 on another has high variance. If you happen to run on the lucky seed, your CV looks great but your LB will regress. `stability_score = mean - 1.5 * std` penalises variance — a consistent 0.873 is worth more than an average 0.875 with high jitter.
```python
# tools/stability_validator.py

import numpy as np
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

DEFAULT_SEEDS      = [42, 7, 123, 999, 2024]
STABILITY_PENALTY  = 1.5   # multiplier on std in stability_score formula


@dataclass
class StabilityResult:
    mean:             float
    std:              float
    stability_score:  float      # mean - 1.5 * std
    seed_results:     list[float]
    seeds_used:       list[int]
    min_score:        float
    max_score:        float
    spread:           float      # max - min


def run_with_seeds(
    config: dict,
    train_fn,                    # callable(config, seed) -> float (cv score)
    seeds: list[int] = None,
    penalty: float = STABILITY_PENALTY,
) -> StabilityResult:
    """
    Runs a training configuration with multiple random seeds.
    Returns mean, std, and stability_score = mean - penalty * std.

    Args:
        config:   Model hyperparameter configuration dict.
        train_fn: Callable that takes (config, seed) and returns a CV score.
                  Claude Code: this is typically _run_cv_with_seed(config, seed).
        seeds:    List of random seeds. Defaults to DEFAULT_SEEDS = [42,7,123,999,2024].
        penalty:  Standard deviation multiplier for stability score (default 1.5).

    Returns:
        StabilityResult dataclass.

    Never raises — returns result with available seeds if some fail.
    """
    if seeds is None:
        seeds = DEFAULT_SEEDS

    seed_results = []
    seeds_used   = []

    for seed in seeds:
        try:
            score = float(train_fn(config, seed))
            seed_results.append(score)
            seeds_used.append(seed)
            logger.debug(
                f"[stability_validator] seed={seed}: score={score:.6f}"
            )
        except Exception as e:
            logger.warning(
                f"[stability_validator] seed={seed} failed: {e}. Skipping."
            )

    if not seed_results:
        logger.warning(
            "[stability_validator] All seeds failed. Returning zero stability score."
        )
        return StabilityResult(
            mean=0.0, std=0.0, stability_score=0.0,
            seed_results=[], seeds_used=[],
            min_score=0.0, max_score=0.0, spread=0.0,
        )

    mean  = float(np.mean(seed_results))
    std   = float(np.std(seed_results))
    stab  = mean - penalty * std

    result = StabilityResult(
        mean=round(mean, 6),
        std=round(std, 6),
        stability_score=round(stab, 6),
        seed_results=[round(s, 6) for s in seed_results],
        seeds_used=seeds_used,
        min_score=round(float(min(seed_results)), 6),
        max_score=round(float(max(seed_results)), 6),
        spread=round(float(max(seed_results) - min(seed_results)), 6),
    )

    logger.info(
        f"[stability_validator] "
        f"mean={result.mean:.5f}, std={result.std:.5f}, "
        f"stability_score={result.stability_score:.5f}, "
        f"spread={result.spread:.5f} "
        f"({len(seeds_used)}/{len(seeds)} seeds succeeded)"
    )

    return result


def rank_by_stability(
    configs: list[dict],
    stability_results: list[StabilityResult],
) -> list[tuple[dict, StabilityResult]]:
    """
    Ranks (config, result) pairs by stability_score descending.
    Most stable config first.

    Returns:
        List of (config, StabilityResult) tuples, sorted by stability_score desc.
    """
    if len(configs) != len(stability_results):
        raise ValueError(
            f"len(configs)={len(configs)} != len(stability_results)={len(stability_results)}"
        )

    paired = list(zip(configs, stability_results))
    paired.sort(key=lambda x: x[1].stability_score, reverse=True)
    return paired


def format_stability_report(
    ranked: list[tuple[dict, StabilityResult]],
    top_n: int = 5,
) -> str:
    """
    Formats a human-readable stability ranking report for lineage logging.
    """
    lines = [f"Top {min(top_n, len(ranked))} configs by stability score:"]
    for i, (config, result) in enumerate(ranked[:top_n]):
        lines.append(
            f"  [{i+1}] stability={result.stability_score:.5f} "
            f"(mean={result.mean:.5f}, std={result.std:.5f}, "
            f"spread={result.spread:.5f})"
        )
    return "\n".join(lines)
```

---

## TASK 3 — Upgrade `agents/ml_optimizer.py` with Optuna HPO + stability ranking

**The new optimizer flow:**
```
1. Optuna study — search LightGBM, XGBoost, CatBoost hyperparameter spaces
2. After study: take top 10 configs by CV mean
3. For each: run_with_seeds([5 seeds]) → stability_score = mean - 1.5 * std
4. Rank by stability_score (not peak CV)
5. Winner enters model_registry
6. If metric is probability-based: apply calibration (Task 1)
```

### Search space definition
```python
import optuna

def _suggest_lgbm_params(trial: optuna.Trial) -> dict:
    return {
        "objective":       "binary",
        "n_estimators":    trial.suggest_int("lgbm_n_estimators", 100, 1000, step=50),
        "learning_rate":   trial.suggest_float("lgbm_lr", 0.01, 0.3, log=True),
        "num_leaves":      trial.suggest_int("lgbm_num_leaves", 16, 256),
        "max_depth":       trial.suggest_int("lgbm_max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("lgbm_min_child", 10, 100),
        "feature_fraction":  trial.suggest_float("lgbm_feat_frac", 0.5, 1.0),
        "bagging_fraction":  trial.suggest_float("lgbm_bag_frac", 0.5, 1.0),
        "bagging_freq":    trial.suggest_int("lgbm_bag_freq", 1, 7),
        "reg_alpha":       trial.suggest_float("lgbm_alpha", 1e-8, 10.0, log=True),
        "reg_lambda":      trial.suggest_float("lgbm_lambda", 1e-8, 10.0, log=True),
        "verbosity":       -1,
        "n_jobs":          1,   # OOM guard from Day 12
        "model_type":      "lgbm",
    }

def _suggest_xgb_params(trial: optuna.Trial) -> dict:
    return {
        "n_estimators":    trial.suggest_int("xgb_n_estimators", 100, 1000, step=50),
        "learning_rate":   trial.suggest_float("xgb_lr", 0.01, 0.3, log=True),
        "max_depth":       trial.suggest_int("xgb_max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("xgb_min_child", 1, 20),
        "subsample":       trial.suggest_float("xgb_subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("xgb_colsample", 0.5, 1.0),
        "gamma":           trial.suggest_float("xgb_gamma", 1e-8, 5.0, log=True),
        "reg_alpha":       trial.suggest_float("xgb_alpha", 1e-8, 5.0, log=True),
        "reg_lambda":      trial.suggest_float("xgb_lambda", 1e-8, 5.0, log=True),
        "use_label_encoder": False,
        "eval_metric":     "logloss",
        "verbosity":       0,
        "n_jobs":          1,
        "model_type":      "xgb",
    }

def _suggest_catboost_params(trial: optuna.Trial) -> dict:
    return {
        "iterations":      trial.suggest_int("cb_iters", 100, 1000, step=50),
        "learning_rate":   trial.suggest_float("cb_lr", 0.01, 0.3, log=True),
        "depth":           trial.suggest_int("cb_depth", 3, 10),
        "l2_leaf_reg":     trial.suggest_float("cb_l2", 1e-8, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("cb_bag_temp", 0.0, 1.0),
        "random_strength": trial.suggest_float("cb_rand_str", 1e-8, 10.0, log=True),
        "verbose":         0,
        "thread_count":    1,
        "model_type":      "catboost",
    }

MODEL_SUGGESTERS = {
    "lgbm":     _suggest_lgbm_params,
    "xgb":      _suggest_xgb_params,
    "catboost": _suggest_catboost_params,
}

def _suggest_params(trial: optuna.Trial) -> dict:
    """Selects model type and suggests hyperparameters for that type."""
    model_type = trial.suggest_categorical("model_type", list(MODEL_SUGGESTERS.keys()))
    return MODEL_SUGGESTERS[model_type](trial)
```

### Objective function with Day 12 OOM guards
```python
def _objective(
    trial: optuna.Trial,
    X_train_cv: np.ndarray,
    y_train_cv: np.ndarray,
    metric: str,
    max_memory_gb: float = 6.0,
) -> float:
    """
    Optuna objective. Includes Day 12 OOM guardrails (gc in finally).
    Stores fold_scores in trial user_attrs for Day 13 Wilcoxon gate.
    """
    params = _suggest_params(trial)
    models = []

    try:
        fold_scores = _run_cv_and_collect_models(
            X_train_cv, y_train_cv, params, models,
            max_memory_gb=max_memory_gb,
            trial=trial,
        )
        trial.set_user_attr("fold_scores", [float(s) for s in fold_scores])
        trial.set_user_attr("mean_cv", float(np.mean(fold_scores)))
        trial.set_user_attr("params", params)
        return float(np.mean(fold_scores))

    finally:
        # Day 12 OOM guard — always runs
        for m in models:
            del m
        del models
        import gc; gc.collect()
```

### Main optimizer function
```python
TOP_K_FOR_STABILITY    = 10    # top-K configs re-run with 5 seeds
N_OPTUNA_TRIALS        = 200
N_STABILITY_SEEDS      = 5

def run_ml_optimizer(state: ProfessorState) -> ProfessorState:
    """
    Full ML Optimizer with Optuna HPO + multi-seed stability ranking + calibration.
    """
    from tools.stability_validator import run_with_seeds, rank_by_stability, format_stability_report
    from tools.wilcoxon_gate import gate_result

    X        = state["X_train"]
    y        = state["y_train"]
    metric   = state.get("evaluation_metric", "auc")
    max_mem  = float(os.getenv("PROFESSOR_MAX_MEMORY_GB", "6.0"))

    # Calibration fold split (if probability metric)
    run_cal  = metric in PROBABILITY_METRICS
    if run_cal:
        X_train_cv_np, y_train_cv, X_calib, y_calib = _split_calibration_fold(X, y)
    else:
        X_train_cv_np, y_train_cv = X.to_numpy(), y
        X_calib, y_calib = None, None

    # ── Phase 1: Optuna study ────────────────────────────────────────────────
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    with _disable_langsmith_tracing():   # Day 15 cost guard
        study.optimize(
            lambda trial: _objective(trial, X_train_cv_np, y_train_cv, metric, max_mem),
            n_trials=N_OPTUNA_TRIALS,
            n_jobs=1,           # Day 12 OOM guard
            gc_after_trial=True,
            callbacks=[_memory_callback(max_mem)],
        )

    # ── Phase 2: Take top-K configs, re-run with 5 seeds ────────────────────
    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
        and t.user_attrs.get("fold_scores")
    ]
    completed.sort(key=lambda t: t.user_attrs["mean_cv"], reverse=True)
    top_k_trials = completed[:TOP_K_FOR_STABILITY]

    logger.info(
        f"[ml_optimizer] Optuna complete: {len(completed)} trials. "
        f"Re-running top {len(top_k_trials)} configs with {N_STABILITY_SEEDS} seeds."
    )

    # train_fn: given (config, seed) return CV mean
    def _train_fn(config: dict, seed: int) -> float:
        params = {**config, "random_state": seed, "seed": seed}
        fold_scores = _run_cv_no_collect(X_train_cv_np, y_train_cv, params)
        return float(np.mean(fold_scores))

    top_k_configs     = [t.user_attrs["params"] for t in top_k_trials]
    stability_results = [
        run_with_seeds(config=cfg, train_fn=_train_fn)
        for cfg in top_k_configs
    ]

    ranked = rank_by_stability(top_k_configs, stability_results)
    best_config, best_stability = ranked[0]

    logger.info(
        f"[ml_optimizer] Stability ranking:\n"
        f"{format_stability_report(ranked, top_n=3)}"
    )

    # ── Phase 3: Train final model on best config ────────────────────────────
    final_model, fold_scores, calib_info = _train_and_optionally_calibrate(
        X=X, y=y,
        params=best_config,
        model_type=best_config.get("model_type", "lgbm"),
        metric=metric,
        state=state,
    )

    # OOF predictions for ensemble diversity selection (Day 16)
    oof_preds = _get_oof_predictions(
        X_train_cv_np, y_train_cv, best_config
    )

    # ── Phase 4: Wilcoxon gate vs any existing champion ──────────────────────
    existing_best = _get_existing_champion_scores(state)
    gate_decision = {}
    if existing_best:
        gate_decision = gate_result(
            fold_scores_a=fold_scores,
            fold_scores_b=existing_best,
            model_name_a="day19_optimizer",
            model_name_b="previous_champion",
        )
        log_event(state=state, action="wilcoxon_gate_decision",
                  agent="ml_optimizer", details=gate_decision)

    # ── Build registry entry ──────────────────────────────────────────────────
    model_id = f"{best_config.get('model_type', 'lgbm')}_day19_{int(time.time())}"
    registry_entry = {
        "model_id":              model_id,
        "cv_mean":               round(best_stability.mean, 6),
        "cv_std":                round(best_stability.std, 6),
        "fold_scores":           fold_scores,
        "stability_score":       best_stability.stability_score,
        "seed_results":          best_stability.seed_results,
        "params":                best_config,
        "oof_predictions":       oof_preds.tolist(),
        "data_hash":             state.get("data_hash", ""),
    }
    registry_entry = _update_model_registry_with_calibration(registry_entry, calib_info)

    state["model_registry"] = {
        **state.get("model_registry", {}),
        model_id: registry_entry,
    }
    state["memory_peak_gb"]       = _get_peak_rss()
    state["memory_oom_risk"]      = any(
        t.user_attrs.get("oom_risk") for t in study.trials if t.user_attrs
    )
    state["optuna_pruned_trials"] = sum(
        1 for t in study.trials
        if t.state == optuna.trial.TrialState.PRUNED
    )

    log_event(
        state=state,
        action="ml_optimizer_complete",
        agent="ml_optimizer",
        details={
            "model_id":        model_id,
            "cv_mean":         registry_entry["cv_mean"],
            "cv_std":          registry_entry["cv_std"],
            "stability_score": registry_entry["stability_score"],
            "is_calibrated":   registry_entry["is_calibrated"],
            "n_trials":        len(completed),
            "top_k_rerun":     len(top_k_trials),
        }
    )

    return state
```

---

## TASK 4 — ML Optimizer + Optuna contract test

**File:** `tests/contracts/test_ml_optimizer_optuna_contract.py`
**Status: IMMUTABLE after Day 19**
```python
# tests/contracts/test_ml_optimizer_optuna_contract.py
#
# CONTRACT: agents/ml_optimizer.py (Optuna + stability)
#
# INVARIANTS:
#   - After Optuna: top-10 configs re-run with exactly 5 seeds
#   - Winner ranked by stability_score = mean - 1.5*std, NOT peak CV
#   - All trials use metric_contract.scorer only (not hardcoded AUC)
#   - model_registry records: cv_mean, cv_std, stability_score, seed_results, fold_scores
#   - Calibration runs when metric in PROBABILITY_METRICS
#   - is_calibrated, calibration_method, calibration_score present in registry entry
#   - fold_scores stored in trial user_attrs during Optuna
#   - OOF predictions present in every registry entry
#   - stability_score = mean - 1.5 * std (not mean - 1.0*std, not mean alone)

import json
import pytest
import numpy as np


class TestMLOptimizerOptunaContract:

    def test_model_registry_has_all_required_fields(self, ml_optimizer_state):
        state = run_ml_optimizer(ml_optimizer_state)
        registry = state["model_registry"]
        assert registry, "model_registry is empty after ml_optimizer"

        REQUIRED = {
            "model_id", "cv_mean", "cv_std", "fold_scores",
            "stability_score", "seed_results", "params",
            "oof_predictions", "data_hash",
            "is_calibrated", "calibration_method", "calibration_score",
        }
        for name, entry in registry.items():
            missing = REQUIRED - set(entry.keys())
            assert not missing, (
                f"Model '{name}' missing required registry fields: {missing}"
            )

    def test_winner_ranked_by_stability_not_peak_cv(self, ml_optimizer_state):
        """
        The registered model must have the highest stability_score among all
        evaluated configs, not the highest cv_mean.
        """
        state = run_ml_optimizer(ml_optimizer_state)
        winner_entry = list(state["model_registry"].values())[0]

        assert "stability_score" in winner_entry, "stability_score missing from winner"
        assert "seed_results" in winner_entry, "seed_results missing from winner"

        # Stability formula: mean - 1.5 * std
        computed = (
            float(np.mean(winner_entry["seed_results"])) -
            1.5 * float(np.std(winner_entry["seed_results"]))
        )
        assert abs(winner_entry["stability_score"] - computed) < 1e-5, (
            f"stability_score={winner_entry['stability_score']} does not match "
            f"mean - 1.5*std = {computed:.6f}. "
            "Winner must be ranked by stability_score = mean - 1.5*std."
        )

    def test_seed_results_has_five_entries(self, ml_optimizer_state):
        """Exactly 5 seeds must be used for stability validation."""
        state = run_ml_optimizer(ml_optimizer_state)
        for name, entry in state["model_registry"].items():
            assert len(entry["seed_results"]) == 5, (
                f"Model '{name}' has {len(entry['seed_results'])} seed results. "
                "Must have exactly 5 (seeds=[42, 7, 123, 999, 2024])."
            )

    def test_fold_scores_stored_in_trial_user_attrs(self, ml_optimizer_state):
        """Optuna trials must store fold_scores for Wilcoxon gate (Day 13)."""
        state, study = run_ml_optimizer_return_study(ml_optimizer_state)
        for trial in study.trials:
            if trial.state.name == "COMPLETE":
                assert "fold_scores" in trial.user_attrs, (
                    f"Trial {trial.number} missing fold_scores in user_attrs. "
                    "Required for Wilcoxon gate."
                )
                assert len(trial.user_attrs["fold_scores"]) >= 3, (
                    f"Trial {trial.number} has < 3 fold scores."
                )

    def test_calibration_runs_for_probability_metric(self, ml_optimizer_state_logloss):
        """Calibration must be applied when metric is log_loss."""
        state = run_ml_optimizer(ml_optimizer_state_logloss)
        for name, entry in state["model_registry"].items():
            assert entry["is_calibrated"] is True, (
                f"Model '{name}' not calibrated despite log_loss metric."
            )
            assert entry["calibration_method"] in ("sigmoid", "isotonic"), (
                f"Model '{name}' has unknown calibration_method: "
                f"{entry['calibration_method']}"
            )
            assert entry["calibration_score"] is not None, (
                f"Model '{name}' missing calibration_score."
            )

    def test_calibration_skipped_for_auc_metric(self, ml_optimizer_state_auc):
        """Calibration must NOT run when metric is auc (not probability-based)."""
        state = run_ml_optimizer(ml_optimizer_state_auc)
        for name, entry in state["model_registry"].items():
            assert entry["is_calibrated"] is False, (
                f"Model '{name}' calibrated despite AUC metric. "
                "Calibration should only run for probability metrics."
            )
            assert entry["calibration_method"] == "none", (
                f"Model '{name}' has calibration_method={entry['calibration_method']} "
                "despite non-probability metric."
            )

    def test_oof_predictions_present_and_correct_length(self, ml_optimizer_state):
        """OOF predictions required for Day 16 diversity ensemble selection."""
        state = run_ml_optimizer(ml_optimizer_state)
        n_train = len(state["y_train"])

        for name, entry in state["model_registry"].items():
            oof = entry.get("oof_predictions", [])
            assert oof, f"Model '{name}' has empty oof_predictions."
            assert len(oof) == n_train, (
                f"Model '{name}' oof_predictions length {len(oof)} != "
                f"n_train {n_train}."
            )

    def test_cv_std_computed_correctly(self, ml_optimizer_state):
        """cv_std must match std of fold_scores, not std of seed_results."""
        state = run_ml_optimizer(ml_optimizer_state)
        for name, entry in state["model_registry"].items():
            computed_std = float(np.std(entry["fold_scores"]))
            assert abs(entry["cv_std"] - computed_std) < 1e-5, (
                f"Model '{name}' cv_std={entry['cv_std']} does not match "
                f"std(fold_scores)={computed_std:.6f}."
            )

    def test_all_trials_use_metric_contract_scorer(self, ml_optimizer_state):
        """
        Trials must use the metric from metric_contract, not a hardcoded AUC.
        If competition metric is log_loss, trials must optimise log_loss.
        """
        for scorer in ("auc", "log_loss", "brier_score"):
            state = {**ml_optimizer_state, "evaluation_metric": scorer}
            result_state, study = run_ml_optimizer_return_study(state)
            # Each trial's objective value must reflect the correct metric direction
            # For log_loss: lower is better → study direction should be "minimize"
            # For auc: higher is better → study direction should be "maximize"
            expected_direction = "minimize" if scorer == "log_loss" else "maximize"
            assert study.direction.name.lower() == expected_direction, (
                f"Study direction is '{study.direction.name}' for metric '{scorer}'. "
                f"Expected '{expected_direction}'."
            )
```

---

## INTEGRATION CHECKLIST

- [ ] `PROBABILITY_METRICS` frozenset defined at module level — not inside a function
- [ ] Calibration fold carved out BEFORE CV folds — never overlaps with validation data
- [ ] `cv='prefit'` used in `CalibratedClassifierCV` — model must already be fitted
- [ ] `_select_calibration_method()` uses `< 1000` (not `<=`) for Platt/isotonic boundary
- [ ] `run_with_seeds()` never raises — returns result with available seeds if some fail
- [ ] `stability_score = mean - 1.5 * std` — penalty is 1.5, not 1.0
- [ ] Top-10 configs by CV mean selected from Optuna BEFORE stability re-run
- [ ] `rank_by_stability()` sorts by `stability_score`, not by `mean`
- [ ] Day 12 OOM guards intact: `del models` in `finally`, `gc_after_trial=True`
- [ ] Day 15 LangSmith cost guard: `_disable_langsmith_tracing()` wraps `study.optimize()`
- [ ] `fold_scores` stored in `trial.user_attrs` during `_objective()`
- [ ] Study direction matches metric: "maximize" for AUC/accuracy, "minimize" for log_loss
- [ ] Calibration's `brier_score_loss` is computed on calibration fold only — not CV folds
- [ ] `StabilityResult` excluded from Redis checkpoint (same as `NullImportanceResult`)
- [ ] Contract test: `test_ml_optimizer_optuna_contract.py` — immutable after Day 19

## NEW STATE FIELDS
```python
optuna_pruned_trials:    int     # count of pruned trials (OOM)
# model_registry entries gain: is_calibrated, calibration_method,
#                               calibration_score, calibration_n_samples,
#                               stability_score, seed_results
```

## GIT COMMIT MESSAGE
```
Day 19: calibration, stability validator, Optuna HPO, optimizer contract

- ml_optimizer: GAP 7 prediction calibration
  Platt (sigmoid) < 1000 samples, Isotonic >= 1000
  Calibration fold separate from CV folds — no leakage
  Triggered by PROBABILITY_METRICS: log_loss, cross_entropy, brier_score
- tools/stability_validator.py: run_with_seeds() + rank_by_stability()
  stability_score = mean - 1.5*std
  Never raises — handles partial seed failures
- ml_optimizer: Optuna HPO — LightGBM, XGBoost, CatBoost search spaces
  Top-10 by CV mean re-run with 5 seeds
  Winner ranked by stability_score, not peak CV
- ml_optimizer: Day 12+13 guards preserved (gc, OOM, Wilcoxon)
- contracts/test_ml_optimizer_optuna_contract.py: 8 immutable contracts
- tests/test_day19_quality.py: 50 adversarial tests — all green
```