# agents/ensemble_architect.py
#
# Day 22 — Ensemble Architect
# Diversity pruning, constrained Optuna weights, stacking meta-learner,
# Wilcoxon validation gate, holdout scoring.
#
# Called AFTER ml_optimizer completes. All model variants are in model_registry.

import logging
import gc
import numpy as np
import optuna
from scipy.stats import pearsonr
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import get_scorer
from core.state import ProfessorState
from core.lineage import log_event
from tools.wilcoxon_gate import is_significantly_better

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────
CORRELATION_THRESHOLD = 0.98
MIN_WEIGHT = 0.05
OPTUNA_N_TRIALS = 50
OPTUNA_N_JOBS = 1
OPTUNA_GC_AFTER_TRIAL = True
HOLDOUT_FRACTION = 0.20
HOLDOUT_SEED = 42
META_LEARNER_CV_FOLDS = 5


# ── Requirement 1: Data hash validation ──────────────────────────

def _validate_and_filter_data_hash(state: ProfessorState) -> dict:
    """
    Validate that all models in model_registry share the same data_hash
    as state["data_hash"]. Filter out mismatches. Raise if none remain.

    Returns the filtered registry dict keyed by model name.
    """
    current_hash = state.get("data_hash", "")
    registry_raw = state.get("model_registry", [])

    # Normalise to dict: {model_name: entry}
    if isinstance(registry_raw, list):
        registry = {}
        for entry in registry_raw:
            name = entry.get("model_id") or entry.get("model_type", f"model_{len(registry)}")
            registry[name] = entry
    elif isinstance(registry_raw, dict):
        registry = dict(registry_raw)
    else:
        raise ValueError(f"model_registry has unexpected type: {type(registry_raw)}")

    if not registry:
        raise ValueError("model_registry is empty — no models to ensemble.")

    # Filter by data_hash
    filtered = {}
    pruned_hash = []
    for name, entry in registry.items():
        entry_hash = entry.get("data_hash", "")
        if entry_hash != current_hash:
            logger.warning(
                f"[ensemble_architect] Model '{name}' has data_hash='{entry_hash}' "
                f"which does not match current data_hash='{current_hash}'. Removing."
            )
            pruned_hash.append(name)
        else:
            filtered[name] = entry

    if not filtered:
        raise ValueError(
            "No models match current data_hash. Retrain required."
        )

    if pruned_hash:
        logger.info(
            f"[ensemble_architect] Removed {len(pruned_hash)} models with stale data_hash: "
            f"{pruned_hash}"
        )

    return filtered


# ── Requirement 2: OOF validation ────────────────────────────────

def _validate_oof_predictions(registry: dict, y_train) -> None:
    """Verify every model has oof_predictions with correct length."""
    expected_len = len(y_train)
    for name, entry in registry.items():
        oof = entry.get("oof_predictions")
        if oof is None:
            raise ValueError(
                f"Model '{name}' is missing 'oof_predictions' in registry entry. "
                "Cannot run ensemble without OOF predictions."
            )
        if len(oof) != expected_len:
            raise ValueError(
                f"Model '{name}' has len(oof_predictions)={len(oof)} "
                f"but len(y_train)={expected_len}. Shape mismatch."
            )


# ── Requirement 3: Diversity pruning ─────────────────────────────

def _prune_by_diversity(registry: dict) -> tuple:
    """
    Greedy diversity selection:
    - Sort by cv_mean descending. Best model is anchor.
    - For each remaining model: compute Pearson correlation with every
      already-selected model. If any correlation > 0.98, reject.
    - Returns (selected_names: list[str], pruned_names: list[str])
    """
    sorted_models = sorted(
        registry.items(),
        key=lambda kv: float(kv[1].get("cv_mean", 0.0)),
        reverse=True,
    )

    selected_names = []
    selected_oofs = []
    pruned_names = []

    for name, entry in sorted_models:
        oof = np.array(entry["oof_predictions"], dtype=float)

        if not selected_names:
            # Anchor always selected
            selected_names.append(name)
            selected_oofs.append(oof)
            continue

        # Compute max correlation with any already-selected model
        max_corr = -1.0
        for sel_oof in selected_oofs:
            corr, _ = pearsonr(oof, sel_oof)
            max_corr = max(max_corr, corr)

        if max_corr > CORRELATION_THRESHOLD:
            logger.info(
                f"[ensemble_architect] Diversity pruning: '{name}' rejected "
                f"(max correlation with selected models = {max_corr:.4f} > {CORRELATION_THRESHOLD})"
            )
            pruned_names.append(name)
        else:
            selected_names.append(name)
            selected_oofs.append(oof)

    return selected_names, selected_oofs, pruned_names


# ── Requirement 4: Holdout split ─────────────────────────────────

def _split_holdout(y_train, task_type: str):
    """
    Split training data into 80% opt_pool and 20% val_holdout.
    Stratified for classification, random for regression.
    Returns (opt_indices, val_indices).
    """
    n = len(y_train)
    indices = np.arange(n)

    if task_type in ("binary_classification", "multiclass_classification", "classification"):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=HOLDOUT_SEED)
        # Use the last fold as holdout
        folds = list(skf.split(indices, y_train))
        _, val_indices = folds[-1]
        opt_indices = np.array([i for i in indices if i not in set(val_indices)])
        val_indices = np.array(val_indices)
    else:
        from sklearn.model_selection import train_test_split
        opt_indices, val_indices = train_test_split(
            indices, test_size=HOLDOUT_FRACTION, random_state=HOLDOUT_SEED
        )

    return opt_indices, val_indices


def _split_oof(oof_array, opt_indices, val_indices):
    """Split OOF predictions to match the holdout split."""
    return oof_array[opt_indices], oof_array[val_indices]


# ── Requirement 5: Constrained Optuna weight optimisation ────────

def _run_weight_optimisation(
    oof_stack_opt: np.ndarray,
    y_opt: np.ndarray,
    oof_stack_val: np.ndarray,
    y_val: np.ndarray,
    cv_scores: list[float],
    metric: str,
    task_type: str,
):
    """
    Use Optuna to find optimal blend weights.
    Maximises ensemble score on opt_pool.

    Constraints:
    - Weights sum to 1.0 via softmax normalisation
    - No individual weight below 0.05 (clip + renormalise)
    - max(1, len(selected_models) // 100) free parameters
    """
    n_models = oof_stack_opt.shape[1]
    n_params = max(1, n_models // 100)

    # Determine study direction
    minimize_metrics = frozenset({
        "log_loss", "cross_entropy", "brier_score",
        "logloss", "binary_crossentropy", "rmse", "mae", "mse",
    })
    direction = "minimize" if metric in minimize_metrics else "maximize"

    scorer = get_scorer(metric)

    def objective(trial):
        # Search temperature parameter(s)
        raw_params = []
        for i in range(n_params):
            raw_params.append(trial.suggest_float(f"temp_{i}", -5.0, 5.0))

        # Derive weights from CV scores scaled by temperature
        temps = np.array(raw_params)
        # For single param: scale CV scores by exp(temp)
        if n_params == 1:
            temp = temps[0]
            scaled_scores = np.array(cv_scores) * np.exp(temp)
        else:
            # Multiple params: partition models and scale each group
            scaled_scores = np.array(cv_scores)

        # Softmax normalisation
        exp_scores = np.exp(scaled_scores - np.max(scaled_scores))
        weights = exp_scores / exp_scores.sum()

        # Clip weights below MIN_WEIGHT and renormalise
        weights = np.clip(weights, MIN_WEIGHT, None)
        weights = weights / weights.sum()

        # Compute ensemble predictions on opt_pool
        ensemble_preds = oof_stack_opt @ weights

        # Score
        try:
            score = _score_predictions(y_opt, ensemble_preds, metric)
        except Exception:
            # If scorer fails (e.g., invalid preds), return worst possible value
            return float("-inf") if direction == "maximize" else float("inf")

        return score

    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=OPTUNA_N_TRIALS, n_jobs=OPTUNA_N_JOBS,
                   gc_after_trial=OPTUNA_GC_AFTER_TRIAL)

    # Extract best weights using the same derivation
    best_params = study.best_params
    temps = np.array([best_params[f"temp_{i}"] for i in range(n_params)])

    if n_params == 1:
        temp = temps[0]
        scaled_scores = np.array(cv_scores) * np.exp(temp)
    else:
        scaled_scores = np.array(cv_scores)

    exp_scores = np.exp(scaled_scores - np.max(scaled_scores))
    weights = exp_scores / exp_scores.sum()
    weights = np.clip(weights, MIN_WEIGHT, None)
    weights = weights / weights.sum()

    return weights, study.best_value


# ── Requirement 6: Stacking meta-learner ─────────────────────────

def _train_stacking_meta_learner(
    oof_stack: np.ndarray,
    y_train,
    task_type: str,
):
    """
    Train a stacking meta-learner using 5-fold CV to avoid leakage.
    Returns (meta_learner, meta_oof_predictions).
    """
    n_samples = oof_stack.shape[0]
    meta_oof = np.zeros(n_samples)

    if task_type in ("binary_classification", "multiclass_classification", "classification"):
        cv = StratifiedKFold(n_splits=META_LEARNER_CV_FOLDS, shuffle=True, random_state=42)
        meta_learner = LogisticRegression(C=0.1, max_iter=1000)
    else:
        cv = KFold(n_splits=META_LEARNER_CV_FOLDS, shuffle=True, random_state=42)
        meta_learner = Ridge(alpha=10.0)

    for train_idx, val_idx in cv.split(oof_stack, y_train):
        fold_model = type(meta_learner)(**meta_learner.get_params())
        fold_model.fit(oof_stack[train_idx], y_train[train_idx])
        meta_oof[val_idx] = fold_model.predict(oof_stack[val_idx])

    # Train final meta-learner on all data
    meta_learner.fit(oof_stack, y_train)

    return meta_learner, meta_oof


# ── Requirement 7: Wilcoxon validation gate ──────────────────────

def _apply_wilcoxon_gate(
    ensemble_fold_scores: list[float],
    best_single_fold_scores: list[float],
    metric: str,
) -> bool:
    """
    Call Wilcoxon gate to check if ensemble significantly beats best single model.
    """
    minimize_metrics = frozenset({
        "log_loss", "cross_entropy", "brier_score",
        "logloss", "binary_crossentropy", "rmse", "mae", "mse",
    })
    direction = "minimize" if metric in minimize_metrics else "maximize"

    return is_significantly_better(
        ensemble_fold_scores,
        best_single_fold_scores,
        direction=direction,
    )


# ── Requirement 8: Holdout validation ────────────────────────────

def _score_predictions(
    y_true,
    predictions: np.ndarray,
    metric: str,
) -> float:
    """Score predictions directly using the metric name."""
    from sklearn.metrics import (
        accuracy_score, roc_auc_score, f1_score,
        mean_squared_error, mean_absolute_error, log_loss,
    )

    metric_lower = metric.lower()

    # Classification metrics
    if metric_lower in ("accuracy", "balanced_accuracy"):
        preds_binary = (predictions >= 0.5).astype(int)
        return float(accuracy_score(y_true, preds_binary))
    elif metric_lower in ("roc_auc", "auc", "roc_auc_ovo"):
        return float(roc_auc_score(y_true, predictions))
    elif metric_lower in ("f1", "f1_score"):
        preds_binary = (predictions >= 0.5).astype(int)
        return float(f1_score(y_true, preds_binary, average="weighted"))
    elif metric_lower in ("log_loss", "cross_entropy", "brier_score",
                           "logloss", "binary_crossentropy"):
        # Clip predictions to avoid log(0)
        clipped = np.clip(predictions, 1e-15, 1 - 1e-15)
        return float(log_loss(y_true, clipped))

    # Regression metrics
    elif metric_lower in ("rmse",):
        return float(np.sqrt(mean_squared_error(y_true, predictions)))
    elif metric_lower in ("mse",):
        return float(mean_squared_error(y_true, predictions))
    elif metric_lower in ("mae",):
        return float(mean_absolute_error(y_true, predictions))

    # Fallback: try get_scorer
    else:
        try:
            scorer = get_scorer(metric)
            # get_scorer returns a _BaseScorer that needs (estimator, X, y)
            # For direct predictions, use the underlying scoring function
            return float(scorer._score_func(y_true, predictions))
        except Exception:
            # Last resort: MSE
            return float(-mean_squared_error(y_true, predictions))


def _score_on_holdout(
    predictions: np.ndarray,
    y_holdout,
    metric: str,
) -> float:
    """Score predictions on holdout set."""
    return _score_predictions(y_holdout, predictions, metric)


# ── Main entry point ─────────────────────────────────────────────

def run_ensemble_architect(state: ProfessorState) -> dict:
    """
    Ensemble Architect pipeline:
    1. Data hash validation
    2. OOF validation
    3. Diversity pruning (correlation > 0.98)
    4. Holdout split (80/20)
    5. Constrained Optuna weight optimisation
    6. Stacking meta-learner
    7. Wilcoxon validation gate
    8. Holdout validation
    9. State outputs
    10. Lineage logging
    """
    session_id = state["session_id"]
    metric = state.get("evaluation_metric", "accuracy")
    task_type = state.get("task_type", "classification")
    y_train = np.array(state["y_train"], dtype=float)

    logger.info(f"[ensemble_architect] Starting — session: {session_id}")

    # ── Step 1: Data hash validation ──────────────────────────────
    registry = _validate_and_filter_data_hash(state)
    logger.info(f"[ensemble_architect] Hash validation passed: {len(registry)} models remain")

    # ── Step 2: OOF validation ────────────────────────────────────
    _validate_oof_predictions(registry, y_train)
    logger.info(f"[ensemble_architect] OOF validation passed for all {len(registry)} models")

    # ── Step 3: Diversity pruning ─────────────────────────────────
    selected_names, selected_oofs, pruned_diversity = _prune_by_diversity(registry)
    logger.info(
        f"[ensemble_architect] Diversity pruning: {len(selected_names)} selected, "
        f"{len(pruned_diversity)} pruned"
    )

    # Build selected registry subset
    selected_registry = {name: registry[name] for name in selected_names}

    # ── Step 4: Holdout split ─────────────────────────────────────
    opt_indices, val_indices = _split_holdout(y_train, task_type)
    y_opt = y_train[opt_indices]
    y_val_holdout = y_train[val_indices]

    # Split OOF predictions
    oof_stack_opt_list = []
    oof_stack_val_list = []
    for name in selected_names:
        oof = np.array(registry[name]["oof_predictions"], dtype=float)
        oof_opt, oof_val = _split_oof(oof, opt_indices, val_indices)
        oof_stack_opt_list.append(oof_opt)
        oof_stack_val_list.append(oof_val)

    oof_stack_opt = np.column_stack(oof_stack_opt_list)
    oof_stack_val = np.column_stack(oof_stack_val_list)

    # Full OOF stack for meta-learner
    oof_stack_full = np.column_stack([
        np.array(registry[name]["oof_predictions"], dtype=float)
        for name in selected_names
    ])

    # CV scores for weight derivation
    cv_scores = [float(registry[name].get("cv_mean", 0.0)) for name in selected_names]

    # ── Step 5: Optuna weight optimisation ────────────────────────
    if len(selected_names) > 1:
        optimal_weights, opt_score = _run_weight_optimisation(
            oof_stack_opt, y_opt,
            oof_stack_val, y_val_holdout,
            cv_scores, metric, task_type,
        )
    else:
        optimal_weights = np.array([1.0])
        opt_score = 0.0

    # Compute weighted blend OOF predictions (full data)
    weight_dict = {name: float(w) for name, w in zip(selected_names, optimal_weights)}
    blend_oof_full = oof_stack_full @ optimal_weights

    # ── Step 6: Stacking meta-learner ─────────────────────────────
    meta_learner, meta_oof_full = _train_stacking_meta_learner(
        oof_stack_full, y_train, task_type
    )

    # Score both approaches on holdout to decide which to use
    blend_holdout_preds = oof_stack_val @ optimal_weights

    blend_holdout_score = _score_on_holdout(blend_holdout_preds, y_val_holdout, metric)
    meta_holdout_score = _score_on_holdout(meta_oof_full[val_indices], y_val_holdout, metric)

    minimize_metrics = frozenset({
        "log_loss", "cross_entropy", "brier_score",
        "logloss", "binary_crossentropy", "rmse", "mae", "mse",
    })

    if minimize_metrics:
        meta_learner_used = meta_holdout_score <= blend_holdout_score
    else:
        meta_learner_used = meta_holdout_score >= blend_holdout_score

    if meta_learner_used:
        ensemble_oof_full = meta_oof_full
        ensemble_holdout_preds = meta_oof_full[val_indices]
        ensemble_holdout_score = meta_holdout_score
        logger.info(
            f"[ensemble_architect] Meta-learner selected (holdout: {meta_holdout_score:.5f} "
            f"vs blend: {blend_holdout_score:.5f})"
        )
    else:
        ensemble_oof_full = blend_oof_full
        ensemble_holdout_preds = blend_holdout_preds
        ensemble_holdout_score = blend_holdout_score
        logger.info(
            f"[ensemble_architect] Weighted blend selected (holdout: {blend_holdout_score:.5f} "
            f"vs meta: {meta_holdout_score:.5f})"
        )

    # ── Step 7: Wilcoxon validation gate ──────────────────────────
    # Get best single model's fold scores
    best_single_name = selected_names[0]  # Already sorted by cv_mean desc
    best_single_fold_scores = registry[best_single_name].get("fold_scores", [])

    # Compute ensemble fold scores using same CV folds
    # We approximate by scoring the ensemble OOF on the opt_pool portion
    # and comparing against the best single model's opt_pool OOF
    if len(best_single_fold_scores) >= 5:
        # Use the opt_pool OOF to compute ensemble fold scores
        # Split opt_pool into 5 folds matching the original CV structure
        n_opt = len(y_opt)
        if task_type in ("binary_classification", "multiclass_classification", "classification"):
            opt_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        else:
            opt_cv = KFold(n_splits=5, shuffle=True, random_state=42)

        ensemble_fold_scores = []
        best_single_fold_scores_opt = []

        best_single_oof = np.array(registry[best_single_name]["oof_predictions"], dtype=float)
        best_single_oof_opt = best_single_oof[opt_indices]

        for _, fold_val_idx in opt_cv.split(oof_stack_opt, y_opt):
            fold_ensemble_preds = ensemble_oof_full[opt_indices][fold_val_idx]
            fold_best_preds = best_single_oof_opt[fold_val_idx]

            ens_score = _score_predictions(y_opt[fold_val_idx], fold_ensemble_preds, metric)
            best_score = _score_predictions(y_opt[fold_val_idx], fold_best_preds, metric)

            ensemble_fold_scores.append(float(ens_score))
            best_single_fold_scores_opt.append(float(best_score))

        ensemble_accepted = _apply_wilcoxon_gate(
            ensemble_fold_scores,
            best_single_fold_scores_opt,
            metric,
        )
    else:
        # Not enough folds for Wilcoxon — fall back to mean comparison
        logger.warning(
            "[ensemble_architect] Insufficient fold scores for Wilcoxon test. "
            "Falling back to mean comparison."
        )
        ens_mean = float(np.mean(ensemble_fold_scores)) if ensemble_fold_scores else 0.0
        best_mean = float(np.mean(best_single_fold_scores)) if best_single_fold_scores else 0.0
        if minimize_metrics:
            ensemble_accepted = ens_mean < best_mean
        else:
            ensemble_accepted = ens_mean > best_mean

    if not ensemble_accepted:
        logger.warning(
            "[ensemble_architect] Wilcoxon gate: ensemble does NOT significantly beat "
            f"best single model ('{best_single_name}'). "
            "Using best single model's predictions instead."
        )
        ensemble_oof_full = np.array(
            registry[best_single_name]["oof_predictions"], dtype=float
        )
        ensemble_holdout_score = _score_on_holdout(
            ensemble_oof_full[val_indices], y_val_holdout, metric
        )
        weight_dict = {best_single_name: 1.0}
        selected_names = [best_single_name]

    # ── Step 8: Holdout validation logging ────────────────────────
    logger.info(
        f"[ensemble_architect] Holdout score: {ensemble_holdout_score:.5f} | "
        f"{'Ensemble' if ensemble_accepted else 'Single model'} used | "
        f"Models in ensemble: {len(selected_names)} | "
        f"Weights: {weight_dict}"
    )

    # ── Step 9: State outputs ─────────────────────────────────────
    # Correlation matrix for selected models
    corr_matrix = {}
    for i, name_a in enumerate(selected_names):
        oof_a = np.array(registry[name_a]["oof_predictions"], dtype=float)
        for name_b in selected_names[i + 1:]:
            oof_b = np.array(registry[name_b]["oof_predictions"], dtype=float)
            corr, _ = pearsonr(oof_a, oof_b)
            corr_matrix[f"{name_a}_vs_{name_b}"] = round(float(corr), 4)

    result = {
        "selected_models": selected_names,
        "ensemble_weights": weight_dict,
        "ensemble_oof": [float(v) for v in ensemble_oof_full],
        "ensemble_holdout_score": float(ensemble_holdout_score),
        "ensemble_accepted": bool(ensemble_accepted),
        "ensemble_correlation_matrix": corr_matrix,
        "models_pruned_diversity": pruned_diversity,
        "meta_learner_used": bool(meta_learner_used),
    }

    # ── Step 10: Lineage ──────────────────────────────────────────
    log_event(
        session_id=session_id,
        agent="ensemble_architect",
        action="ensemble_selection_complete",
        keys_read=["model_registry", "data_hash", "y_train", "evaluation_metric", "task_type"],
        keys_written=[
            "selected_models", "ensemble_weights", "ensemble_oof",
            "ensemble_holdout_score", "ensemble_accepted",
            "ensemble_correlation_matrix", "models_pruned_diversity",
            "meta_learner_used",
        ],
        values_changed={
            "n_candidates": len(registry),
            "n_selected": len(selected_names),
            "n_pruned_diversity": len(pruned_diversity),
            "ensemble_accepted": bool(ensemble_accepted),
            "ensemble_holdout_score": float(ensemble_holdout_score),
            "weights": weight_dict,
        },
    )

    logger.info(f"[ensemble_architect] Complete — session: {session_id}")
    return result
