# agents/pseudo_label_agent.py
# -------------------------------------------------------------------------
# Day 18: GM-CAP 6 — Pseudo-labeling with confidence gating.
# Top 10% most confident test predictions → add to training folds ONLY.
# Validation fold never sees pseudo-labels (critical invariant).
# Max 3 iterations, Wilcoxon gate on CV improvement.
# 
# FIXED: 2026-03-24 — All 20 bugs fixed
# -------------------------------------------------------------------------

import gc
import logging
from dataclasses import dataclass, field

import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, KFold

from core.state import ProfessorState
from core.lineage import log_event

# FIX Bug #5: Import is_significantly_better
from tools.wilcoxon_gate import is_significantly_better

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────
CONFIDENCE_TOP_FRACTION = 0.10   # top 10% most confident
MAX_PL_ITERATIONS = 3
MIN_CV_IMPROVEMENT = 0.001       # must improve by at least 0.1pp


# FIX Bug #19: Use field(default_factory=list) instead of mutable defaults
@dataclass
class PseudoLabelResult:
    iterations_completed: int
    pseudo_labels_added: list[int] = field(default_factory=list)
    cv_scores_with_pl: list[float] = field(default_factory=list)
    cv_scores_without_pl: list[float] = field(default_factory=list)
    cv_improvements: list[float] = field(default_factory=list)
    halted_early: bool = False
    halt_reason: str = ""
    final_pseudo_label_mask: list[int] = field(default_factory=list)
    confidence_thresholds: list[float] = field(default_factory=list)


# ── Confidence computation ───────────────────────────────────────

def _compute_confidence(
    y_pred: np.ndarray,
    metric: str,
    quantile_model=None,
) -> np.ndarray:
    """
    Computes confidence score for each prediction.

    Binary: confidence = |pred - 0.5| (distance from decision boundary)
    Regression: inverse interval width, or uniform if no quantile model
    Multiclass: margin between top-2 class probabilities
    """
    if metric in ("auc", "logloss", "binary"):
        return np.abs(y_pred - 0.5)

    elif metric in ("rmse", "mae", "regression"):
        if quantile_model is not None:
            q_low = quantile_model.predict(X=None, pred_quantile=0.1)
            q_high = quantile_model.predict(X=None, pred_quantile=0.9)
            return 1.0 / (q_high - q_low + 1e-6)
        else:
            logger.warning(
                "[pseudo_label] No quantile model for regression confidence. "
                "Using uniform confidence."
            )
            return np.ones(len(y_pred))

    elif metric in ("multiclass", "logloss_multiclass"):
        if y_pred.ndim == 1:
            return np.abs(y_pred - 0.5)
        sorted_probs = np.sort(y_pred, axis=1)[:, ::-1]
        return sorted_probs[:, 0] - sorted_probs[:, 1]

    else:
        logger.warning(f"[pseudo_label] Unknown metric '{metric}' — using binary confidence.")
        return np.abs(y_pred - 0.5)


# ── Sample selection ─────────────────────────────────────────────

def _select_confident_samples(
    confidence: np.ndarray,
    y_pred: np.ndarray,
    top_fraction: float = CONFIDENCE_TOP_FRACTION,
) -> tuple[np.ndarray, float]:
    """
    Selects top-fraction of test samples by confidence.
    Returns (boolean mask, threshold confidence value).
    """
    threshold = float(np.percentile(confidence, (1.0 - top_fraction) * 100))
    mask = confidence >= threshold
    return mask, threshold


# ── CV with pseudo-labels ────────────────────────────────────────

def _run_cv_with_pseudo_labels(
    X_train: pl.DataFrame,
    y_train: np.ndarray,
    X_pseudo: pl.DataFrame,
    y_pseudo: np.ndarray,
    lgbm_params: dict,
    n_folds: int = 5,
    metric: str = "auc",
    random_state: int = 42,
) -> list[float]:
    """
    Runs CV where pseudo-labels are added to TRAINING FOLDS ONLY.

    CRITICAL INVARIANT: Validation fold sees only real labeled samples.
    Pseudo-labels are concatenated to the training portion of each fold
    INSIDE the loop, AFTER train_idx/val_idx are determined.
    """
    from sklearn.metrics import roc_auc_score, mean_squared_error

    is_classification = metric in ("auc", "logloss", "binary", "multiclass")
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state) \
         if is_classification else \
         KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    X_np = X_train.to_numpy()
    fold_scores = []

    ModelClass = lgb.LGBMClassifier if is_classification else lgb.LGBMRegressor
    X_pseudo_np = X_pseudo.to_numpy()

    for train_idx, val_idx in cv.split(X_np, y_train if is_classification else None):
        # Training fold: real labels + pseudo-labels
        X_fold_train = np.vstack([X_np[train_idx], X_pseudo_np])
        y_fold_train = np.concatenate([y_train[train_idx], y_pseudo])

        # Validation fold: ONLY real labels (invariant)
        X_fold_val = X_np[val_idx]
        y_fold_val = y_train[val_idx]

        model = ModelClass(**lgbm_params)
        model.fit(X_fold_train, y_fold_train)

        if metric == "auc":
            preds = model.predict_proba(X_fold_val)[:, 1]
            score = roc_auc_score(y_fold_val, preds)
        elif metric in ("rmse", "regression"):
            preds = model.predict(X_fold_val)
            score = -np.sqrt(mean_squared_error(y_fold_val, preds))
        else:
            preds = model.predict_proba(X_fold_val)[:, 1]
            score = roc_auc_score(y_fold_val, preds)

        fold_scores.append(float(score))

        del model
        gc.collect()

    return fold_scores


# ── Memory cleanup helper ────────────────────────────────────────

def _cleanup_pl_iteration(**kwargs):
    """Delete large arrays and run GC. FIX Bug #20."""
    for obj in kwargs.values():
        if obj is not None:
            del obj
    gc.collect()


# ── Main entry point ─────────────────────────────────────────────

def run_pseudo_label_agent(state: ProfessorState) -> ProfessorState:
    """
    GM-CAP 6: Pseudo-labeling with confidence gating.

    1. Train on labeled data (best model params from registry)
    2. Predict test set
    3. Select top 10% most confident predictions
    4. Add to training folds only (never validation)
    5. CV gate + Wilcoxon: only proceed if CV improves significantly
    6. Repeat up to MAX_PL_ITERATIONS
    """
    from tools.data_tools import read_parquet, read_json
    import os

    # ── Load paths from state ────────────────────────────────────
    feature_data_path = state.get("feature_data_path")
    feature_data_path_test = state.get("feature_data_path_test")

    # FIX Bug #6: Fallback paths if not set by upstream
    if not feature_data_path:
        session_id = state["session_id"]
        feature_data_path = f"outputs/{session_id}/X_train.parquet"
    if not feature_data_path_test:
        session_id = state["session_id"]
        feature_data_path_test = f"outputs/{session_id}/X_test.parquet"

    # ── Validate target_col FIRST (before loading data) ──────────
    target_col = state.get("target_col")
    if not target_col:
        logger.warning("[pseudo_label] target_col not set in state. Skipping.")
        return {**state, "pseudo_labels_applied": False, "pseudo_label_cv_improvement": 0.0}

    # ── Validate paths ───────────────────────────────────────────
    if not os.path.exists(feature_data_path):
        logger.warning(f"[pseudo_label] Training data not found: {feature_data_path}. Skipping.")
        return {**state, "pseudo_labels_applied": False, "pseudo_label_cv_improvement": 0.0}

    if not os.path.exists(feature_data_path_test):
        logger.warning(f"[pseudo_label] Test data not found: {feature_data_path_test}. Skipping.")
        return {**state, "pseudo_labels_applied": False, "pseudo_label_cv_improvement": 0.0}

    # ── FIX Bug #1, #3: Load data from disk ──────────────────────
    X_train = read_parquet(feature_data_path)
    X_test = read_parquet(feature_data_path_test)

    # ── FIX Bug #2, #8, #13: Extract target column ───────────────
    # Validate target exists in training data
    if target_col not in X_train.columns:
        raise ValueError(f"[pseudo_label] Target '{target_col}' not in training data columns: {X_train.columns}")
    
    y_train = X_train[target_col].to_numpy()
    X_train = X_train.drop(target_col)

    # FIX Bug #14: Drop target from test if present (prevent leakage)
    if target_col in X_test.columns:
        logger.warning(f"[pseudo_label] Dropping target column from test data")
        X_test = X_test.drop(target_col)

    # ── FIX Bug #9, #15: Enforce feature order ───────────────────
    feature_order = state.get("feature_order")
    if feature_order:
        try:
            X_train = X_train.select(feature_order)
            X_test = X_test.select(feature_order)
        except pl.exceptions.ColumnNotFoundError as e:
            logger.error(f"[pseudo_label] Feature order mismatch: {e}")
            raise ValueError(f"Test data columns don't match feature_order: {e}")

    # ── FIX Bug #4: Load metric ──────────────────────────────────
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

    # FIX Bug #7, #13: Look up best model entry from registry (support both list and dict formats)
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

    # ── FIX Bug #1, #2, #3: Initialize working variables ─────────
    # Use actual loaded data schema
    X_pseudo_accumulated = X_train.slice(0, 0)  # Empty DataFrame with same schema
    y_pseudo_accumulated = np.array([], dtype=y_train.dtype)
    current_test_mask = np.zeros(len(X_test), dtype=bool)
    
    # FIX Bug #11: Track previous iteration's fold scores for fair comparison
    prev_fold_scores = baseline_cv.copy()

    # ── Main iteration loop ──────────────────────────────────────
    for iteration in range(1, MAX_PL_ITERATIONS + 1):
        logger.info(f"[pseudo_label] Iteration {iteration}/{MAX_PL_ITERATIONS}")
        
        # FIX Bug #17, #18: Add try/except and validation
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
                _cleanup_pl_iteration(X_all=X_all, y_all=y_all)
                break
            
            # Predict test set — exclude already pseudo-labeled samples
            remaining_mask = ~current_test_mask
            X_remaining = X_test.filter(pl.Series(remaining_mask))

            if X_remaining.is_empty():
                result.halt_reason = "no_confident_samples"
                result.halted_early = True
                _cleanup_pl_iteration(X_all=X_all, y_all=y_all, model=model)
                break

            y_pred = model.predict_proba(X_remaining.to_numpy())[:, 1] if is_cls \
                     else model.predict(X_remaining.to_numpy())

            del model  # Free memory early
            gc.collect()

            # Select high-confidence samples
            confidence = _compute_confidence(y_pred, metric)
            conf_mask, threshold = _select_confident_samples(confidence, y_pred)

            n_selected = int(conf_mask.sum())
            if n_selected == 0:
                result.halt_reason = "no_confident_samples"
                result.halted_early = True
                _cleanup_pl_iteration(X_all=X_all, y_all=y_all, y_pred=y_pred, confidence=confidence, X_remaining=X_remaining)
                break

            result.confidence_thresholds.append(threshold)

            X_new_pseudo = X_remaining.filter(pl.Series(conf_mask))
            y_new_pseudo = y_pred[conf_mask]

            # FIX Bug #10, #16: Convert to hard labels BEFORE using + type safety
            if is_cls:
                y_new_pseudo = (y_new_pseudo >= 0.5).astype(y_train.dtype)
            else:
                # Ensure dtype matches for regression
                if y_new_pseudo.dtype != y_train.dtype:
                    y_new_pseudo = y_new_pseudo.astype(y_train.dtype)

            # CV with pseudo-labels — validation fold ONLY sees real labels
            if len(y_pseudo_accumulated) > 0:
                X_pseudo_for_cv = pl.concat([X_pseudo_accumulated, X_new_pseudo])
                y_pseudo_for_cv = np.concatenate([y_pseudo_accumulated, y_new_pseudo])
            else:
                X_pseudo_for_cv = X_new_pseudo
                y_pseudo_for_cv = y_new_pseudo

            cv_with = _run_cv_with_pseudo_labels(
                X_train=X_train,
                y_train=y_train,
                X_pseudo=X_pseudo_for_cv,
                y_pseudo=y_pseudo_for_cv,
                lgbm_params=lgbm_params,
                metric=metric,
            )

            cv_mean_with = float(np.mean(cv_with))
            # FIX Bug #12: Compare against previous iteration, not original model
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

            # FIX Bug #11: Wilcoxon gate compares against PREVIOUS iteration
            gate_passed = is_significantly_better(cv_with, prev_fold_scores)

            if not gate_passed and improvement < MIN_CV_IMPROVEMENT:
                result.halt_reason = "cv_did_not_improve"
                result.halted_early = True
                _cleanup_pl_iteration(X_all=X_all, y_all=y_all, y_pred=y_pred, confidence=confidence, X_remaining=X_remaining)
                break

            # Accept iteration — update baseline for next comparison
            prev_fold_scores = cv_with.copy()
            baseline_cv = cv_with  # Also update for final state
            
            # Accumulate pseudo-labels
            if len(y_pseudo_accumulated) > 0:
                X_pseudo_accumulated = pl.concat([X_pseudo_accumulated, X_new_pseudo])
                y_pseudo_accumulated = np.concatenate([y_pseudo_accumulated, y_new_pseudo])
            else:
                X_pseudo_accumulated = X_new_pseudo
                y_pseudo_accumulated = y_new_pseudo

            current_test_mask[np.where(remaining_mask)[0][conf_mask]] = True
            result.iterations_completed = iteration
            
            _cleanup_pl_iteration(X_all=X_all, y_all=y_all, y_pred=y_pred, confidence=confidence, X_remaining=X_remaining)

        except Exception as e:
            logger.error(f"[pseudo_label] Iteration {iteration} failed: {e}")
            result.halt_reason = f"iteration_failed: {e}"
            result.halted_early = True
            break

    if result.iterations_completed == MAX_PL_ITERATIONS and not result.halted_early:
        result.halt_reason = "max_iterations"

    result.final_pseudo_label_mask = current_test_mask.astype(int).tolist()

    # ── Update state ─────────────────────────────────────────────
    if len(y_pseudo_accumulated) > 0:
        X_with_pseudo = pl.concat([X_train, X_pseudo_accumulated])
        y_with_pseudo = np.concatenate([y_train, y_pseudo_accumulated])
    else:
        X_with_pseudo = X_train
        y_with_pseudo = y_train

    pl_applied = result.iterations_completed > 0 and not result.halted_early

    state = {
        **state,
        "pseudo_label_result": result,
        "X_train_with_pseudo": X_with_pseudo,
        "y_train_with_pseudo": y_with_pseudo,
        "pseudo_labels_applied": pl_applied,
        "pseudo_label_cv_improvement": sum(result.cv_improvements) if result.cv_improvements else 0.0,
    }

    log_event(
        session_id=state["session_id"],
        agent="pseudo_label_agent",
        action="pseudo_label_complete",
        keys_read=["model_registry", "selected_models", "feature_data_path", "feature_data_path_test"],
        keys_written=["pseudo_label_result", "X_train_with_pseudo",
                       "y_train_with_pseudo", "pseudo_labels_applied"],
        values_changed={
            "iterations": result.iterations_completed,
            "total_pl_added": sum(result.pseudo_labels_added),
            "cv_improvement": state["pseudo_label_cv_improvement"],
            "halt_reason": result.halt_reason,
        },
    )

    return state
