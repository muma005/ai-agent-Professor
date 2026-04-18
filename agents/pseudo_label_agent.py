# agents/pseudo_label_agent.py
# -------------------------------------------------------------------------
# Day 18/25: GM-CAP 6 — Pseudo-labeling with confidence gating.
# Three activation gates: probability metric, data size, calibration.
# Critic verification of confidence distribution.
# Validation fold never sees pseudo-labels (critical invariant).
# Max 3 iterations, Wilcoxon gate on CV improvement.
# Max pseudo-label fraction: 30% of training data.
# -------------------------------------------------------------------------

import gc
import json
import logging
import os
from dataclasses import dataclass, field

import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, KFold

from core.state import ProfessorState
from core.lineage import log_event
from tools.wilcoxon_gate import is_significantly_better

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────
PROBABILITY_METRICS     = frozenset({"log_loss", "logloss", "cross_entropy", "brier_score", "auc"})
MIN_TEST_TO_TRAIN_RATIO = 2.0    # test set must have > 2x rows of train set
MIN_CALIBRATION_SCORE   = 0.80   # model calibration above this threshold required
HIGH_CONFIDENCE_THRESHOLD = 0.95  # probability threshold for pseudo-label selection
MAX_PSEUDO_LABEL_FRACTION = 0.30  # pseudo-labels never exceed 30% of training data
MAX_ITERATIONS          = 3
CONFIDENCE_TOP_FRACTION = 0.10   # top 10% most confident (fallback)
MIN_CV_IMPROVEMENT = 0.001       # must improve by at least 0.1pp


@dataclass
class PseudoLabelResult:
    iterations_completed: int
    pseudo_labels_added: list[int] = field(default_factory=list)
    cv_scores_with_pl: list[float] = field(default_factory=list)
    cv_improvements: list[float] = field(default_factory=list)
    halted_early: bool = False
    halt_reason: str = ""
    final_pseudo_label_mask: list[int] = field(default_factory=list)
    confidence_thresholds: list[float] = field(default_factory=list)


# =========================================================================
# Activation Gates — ALL must pass before pseudo-labeling runs
# =========================================================================

def _count_test_rows(state: dict) -> int:
    """Count test rows from state or test data file."""
    n_test = state.get("n_test_rows", 0)
    if n_test > 0:
        return n_test
    test_path = state.get("test_path") or state.get("clean_test_path")
    if test_path:
        try:
            return len(pl.scan_csv(test_path).select(pl.count()).collect())
        except Exception:
            pass
    return 0


def _get_best_calibration_score(state: dict) -> float:
    """
    Get calibration score from model registry.
    Returns 1.0 - Brier score as calibration quality metric.
    """
    registry = state.get("model_registry", {})
    if isinstance(registry, dict):
        for name, entry in registry.items():
            if entry.get("is_calibrated"):
                brier = entry.get("calibration_score")
                if brier is not None:
                    return max(0.0, 1.0 - brier)
    elif isinstance(registry, list):
        for entry in registry:
            if entry.get("is_calibrated"):
                brier = entry.get("calibration_score")
                if brier is not None:
                    return max(0.0, 1.0 - brier)
    return None


def _check_activation_gates(state: dict) -> tuple[bool, str]:
    """
    Returns (should_run, reason).
    ALL three gates must pass. Returns the first failing gate's reason.
    """
    metric = state.get("evaluation_metric", "")
    if metric not in PROBABILITY_METRICS:
        return False, f"metric '{metric}' is not probability-based"

    n_train = len(state.get("y_train", []))
    n_test = _count_test_rows(state)
    if n_test <= n_train * MIN_TEST_TO_TRAIN_RATIO:
        return False, (
            f"test set ({n_test} rows) is not > {MIN_TEST_TO_TRAIN_RATIO}x "
            f"training set ({n_train} rows)"
        )

    calibration = _get_best_calibration_score(state)
    if calibration is None or calibration < MIN_CALIBRATION_SCORE:
        return False, (
            f"model calibration ({calibration}) below threshold {MIN_CALIBRATION_SCORE}"
        )

    return True, "all gates passed"


# =========================================================================
# Critic Verification — confidence distribution check
# =========================================================================

def _critic_verifies_confidence_distribution(
    confidences: np.ndarray,
    state: dict,
) -> tuple[bool, str]:
    """
    Critic check: pseudo-label confidence distribution must be realistic.

    Rejects if:
    1. > 50% of predictions are above HIGH_CONFIDENCE_THRESHOLD
    2. Mean confidence < 0.55
    3. Std of confidences < 0.05
    """
    high_conf_fraction = float(np.mean(confidences >= HIGH_CONFIDENCE_THRESHOLD))
    mean_conf = float(np.mean(confidences))
    std_conf = float(np.std(confidences))

    if high_conf_fraction > 0.50:
        return False, (
            f"distribution collapse: {high_conf_fraction:.1%} of predictions "
            f"above {HIGH_CONFIDENCE_THRESHOLD}. Model is overconfident."
        )
    if mean_conf < 0.55:
        return False, (
            f"mean confidence {mean_conf:.3f} too low. "
            "Model has insufficient discriminative power for pseudo-labeling."
        )
    if std_conf < 0.05:
        return False, (
            f"confidence std {std_conf:.4f} < 0.05. "
            "All predictions nearly identical — constant predictor."
        )

    return True, f"distribution OK (mean={mean_conf:.3f}, std={std_conf:.4f}, high_conf={high_conf_fraction:.1%})"


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
    Validation fold sees ONLY real labels (critical invariant).
    """
    from sklearn.metrics import roc_auc_score, mean_squared_error

    is_classification = metric in ("auc", "logloss", "binary", "multiclass", "log_loss", "cross_entropy", "brier_score")
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state) \
         if is_classification else \
         KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    X_np = X_train.to_numpy()
    fold_scores = []
    ModelClass = lgb.LGBMClassifier if is_classification else lgb.LGBMRegressor
    X_pseudo_np = X_pseudo.to_numpy()

    for train_idx, val_idx in cv.split(X_np, y_train if is_classification else None):
        X_fold_train = np.vstack([X_np[train_idx], X_pseudo_np])
        y_fold_train = np.concatenate([y_train[train_idx], y_pseudo])
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
    """Delete large arrays and run GC."""
    for obj in kwargs.values():
        if obj is not None:
            del obj
    gc.collect()


# ── Main entry point ─────────────────────────────────────────────

def run_pseudo_label_agent(state: ProfessorState) -> ProfessorState:
    """
    GM-CAP 6: Pseudo-labeling with confidence gating.

    Three activation gates (ALL must pass):
    1. Probability-based metric (logloss, auc, etc.)
    2. Test set > 2x training set size
    3. Model calibration >= MIN_CALIBRATION_SCORE

    If gates pass:
    1. Train on labeled data
    2. Predict test set
    3. Select predictions >= HIGH_CONFIDENCE_THRESHOLD
    4. Add to training folds only (never validation)
    5. CV gate + Wilcoxon
    6. Repeat up to MAX_ITERATIONS
    7. Max pseudo-label cap: 30%
    """
    session_id = state.get("session_id", "unknown")

    # ── Check activation gates FIRST (before loading any data) ───
    should_run, reason = _check_activation_gates(state)
    if not should_run:
        logger.info(f"[pseudo_label] Skipped: {reason}")
        return {
            **state,
            "pseudo_labels_applied": False,
            "pseudo_label_skip_reason": reason,
            "pseudo_label_iterations": 0,
            "pseudo_label_n_added": 0,
            "pseudo_label_cv_improvement": 0.0,
            "pseudo_label_confidence_mean": 0.0,
            "pseudo_label_confidence_std": 0.0,
            "pseudo_label_critic_accepted": False,
            "clean_train_with_pseudo_path": "",
        }

    # ── Load paths from state ────────────────────────────────────
    train_path = state.get("clean_train_path") or state.get("train_path")
    test_path = state.get("clean_test_path") or state.get("test_path")
    if not train_path or not test_path:
        logger.warning("[pseudo_label] No train/test path in state. Skipping.")
        return {
            **state,
            "pseudo_labels_applied": False,
            "pseudo_label_skip_reason": "no_train_test_paths",
            "pseudo_label_iterations": 0,
            "pseudo_label_n_added": 0,
            "pseudo_label_cv_improvement": 0.0,
            "pseudo_label_confidence_mean": 0.0,
            "pseudo_label_confidence_std": 0.0,
            "pseudo_label_critic_accepted": False,
            "clean_train_with_pseudo_path": "",
        }

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        logger.warning(f"[pseudo_label] Train or test path not found. Skipping.")
        return {
            **state,
            "pseudo_labels_applied": False,
            "pseudo_label_skip_reason": "data_files_not_found",
            "pseudo_label_iterations": 0,
            "pseudo_label_n_added": 0,
            "pseudo_label_cv_improvement": 0.0,
            "pseudo_label_confidence_mean": 0.0,
            "pseudo_label_confidence_std": 0.0,
            "pseudo_label_critic_accepted": False,
            "clean_train_with_pseudo_path": "",
        }

    target_col = state.get("target_column") or state.get("target_col")
    if not target_col:
        logger.warning("[pseudo_label] target_column not set in state. Skipping.")
        return {
            **state,
            "pseudo_labels_applied": False,
            "pseudo_label_skip_reason": "no_target_column",
            "pseudo_label_iterations": 0,
            "pseudo_label_n_added": 0,
            "pseudo_label_cv_improvement": 0.0,
            "pseudo_label_confidence_mean": 0.0,
            "pseudo_label_confidence_std": 0.0,
            "pseudo_label_critic_accepted": False,
            "clean_train_with_pseudo_path": "",
        }

    # ── Load data from disk ──────────────────────────────────────
    X_train_df = pl.read_csv(train_path)
    y_train = X_train_df[target_col].to_numpy()
    X_train = X_train_df.drop(target_col)
    X_test = pl.read_csv(test_path)

    # ── Enforce feature order ────────────────────────────────────
    feature_order = state.get("feature_order", [])
    if feature_order:
        available = [c for c in feature_order if c in X_train.columns]
        X_train = X_train.select(available)
        X_test = X_test.select([c for c in available if c in X_test.columns])

    metric = state.get("evaluation_metric", "logloss")

    # ── Get model params from registry ───────────────────────────
    registry = state.get("model_registry", {})
    best_entry = None
    if isinstance(registry, dict):
        best_entry = next(iter(registry.values()), None)
    elif isinstance(registry, list):
        best_entry = registry[0] if registry else None

    if best_entry is None:
        return {
            **state,
            "pseudo_labels_applied": False,
            "pseudo_label_skip_reason": "no_model_in_registry",
            "pseudo_label_iterations": 0,
            "pseudo_label_n_added": 0,
            "pseudo_label_cv_improvement": 0.0,
            "pseudo_label_confidence_mean": 0.0,
            "pseudo_label_confidence_std": 0.0,
            "pseudo_label_critic_accepted": False,
            "clean_train_with_pseudo_path": "",
        }

    lgbm_params = best_entry.get("params", {"n_estimators": 500, "learning_rate": 0.05, "verbosity": -1})
    baseline_cv = best_entry.get("fold_scores", [])
    if not baseline_cv:
        return {
            **state,
            "pseudo_labels_applied": False,
            "pseudo_label_skip_reason": "no_baseline_cv_scores",
            "pseudo_label_iterations": 0,
            "pseudo_label_n_added": 0,
            "pseudo_label_cv_improvement": 0.0,
            "pseudo_label_confidence_mean": 0.0,
            "pseudo_label_confidence_std": 0.0,
            "pseudo_label_critic_accepted": False,
            "clean_train_with_pseudo_path": "",
        }

    # ── Initialize working variables ─────────────────────────────
    X_pseudo_accumulated = X_train.slice(0, 0)
    y_pseudo_accumulated = np.array([], dtype=y_train.dtype)
    current_test_mask = np.zeros(len(X_test), dtype=bool)
    prev_fold_scores = baseline_cv.copy()
    all_confidences = []
    iterations_completed = 0
    total_pseudo_added = 0
    cv_improvements = []
    halt_reason = ""

    # ── Lightning Offload Hook ─────────────────────────────────────────
    from tools.lightning_runner import is_lightning_configured, run_on_lightning, sync_files_to_lightning
    USE_LIGHTNING_PL = (
        is_lightning_configured() and
        os.getenv("LIGHTNING_OFFLOAD_PSEUDO_LABEL", "0") == "1"
    )
    
    if USE_LIGHTNING_PL:
        logger.info("[pseudo_label] ⚡ Offloading pseudo labeling to Lightning AI...")
        output_dir = f"outputs/{session_id}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Write model registry for the cloud script
        registry_tmp = os.path.join(output_dir, "model_registry.json")
        with open(registry_tmp, "w") as f:
            json.dump(state.get("model_registry", []), f)
        
        files_to_sync = {
            train_path: "train.csv",
            test_path: "test.csv",
            registry_tmp: "model_registry.json",
        }
        
        synced = sync_files_to_lightning(session_id=session_id, files=files_to_sync)
        if synced:
            machine = os.getenv("LIGHTNING_PSEUDO_LABEL_MACHINE", "CPU")
            task_type = "binary" if metric in ("auc", "logloss", "binary", "log_loss") else "regression"
            res = run_on_lightning(
                script="tools/lightning_jobs/run_pseudo_label.py",
                args={"session_id": session_id, "target_col": target_col, "task_type": task_type},
                job_name=f"pseudo_label_{session_id}",
                machine=machine,
                interruptible=True,
                result_path=f"{output_dir}/pseudo_label_result.json",
            )
            if res["success"] and res["result"].get("success"):
                lightning_data = res["result"]
                state["lightning_pseudo_label_link"] = res["job_link"]
                state["lightning_pseudo_label_runtime"] = res["runtime_s"]
                
                return {
                    **state,
                    "pseudo_labels_applied": lightning_data.get("pseudo_labels_applied", False),
                    "pseudo_label_skip_reason": "",
                    "pseudo_label_iterations": lightning_data.get("iterations_completed", 0),
                    "pseudo_label_n_added": lightning_data.get("n_labels_added", 0),
                    "pseudo_label_cv_improvement": sum(lightning_data.get("cv_improvements", [])),
                    "pseudo_label_confidence_mean": 0.0,
                    "pseudo_label_confidence_std": 0.0,
                    "pseudo_label_critic_accepted": lightning_data.get("pseudo_labels_applied", False),
                    "clean_train_with_pseudo_path": "",
                    "pseudo_label_halt_reason": lightning_data.get("halt_reason", ""),
                }
            else:
                logger.warning(f"[pseudo_label] Lightning failed: {res.get('error')}. Running locally.")
        else:
            logger.warning("[pseudo_label] File sync to Lightning failed. Running locally.")

    # ── Main iteration loop ──────────────────────────────────────
    for iteration in range(1, MAX_ITERATIONS + 1):
        logger.info(f"[pseudo_label] Iteration {iteration}/{MAX_ITERATIONS}")

        try:
            max_pseudo = int(len(y_train) * MAX_PSEUDO_LABEL_FRACTION)
            if len(y_pseudo_accumulated) >= max_pseudo:
                logger.info(
                    f"[pseudo_label] Reached max pseudo-label fraction "
                    f"({MAX_PSEUDO_LABEL_FRACTION:.0%}). Stopping."
                )
                halt_reason = "max_fraction_reached"
                break

            is_cls = metric in ("auc", "logloss", "binary", "log_loss", "cross_entropy", "brier_score")
            ModelClass = lgb.LGBMClassifier if is_cls else lgb.LGBMRegressor

            if len(y_pseudo_accumulated) > 0:
                X_all = pl.concat([X_train, X_pseudo_accumulated])
                y_all = np.concatenate([y_train, y_pseudo_accumulated])
            else:
                X_all = X_train
                y_all = y_train

            model = ModelClass(**lgbm_params)
            model.fit(X_all.to_numpy(), y_all)

            remaining_mask = ~current_test_mask
            X_remaining = X_test.filter(pl.Series(remaining_mask))
            if X_remaining.is_empty():
                halt_reason = "no_confident_samples"
                break

            y_pred = model.predict_proba(X_remaining.to_numpy())[:, 1] if is_cls \
                     else model.predict(X_remaining.to_numpy())
            del model
            gc.collect()

            # Select samples above confidence threshold
            confidence = np.abs(y_pred - 0.5) if is_cls else np.ones(len(y_pred))
            high_conf_mask = confidence >= (HIGH_CONFIDENCE_THRESHOLD - 0.5) if is_cls else np.ones(len(y_pred), dtype=bool)

            n_available = int(high_conf_mask.sum())
            n_can_add = max_pseudo - len(y_pseudo_accumulated)
            if n_can_add <= 0:
                halt_reason = "max_fraction_reached"
                break

            high_conf_indices = np.where(high_conf_mask)[0]
            if len(high_conf_indices) > n_can_add:
                high_conf_indices = high_conf_indices[:n_can_add]

            if len(high_conf_indices) == 0:
                halt_reason = "no_confident_samples"
                break

            n_selected = len(high_conf_indices)
            conf_mask = np.zeros(len(high_conf_mask), dtype=bool)
            conf_mask[high_conf_indices] = True

            X_new_pseudo = X_remaining.filter(pl.Series(conf_mask))
            y_new_pseudo = y_pred[conf_mask]

            if is_cls:
                y_new_pseudo = (y_new_pseudo >= 0.5).astype(y_train.dtype)
            else:
                if y_new_pseudo.dtype != y_train.dtype:
                    y_new_pseudo = y_new_pseudo.astype(y_train.dtype)

            # Critic verification
            selected_confidences = confidence[conf_mask]
            critic_accepted, critic_reason = _critic_verifies_confidence_distribution(selected_confidences, state)
            if not critic_accepted:
                logger.info(f"[pseudo_label] Iteration {iteration}: Critic rejected: {critic_reason}")
                halt_reason = f"critic_rejected: {critic_reason}"
                break

            all_confidences.extend(selected_confidences.tolist())

            # CV with pseudo-labels
            if len(y_pseudo_accumulated) > 0:
                X_pseudo_for_cv = pl.concat([X_pseudo_accumulated, X_new_pseudo])
                y_pseudo_for_cv = np.concatenate([y_pseudo_accumulated, y_new_pseudo])
            else:
                X_pseudo_for_cv = X_new_pseudo
                y_pseudo_for_cv = y_new_pseudo

            cv_with = _run_cv_with_pseudo_labels(
                X_train=X_train, y_train=y_train,
                X_pseudo=X_pseudo_for_cv, y_pseudo=y_pseudo_for_cv,
                lgbm_params=lgbm_params, metric=metric,
            )

            cv_mean_with = float(np.mean(cv_with))
            cv_mean_without = float(np.mean(prev_fold_scores))
            improvement = cv_mean_with - cv_mean_without

            logger.info(
                f"[pseudo_label] Iteration {iteration}: n_added={n_selected}, "
                f"cv_before={cv_mean_without:.5f}, cv_after={cv_mean_with:.5f}, "
                f"improvement={improvement:+.5f}"
            )

            if not is_significantly_better(cv_with, prev_fold_scores):
                logger.info(
                    f"[pseudo_label] Iteration {iteration}: Wilcoxon gate failed. Stopping."
                )
                halt_reason = "wilcoxon_gate_failed"
                break

            cv_improvements.append(improvement)
            prev_fold_scores = cv_with.copy()

            if len(y_pseudo_accumulated) > 0:
                X_pseudo_accumulated = pl.concat([X_pseudo_accumulated, X_new_pseudo])
                y_pseudo_accumulated = np.concatenate([y_pseudo_accumulated, y_new_pseudo])
            else:
                X_pseudo_accumulated = X_new_pseudo
                y_pseudo_accumulated = y_new_pseudo

            current_test_mask[np.where(remaining_mask)[0][conf_mask]] = True
            iterations_completed = iteration
            total_pseudo_added += n_selected
            gc.collect()

        except Exception as e:
            logger.error(f"[pseudo_label] Iteration {iteration} failed: {e}")
            halt_reason = f"iteration_failed: {e}"
            break

    if iterations_completed == MAX_ITERATIONS and not halt_reason:
        halt_reason = "max_iterations_reached"

    pl_applied = iterations_completed > 0

    # Write augmented training CSV
    clean_train_with_pseudo_path = ""
    if pl_applied and len(y_pseudo_accumulated) > 0:
        output_dir = state.get("output_dir", f"outputs/{session_id}")
        os.makedirs(output_dir, exist_ok=True)
        clean_train_with_pseudo_path = os.path.join(output_dir, "train_with_pseudo.csv")
        X_with_pseudo = pl.concat([X_train, X_pseudo_accumulated])
        y_with_pseudo = np.concatenate([y_train, y_pseudo_accumulated])
        df_with_pseudo = X_with_pseudo.with_columns(
            pl.Series(target_col, y_with_pseudo)
        )
        df_with_pseudo.write_csv(clean_train_with_pseudo_path)

    state = {
        **state,
        "pseudo_labels_applied": pl_applied,
        "pseudo_label_skip_reason": "" if pl_applied else reason,
        "pseudo_label_halt_reason": halt_reason,
        "pseudo_label_iterations": iterations_completed,
        "pseudo_label_n_added": total_pseudo_added,
        "pseudo_label_cv_improvement": float(np.sum(cv_improvements)) if cv_improvements else 0.0,
        "pseudo_label_confidence_mean": float(np.mean(all_confidences)) if all_confidences else 0.0,
        "pseudo_label_confidence_std": float(np.std(all_confidences)) if all_confidences else 0.0,
        "pseudo_label_critic_accepted": pl_applied and len(all_confidences) > 0,
        "clean_train_with_pseudo_path": clean_train_with_pseudo_path,
    }

    log_event(
        session_id=session_id,
        agent="pseudo_label_agent",
        action="pseudo_label_complete",
        keys_read=["model_registry", "train_path", "test_path"],
        keys_written=["pseudo_labels_applied", "pseudo_label_iterations",
                       "pseudo_label_n_added", "pseudo_label_cv_improvement"],
        values_changed={
            "iterations": iterations_completed,
            "total_pl_added": total_pseudo_added,
            "cv_improvement": state["pseudo_label_cv_improvement"],
            "halt_reason": halt_reason,
        },
    )

    return state
