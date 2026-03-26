# tools/wilcoxon_gate.py

import logging
import numpy as np
from scipy.stats import wilcoxon

logger = logging.getLogger(__name__)

MIN_FOLDS_REQUIRED = 5   # Wilcoxon unreliable below 5 pairs
P_VALUE_THRESHOLD = 0.05


def is_significantly_better(
    fold_scores_a: list[float],
    fold_scores_b: list[float],
    p_threshold: float = P_VALUE_THRESHOLD,
    alternative: str = "greater",
    direction: str = "maximize",
) -> bool:
    """
    Returns True iff fold_scores_a is statistically significantly better
    than fold_scores_b at p < p_threshold using the Wilcoxon signed-rank test.

    Args:
        direction: "maximize" (higher=better, e.g. AUC) or
                   "minimize" (lower=better, e.g. RMSE, log_loss).
                   When "minimize", the alternative is flipped to "less".

    Never raises — returns False on any error (conservative default).
    """
    # For minimize metrics, "better" means LOWER scores
    if direction == "minimize":
        alternative = "less"

    if len(fold_scores_a) != len(fold_scores_b):
        logger.warning(
            f"[WilcoxonGate] fold count mismatch: "
            f"len(a)={len(fold_scores_a)}, len(b)={len(fold_scores_b)}. "
            f"Cannot compare — returning False (keep existing model)."
        )
        return False

    if len(fold_scores_a) < MIN_FOLDS_REQUIRED:
        logger.warning(
            f"[WilcoxonGate] Only {len(fold_scores_a)} folds — "
            f"minimum {MIN_FOLDS_REQUIRED} required for reliable Wilcoxon test. "
            f"Falling back to mean comparison."
        )
        mean_a = float(np.mean(fold_scores_a))
        mean_b = float(np.mean(fold_scores_b))
        if direction == "minimize":
            return bool(mean_a < mean_b)
        return bool(mean_a > mean_b)

    differences = np.array(fold_scores_a) - np.array(fold_scores_b)

    # If all differences are zero, models are identical
    if np.all(differences == 0):
        logger.info("[WilcoxonGate] All fold differences are zero — models identical.")
        return False

    try:
        stat, p_value = wilcoxon(differences, alternative=alternative, zero_method="wilcox")
    except Exception as e:
        logger.warning(
            f"[WilcoxonGate] scipy.stats.wilcoxon raised: {e}. "
            f"Falling back to mean comparison."
        )
        mean_a = float(np.mean(fold_scores_a))
        mean_b = float(np.mean(fold_scores_b))
        if direction == "minimize":
            return bool(mean_a < mean_b)
        return bool(mean_a > mean_b)

    result = bool(p_value < p_threshold)
    logger.info(
        f"[WilcoxonGate] stat={stat:.4f}, p={p_value:.4f}, "
        f"threshold={p_threshold}, significant={result}, direction={direction}. "
        f"mean_a={np.mean(fold_scores_a):.5f}, mean_b={np.mean(fold_scores_b):.5f}, "
        f"delta={np.mean(differences):+.5f}"
    )
    return result


def gate_result(
    fold_scores_a: list[float],
    fold_scores_b: list[float],
    model_name_a: str = "challenger",
    model_name_b: str = "champion",
    p_threshold: float = P_VALUE_THRESHOLD,
) -> dict:
    """
    Returns a structured gate result dict — used for lineage logging.
    """
    significant = is_significantly_better(fold_scores_a, fold_scores_b, p_threshold)
    differences = np.array(fold_scores_a) - np.array(fold_scores_b)

    return {
        "gate_passed":    significant,
        "selected_model": model_name_a if significant else model_name_b,
        "mean_a":         round(float(np.mean(fold_scores_a)), 6),
        "mean_b":         round(float(np.mean(fold_scores_b)), 6),
        "mean_delta":     round(float(np.mean(differences)), 6),
        "p_threshold":    p_threshold,
        "n_folds":        len(fold_scores_a),
        "model_name_a":   model_name_a,
        "model_name_b":   model_name_b,
        "reason": (
            f"{model_name_a} significantly better (p<{p_threshold})"
            if significant else
            f"Difference not significant — keeping {model_name_b}"
        ),
    }


# ── Day 17: Feature-level gate functions ─────────────────────────

def is_feature_worth_adding(
    baseline_fold_scores: list[float],
    augmented_fold_scores: list[float],
    feature_name: str = "candidate",
    p_threshold: float = P_VALUE_THRESHOLD,
) -> bool:
    """
    Returns True iff adding the candidate feature produces a statistically
    significant improvement in CV fold scores (p < p_threshold, Wilcoxon).

    Thin wrapper over is_significantly_better() with explicit naming
    for the feature selection context.

    Never raises. Conservative default (False) on any error.
    """
    result = is_significantly_better(
        fold_scores_a=augmented_fold_scores,
        fold_scores_b=baseline_fold_scores,
        p_threshold=p_threshold,
        alternative="greater",
    )

    try:
        logger.info(
            f"[WilcoxonGate] Feature '{feature_name}': "
            f"{'KEEP' if result else 'DROP'} "
            f"(baseline_mean={np.mean(baseline_fold_scores):.5f}, "
            f"augmented_mean={np.mean(augmented_fold_scores):.5f}, "
            f"delta={np.mean(augmented_fold_scores) - np.mean(baseline_fold_scores):+.5f})"
        )
    except Exception:
        pass

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
