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
) -> bool:
    """
    Returns True iff fold_scores_a is statistically significantly better
    than fold_scores_b at p < p_threshold using the Wilcoxon signed-rank test.

    Never raises — returns False on any error (conservative default).
    """
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
        return bool(float(np.mean(fold_scores_a)) > float(np.mean(fold_scores_b)))

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
        return bool(float(np.mean(fold_scores_a)) > float(np.mean(fold_scores_b)))

    result = bool(p_value < p_threshold)
    logger.info(
        f"[WilcoxonGate] stat={stat:.4f}, p={p_value:.4f}, "
        f"threshold={p_threshold}, significant={result}. "
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
