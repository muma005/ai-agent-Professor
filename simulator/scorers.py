"""
Metric Scorers — Implementations matching exact Kaggle evaluation.

CRITICAL: The scorer must produce IDENTICAL results to Kaggle's
evaluation. Even a rounding difference can shift percentile rank.
Uses the same implementations Kaggle uses (sklearn for standard
metrics, custom for Kaggle-specific ones like balanced log loss, QWK).

Available metrics:
- accuracy, auc, f1, macro_f1
- log_loss, balanced_log_loss
- rmse, rmsle, mae
- qwk (Quadratic Weighted Kappa)
- map_at_k (Mean Average Precision @ K)
"""

from typing import Callable, Dict
import numpy as np
from scipy.special import softmax


def get_scorer(metric: str, direction: str = "maximize") -> Callable:
    """
    Build a scorer function that matches the exact Kaggle evaluation.
    
    Args:
        metric: Metric name
        direction: "maximize" or "minimize"
    
    Returns:
        Scorer function: scorer(y_true, y_pred) -> float
    """
    metric = metric.lower().replace("-", "_")
    
    SCORERS: Dict[str, Callable] = {
        "accuracy": _accuracy,
        "auc": _auc,
        "f1": _f1_binary,
        "macro_f1": _f1_macro,
        "log_loss": _log_loss,
        "balanced_log_loss": _balanced_log_loss,
        "rmse": _rmse,
        "rmsle": _rmsle,
        "mae": _mae,
        "qwk": _quadratic_weighted_kappa,
        "map_at_k": _mean_average_precision_at_k,
    }
    
    if metric not in SCORERS:
        raise ValueError(
            f"Unknown metric '{metric}'. "
            f"Available: {list(SCORERS.keys())}"
        )
    
    return SCORERS[metric]


# =============================================================================
# Standard Metrics (sklearn-backed)
# =============================================================================

def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Accuracy score for classification."""
    from sklearn.metrics import accuracy_score
    # Convert probabilities to class predictions if needed
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=-1)
    elif y_pred.dtype.kind == 'f':
        # For binary classification, round to 0/1
        if len(np.unique(y_true)) == 2:
            y_pred = (y_pred >= 0.5).astype(int)
        else:
            y_pred = np.round(y_pred).astype(int)
    return float(accuracy_score(y_true, y_pred))


def _auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """ROC AUC score for binary classification."""
    from sklearn.metrics import roc_auc_score
    # Handle multiclass AUC (one-vs-rest)
    if y_pred.ndim > 1:
        return float(roc_auc_score(y_true, y_pred, average="macro", multi_class="ovr"))
    return float(roc_auc_score(y_true, y_pred))


def _f1_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """F1 score for binary classification."""
    from sklearn.metrics import f1_score
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=-1)
    elif y_pred.dtype.kind == 'f':
        # For binary classification, round to 0/1
        y_pred = (y_pred >= 0.5).astype(int)
    return float(f1_score(y_true, y_pred, average="binary"))


def _f1_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro F1 score for multiclass classification."""
    from sklearn.metrics import f1_score
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=-1)
    elif y_pred.dtype.kind == 'f':
        y_pred = np.round(y_pred).astype(int)
    return float(f1_score(y_true, y_pred, average="macro"))


def _log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Log loss (cross-entropy) for classification."""
    from sklearn.metrics import log_loss
    # Clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    
    # Handle binary vs multiclass
    if y_pred.ndim == 1:
        y_pred = np.vstack([1 - y_pred, y_pred]).T
    
    return float(log_loss(y_true, y_pred, labels=list(range(len(np.unique(y_true))))))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error for regression."""
    from sklearn.metrics import mean_squared_error
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Log Error for regression.
    
    Matches Kaggle's RMSLE exactly:
    - Clips negative predictions to 0 before log transform
    - Uses log1p (log(1 + x)) for numerical stability
    """
    from sklearn.metrics import mean_squared_error
    # Clip to non-negative before log transform
    y_true = np.clip(y_true.astype(float), 0, None)
    y_pred = np.clip(y_pred.astype(float), 0, None)
    # Log transform
    y_true = np.log1p(y_true)
    y_pred = np.log1p(y_pred)
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error for regression."""
    from sklearn.metrics import mean_absolute_error
    return float(mean_absolute_error(y_true, y_pred))


# =============================================================================
# Custom Metrics (Kaggle-specific)
# =============================================================================

def _balanced_log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Balanced Log Loss — used in imbalanced binary classification.
    
    This metric weights positive and negative classes equally,
    regardless of their frequency in the dataset.
    
    Formula:
        BLL = -0.5 * (mean(log(p_i)) for positive + mean(log(1-p_i)) for negative)
    
    Matches the implementation used in ICR competition.
    """
    # Clip predictions to avoid log(0)
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    
    # Separate positive and negative classes
    pos_mask = y_true == 1
    neg_mask = y_true == 0
    
    n_pos = np.sum(pos_mask)
    n_neg = np.sum(neg_mask)
    
    if n_pos == 0 or n_neg == 0:
        # Fall back to standard log loss if one class is missing
        return _log_loss(y_true, y_pred)
    
    # Compute log loss separately for each class
    pos_loss = -np.mean(np.log(y_pred[pos_mask]))
    neg_loss = -np.mean(np.log(1 - y_pred[neg_mask]))
    
    return float(0.5 * (pos_loss + neg_loss))


def _quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Quadratic Weighted Kappa (QWK) — used in ordinal classification.
    
    Measures agreement between two raters, weighted by the squared
    difference between ratings.
    
    Range: -1 (complete disagreement) to 1 (perfect agreement)
    0 = agreement equivalent to chance
    
    Used in competitions with ordinal targets (e.g., essay scoring).
    """
    from sklearn.metrics import cohen_kappa_score
    
    # Convert to integers if needed
    if y_pred.dtype.kind == 'f':
        if y_pred.ndim > 1:
            y_pred = np.argmax(y_pred, axis=-1)
        else:
            y_pred = np.round(y_pred).astype(int)
    
    if y_true.dtype.kind == 'f':
        y_true = np.round(y_true).astype(int)
    
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))


def _mean_average_precision_at_k(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: int = None,
) -> float:
    """
    Mean Average Precision @ K — used in ranking/retrieval tasks.
    
    Args:
        y_true: Binary relevance labels (1 = relevant, 0 = not relevant)
        y_pred: Prediction scores (higher = more relevant)
        k: Number of top predictions to consider. If None, uses all.
    
    Returns:
        MAP@K score
    """
    # Sort by predicted score (descending)
    sorted_indices = np.argsort(-y_pred)
    y_true_sorted = y_true[sorted_indices]
    
    if k is None:
        k = len(y_true_sorted)
    else:
        k = min(k, len(y_true_sorted))
    
    # Take top k
    y_true_k = y_true_sorted[:k]
    
    # Compute precision at each position
    n_relevant = np.sum(y_true_k)
    if n_relevant == 0:
        return 0.0
    
    # Cumulative sum of relevant items
    cumsum_relevant = np.cumsum(y_true_k)
    
    # Precision at each position
    positions = np.arange(1, k + 1)
    precision_at_k = cumsum_relevant / positions
    
    # Average precision
    ap = np.sum(precision_at_k * y_true_k) / n_relevant
    
    return float(ap)


# =============================================================================
# Scoring Helper
# =============================================================================

def score_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str,
    direction: str = "maximize",
) -> float:
    """
    Compute score between true and predicted values.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        metric: Metric name
        direction: "maximize" or "minimize"
    
    Returns:
        Score value
    """
    scorer = get_scorer(metric, direction)
    return scorer(y_true, y_pred)
