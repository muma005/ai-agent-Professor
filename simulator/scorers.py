"""
Metric scorers — must match Kaggle's evaluation exactly.

Every scorer takes (true_labels_df, predictions_df, id_column, target_column)
and returns a float score.

CRITICAL: These must produce IDENTICAL results to Kaggle's evaluation.
Verify each scorer against known Kaggle submissions when possible.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, log_loss,
    mean_squared_error, mean_absolute_error
)


def score_accuracy(labels_df, preds_df, id_col, target_col):
    merged = labels_df.join(preds_df, on=id_col, suffix="_pred")
    y_true = merged[target_col].to_numpy()
    y_pred = merged[f"{target_col}_pred"].to_numpy()
    return float(accuracy_score(y_true, y_pred))


def score_auc(labels_df, preds_df, id_col, target_col):
    merged = labels_df.join(preds_df, on=id_col, suffix="_pred")
    y_true = merged[target_col].to_numpy()
    y_pred = merged[f"{target_col}_pred"].to_numpy().astype(float)
    return float(roc_auc_score(y_true, y_pred))


def score_f1_binary(labels_df, preds_df, id_col, target_col):
    merged = labels_df.join(preds_df, on=id_col, suffix="_pred")
    y_true = merged[target_col].to_numpy()
    y_pred = merged[f"{target_col}_pred"].to_numpy()
    return float(f1_score(y_true, y_pred, average="binary"))


def score_macro_f1(labels_df, preds_df, id_col, target_col):
    merged = labels_df.join(preds_df, on=id_col, suffix="_pred")
    y_true = merged[target_col].to_numpy()
    y_pred = merged[f"{target_col}_pred"].to_numpy()
    return float(f1_score(y_true, y_pred, average="macro"))


def score_log_loss(labels_df, preds_df, id_col, target_col):
    merged = labels_df.join(preds_df, on=id_col, suffix="_pred")
    y_true = merged[target_col].to_numpy()
    y_pred = merged[f"{target_col}_pred"].to_numpy().astype(float)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return float(log_loss(y_true, y_pred))


def score_rmse(labels_df, preds_df, id_col, target_col):
    merged = labels_df.join(preds_df, on=id_col, suffix="_pred")
    y_true = merged[target_col].to_numpy().astype(float)
    y_pred = merged[f"{target_col}_pred"].to_numpy().astype(float)
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def score_rmsle(labels_df, preds_df, id_col, target_col):
    merged = labels_df.join(preds_df, on=id_col, suffix="_pred")
    y_true = merged[target_col].to_numpy().astype(float)
    y_pred = merged[f"{target_col}_pred"].to_numpy().astype(float)
    y_pred = np.maximum(y_pred, 0)  # RMSLE requires non-negative
    return float(np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred))))


def score_mae(labels_df, preds_df, id_col, target_col):
    merged = labels_df.join(preds_df, on=id_col, suffix="_pred")
    y_true = merged[target_col].to_numpy().astype(float)
    y_pred = merged[f"{target_col}_pred"].to_numpy().astype(float)
    return float(mean_absolute_error(y_true, y_pred))


SCORERS = {
    "accuracy":     score_accuracy,
    "auc":          score_auc,
    "f1":           score_f1_binary,
    "macro_f1":     score_macro_f1,
    "log_loss":     score_log_loss,
    "rmse":         score_rmse,
    "rmsle":        score_rmsle,
    "mae":          score_mae,
}


def get_scorer(metric: str):
    """Get scorer function by metric name. Raises ValueError if unknown."""
    if metric not in SCORERS:
        raise ValueError(
            f"Unknown metric '{metric}'. Available: {list(SCORERS.keys())}. "
            f"Add new scorers to simulator/scorers.py"
        )
    return SCORERS[metric]
