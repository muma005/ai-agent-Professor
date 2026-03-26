"""
Metric-aware scorer: computes the official competition metric
between Professor's predictions and the private held-out set.
"""

import numpy as np
import polars as pl
from sklearn.metrics import (
    accuracy_score, roc_auc_score, mean_squared_error,
    log_loss, mean_absolute_error,
)


def score_predictions(private_test: pl.DataFrame, predictions: pl.DataFrame, spec) -> dict:
    """
    Aligns predictions with private_test on id_column, computes official metric.
    predictions must have columns: [id_column, "prediction"].
    """
    merged = private_test.join(
        predictions.rename({"prediction": "prof_pred"}),
        on=spec.id_column,
        how="left",
    )

    n_missing = int(merged["prof_pred"].is_null().sum())
    if n_missing > 0:
        print(f"[scorer] WARNING: {n_missing} IDs missing. Filling with baseline.")
        if spec.task_type == "regression":
            fill = float(merged[spec.target_column].mean())
        else:
            vals, counts = np.unique(
                merged[spec.target_column].drop_nulls().to_numpy(), return_counts=True
            )
            fill = float(vals[counts.argmax()])
        merged = merged.with_columns(pl.col("prof_pred").fill_null(fill))

    y_true = merged[spec.target_column].to_numpy()
    y_pred = merged["prof_pred"].to_numpy()
    metric = spec.evaluation_metric.lower().replace("-", "_")

    return {
        "private_score":    round(float(_compute(y_true, y_pred, metric)), 6),
        "metric":           metric,
        "n_scored":         len(merged),
        "n_missing_ids":    n_missing,
        "higher_is_better": metric in {"accuracy", "auc", "f1"},
    }


def _compute(y_true, y_pred, metric: str) -> float:
    if metric == "accuracy":
        return accuracy_score(y_true, (y_pred >= 0.5).astype(int))
    elif metric == "auc":
        return roc_auc_score(y_true, y_pred)
    elif metric in ("rmsle", "root_mean_squared_log_error"):
        yt = np.log1p(np.clip(y_true.astype(float), 0, None))
        yp = np.log1p(np.clip(y_pred.astype(float), 0, None))
        return float(np.sqrt(mean_squared_error(yt, yp)))
    elif metric == "rmse":
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))
    elif metric == "mae":
        return float(mean_absolute_error(y_true, y_pred))
    elif metric in ("logloss", "log_loss"):
        return float(log_loss(y_true, np.clip(y_pred, 1e-7, 1 - 1e-7)))
    else:
        raise ValueError(f"Unknown metric: '{metric}'")
