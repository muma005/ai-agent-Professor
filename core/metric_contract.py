# core/metric_contract.py

from dataclasses import dataclass, field
from typing import Callable, Optional
import json
import os
from sklearn import metrics as skmetrics


# ── All supported scorers ─────────────────────────────────────────
SCORER_REGISTRY = {
    # Classification
    "accuracy":          (skmetrics.accuracy_score,          "maximize"),
    "auc":               (skmetrics.roc_auc_score,           "maximize"),
    "roc_auc":           (skmetrics.roc_auc_score,           "maximize"),
    "log_loss":          (skmetrics.log_loss,                "minimize"),
    "f1":                (skmetrics.f1_score,                "maximize"),
    "f1_macro":          (lambda y, p: skmetrics.f1_score(y, p, average="macro"), "maximize"),
    "f1_weighted":       (lambda y, p: skmetrics.f1_score(y, p, average="weighted"), "maximize"),
    "matthews_corrcoef": (skmetrics.matthews_corrcoef,       "maximize"),
    # Regression
    "rmse":              (lambda y, p: skmetrics.root_mean_squared_error(y, p), "minimize"),
    "mae":               (skmetrics.mean_absolute_error,     "minimize"),
    "r2":                (skmetrics.r2_score,                "maximize"),
    "rmsle":             (lambda y, p: skmetrics.mean_squared_log_error(y, p) ** 0.5, "minimize"),
    "mape":              (skmetrics.mean_absolute_percentage_error, "minimize"),
}

# Metrics that require predict_proba instead of predict
PROBABILITY_METRICS = {"auc", "roc_auc", "log_loss"}

# Metrics that are FORBIDDEN — never optimise toward these
# (proxies that look good but don't reflect true performance)
FORBIDDEN_METRICS = {"accuracy_on_train", "train_loss", "overfit_score"}


@dataclass
class MetricContract:
    """
    The single source of truth for what Professor is optimising toward.
    Written once per competition. Injected into every agent system prompt.
    Never changed mid-pipeline without explicit user approval.
    """
    scorer_name:       str              # e.g. "auc", "rmse"
    direction:         str              # "maximize" or "minimize"
    scorer_fn:         Callable         # the actual sklearn function
    requires_proba:    bool             # True if predict_proba needed
    forbidden_metrics: list             # metrics never to optimise toward
    task_type:         str              # "classification" or "regression"
    competition_name:  str = ""
    locked:            bool = False     # True after first submission
    notes:             str = ""


def build_metric_contract(
    scorer_name: str,
    task_type: str,
    competition_name: str = "",
    notes: str = ""
) -> MetricContract:
    """
    Build a MetricContract from a scorer name string.
    Used by the Validation Architect (Phase 2). For Phase 1: hardcode AUC.
    """
    scorer_name = scorer_name.lower().strip()

    if scorer_name not in SCORER_REGISTRY:
        raise ValueError(
            f"Unknown scorer: '{scorer_name}'. "
            f"Supported: {list(SCORER_REGISTRY.keys())}"
        )

    scorer_fn, direction = SCORER_REGISTRY[scorer_name]

    return MetricContract(
        scorer_name=scorer_name,
        direction=direction,
        scorer_fn=scorer_fn,
        requires_proba=scorer_name in PROBABILITY_METRICS,
        forbidden_metrics=list(FORBIDDEN_METRICS),
        task_type=task_type,
        competition_name=competition_name,
        locked=False,
        notes=notes
    )


def default_contract(competition_name: str = "") -> MetricContract:
    """
    Phase 1 default: AUC for binary classification.
    Replaced by Validation Architect auto-detection in Phase 2.
    """
    return build_metric_contract(
        scorer_name="auc",
        task_type="classification",
        competition_name=competition_name,
        notes="Phase 1 default — hardcoded AUC. Auto-detected from Day 8."
    )


def save_contract(contract: MetricContract, path: str) -> str:
    """Save MetricContract as metric_contract.json."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = {
        "scorer_name":       contract.scorer_name,
        "direction":         contract.direction,
        "requires_proba":    contract.requires_proba,
        "forbidden_metrics": contract.forbidden_metrics,
        "task_type":         contract.task_type,
        "competition_name":  contract.competition_name,
        "locked":            contract.locked,
        "notes":             contract.notes
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def load_contract(path: str) -> MetricContract:
    """Load MetricContract from metric_contract.json."""
    with open(path) as f:
        data = json.load(f)
    scorer_fn, _ = SCORER_REGISTRY[data["scorer_name"]]
    return MetricContract(
        scorer_name=data["scorer_name"],
        direction=data["direction"],
        scorer_fn=scorer_fn,
        requires_proba=data["requires_proba"],
        forbidden_metrics=data["forbidden_metrics"],
        task_type=data["task_type"],
        competition_name=data["competition_name"],
        locked=data.get("locked", False),
        notes=data.get("notes", "")
    )


def contract_to_prompt_snippet(contract: MetricContract) -> str:
    """
    Returns a string injected into every agent system prompt.
    Makes every agent aware of what it is optimising toward.
    """
    better = "higher" if contract.direction == "maximize" else "lower"
    return f"""
METRIC CONTRACT (read-only — never change this mid-pipeline):
  Competition:     {contract.competition_name}
  Optimise for:    {contract.scorer_name.upper()} ({better} is better)
  Task type:       {contract.task_type}
  Requires proba:  {contract.requires_proba}
  FORBIDDEN:       Never report or optimise toward: {contract.forbidden_metrics}
  Locked:          {contract.locked}
"""
