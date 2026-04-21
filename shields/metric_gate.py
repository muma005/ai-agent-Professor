# shields/metric_gate.py

import json
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from sklearn import metrics as skmetrics

logger = logging.getLogger(__name__)

# ── Metric Verification Cases ───────────────────────────────────────────────

METRIC_VERIFICATION_CASES = {
    "binary_logloss": [
        {"y_true": [1, 0, 1, 0], "y_pred": [0.9, 0.1, 0.9, 0.1], "expected_range": (0.05, 0.15)},
        {"y_true": [1, 0], "y_pred": [0.5, 0.5], "expected_range": (0.6, 0.8)}
    ],
    "auc": [
        {"y_true": [1, 0, 1, 0], "y_pred": [0.8, 0.2, 0.8, 0.2], "expected_value": 1.0},
        {"y_true": [1, 0, 1, 0], "y_pred": [0.2, 0.8, 0.2, 0.8], "expected_value": 0.0}
    ],
    "f1": [
        {"y_true": [1, 0, 1, 0], "y_pred": [1, 0, 1, 0], "expected_value": 1.0},
        {"y_true": [1, 0, 1, 0], "y_pred": [0, 0, 0, 0], "expected_value": 0.0}
    ],
    "rmse": [
        {"y_true": [10, 20], "y_pred": [10, 20], "expected_value": 0.0},
        {"y_true": [10, 20], "y_pred": [11, 21], "expected_value": 1.0}
    ],
    "mae": [
        {"y_true": [10, 20], "y_pred": [11, 21], "expected_value": 1.0}
    ],
    "r2": [
        {"y_true": [10, 20, 30], "y_pred": [10, 20, 30], "expected_value": 1.0}
    ],
    # Edge cases
    "imbalanced_auc": [
        {"y_true": [1] + [0]*99, "y_pred": [0.9] + [0.1]*99, "expected_value": 1.0}
    ],
    "constant_pred_logloss": [
        {"y_true": [1, 0], "y_pred": [0.5, 0.5], "expected_range": (0.6, 0.8)}
    ]
}

# ── Core Verification Logic ──────────────────────────────────────────────────

def verify_metric(metric_name: str, task_type: str) -> Tuple[bool, str]:
    """Tests a metric against ground-truth cases."""
    cases = METRIC_VERIFICATION_CASES.get(metric_name)
    if not cases:
        # Check for generic patterns
        if "auc" in metric_name: cases = METRIC_VERIFICATION_CASES["auc"]
        elif "logloss" in metric_name: cases = METRIC_VERIFICATION_CASES["binary_logloss"]
        elif "f1" in metric_name: cases = METRIC_VERIFICATION_CASES["f1"]
        elif "rmse" in metric_name: cases = METRIC_VERIFICATION_CASES["rmse"]
        elif "mae" in metric_name: cases = METRIC_VERIFICATION_CASES["mae"]
        else:
            return False, f"Unknown metric: {metric_name}. No verification cases found."

    for i, case in enumerate(cases):
        y_true = np.array(case["y_true"])
        y_pred = np.array(case["y_pred"])
        
        try:
            if metric_name in ["binary_logloss", "logloss"]:
                score = skmetrics.log_loss(y_true, y_pred)
            elif "auc" in metric_name:
                score = skmetrics.roc_auc_score(y_true, y_pred)
            elif "f1" in metric_name:
                score = skmetrics.f1_score(y_true, y_pred)
            elif "rmse" in metric_name:
                score = np.sqrt(skmetrics.mean_squared_error(y_true, y_pred))
            elif "mae" in metric_name:
                score = skmetrics.mean_absolute_error(y_true, y_pred)
            elif "r2" in metric_name:
                score = skmetrics.r2_score(y_true, y_pred)
            else:
                return False, f"Implementation missing for {metric_name}"

            # Validate
            if "expected_value" in case:
                if not np.isclose(score, case["expected_value"]):
                    return False, f"Case {i} failed: expected {case['expected_value']}, got {score}"
            elif "expected_range" in case:
                low, high = case["expected_range"]
                if not (low <= score <= high):
                    return False, f"Case {i} failed: {score} out of range ({low}, {high})"
                    
        except Exception as e:
            return False, f"Case {i} crashed: {e}"

    return True, f"Verified {len(cases)} cases."

def run_metric_verification_gate(state: Any) -> Any:
    """LangGraph node: Shield 1."""
    # Import state here to avoid circular dependencies
    from core.state import ProfessorState
    
    metric_name = state.get("metric_contract", {}).get("metric_name", "unknown")
    task_type = state.get("task_type", "unknown")
    
    logger.info(f"[Shield 1] Verifying metric: {metric_name} for {task_type}")
    
    success, message = verify_metric(metric_name, task_type)
    
    if not success:
        logger.error(f"[Shield 1] GATE FAILED: {message}")
        # In v2, we trigger HITL if the metric is unverified
        return ProfessorState.validated_update(state, "preflight", {
            "preflight_passed": False,
            "preflight_warnings": state.get("preflight_warnings", []) + [f"Metric Verification Failed: {message}"]
        })

    logger.info(f"[Shield 1] Gate Passed: {message}")
    return state
