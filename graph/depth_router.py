# graph/depth_router.py

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# ── Configuration ───────────────────────────────────────────────────────────

STANDARD_METRICS = {
    "roc_auc", "auc", "log_loss", "logloss", "rmse", "rmsle", "mae", "f1", 
    "accuracy", "r2", "mcc", "binary_crossentropy", "cross_entropy"
}

SPRINT_SKIP_AGENTS = [
    "competition_intel", "domain_researcher", "shift_detector", 
    "creative_hypothesis", "problem_reframer", "pseudo_label_agent", 
    "post_processor"
]

# ── Core Logic ──────────────────────────────────────────────────────────────

def classify_pipeline_depth(
    preflight_data_files: List[Dict],
    preflight_warnings: List[Dict], 
    preflight_target_type: str,
    preflight_data_size_mb: float,
    n_rows: int,
    n_features: int,
    metric_name: str,
    operator_override: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Classify competition complexity and select pipeline depth.
    
    Depth Tiers:
    - SPRINT: Simple, fast, low-overhead.
    - STANDARD: Default, balanced.
    - MARATHON: Complex, high-resource, exhaustive.
    """
    if operator_override in ["sprint", "standard", "marathon"]:
        logger.info(f"[DepthRouter] Using operator override: {operator_override}")
        depth = operator_override
        auto_detected = False
    else:
        # Auto-detection Logic
        auto_detected = True
        
        # 1. SPRINT triggers (ALL must be true)
        is_sprint_candidate = (
            n_rows < 10000 and
            n_features < 30 and
            metric_name.lower() in STANDARD_METRICS and
            preflight_target_type in ["binary", "multiclass", "regression"] and
            not any(w.get("type") == "blocking" for w in preflight_warnings)
        )
        
        # 2. MARATHON triggers (ANY may be true)
        is_marathon_candidate = (
            n_rows > 100000 or
            n_features > 200 or
            metric_name.lower() not in STANDARD_METRICS or
            preflight_data_size_mb > 500
        )
        
        if is_sprint_candidate and not is_marathon_candidate:
            depth = "sprint"
        elif is_marathon_candidate:
            depth = "marathon"
        else:
            depth = "standard"

    # 3. Setting Configurations based on Depth
    if depth == "sprint":
        config = {
            "depth": "sprint",
            "agents_skipped": SPRINT_SKIP_AGENTS,
            "optuna_trials": 50,
            "feature_rounds": 2,
            "critic_vectors": 4
        }
    elif depth == "marathon":
        config = {
            "depth": "marathon",
            "agents_skipped": [],
            "optuna_trials": 200,
            "feature_rounds": 7,
            "critic_vectors": 9
        }
    else:
        config = {
            "depth": "standard",
            "agents_skipped": [],
            "optuna_trials": 100,
            "feature_rounds": 3,
            "critic_vectors": 9
        }

    config["auto_detected"] = auto_detected
    config["reason"] = f"rows={n_rows}, features={n_features}, metric={metric_name} -> {depth}"
    
    return config
