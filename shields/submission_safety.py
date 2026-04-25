# shields/submission_safety.py

import os
import json
import logging
import numpy as np
import polars as pl
from scipy.stats import spearmanr
from typing import Dict, Any, Tuple, Optional
from core.state import ProfessorState

logger = logging.getLogger(__name__)

AGENT_NAME = "submission_safety"

def calculate_diversity(path_a: str, path_b: str) -> float:
    """Calculates Spearman correlation between two submission files."""
    if not (path_a and path_b and os.path.exists(path_a) and os.path.exists(path_b)):
        return 0.0
    
    try:
        df_a = pl.read_csv(path_a) if path_a.endswith(".csv") else pl.read_parquet(path_a)
        df_b = pl.read_csv(path_b) if path_b.endswith(".csv") else pl.read_parquet(path_b)
        
        # Second column is usually the target
        col_a = df_a.columns[1]
        col_b = df_b.columns[1]
        
        s_a = df_a[col_a].to_numpy()
        s_b = df_b[col_b].to_numpy()

        if len(s_a) < 2:
            return 1.0
        
        std_a = np.std(s_a)
        std_b = np.std(s_b)

        if std_a == 0 or std_b == 0:
            return 1.0
            
        corr, _ = spearmanr(s_a, s_b)
        return float(corr)
    except Exception as e:
        logger.warning(f"Diversity calculation failed: {e}")
        return 0.0

def estimate_lb_noise(ewma_initial: Optional[list], ewma_current: Optional[list]) -> float:
    """Estimates LB noise based on divergence of EMA from initial state."""
    if not ewma_initial or not ewma_current:
        return 0.0
        
    try:
        a = np.array(ewma_initial)
        b = np.array(ewma_current)
        if len(a) != len(b): return 0.0
        
        noise = np.mean(np.abs(a - b)) / (np.mean(np.abs(a)) + 1e-9)
        return float(noise)
    except:
        return 0.0

def check_submission_safety(state: ProfessorState) -> ProfessorState:
    """
    Shield: Performs safety analysis on submissions and EMA stability.
    """
    # 1. Diversity Rating
    path_a = state.get("submission_a_path")
    path_b = state.get("submission_b_path")
    diversity_corr = calculate_diversity(path_a, path_b)
    
    # 2. LB Noise Estimation
    noise = estimate_lb_noise(state.get("ewma_initial"), state.get("ewma_current"))
    
    # 3. Freeze Override Logic
    context = state.get("competition_context") or {}
    days_left = context.get("days_remaining", 99)
    freeze_override_active = (days_left < 7)
    
    # 4. Generate Report
    safety_report = {
        "diversity_spearman": round(diversity_corr, 4),
        "diversity_rating": "HIGH" if diversity_corr < 0.90 else ("MEDIUM" if diversity_corr < 0.98 else "LOW"),
        "lb_noise_estimate": round(noise, 6),
        "freeze_override_active": bool(freeze_override_active),
        "days_remaining": int(days_left),
        "risk_level": "CRITICAL" if noise > 0.1 else ("WARNING" if noise > 0.05 else "SAFE")
    }
    
    logger.info(f"[{AGENT_NAME}] Safety Check: {safety_report['risk_level']} (Noise: {safety_report['lb_noise_estimate']})")

    updates = {
        "submission_safety_report": safety_report,
        "submission_b_correlation_with_a": diversity_corr
    }
    
    return ProfessorState.validated_update(state, AGENT_NAME, updates)
