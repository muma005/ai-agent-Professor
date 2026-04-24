# tools/submission_freeze.py

import numpy as np
from scipy.stats import spearmanr
from typing import Tuple
from core.state import ProfessorState

def apply_submission_freeze(state: ProfessorState, new_preds: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Applies Exponential Moving Average (EWMA) smoothing to submissions.
    Freezes (rejects) the new predictions if they diverge too much from the running average.
    """
    n_subs = state.get("n_submissions_with_lb", 0)
    ewma_current = state.get("ewma_current")
    
    if n_subs == 0 or ewma_current is None:
        return new_preds, False
        
    ewma_current = np.array(ewma_current)
    
    if len(new_preds) != len(ewma_current):
        # Mismatched lengths, cannot compare safely
        return ewma_current, True
        
    task_type = state.get("task_type", "classification")
    threshold = 0.95 if task_type == "regression" else 0.98
    
    # Calculate Spearman correlation
    # Handle cases where predictions are constant (std=0)
    if np.std(new_preds) == 0 or np.std(ewma_current) == 0:
        corr = 0.0
    else:
        corr, _ = spearmanr(new_preds, ewma_current)
        
    if corr < threshold:
        # Frozen
        return ewma_current, True
    else:
        # Updated
        new_ewma = 0.7 * ewma_current + 0.3 * new_preds
        return new_ewma, False
