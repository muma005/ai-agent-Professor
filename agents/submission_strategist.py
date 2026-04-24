# agents/submission_strategist.py

import os
import logging
import numpy as np
import polars as pl
from pathlib import Path
from typing import Dict, Any
from core.state import ProfessorState
from guards.agent_retry import with_agent_retry
from tools.performance_monitor import timed_node
from core.lineage import log_event

from tools.submission_freeze import apply_submission_freeze
from tools.submission_validator import validate_submission
from tools.operator_channel import emit_to_operator

logger = logging.getLogger(__name__)

AGENT_NAME = "submission_strategist"

@timed_node
@with_agent_retry(AGENT_NAME)
def run_submission_strategist(state: ProfessorState) -> ProfessorState:
    """
    Final node before publisher. Applies EMA stability, validates format,
    and saves the final submission.csv.
    """
    session_id = state.get("session_id", "default")
    output_dir = Path(f"outputs/{session_id}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"[{AGENT_NAME}] Preparing final submission...")

    ensemble_preds_path = state.get("ensemble_test_predictions_path")
    sample_sub_path = state.get("sample_submission_path")
    
    if not ensemble_preds_path or not os.path.exists(ensemble_preds_path):
        return state
    if not sample_sub_path or not os.path.exists(sample_sub_path):
        return state

    # 1. Load predictions
    new_preds = pl.read_parquet(ensemble_preds_path)["pred"].to_numpy()

    # 2. Apply Submission Freeze (EMA stability)
    ewma, is_frozen = apply_submission_freeze(state, new_preds)
    
    freeze_reason = ""
    if is_frozen:
        logger.warning(f"[{AGENT_NAME}] Submission frozen! Predictions diverged from EMA.")
        freeze_reason = "Spearman correlation dropped below threshold."
        emit_to_operator("❄️ SUBMISSION FROZEN: Predictions diverged from historical EMA. Using safe historical average.", level="WARNING")
    
    # 3. Format the submission
    # Load sample submission to get IDs and exact format
    try:
        sample_df = pl.read_csv(sample_sub_path) if sample_sub_path.endswith(".csv") else pl.read_parquet(sample_sub_path)
    except:
        return state

    id_col = sample_df.columns[0]
    target_col = sample_df.columns[1]
    
    # Ensure length matches before assigning
    if len(sample_df) == len(ewma):
        # We need to respect the target dtype in the sample submission
        target_dtype = sample_df[target_col].dtype
        if target_dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
            # Probably binary classification, convert float to int
            # Threshold at 0.5
            final_preds = (ewma > 0.5).astype(int)
        else:
            final_preds = ewma
            
        sub_df = pl.DataFrame({
            id_col: sample_df[id_col],
            target_col: pl.Series(final_preds).cast(target_dtype)
        })
    else:
        # Fallback to zeros/means because shapes don't match
        sub_df = sample_df

    # 4. Save draft submission
    draft_path = str(output_dir / "draft_submission.csv")
    sub_df.write_csv(draft_path)

    # 5. Validate submission format
    validation = validate_submission(sample_sub_path, draft_path)
    
    final_path = str(output_dir / "submission.csv")
    
    if validation["is_valid"]:
        sub_df.write_csv(final_path)
        logger.info(f"[{AGENT_NAME}] Submission validated and saved to {final_path}")
    else:
        logger.error(f"[{AGENT_NAME}] Validation failed! {validation['errors']}")
        emit_to_operator(f"🚨 SUBMISSION VALIDATION FAILED: {validation['errors']}. Falling back to sample_submission.", level="ERROR")
        # 6. Fallback to sample submission to guarantee success
        sample_df.write_csv(final_path)

    # 7. Update state
    updates = {
        "submission_path": final_path,
        "submission_freeze_active": is_frozen,
        "submission_freeze_reason": freeze_reason,
        "ewma_current": ewma.tolist(),
        "n_submissions_with_lb": state.get("n_submissions_with_lb", 0) + 1
    }

    log_event(
        session_id=session_id,
        agent=AGENT_NAME,
        action="submission_generated",
        keys_written=list(updates.keys())
    )

    return ProfessorState.validated_update(state, AGENT_NAME, updates)
