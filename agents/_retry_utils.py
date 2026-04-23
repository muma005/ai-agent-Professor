# agents/_retry_utils.py

from typing import Dict, Any
from guards.circuit_breaker import _classify_error

RETRY_GUIDANCE = {
    "data_quality": (
        "ERROR CLASSIFICATION: Data quality issue.\n"
        "MOST LIKELY CAUSES:\n"
        "- Missing column (check column names against data schema)\n"
        "- Type mismatch (categorical treated as numeric or vice versa)\n"
        "- NaN/null values in unexpected places\n"
        "- Empty dataframe after filtering or join\n"
        "FIX STRATEGY: Print df.columns and df.dtypes FIRST. "
        "Verify column exists before accessing. "
        "Add .drop_nulls() or .fill_null() before operations that don't accept nulls."
    ),
    "model_failure": (
        "ERROR CLASSIFICATION: Model training failure.\n"
        "MOST LIKELY CAUSES:\n"
        "- Hyperparameters too aggressive (learning_rate too high, max_depth too deep)\n"
        "- Feature matrix contains inf or NaN values\n"
        "- Target variable has wrong format or type\n"
        "- Too few samples for the model complexity\n"
        "FIX STRATEGY: Use conservative defaults: n_estimators=100, max_depth=6, "
        "learning_rate=0.1. Add np.isfinite() check on feature matrix before training. "
        "Verify target dtype matches task type."
    ),
    "memory": (
        "ERROR CLASSIFICATION: Memory exhaustion.\n"
        "MOST LIKELY CAUSES:\n"
        "- Dataset too large for available RAM\n"
        "- Too many features created simultaneously\n"
        "- Model with n_jobs=-1 using all cores\n"
        "FIX STRATEGY: Set n_jobs=1. Reduce n_estimators to 100. "
        "Process features in batches of 10. "
        "Use float32 instead of float64: df = df.cast({col: pl.Float32 for col in float_cols})."
    ),
    "api_timeout": (
        "ERROR CLASSIFICATION: External API timeout.\n"
        "This is NOT a code bug. The LLM API rate limit was hit or network is slow.\n"
        "FIX STRATEGY: Reduce prompt size. Truncate context. "
        "The service_health wrapper handles retries automatically. "
        "Do NOT change the actual ML code — the issue is infrastructure, not logic."
    ),
    "unknown": (
        "ERROR CLASSIFICATION: Unclassified error.\n"
        "FIX STRATEGY: Read the traceback carefully. Identify the exact line. "
        "Print the variable state on the line BEFORE the failure. "
        "Do NOT change unrelated code — isolate the fix to the failing operation."
    ),
}

def build_retry_prompt(
    error: Exception,
    traceback_str: str,
    attempt: int,
    agent_name: str,
) -> str:
    """
    Build a targeted retry prompt based on error classification.
    """
    # Classify using existing circuit breaker logic
    error_class = _classify_error(agent_name, error)
    guidance = RETRY_GUIDANCE.get(error_class, RETRY_GUIDANCE["unknown"])
    
    # Truncate traceback to keep LLM context efficient
    tb_lines = traceback_str.strip().split("\n")
    if len(tb_lines) > 30:
        tb_truncated = "\n".join(
            tb_lines[:5] + ["... (truncated %d lines) ..." % (len(tb_lines) - 30)] + tb_lines[-25:]
        )
        label = "ACTUAL ERROR (truncated):"
    else:
        tb_truncated = traceback_str
        label = "ACTUAL ERROR:"
    
    return (
        f"\n{'='*60}\n"
        f"ATTEMPT {attempt} FAILED. DO NOT REPEAT THE SAME MISTAKE.\n"
        f"{'='*60}\n\n"
        f"{guidance}\n\n"
        f"EXCEPTION: {str(error)}\n\n"
        f"{label}\n{tb_truncated}\n\n"
        f"CRITICAL: Fix the SPECIFIC issue identified above. "
        f"Do not rewrite unrelated code.\n"
        f"{'='*60}\n"
    )
