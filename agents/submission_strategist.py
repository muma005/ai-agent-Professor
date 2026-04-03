# agents/submission_strategist.py
#
# Day 23 — Submission Strategist
# EWMA monitor, final submission pair, format validation.
#
# Runs after ensemble_architect. Selects the final submission pair,
# monitors EWMA of CV/LB gap over time, and writes submission CSVs.

import json
import logging
import os
import numpy as np
import polars as pl
from datetime import datetime, timezone
from pathlib import Path
from scipy.stats import pearsonr
from core.state import ProfessorState
from core.lineage import log_event

logger = logging.getLogger(__name__)

# ── EWMA monitor constants ────────────────────────────────────────
MIN_SUBMISSIONS_BEFORE_MONITOR = 5
EWMA_ALPHA = 0.3


# ── EWMA computation ──────────────────────────────────────────────

def compute_ewma_gap(gap_history: list[float], alpha: float = EWMA_ALPHA) -> float:
    """
    gap_history: list of cv_lb_gap values in chronological order (oldest first).
    Returns the EWMA of gaps. First value initialises the EWMA.
    """
    if not gap_history:
        return 0.0
    ewma = gap_history[0]
    for gap in gap_history[1:]:
        ewma = alpha * gap + (1 - alpha) * ewma
    return ewma


# ── Submission log helpers ────────────────────────────────────────

def _get_submission_log_path(state: ProfessorState) -> Path:
    session_id = state.get("session_id", "unknown")
    # Support explicit output_dir from tests
    output_dir = state.get("output_dir")
    if output_dir:
        return Path(output_dir) / "submission_log.json"
    return Path(f"outputs/{session_id}/submission_log.json")


def _read_submission_log(state: ProfessorState) -> list:
    log_path = _get_submission_log_path(state)
    if log_path.exists():
        try:
            return json.loads(log_path.read_text())
        except (json.JSONDecodeError, IOError):
            logger.warning(f"[submission_strategist] Corrupt submission log at {log_path}. Starting fresh.")
            return []
    return []


def _write_submission_log(state: ProfessorState, log: list) -> None:
    log_path = _get_submission_log_path(state)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log, indent=2, default=str))


def _append_submission_record(state: ProfessorState, log: list, submission_path: str) -> None:
    """Append a new submission record to the log."""
    session_id = state.get("session_id", "unknown")
    competition = state.get("competition_name", "unknown")
    cv_score = state.get("cv_mean")
    ensemble_accepted = state.get("ensemble_accepted", False)

    # Determine model name
    selected_models = state.get("selected_models", [])
    if ensemble_accepted and selected_models:
        model_used = f"ensemble_{'_'.join(selected_models)}"
    elif selected_models:
        model_used = selected_models[0]
    else:
        model_used = "unknown"

    record = {
        "submission_number": len(log) + 1,
        "session_id": session_id,
        "competition_id": competition,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cv_score": float(cv_score) if cv_score is not None else None,
        "lb_score": None,  # null at write time — filled by harness later
        "cv_lb_gap": None,
        "model_used": model_used,
        "ensemble_accepted": bool(ensemble_accepted),
        "n_models_in_ensemble": len(selected_models) if selected_models else 0,
        "submission_path": submission_path,
        "is_final_pair_submission": False,
    }
    log.append(record)


# ── EWMA monitor ──────────────────────────────────────────────────

def _run_ewma_monitor(state: ProfessorState, submission_log: list) -> dict:
    """
    Monitor CV/LB gap using EWMA. Returns freeze status dict.

    Activation: only after MIN_SUBMISSIONS_BEFORE_MONITOR submissions
    with non-null lb_score.
    """
    # Filter submissions that have lb_score
    submissions_with_lb = [s for s in submission_log if s.get("lb_score") is not None]
    n_with_lb = len(submissions_with_lb)

    result = {
        "n_submissions_with_lb": n_with_lb,
        "submission_freeze_active": False,
        "submission_freeze_reason": "",
        "ewma_current": None,
        "ewma_initial": None,
    }

    if n_with_lb < MIN_SUBMISSIONS_BEFORE_MONITOR:
        return result

    # Extract gap history
    gaps = [s["cv_lb_gap"] for s in submissions_with_lb if s.get("cv_lb_gap") is not None]
    if len(gaps) < MIN_SUBMISSIONS_BEFORE_MONITOR:
        return result

    # Compute initial EWMA from first 5 submissions
    initial_ewma = compute_ewma_gap(gaps[:MIN_SUBMISSIONS_BEFORE_MONITOR])
    current_ewma = compute_ewma_gap(gaps)

    result["ewma_initial"] = float(initial_ewma)
    result["ewma_current"] = float(current_ewma)

    # Freeze Condition A: current_ewma > 2.0 * initial_ewma
    if initial_ewma > 0 and current_ewma > 2.0 * initial_ewma:
        result["submission_freeze_active"] = True
        result["submission_freeze_reason"] = "ewma_exceeded_2x_initial"
        logger.warning(
            f"[submission_strategist] EWMA FREEZE TRIGGERED: "
            f"current_ewma={current_ewma:.6f} > 2 * initial_ewma={initial_ewma:.6f}. "
            f"CV/LB gap is systematically worsening."
        )
        return result

    # Freeze Condition B: gap increased in 5 of last 7 submissions
    if len(gaps) >= 7:
        last_7 = gaps[-7:]
        increases = sum(1 for i in range(1, len(last_7)) if last_7[i] > last_7[i - 1])
        if increases >= 5:
            result["submission_freeze_active"] = True
            result["submission_freeze_reason"] = "gap_increasing_5_of_7"
            logger.warning(
                f"[submission_strategist] EWMA FREEZE TRIGGERED: "
                f"gap increased in {increases} of last 7 submissions. "
                f"CV/LB gap is trending upward."
            )
            return result

    return result


# ── Submission pair selection ─────────────────────────────────────

def _select_submission_pair(state: ProfessorState) -> tuple:
    """
    Select Submission A (best stability_score) and Submission B
    (lowest correlation with A's OOF).

    Returns (a_name, a_oof, b_name, b_oof, corr_ab).
    """
    registry_raw = state.get("model_registry", [])

    # Normalise to dict
    if isinstance(registry_raw, list):
        registry = {}
        for entry in registry_raw:
            name = entry.get("model_id") or entry.get("model_type", f"model_{len(registry)}")
            registry[name] = entry
    elif isinstance(registry_raw, dict):
        registry = dict(registry_raw)
    else:
        raise ValueError(f"model_registry has unexpected type: {type(registry_raw)}")

    if not registry:
        raise ValueError("model_registry is empty — cannot select submissions.")

    # Submission A: highest stability_score
    a_name = max(registry.keys(), key=lambda n: registry[n].get("stability_score", 0.0))

    # If ensemble accepted, use ensemble OOF for A's predictions
    ensemble_accepted = state.get("ensemble_accepted", False)
    if ensemble_accepted:
        a_oof = np.array(state.get("ensemble_oof", []), dtype=float)
        if len(a_oof) == 0:
            a_oof = np.array(registry[a_name].get("oof_predictions", []), dtype=float)
    else:
        a_oof = np.array(registry[a_name].get("oof_predictions", []), dtype=float)

    # Submission B: lowest correlation with A
    if len(registry) == 1:
        # Only one model — use it for both
        b_name = a_name
        b_oof = a_oof
        corr_ab = 1.0
    else:
        b_name = None
        b_corr = 2.0  # start high
        for name, entry in registry.items():
            if name == a_name:
                continue
            oof = np.array(entry.get("oof_predictions", []), dtype=float)
            if len(oof) != len(a_oof):
                continue
            corr, _ = pearsonr(a_oof, oof)
            if corr < b_corr:
                b_corr = corr
                b_name = name
                b_oof = oof

        if b_name is None:
            # Fallback: use second best stability
            sorted_names = sorted(registry.keys(),
                                  key=lambda n: registry[n].get("stability_score", 0.0),
                                  reverse=True)
            b_name = sorted_names[1] if len(sorted_names) > 1 else sorted_names[0]
            b_oof = np.array(registry[b_name].get("oof_predictions", []), dtype=float)
            corr_ab, _ = pearsonr(a_oof, b_oof)
        else:
            corr_ab = b_corr

    return a_name, a_oof, b_name, b_oof, float(corr_ab)


# ── Submission format validation ──────────────────────────────────

def validate_submission(submission_df: pl.DataFrame, sample_submission_path: str, spec: dict) -> None:
    """
    Validate submission against sample_submission.csv before writing.
    Raises ValueError on any mismatch.
    """
    sample = pl.read_csv(sample_submission_path)

    # Check 1: column names match exactly
    if list(submission_df.columns) != list(sample.columns):
        raise ValueError(
            f"Column mismatch. Expected {list(sample.columns)}, "
            f"got {list(submission_df.columns)}"
        )

    # Check 2: row count matches test set
    if len(submission_df) != len(sample):
        raise ValueError(
            f"Row count mismatch. Expected {len(sample)}, got {len(submission_df)}"
        )

    # Check 3: id column values match (same order)
    id_col = spec.get("id_column", sample.columns[0])
    if id_col in submission_df.columns and id_col in sample.columns:
        if submission_df[id_col].to_list() != sample[id_col].to_list():
            raise ValueError(
                "ID column values or order do not match sample_submission.csv"
            )

    # Check 4: target column dtype matches sample
    target_col = spec.get("target_column", sample.columns[1])
    if target_col in submission_df.columns and target_col in sample.columns:
        if submission_df[target_col].dtype != sample[target_col].dtype:
            raise ValueError(
                f"Target column dtype mismatch. Expected {sample[target_col].dtype}, "
                f"got {submission_df[target_col].dtype}"
            )


# ── Write submission CSV ──────────────────────────────────────────

def _write_submission_csv(
    predictions: list,
    sample_submission_path: str,
    output_path: str,
    state: ProfessorState,
) -> str:
    """
    Build and validate submission CSV, then write to disk.
    Returns the output path.
    """
    sample = pl.read_csv(sample_submission_path)
    id_col = sample.columns[0]
    target_col = sample.columns[1]

    # Build submission DataFrame
    submission = pl.DataFrame({
        id_col: sample[id_col].to_list(),
        target_col: predictions,
    })

    # Validate format
    spec = {
        "id_column": id_col,
        "target_column": target_col,
    }
    validate_submission(submission, sample_submission_path, spec)

    # Write
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    submission.write_csv(output_path)
    return output_path


def _convert_predictions_to_target_dtype(
    predictions: list,
    sample_submission_path: str,
) -> list:
    """Convert predictions to match the dtype expected by sample_submission."""
    sample = pl.read_csv(sample_submission_path)
    target_col = sample.columns[1]
    sample_dtype = sample[target_col].dtype

    if sample_dtype == pl.Boolean:
        return [bool(float(p) > 0.5) for p in predictions]
    elif sample_dtype in (pl.Float32, pl.Float64):
        return [float(p) for p in predictions]
    else:
        return [int(float(p)) for p in predictions]


# ── Main entry point ──────────────────────────────────────────────

def run_submission_strategist(state: ProfessorState) -> ProfessorState:
    """
    Submission Strategist pipeline:
    1. Read/initialise submission log
    2. Run EWMA monitor
    3. Select submission pair (A = best stability, B = most diverse)
    4. Write submission_a.csv, submission_b.csv, submission.csv
    5. Append to submission log
    6. Set all state outputs
    7. Log lineage event
    """
    session_id = state.get("session_id", "unknown")
    output_dir = Path(f"outputs/{session_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"[submission_strategist] Starting — session: {session_id}")

    # ── Step 1: Read submission log ───────────────────────────────
    submission_log = _read_submission_log(state)

    # ── Step 2: EWMA monitor ──────────────────────────────────────
    ewma_result = _run_ewma_monitor(state, submission_log)

    # ── Step 3: Select submission pair ────────────────────────────
    a_name, a_oof, b_name, b_oof, corr_ab = _select_submission_pair(state)

    # ── Step 4: Write submission CSVs ─────────────────────────────
    sample_path = state.get("sample_submission_path", "")
    if not sample_path or not Path(sample_path).exists():
        raise ValueError(
            f"sample_submission.csv not found at '{sample_path}'. "
            "Cannot write submission without format reference."
        )

    # Convert predictions to correct dtype
    a_preds = _convert_predictions_to_target_dtype(a_oof.tolist(), sample_path)
    b_preds = _convert_predictions_to_target_dtype(b_oof.tolist(), sample_path)

    submission_a_path = str(output_dir / "submission_a.csv")
    submission_b_path = str(output_dir / "submission_b.csv")
    submission_path = str(output_dir / "submission.csv")

    _write_submission_csv(a_preds, sample_path, submission_a_path, state)
    _write_submission_csv(b_preds, sample_path, submission_b_path, state)

    # submission.csv is same as submission_a.csv (default)
    import shutil
    shutil.copy2(submission_a_path, submission_path)

    # ── Step 5: Append to submission log ──────────────────────────
    _append_submission_record(state, submission_log, submission_path)
    _write_submission_log(state, submission_log)
    log_path = str(_get_submission_log_path(state))

    # ── Step 6: Set state outputs ─────────────────────────────────
    result = {
        **state,
        "submission_a_path": submission_a_path,
        "submission_b_path": submission_b_path,
        "submission_path": submission_path,
        "submission_a_model": a_name,
        "submission_b_model": b_name,
        "submission_b_correlation_with_a": corr_ab,
        "submission_log_path": log_path,
        "submission_freeze_active": ewma_result["submission_freeze_active"],
        "submission_freeze_reason": ewma_result["submission_freeze_reason"],
        "ewma_current": ewma_result["ewma_current"],
        "ewma_initial": ewma_result["ewma_initial"],
        "n_submissions_with_lb": ewma_result["n_submissions_with_lb"],
    }

    # ── Step 7: Lineage event ─────────────────────────────────────
    log_event(
        session_id=session_id,
        agent="submission_strategist",
        action="submission_strategist_complete",
        keys_read=["model_registry", "ensemble_accepted", "ensemble_oof", "cv_mean"],
        keys_written=[
            "submission_a_path", "submission_b_path", "submission_path",
            "submission_a_model", "submission_b_model",
            "submission_b_correlation_with_a", "submission_log_path",
            "submission_freeze_active", "submission_freeze_reason",
            "ewma_current", "ewma_initial", "n_submissions_with_lb",
        ],
        values_changed={
            "submission_a_model": a_name,
            "submission_b_model": b_name,
            "submission_b_correlation": corr_ab,
            "freeze_active": ewma_result["submission_freeze_active"],
            "freeze_reason": ewma_result["submission_freeze_reason"],
            "n_submissions_with_lb": ewma_result["n_submissions_with_lb"],
        },
    )

    logger.info(
        f"[submission_strategist] Complete — A: {a_name}, B: {b_name}, "
        f"corr={corr_ab:.4f}, freeze={ewma_result['submission_freeze_active']}"
    )
    return result
