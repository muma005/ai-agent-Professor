# agents/ensemble_architect.py

import logging
import numpy as np
from scipy.stats import pearsonr
from core.state import ProfessorState
from core.lineage import log_event

logger = logging.getLogger(__name__)

# ── Diversity ensemble constants ──────────────────────────────────
DIVERSITY_WEIGHT = 1.0
MAX_CORRELATION_REJECT = 0.97
PRIZE_CORRELATION_CEIL = 0.85
PRIZE_CV_WITHIN = 0.01
MIN_ENSEMBLE_SIZE = 2
MAX_ENSEMBLE_SIZE = 8


def _validate_data_hash_consistency(state: ProfessorState) -> ProfessorState:
    """
    Ensures all models in registry were trained on the same data version.
    Raises ValueError if models are mixed across data versions and none match current.
    Filters to current-hash models if mismatch detected.
    Logs WARNING if any filtering occurs.
    """
    registry = state.get("model_registry", [])
    if not registry:
        raise ValueError("model_registry is empty — no models to ensemble.")

    current_hash = state.get("data_hash")
    if not current_hash:
        logger.warning(
            "[ensemble_architect] state['data_hash'] is None. "
            "Cannot verify data version consistency. Proceeding without check."
        )
        return state

    # Extract hash from every registry entry (list format per state spec)
    hashes = {}
    for entry in registry:
        name = entry.get("model_type", "unknown")
        hashes[name] = entry.get("data_hash")

    unique_hashes = set(h for h in hashes.values() if h is not None)

    if unique_hashes and current_hash not in unique_hashes:
        logger.warning(
            f"[ensemble_architect] DATA VERSION MISMATCH DETECTED. "
            f"No models match current data_hash={current_hash}. "
            f"Registry hashes: {unique_hashes}. Retrain required."
        )
        raise ValueError(
            f"No models in registry match current data_hash={current_hash}. "
            f"All {len(registry)} models were trained on stale data versions. "
            f"Retrain required: run ml_optimizer from the beginning."
        )

    if len(unique_hashes) > 1:
        logger.warning(
            f"[ensemble_architect] DATA VERSION MISMATCH DETECTED. "
            f"Registry contains models trained on {len(unique_hashes)} different data versions: "
            f"{unique_hashes}. "
            f"Filtering to only models matching current data_hash={current_hash}."
        )
        filtered_registry = [
            entry for entry in registry
            if entry.get("data_hash") == current_hash
        ]

        if not filtered_registry:
            raise ValueError(
                f"No models in registry match current data_hash={current_hash}. "
                f"All {len(registry)} models were trained on stale data versions. "
                f"Retrain required: run ml_optimizer from the beginning."
            )

        logger.info(
            f"[ensemble_architect] Filtered registry: "
            f"{len(filtered_registry)}/{len(registry)} models retained "
            f"(matching data_hash={current_hash})."
        )

        state = {**state, "model_registry": filtered_registry}

    log_event(
        session_id=state["session_id"],
        agent="ensemble_architect",
        action="data_hash_validated",
        keys_read=["data_hash", "model_registry"],
        keys_written=[],
        values_changed={
            "data_hash": current_hash,
            "models_checked": len(registry),
            "models_retained": len(state["model_registry"]),
        },
    )
    return state


# ── OOF validation ────────────────────────────────────────────────

def _validate_oof_present(model_registry: dict) -> None:
    """Raises ValueError if any model is missing OOF predictions."""
    missing = [
        name for name, entry in model_registry.items()
        if not entry.get("oof_predictions")
    ]
    if missing:
        raise ValueError(
            f"OOF predictions missing for models: {missing}. "
            "Cannot run diversity ensemble selection without OOF predictions. "
            "Verify ml_optimizer contract (Day 4) is met."
        )


# ── Correlation matrix ───────────────────────────────────────────

def _build_correlation_matrix(oof_map: dict[str, np.ndarray]) -> dict:
    """Returns pairwise Pearson correlation for all selected models."""
    names = list(oof_map.keys())
    matrix = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            corr, _ = pearsonr(oof_map[a], oof_map[b])
            matrix[f"{a}_vs_{b}"] = round(float(corr), 4)
    return matrix


# ── Selection report ─────────────────────────────────────────────

def _write_selection_report(state: ProfessorState, result: dict) -> None:
    """Write diversity selection report to session output dir."""
    import json
    from pathlib import Path
    session_id = state["session_id"]
    report_dir = Path(f"outputs/{session_id}")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "ensemble_selection_report.json"
    report_path.write_text(json.dumps(result, indent=2, default=str))


# ── GM-CAP 5: Diversity-first ensemble selection ─────────────────

def select_diverse_ensemble(
    model_registry: dict,
    state: ProfessorState,
    diversity_weight: float = DIVERSITY_WEIGHT,
    max_correlation: float = MAX_CORRELATION_REJECT,
    max_ensemble_size: int = MAX_ENSEMBLE_SIZE,
) -> dict:
    """
    Diversity-first ensemble selection.

    Algorithm:
      1. Validate OOF predictions present for all models.
      2. Start with the highest CV model as anchor.
      3. Greedily add models that maximise: cv_score * (1 - correlation) * diversity_weight
      4. Reject any model with correlation > max_correlation (adds nothing).
      5. Flag prize candidates: correlation < 0.85 AND CV within 0.01 of best model.
      6. Stop when max_ensemble_size reached or all models evaluated.

    Returns:
        dict with keys: selected_models, prize_candidates, selection_log,
                        ensemble_weights, correlation_matrix
    """
    _validate_oof_present(model_registry)

    models = list(model_registry.items())
    if not models:
        raise ValueError("model_registry is empty — no models to select from.")

    # Sort by CV score descending — anchor is the best model
    models.sort(key=lambda x: float(x[1].get("cv_mean", 0.0)), reverse=True)
    best_cv = float(models[0][1]["cv_mean"])

    anchor_name, anchor_entry = models[0]
    selected = [anchor_name]
    selection_log = []
    prize_candidates = []

    # Build current ensemble OOF mean (starts as just the anchor's OOF)
    oof_arrays = {anchor_name: np.array(anchor_entry["oof_predictions"], dtype=float)}
    ensemble_oof_mean = oof_arrays[anchor_name].copy()

    selection_log.append({
        "model": anchor_name,
        "decision": "SELECTED_ANCHOR",
        "cv_mean": round(float(anchor_entry["cv_mean"]), 6),
        "correlation": None,
        "diversity_score": None,
        "reason": "Highest CV model — anchor",
    })

    # Greedy selection loop
    for model_name, entry in models[1:]:
        if len(selected) >= max_ensemble_size:
            selection_log.append({
                "model": model_name,
                "decision": "SKIPPED_MAX_SIZE",
                "cv_mean": round(float(entry.get("cv_mean", 0.0)), 6),
                "correlation": None,
                "diversity_score": None,
                "reason": f"Max ensemble size ({max_ensemble_size}) reached",
            })
            continue

        candidate_oof = np.array(entry["oof_predictions"], dtype=float)
        cv = float(entry.get("cv_mean", 0.0))

        # Pearson correlation between candidate OOF and current ensemble mean
        corr, _ = pearsonr(candidate_oof, ensemble_oof_mean)

        # Reject if too correlated — strict > not >=
        if corr > max_correlation:
            selection_log.append({
                "model": model_name,
                "decision": "REJECTED_TOO_CORRELATED",
                "cv_mean": round(cv, 6),
                "correlation": round(float(corr), 4),
                "diversity_score": None,
                "reason": f"correlation={corr:.4f} > {max_correlation}",
            })
            continue

        # Diversity-weighted score: higher CV + lower correlation = higher score
        diversity_score = cv * (1.0 - corr) * diversity_weight

        # Prize candidate check — both conditions required (AND, not OR)
        is_prize = (corr < PRIZE_CORRELATION_CEIL) and (abs(cv - best_cv) <= PRIZE_CV_WITHIN)
        if is_prize:
            prize_candidates.append({
                "model": model_name,
                "cv_mean": round(cv, 6),
                "correlation": round(float(corr), 4),
                "cv_delta_from_best": round(abs(cv - best_cv), 6),
                "diversity_score": round(diversity_score, 6),
                "reason": "Low correlation + competitive CV — prize candidate",
            })

        selection_log.append({
            "model": model_name,
            "decision": "SELECTED",
            "cv_mean": round(cv, 6),
            "correlation": round(float(corr), 4),
            "diversity_score": round(diversity_score, 6),
            "is_prize": is_prize,
            "reason": f"diversity_score={diversity_score:.4f}",
        })

        selected.append(model_name)
        oof_arrays[model_name] = candidate_oof

        # Update ensemble OOF mean (equal weights during selection)
        n = len(selected)
        ensemble_oof_mean = sum(oof_arrays[m] for m in selected) / n

    # Compute final ensemble weights (equal for now — Nelder-Mead in Phase 3)
    ensemble_weights = _compute_ensemble_weights(selected)

    # Build correlation matrix for logging
    correlation_matrix = _build_correlation_matrix(
        {m: oof_arrays[m] for m in selected}
    )

    logger.info(
        f"[ensemble_architect] Diversity selection: "
        f"{len(model_registry)} candidates → {len(selected)} selected. "
        f"Prize candidates: {len(prize_candidates)}. "
        f"Selections: {selected}"
    )

    return {
        "selected_models": selected,
        "prize_candidates": prize_candidates,
        "selection_log": selection_log,
        "ensemble_weights": ensemble_weights,
        "correlation_matrix": correlation_matrix,
        "anchor": anchor_name,
        "best_cv": round(best_cv, 6),
    }


def _compute_ensemble_weights(selected_models: list) -> dict:
    """Compute equal weights for selected ensemble models."""
    return {m: 1.0 / len(selected_models) for m in selected_models}


def blend_models(state: ProfessorState) -> ProfessorState:
    """
    Blend models from registry into an ensemble.
    1. Validates data_hash consistency FIRST (Day 13).
    2. Diversity-first ensemble selection (Day 16).
    3. Blend using selected models and weights.
    """
    # Data hash validation MUST come first — before any blending
    state = _validate_data_hash_consistency(state)

    # Convert list-format registry to dict for internal ensemble logic
    registry_list = state.get("model_registry", [])
    if isinstance(registry_list, list):
        registry_dict = {entry.get("model_type", f"model_{i}"): entry
                         for i, entry in enumerate(registry_list)}
    else:
        registry_dict = registry_list

    # Diversity selection — replaces naive top-N
    selection_result = select_diverse_ensemble(
        model_registry=registry_dict,
        state=state,
    )

    # Log to lineage
    log_event(
        session_id=state["session_id"],
        agent="ensemble_architect",
        action="ensemble_selection_complete",
        keys_read=["model_registry"],
        keys_written=["ensemble_selection", "selected_models", "ensemble_weights",
                       "ensemble_oof", "prize_candidates"],
        values_changed={
            "selected": selection_result["selected_models"],
            "prize_candidates_count": len(selection_result["prize_candidates"]),
            "anchor": selection_result["anchor"],
        },
    )

    # Write selection report
    _write_selection_report(state, selection_result)

    # Blend using selected models and weights
    oof_stack = np.column_stack([
        np.array(registry_dict[m]["oof_predictions"], dtype=float)
        for m in selection_result["selected_models"]
    ])
    weights = np.array([
        selection_result["ensemble_weights"][m]
        for m in selection_result["selected_models"]
    ])
    ensemble_oof = oof_stack @ weights

    state = {
        **state,
        "ensemble_selection": selection_result,
        "selected_models": selection_result["selected_models"],
        "ensemble_weights": selection_result["ensemble_weights"],
        "ensemble_oof": ensemble_oof.tolist(),
        "prize_candidates": selection_result["prize_candidates"],
    }
    return state
