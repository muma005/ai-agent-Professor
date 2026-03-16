# agents/ensemble_architect.py

import logging
from core.state import ProfessorState
from core.lineage import log_event

logger = logging.getLogger(__name__)


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

    # Extract hash from every registry entry
    hashes = {}
    for i, entry in enumerate(registry):
        name = entry.get("model_type", f"model_{i}")
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
        # Filter to models matching current data version only
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

    # All hashes match current — clean path
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


def _compute_ensemble_weights(state: ProfessorState) -> dict:
    """Compute ensemble weights from model registry CV scores."""
    registry = state.get("model_registry", [])
    if not registry:
        return {}
    # Simple equal-weight ensemble for now
    n = len(registry)
    return {i: 1.0 / n for i in range(n)}


def blend_models(state: ProfessorState) -> ProfessorState:
    """
    Blend models from registry into an ensemble.
    Validates data_hash consistency FIRST, before any weight computation.
    """
    # Data hash validation MUST come first — before any blending
    state = _validate_data_hash_consistency(state)

    weights = _compute_ensemble_weights(state)
    return {**state, "ensemble_weights": weights}
