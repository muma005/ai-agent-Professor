# memory/memory_schema.py
# -------------------------------------------------------------------------
# Day 10 — Competition fingerprints + pattern memory (v2)
# Stores structural insights, not hyperparams.
# -------------------------------------------------------------------------

import uuid
import json
import logging
from datetime import datetime, timezone
from typing import Optional

from memory.chroma_client import build_chroma_client, get_or_create_collection, CHROMADB_AVAILABLE
from core.state import ProfessorState

if not CHROMADB_AVAILABLE:
    logger.warning("[memory_schema] chromadb not installed — memory features disabled.")

logger = logging.getLogger(__name__)

PATTERNS_COLLECTION = "professor_patterns_v2"  # v2 — do NOT touch v1
CRITIC_FAILURE_COLLECTION = "critic_failure_patterns"  # Day 11


def build_competition_fingerprint(state: ProfessorState) -> dict:
    """
    Builds a competition fingerprint from the current pipeline state.
    Called after EDA completes — requires eda_report and validation_strategy in state.

    Never raises. Returns a fingerprint with sensible defaults if data is missing.
    """
    eda    = state.get("eda_report", {})
    schema = {}
    if state.get("schema_path"):
        import os, json as _json
        if os.path.exists(state["schema_path"]):
            try:
                with open(state["schema_path"]) as _f:
                    schema = _json.load(_f)
            except Exception:
                pass

    # -- Row count bucket -------------------------------------------------------
    n_rows = schema.get("n_rows", 0)
    if n_rows == 0:
        # Fallback: try shape field
        shape = schema.get("shape", [])
        n_rows = shape[0] if shape else 0
    if n_rows < 1_000:
        n_rows_bucket = "tiny"
    elif n_rows < 10_000:
        n_rows_bucket = "small"
    elif n_rows < 100_000:
        n_rows_bucket = "medium"
    elif n_rows < 1_000_000:
        n_rows_bucket = "large"
    else:
        n_rows_bucket = "huge"

    # -- Feature count bucket ---------------------------------------------------
    n_features = len(schema.get("columns", [])) - 1  # exclude target
    if n_features < 10:
        n_features_bucket = "narrow"
    elif n_features < 50:
        n_features_bucket = "medium"
    elif n_features < 200:
        n_features_bucket = "wide"
    else:
        n_features_bucket = "very_wide"

    # -- Imbalance ratio --------------------------------------------------------
    target_dist     = eda.get("target_distribution", {})
    imbalance_ratio = target_dist.get("imbalance_ratio", 1.0)  # 1.0 = balanced

    # -- High-cardinality categorical count -------------------------------------
    types      = schema.get("types", {})
    n_unique_map = schema.get("n_unique", {})
    target_col = state.get("target_col", "")
    n_categorical_high_cardinality = sum(
        1 for col, dtype in types.items()
        if col != target_col
        and str(dtype) in {"Utf8", "Categorical", "str"}
        and n_unique_map.get(col, 0) > 50
    )

    # -- Temporal feature -------------------------------------------------------
    temporal = eda.get("temporal_profile", {})
    has_temporal_feature = temporal.get("has_dates", False)

    # -- Task and target types --------------------------------------------------
    task_type   = state.get("task_type", "unknown")
    vs          = state.get("validation_strategy", {})
    target_type = vs.get("target_type", "unknown")

    fingerprint = {
        "task_type":                      task_type,
        "imbalance_ratio":                round(float(imbalance_ratio), 4),
        "n_categorical_high_cardinality": int(n_categorical_high_cardinality),
        "n_rows_bucket":                  n_rows_bucket,
        "has_temporal_feature":           bool(has_temporal_feature),
        "n_features_bucket":              n_features_bucket,
        "target_type":                    target_type,
        "cv_lb_gap_typical":              0.0,  # populated post-competition
    }

    logger.info(f"[MemorySchema] Competition fingerprint built: {fingerprint}")
    return fingerprint


def fingerprint_to_text(fingerprint: dict) -> str:
    """Converts fingerprint dict to embeddable natural language text."""
    imbalance = fingerprint.get("imbalance_ratio", 1.0)
    parts = [
        f"{fingerprint.get('task_type', 'tabular')} machine learning task",
        f"target: {fingerprint.get('target_type', 'unknown')}",
        f"dataset size: {fingerprint.get('n_rows_bucket', 'medium')}",
        f"feature count: {fingerprint.get('n_features_bucket', 'medium')}",
    ]
    if imbalance < 0.05:
        parts.append(f"severely imbalanced dataset {imbalance:.1%} minority class fraud-like")
    elif imbalance < 0.15:
        parts.append(f"highly imbalanced {imbalance:.1%} minority class medical or anomaly detection")
    elif imbalance < 0.35:
        parts.append(f"moderately imbalanced {imbalance:.1%} minority class")
    else:
        parts.append("balanced or near-balanced class distribution")

    n_hc = fingerprint.get("n_categorical_high_cardinality", 0)
    if n_hc > 5:
        parts.append(f"many high-cardinality categorical features {n_hc} columns over 50 unique values")
    elif n_hc > 0:
        parts.append(f"some high-cardinality categoricals {n_hc} columns")
    else:
        parts.append("no high-cardinality categorical features")

    if fingerprint.get("has_temporal_feature"):
        parts.append("temporal structure present time-series date features")

    return ". ".join(parts) + "."


def store_pattern(
    fingerprint: dict,
    validated_approaches: list,
    failed_approaches: list,
    competition_name: str,
    confidence: float = 0.5,
    cv_lb_gap: float = 0.0,
) -> str:
    """
    Stores a pattern in ChromaDB v2 collection.
    Called by post_mortem_agent (Day 11) after a competition closes.
    Returns the pattern_id.
    """
    client     = build_chroma_client()
    collection = get_or_create_collection(client, PATTERNS_COLLECTION)

    pattern_id = str(uuid.uuid4())
    fingerprint["cv_lb_gap_typical"] = cv_lb_gap

    pattern = {
        "pattern_id":               pattern_id,
        "competition_fingerprint":  fingerprint,
        "validated_approaches":     validated_approaches,
        "failed_approaches":        failed_approaches,
        "confidence":               confidence,
        "competitions_validated_on": [competition_name],
        "created_at":               datetime.now(timezone.utc).isoformat(),
        "last_updated":             datetime.now(timezone.utc).isoformat(),
    }

    document_text = (
        fingerprint_to_text(fingerprint)
        + " Validated approaches: "
        + "; ".join(a["approach"] for a in validated_approaches[:3])
    )

    collection.add(
        documents=[document_text],
        metadatas=[{"pattern_json": json.dumps(pattern), "competition": competition_name}],
        ids=[pattern_id],
    )
    logger.info(f"[MemorySchema] Pattern stored: {pattern_id} for {competition_name}")
    return pattern_id


def query_similar_competitions(
    fingerprint: dict,
    n_results: int = 5,
) -> list:
    """
    Returns up to n_results patterns from structurally similar past competitions.
    Sorted by semantic similarity — most similar first.

    Returns list of dicts, each containing the full pattern_json and distance.
    Returns [] (not None) if ChromaDB is unavailable or no patterns exist.
    """
    try:
        client     = build_chroma_client()
        collection = get_or_create_collection(client, PATTERNS_COLLECTION)

        count = collection.count()
        if count == 0:
            logger.info("[MemorySchema] Pattern collection is empty — no warm-start priors available.")
            return []

        query_text = fingerprint_to_text(fingerprint)
        results    = collection.query(
            query_texts=[query_text],
            n_results=min(n_results, count),
            include=["documents", "metadatas", "distances"],
        )

        patterns = []
        for i, metadata in enumerate(results["metadatas"][0]):
            try:
                pattern = json.loads(metadata["pattern_json"])
                pattern["_similarity_distance"] = results["distances"][0][i]
                patterns.append(pattern)
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"[MemorySchema] Skipping malformed pattern: {e}")

        logger.info(f"[MemorySchema] Retrieved {len(patterns)} similar patterns.")
        return patterns

    except Exception as e:
        logger.warning(f"[MemorySchema] Pattern query failed: {e}. Returning empty priors.")
        return []


def get_warm_start_priors(state: ProfessorState) -> list:
    """
    High-level entry point called by ml_optimizer at the start of each session.
    Builds fingerprint from state, queries memory, returns top validated approaches.
    """
    fingerprint = build_competition_fingerprint(state)
    patterns    = query_similar_competitions(fingerprint, n_results=3)

    priors = []
    for pattern in patterns:
        distance = pattern.get("_similarity_distance", 1.0)
        if distance > 0.8:
            continue  # too dissimilar — don't trust it
        for approach in pattern.get("validated_approaches", []):
            priors.append({
                "approach":    approach["approach"],
                "confidence":  pattern["confidence"] * (1.0 - distance),
                "source":      pattern["competitions_validated_on"],
                "improvement": approach.get("cv_improvement", 0.0),
            })

    priors.sort(key=lambda x: x["confidence"], reverse=True)
    logger.info(f"[MemorySchema] Warm-start priors: {len(priors)} approaches")
    return priors[:10]  # top 10 only


def store_critic_failure_pattern(
    fingerprint: dict,
    missed_issue: str,
    competition_name: str,
    feature_flagged: str = "",
    failure_mode: str = "",
    cv_lb_gap: float = 0.0,
    confidence: float = 0.5,
) -> str:
    """
    Stores a critic failure pattern — an issue the critic failed to detect.
    Written by post_mortem_agent when CV/LB gap indicates an undetected problem.
    Returns the failure_id.
    """
    client     = build_chroma_client()
    collection = get_or_create_collection(client, CRITIC_FAILURE_COLLECTION)

    failure_id = str(uuid.uuid4())

    document_text = (
        fingerprint_to_text(fingerprint)
        + f" Critic missed: {missed_issue}"
    )

    metadata = {
        "failure_id":       failure_id,
        "missed_issue":     missed_issue,
        "competition":      competition_name,
        "competition_name": competition_name,
        "fingerprint_json": json.dumps(fingerprint),
        "fingerprint_text": fingerprint_to_text(fingerprint),
        "feature_flagged":  feature_flagged,
        "failure_mode":     failure_mode or missed_issue,
        "cv_lb_gap":        str(cv_lb_gap),
        "confidence":       str(confidence),
        "stored_at":        datetime.now(timezone.utc).isoformat(),
        "created_at":       datetime.now(timezone.utc).isoformat(),
    }

    collection.add(
        documents=[document_text],
        metadatas=[metadata],
        ids=[failure_id],
    )
    logger.info(f"[MemorySchema] Critic failure pattern stored: {failure_id} for {competition_name}")
    return failure_id


def query_critic_failure_patterns(
    fingerprint: dict,
    n_results: int = 5,
    max_distance: float = 0.75,
) -> list[dict]:
    """
    Queries the critic_failure_patterns ChromaDB collection.
    Returns [] if chromadb not installed, collection doesn't exist, or is empty.
    Never raises.
    """
    if not CHROMADB_AVAILABLE:
        return []
    try:
        client = build_chroma_client()
        if client is None:
            return []
        ef = getattr(client, "_professor_ef", None)
        try:
            collection = client.get_collection(
                name=CRITIC_FAILURE_COLLECTION,
                embedding_function=ef,
            )
        except Exception:
            return []   # collection doesn't exist yet — first competition

        count = collection.count()
        if count == 0:
            return []

        query_text = fingerprint_to_text(fingerprint)
        results = collection.query(
            query_texts=[query_text],
            n_results=min(n_results, count),
            include=["documents", "metadatas", "distances"],
        )

        patterns = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            if dist > max_distance:
                continue   # too dissimilar to be useful
            patterns.append({**meta, "distance": dist, "document": doc})

        # Sort by distance ascending — most similar first
        patterns.sort(key=lambda p: p["distance"])
        return patterns

    except Exception as e:
        logger.warning(f"[query_critic_failure_patterns] Failed: {e}")
        return []

