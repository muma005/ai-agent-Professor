# memory/pinecone_memory.py
"""
Vector memory for experiment results, domain briefings, and feature findings.

Currently implemented as an in-memory dict (stub).
Interface is identical to what a Pinecone backend would expose.
Swap backend by changing _get_client() only.

Quality scoring fields on every entry:
  helpfulness_rate: float  — fraction of retrievals that led to improvement
  validated_by_critic: bool — whether red_team_critic validated this finding
  confidence: float        — 0.0 to 1.0

Retrieval threshold: helpfulness_rate > 0.6 AND confidence > 0.7
"""

import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

_STORE: dict[str, dict] = {}   # in-memory backend


def upsert(
    collection: str,
    id: str,
    text: str,
    metadata: dict = None,
    confidence: float = 0.70,
    validated_by_critic: bool = False,
) -> bool:
    """
    Stores or updates an entry. Returns True on success.
    Never raises.
    """
    try:
        key = f"{collection}::{id}"
        _STORE[key] = {
            "id": id,
            "collection": collection,
            "text": text,
            "metadata": metadata or {},
            "confidence": confidence,
            "validated_by_critic": validated_by_critic,
            "helpfulness_rate": _STORE.get(key, {}).get("helpfulness_rate", 1.0),
            "n_retrieved": _STORE.get(key, {}).get("n_retrieved", 0),
            "n_helpful": _STORE.get(key, {}).get("n_helpful", 0),
            "stored_at": datetime.now(timezone.utc).isoformat(),
        }
        return True
    except Exception as e:
        logger.warning(f"[pinecone_memory.upsert] Failed: {e}")
        return False


def query(
    collection: str,
    query_text: str,
    n_results: int = 5,
    min_confidence: float = 0.60,
    min_helpfulness: float = 0.60,
) -> list[dict]:
    """
    Returns up to n_results entries from the collection that meet quality thresholds.
    Uses simple text overlap scoring (stub behaviour — Pinecone would use embeddings).
    Never raises. Returns [] on any error.
    """
    try:
        entries = [
            v for k, v in _STORE.items()
            if k.startswith(f"{collection}::")
            and float(v.get("confidence", 0)) >= min_confidence
            and (
                int(v.get("n_retrieved", 0)) == 0 or
                float(v.get("helpfulness_rate", 1.0)) >= min_helpfulness
            )
        ]

        # Stub scoring: fraction of query words found in text
        query_words = set(query_text.lower().split())

        def _score(entry):
            text_words = set(entry["text"].lower().split())
            return len(query_words & text_words) / max(len(query_words), 1)

        entries.sort(key=_score, reverse=True)
        return entries[:n_results]

    except Exception as e:
        logger.warning(f"[pinecone_memory.query] Failed: {e}")
        return []


def update_helpfulness(collection: str, id: str, was_helpful: bool) -> bool:
    """Updates helpfulness_rate after a competition result is known."""
    try:
        key = f"{collection}::{id}"
        if key not in _STORE:
            return False
        entry = _STORE[key]
        entry["n_retrieved"] += 1
        if was_helpful:
            entry["n_helpful"] += 1
        entry["helpfulness_rate"] = round(
            entry["n_helpful"] / entry["n_retrieved"], 4
        )
        if not was_helpful:
            entry["confidence"] = round(
                max(0.0, entry["confidence"] - 0.05), 4
            )
        return True
    except Exception as e:
        logger.warning(f"[pinecone_memory.update_helpfulness] Failed: {e}")
        return False


def count(collection: str) -> int:
    """Returns the number of entries in a collection."""
    return sum(1 for k in _STORE if k.startswith(f"{collection}::"))


def clear(collection: str) -> None:
    """Remove all entries from a collection. Used in tests."""
    to_delete = [k for k in _STORE if k.startswith(f"{collection}::")]
    for k in to_delete:
        del _STORE[k]
