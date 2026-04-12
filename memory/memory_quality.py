# memory/memory_quality.py
"""
Quality scoring for ChromaDB memory entries.

helpfulness_rate: fraction of times a retrieved memory led to a CV improvement.
Computed after each competition by post_mortem_agent.

Retrieval threshold: helpfulness_rate > 0.6 AND confidence > 0.7
Memories that fall below threshold are flagged for decay.

Decay: confidence is reduced by 0.05 per competition where memory was retrieved
but did not help. Memories below confidence=0.50 are removed.
"""

import logging
from datetime import datetime, timezone

from memory.memory_schema import build_chroma_client, CHROMADB_AVAILABLE
from memory.seed_memory import SEED_CONFIDENCE

logger = logging.getLogger(__name__)

RETRIEVAL_THRESHOLD_HELPFULNESS = 0.60
RETRIEVAL_THRESHOLD_CONFIDENCE  = 0.70
DECAY_RATE                      = 0.05
REMOVAL_THRESHOLD               = 0.50


def update_memory_helpfulness(
    collection_name: str,
    memory_id: str,
    was_helpful: bool,
) -> bool:
    """
    Updates the helpfulness_rate of a memory entry after a competition.

    was_helpful=True:  this memory contributed to a CV improvement
    was_helpful=False: this memory was retrieved but did not help

    Updates the entry in ChromaDB. Returns True on success, False on failure.
    Never raises.
    """
    try:
        if not CHROMADB_AVAILABLE:
            return False
        client = build_chroma_client()
        if client is None:
            return False
        collection = client.get_collection(collection_name)

        result = collection.get(ids=[memory_id], include=["metadatas"])
        if not result["metadatas"]:
            return False

        meta = result["metadatas"][0]
        n_retrieved = int(meta.get("n_retrieved", 0)) + 1
        n_helpful = int(meta.get("n_helpful", 0)) + (1 if was_helpful else 0)
        helpfulness = round(n_helpful / n_retrieved, 4)

        # Apply decay if not helpful
        confidence = float(meta.get("confidence", 0.70))
        if not was_helpful:
            confidence = round(max(0.0, confidence - DECAY_RATE), 4)

        new_meta = {
            **meta,
            "n_retrieved": n_retrieved,
            "n_helpful": n_helpful,
            "helpfulness_rate": helpfulness,
            "confidence": confidence,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }

        collection.update(ids=[memory_id], metadatas=[new_meta])
        return True

    except Exception as e:
        logger.warning(f"[update_memory_helpfulness] Failed for {memory_id}: {e}")
        return False


def remove_decayed_memories(collection_name: str) -> int:
    """
    Removes all entries with confidence < REMOVAL_THRESHOLD (0.50).
    Returns count of removed entries.
    Called by post_mortem_agent after updating helpfulness.
    Never raises.
    """
    try:
        if not CHROMADB_AVAILABLE:
            return 0
        client = build_chroma_client()
        if client is None:
            return 0
        collection = client.get_collection(collection_name)

        if collection.count() == 0:
            return 0

        result = collection.get(include=["metadatas", "ids"])
        to_remove = [
            id_ for id_, meta in zip(result["ids"], result["metadatas"])
            if float(meta.get("confidence", 1.0)) < REMOVAL_THRESHOLD
        ]

        if to_remove:
            collection.delete(ids=to_remove)
            logger.info(
                f"[memory_quality] Removed {len(to_remove)} decayed entries "
                f"from '{collection_name}'."
            )

        return len(to_remove)

    except Exception as e:
        logger.warning(f"[remove_decayed_memories] Failed: {e}")
        return 0


def should_retrieve(meta: dict) -> bool:
    """
    Returns True if a memory entry meets the retrieval quality threshold.
    Applied as a post-filter after ChromaDB distance filtering.
    """
    helpfulness = float(meta.get("helpfulness_rate", 1.0))  # default: always retrieve if new
    confidence = float(meta.get("confidence", SEED_CONFIDENCE))
    n_retrieved = int(meta.get("n_retrieved", 0))

    # New memories (never retrieved) get benefit of the doubt
    if n_retrieved == 0:
        return confidence >= RETRIEVAL_THRESHOLD_CONFIDENCE

    return (
        helpfulness >= RETRIEVAL_THRESHOLD_HELPFULNESS and
        confidence >= RETRIEVAL_THRESHOLD_CONFIDENCE
    )
