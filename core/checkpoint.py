# core/checkpoint.py
"""
Pipeline checkpointing for recovery.

Saves pipeline state at critical points to enable:
1. Recovery from failures
2. Resume from checkpoints
3. Debugging state at each node
"""

import json
import os
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


def _is_serializable(value: Any) -> bool:
    """Check if value is JSON-serializable."""
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False


def save_checkpoint(
    state: Dict[str, Any],
    path: str,
    node_name: str = None,
    metadata: Optional[Dict] = None,
):
    """
    Save pipeline checkpoint for recovery.
    
    Args:
        state: Current ProfessorState
        path: Path to save checkpoint
        node_name: Name of node that just completed
        metadata: Additional metadata to save
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    
    # Filter to serializable fields only
    serializable_state = {
        k: v for k, v in state.items()
        if _is_serializable(v)
    }
    
    # Add metadata
    checkpoint = {
        "state": serializable_state,
        "metadata": {
            "node_completed": node_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "1.0",
            **(metadata or {}),
        }
    }
    
    with open(path, "w") as f:
        json.dump(checkpoint, f, indent=2)
    
    logger.info(f"Checkpoint saved to {path}")


def load_checkpoint(path: str) -> Dict:
    """
    Load pipeline checkpoint for recovery.
    
    Returns:
        Dict with "state" and "metadata" keys
    """
    with open(path) as f:
        return json.load(f)


def get_latest_checkpoint(session_id: str) -> Optional[str]:
    """
    Get path to latest checkpoint for session.
    
    Returns:
        Path to latest checkpoint, or None if no checkpoints exist
    """
    checkpoint_dir = f"outputs/{session_id}/checkpoints"
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoints = [
        f for f in os.listdir(checkpoint_dir)
        if f.endswith(".json")
    ]
    
    if not checkpoints:
        return None
    
    # Return most recent (sorted by filename which includes timestamp)
    checkpoints.sort()
    return os.path.join(checkpoint_dir, checkpoints[-1])


def list_checkpoints(session_id: str) -> List[str]:
    """
    List all checkpoints for session.
    
    Returns:
        List of checkpoint paths, sorted by timestamp
    """
    checkpoint_dir = f"outputs/{session_id}/checkpoints"
    if not os.path.exists(checkpoint_dir):
        return []
    
    checkpoints = [
        os.path.join(checkpoint_dir, f)
        for f in os.listdir(checkpoint_dir)
        if f.endswith(".json")
    ]
    
    checkpoints.sort()
    return checkpoints


def save_node_checkpoint(
    state: Dict[str, Any],
    session_id: str,
    node_name: str,
):
    """
    Save checkpoint after node completion.
    
    Args:
        state: Current ProfessorState
        session_id: Session ID
        node_name: Name of completed node
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{node_name}.json"
    path = f"outputs/{session_id}/checkpoints/{filename}"
    
    save_checkpoint(state, path, node_name=node_name)


def load_last_checkpoint(session_id: str) -> Optional[Dict]:
    """
    Load last checkpoint for session.
    
    Returns:
        Checkpoint dict with "state" and "metadata", or None
    """
    path = get_latest_checkpoint(session_id)
    if path:
        logger.info(f"Loading checkpoint: {path}")
        return load_checkpoint(path)
    return None
