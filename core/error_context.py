# core/error_context.py
"""
Error context management for debugging and recovery.

Saves error context to disk for:
1. Debugging failures
2. Resuming from failures
3. Analyzing failure patterns
"""

import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
import logging
import traceback as tb_module

logger = logging.getLogger(__name__)


class ErrorContextManager:
    """
    Manages error context for debugging and recovery.
    """
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.context_path = f"outputs/{session_id}/error_context.json"
        self.context = {
            "session_id": session_id,
            "start_time": None,
            "end_time": None,
            "status": "running",  # running, success, failed
            "nodes_completed": [],
            "errors": [],
            "state_snapshots": [],
        }
    
    def start(self):
        """Mark pipeline start."""
        self.context["start_time"] = datetime.now(timezone.utc).isoformat()
        self._save()
    
    def complete_node(self, node_name: str, state_snapshot: Optional[Dict] = None):
        """Mark node completion."""
        self.context["nodes_completed"].append({
            "node": node_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        
        if state_snapshot:
            # Save serializable subset
            snapshot = {k: v for k, v in state_snapshot.items() if self._is_serializable(v)}
            self.context["state_snapshots"].append({
                "node": node_name,
                "state": snapshot,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        
        self._save()
    
    def record_error(
        self,
        error: Exception,
        node_name: Optional[str] = None,
        traceback_str: Optional[str] = None,
    ):
        """Record error."""
        self.context["errors"].append({
            "node": node_name,
            "error": str(error),
            "error_type": type(error).__name__,
            "traceback": traceback_str,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self._save()
    
    def fail(self):
        """Mark pipeline failure."""
        self.context["status"] = "failed"
        self.context["end_time"] = datetime.now(timezone.utc).isoformat()
        self._save()
    
    def success(self):
        """Mark pipeline success."""
        self.context["status"] = "success"
        self.context["end_time"] = datetime.now(timezone.utc).isoformat()
        self._save()
    
    def _is_serializable(self, value: Any) -> bool:
        """Check if value is JSON-serializable."""
        try:
            json.dumps(value)
            return True
        except (TypeError, ValueError):
            return False
    
    def _save(self):
        """Save context to disk."""
        os.makedirs(os.path.dirname(self.context_path) or ".", exist_ok=True)
        with open(self.context_path, "w") as f:
            json.dump(self.context, f, indent=2)
    
    def load(self) -> Dict:
        """Load context from disk."""
        if os.path.exists(self.context_path):
            with open(self.context_path) as f:
                return json.load(f)
        return self.context


def get_error_context(session_id: str) -> ErrorContextManager:
    """Get or create error context manager for session."""
    return ErrorContextManager(session_id)


def save_error_context(session_id: str, context: Dict):
    """Save error context to disk."""
    os.makedirs(f"outputs/{session_id}", exist_ok=True)
    path = f"outputs/{session_id}/error_context.json"
    with open(path, "w") as f:
        json.dump(context, f, indent=2)


def load_error_context(session_id: str) -> Optional[Dict]:
    """Load error context from disk."""
    path = f"outputs/{session_id}/error_context.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None
