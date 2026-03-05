# core/lineage.py

import os
import json
from datetime import datetime
from typing import Any


def log_event(
    session_id: str,
    agent: str,
    action: str,
    keys_read: list = None,
    keys_written: list = None,
    values_changed: dict = None,
    notes: str = ""
) -> None:
    """
    Append a single event to the session's lineage log.
    Append-only. Never reads or rewrites existing entries.

    Each entry: timestamp, agent, action, keys_read,
                keys_written, values_changed, notes.
    One file per session: outputs/{session_id}/logs/lineage.jsonl
    """
    log_dir  = f"outputs/{session_id}/logs"
    log_path = f"{log_dir}/lineage.jsonl"
    os.makedirs(log_dir, exist_ok=True)

    entry = {
        "timestamp":      datetime.utcnow().isoformat(),
        "session_id":     session_id,
        "agent":          agent,
        "action":         action,
        "keys_read":      keys_read or [],
        "keys_written":   keys_written or [],
        "values_changed": _sanitize_values(values_changed or {}),
        "notes":          notes,
    }

    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _sanitize_values(values: dict) -> dict:
    """Ensure all values are JSON-serializable."""
    sanitized = {}
    for k, v in values.items():
        if isinstance(v, (str, int, float, bool, type(None))):
            sanitized[k] = v
        elif isinstance(v, (list, tuple)):
            sanitized[k] = [str(x) for x in v]
        else:
            sanitized[k] = str(v)
    return sanitized


def read_lineage(session_id: str) -> list:
    """Read all lineage entries for a session. Returns list of dicts."""
    log_path = f"outputs/{session_id}/logs/lineage.jsonl"
    if not os.path.exists(log_path):
        return []
    with open(log_path) as f:
        return [json.loads(line) for line in f if line.strip()]


def print_lineage(session_id: str) -> None:
    """Print a human-readable lineage trace for a session."""
    entries = read_lineage(session_id)
    if not entries:
        print(f"No lineage entries for session: {session_id}")
        return
    print(f"\n-- Lineage: {session_id} ({len(entries)} events) --")
    for e in entries:
        ts    = e["timestamp"][11:19]  # HH:MM:SS only
        wrote = ", ".join(e["keys_written"]) or "--"
        print(f"  {ts} [{e['agent']}] {e['action']} -> wrote: {wrote}")
    print()
