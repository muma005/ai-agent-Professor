import os
import json
from datetime import datetime


def create_session(competition_name: str, base_dir: str = "outputs") -> str:
    """Create a new session directory with standard structure."""
    session_id = f"{competition_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    session_dir = os.path.join(base_dir, session_id)
    
    # Create directory structure
    os.makedirs(os.path.join(session_dir, "eda_plots"), exist_ok=True)
    os.makedirs(os.path.join(session_dir, "narrative_plots"), exist_ok=True)
    os.makedirs(os.path.join(session_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(session_dir, "submission_backups"), exist_ok=True)
    
    # Initialize empty code ledger
    open(os.path.join(session_dir, "code_ledger.jsonl"), "w").close()
    
    # Save session metadata
    metadata = {
        "session_id": session_id,
        "competition": competition_name,
        "created_at": datetime.utcnow().isoformat(),
        "professor_version": "2.0",
        "status": "running",
    }
    with open(os.path.join(session_dir, "session_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    return session_id


def list_sessions(base_dir: str = "outputs") -> list[dict]:
    """List all sessions with their status."""
    sessions = []
    if not os.path.exists(base_dir):
        return sessions
    
    for dirname in sorted(os.listdir(base_dir), reverse=True):
        meta_path = os.path.join(base_dir, dirname, "session_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            meta["dir"] = os.path.join(base_dir, dirname)
            sessions.append(meta)
    
    return sessions
