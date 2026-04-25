# tools/code_ledger.py

import os
import json
import logging
import hashlib
from datetime import datetime, timezone
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

@dataclass
class CodeLedgerEntry:
    entry_id: str
    timestamp: str
    agent: str
    purpose: str
    round: int
    attempt: int
    code: str
    code_hash: str
    success: bool
    stdout: str = ""
    stderr: str = ""
    runtime_seconds: float = 0.0
    llm_prompt: str = ""
    llm_reasoning: str = ""
    is_winning_component: bool = False # Flag for solution assembly

class CodeLedger:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.output_dir = os.path.join("outputs", session_id)
        self.ledger_path = os.path.join(self.output_dir, "code_ledger.json")
        self.entries: List[CodeLedgerEntry] = []
        self._load()

    def add_entry(self, entry_data: Dict[str, Any]):
        """Creates and adds a new entry to the ledger."""
        if "entry_id" not in entry_data:
            entry_data["entry_id"] = f"ledger_{int(datetime.now(timezone.utc).timestamp())}_{len(self.entries)}"
        if "timestamp" not in entry_data:
            entry_data["timestamp"] = datetime.now(timezone.utc).isoformat()
        
        # Determine if this code is a 'winning' piece (successful final round)
        # Simplified: any successful round-final attempt is part of the solution
        if entry_data.get("success"):
            # Mark as potential winning component
            entry_data["is_winning_component"] = True
            
        entry = CodeLedgerEntry(**entry_data)
        self.entries.append(entry)
        self.save()
        return entry.entry_id

    def get_winning_sequence(self) -> List[CodeLedgerEntry]:
        """Returns the ordered list of code blocks that form the final solution."""
        # Order by timestamp, keep only those marked as winning components
        # (Filtering out older versions of the same component if they were refined)
        winning = [e for e in self.entries if e.is_winning_component]
        # Keep only the LATEST success per (agent, purpose)
        latest_wins = {}
        for e in winning:
            key = (e.agent, e.purpose)
            latest_wins[key] = e
            
        return sorted(latest_wins.values(), key=lambda x: x.timestamp)

    def save(self):
        os.makedirs(self.output_dir, exist_ok=True)
        data = [asdict(e) for e in self.entries]
        with open(self.ledger_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self):
        if os.path.exists(self.ledger_path):
            try:
                with open(self.ledger_path, "r") as f:
                    data = json.load(f)
                    self.entries = [CodeLedgerEntry(**e) for e in data]
            except Exception as e:
                logger.error(f"Failed to load ledger: {e}")
