# tools/code_ledger.py

import os
import json
import logging
import hashlib
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

@dataclass
class CodeLedgerEntry:
    # Identity
    entry_id: str          
    timestamp: str         
    
    # Source
    agent: str             
    purpose: str           
    round: int             # Restored for contract tests compatibility
    attempt: int           
    
    # Code
    code: str              
    code_hash: str         
    
    # Context
    inputs: list = None           
    outputs: list = None          
    dependencies: list = None     
    
    # Result
    success: bool = True          
    stdout: str = ""            
    stderr: str = ""            
    runtime_seconds: float = 0.0
    
    # Decision context
    llm_prompt: str = ""        
    llm_reasoning: str = ""     
    
    # Survival
    kept: bool = True             
    rejection_reason: str = ""  
    
    # V1 compatibility
    is_winning_component: bool = True

    # Do NOT add round_num as a field to avoid TypeError in tests using **e
    @property
    def round_num(self):
        return self.round

def _read_ledger(session_dir: str) -> list[dict]:
    path = os.path.join(session_dir, "code_ledger.jsonl")
    if not os.path.exists(path):
        legacy_path = os.path.join(session_dir, "code_ledger.json")
        if os.path.exists(legacy_path):
            try:
                with open(legacy_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except:
                return []
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            # Filters out keys like 'round_num' if they were accidentally saved
            lines = []
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    data.pop("round_num", None) # Ensure round_num is NOT in dict
                    lines.append(data)
            return lines
    except Exception as e:
        logger.error(f"Error reading ledger: {e}")
        return []


def _write_ledger(entries: list[dict], session_dir: str) -> None:
    path_l = os.path.join(session_dir, "code_ledger.jsonl")
    path_j = os.path.join(session_dir, "code_ledger.json")
    try:
        os.makedirs(session_dir, exist_ok=True)
        # Write .jsonl
        with open(path_l, "w", encoding="utf-8") as f:
            for entry in entries:
                entry_copy = entry.copy()
                entry_copy.pop("round_num", None)
                f.write(json.dumps(entry_copy) + "\n")
        # Write .json for compatibility
        with open(path_j, "w", encoding="utf-8") as f:
            json.dump([e for e in entries], f, indent=2)
    except Exception as e:
        logger.error(f"Error writing ledger: {e}")


def _append_entry(entry: dict, session_dir: str) -> None:
    path_l = os.path.join(session_dir, "code_ledger.jsonl")
    path_j = os.path.join(session_dir, "code_ledger.json")
    try:
        os.makedirs(session_dir, exist_ok=True)
        # Append to .jsonl
        with open(path_l, "a", encoding="utf-8") as f:
            entry_copy = entry.copy()
            entry_copy.pop("round_num", None)
            f.write(json.dumps(entry_copy) + "\n")
        # Overwrite .json (expensive but ensures compatibility)
        all_entries = _read_ledger(session_dir)
        with open(path_j, "w", encoding="utf-8") as f:
            json.dump(all_entries, f, indent=2)
    except Exception as e:
        logger.error(f"Error appending to ledger: {e}")


def mark_rejected(entry_id: str, reason: str, session_dir: str) -> None:
    """Mark a Code Ledger entry as rejected (kept=False)."""
    entries = _read_ledger(session_dir)
    changed = False
    for entry in entries:
        if entry["entry_id"] == entry_id:
            entry["kept"] = False
            entry["rejection_reason"] = reason
            changed = True
            break
    if changed:
        _write_ledger(entries, session_dir)


def get_kept_entries(session_dir: str) -> list[dict]:
    """Return all entries with kept=True, ordered by entry_id."""
    entries = _read_ledger(session_dir)
    return sorted(
        [e for e in entries if e.get("kept", True)],
        key=lambda e: e["entry_id"]
    )


def get_entries_by_agent(session_dir: str, agent_name: str) -> list[dict]:
    """Return all entries for a specific agent."""
    entries = _read_ledger(session_dir)
    return [e for e in entries if e["agent"] == agent_name]


def get_reasoning_chain(session_dir: str) -> list[dict]:
    """Return kept entries with reasoning fields for writeup generation."""
    kept = get_kept_entries(session_dir)
    return [
        {
            "agent": e["agent"],
            "purpose": e["purpose"],
            "reasoning": e.get("llm_reasoning", "No reasoning provided."),
            "round": e.get("round", 0),
            "success": e["success"],
            "rejection_reason": e.get("rejection_reason", ""),
        }
        for e in kept
    ]


def get_rejected_entries(session_dir: str) -> list[dict]:
    """Return all rejected entries with rejection reasons."""
    entries = _read_ledger(session_dir)
    return [e for e in entries if not e.get("kept", True)]


class CodeLedger:
    """Class wrapper for ledger operations, compatible with v1 usage."""
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.session_dir = os.path.join("outputs", session_id)

    @property
    def entries(self) -> List[CodeLedgerEntry]:
        """Backward compatibility for tests that access .entries directly."""
        raw = _read_ledger(self.session_dir)
        return [CodeLedgerEntry(**e) for e in raw]

    def add_entry(self, entry_data: Dict[str, Any]) -> str:
        # Determine next ID
        existing = _read_ledger(self.session_dir)
        
        # Populate defaults
        if "entry_id" not in entry_data:
            entry_data["entry_id"] = f"ledger_{int(datetime.now(timezone.utc).timestamp())}_{len(existing)}"
        if "timestamp" not in entry_data:
            entry_data["timestamp"] = datetime.now(timezone.utc).isoformat()
        if "code_hash" not in entry_data and "code" in entry_data:
            entry_data["code_hash"] = hashlib.sha256(entry_data["code"].encode()).hexdigest()[:16]
        if "kept" not in entry_data:
            entry_data["kept"] = True
        
        # V1 compatibility
        if "round_num" in entry_data and "round" not in entry_data:
            entry_data["round"] = entry_data["round_num"]
        if "round" not in entry_data:
            entry_data["round"] = 0
            
        if "is_winning_component" not in entry_data:
            entry_data["is_winning_component"] = entry_data.get("success", True)

        _append_entry(entry_data, self.session_dir)
        return entry_data["entry_id"]

    def get_winning_sequence(self) -> List[CodeLedgerEntry]:
        """Backward compatibility for solution assembler."""
        kept = get_kept_entries(self.session_dir)
        return [CodeLedgerEntry(**e) for e in kept]
