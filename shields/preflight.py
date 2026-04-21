# shields/preflight.py

import os
import logging
import polars as pl
from typing import Dict, List, Any, Optional, Tuple
from core.state import ProfessorState

logger = logging.getLogger(__name__)

# ── Pre-flight Audits ────────────────────────────────────────────────────────

def _audit_files(state: ProfessorState) -> List[str]:
    """Verify all required files exist and are readable."""
    errors = []
    required = {
        "train": state.raw_data_path,
        "test": state.test_data_path,
        "sample_submission": state.sample_submission_path
    }
    
    for label, path in required.items():
        if not path:
            errors.append(f"Missing path for {label}")
            continue
        if not os.path.exists(path):
            errors.append(f"File not found: {path} ({label})")
        elif os.path.getsize(path) == 0:
            errors.append(f"File is empty: {path} ({label})")
            
    return errors

def _profile_columns(path: str) -> Dict[str, List[str]]:
    """Detect special media types in columns."""
    if not path or not os.path.exists(path):
        return {}
        
    try:
        df = pl.scan_csv(path).collect(streaming=True).head(100)
        profiles = {
            "nlp": [],
            "image": [],
            "audio": [],
            "json": []
        }
        
        for col in df.columns:
            # Simple heuristic detection
            sample = [str(x) for x in df[col].head(5).to_list() if x is not None]
            if not sample: continue
            
            combined = " ".join(sample).lower()
            
            # NLP: Long text
            if any(len(s) > 100 for s in sample):
                profiles["nlp"].append(col)
            # Image: common extensions
            elif any(ext in combined for ext in [".jpg", ".png", ".jpeg"]):
                profiles["image"].append(col)
            # Audio: common extensions
            elif any(ext in combined for ext in [".wav", ".mp3", ".flac"]):
                profiles["audio"].append(col)
            # JSON: starts with { or [
            elif any(s.strip().startswith(("{", "[")) for s in sample):
                profiles["json"].append(col)
                
        return {k: v for k, v in profiles.items() if v}
    except Exception as e:
        logger.warning(f"Column profiling failed for {path}: {e}")
        return {}

def run_preflight_checks(state: Any) -> Any:
    """LangGraph node: Shield 6."""
    logger.info("[Shield 6] Running Pre-flight Checks...")
    
    warnings = []
    
    # 1. File Inventory
    file_errors = _audit_files(state)
    if file_errors:
        warnings.extend(file_errors)
        
    # 2. Column Profiling
    train_profile = _profile_columns(state.raw_data_path)
    if train_profile:
        for k, v in train_profile.items():
            warnings.append(f"Detected {k.upper()} columns in train: {v}")
            
    # 3. Format Verification (Stub: Check if train/test have common columns)
    try:
        if state.raw_data_path and state.test_data_path:
            train_cols = set(pl.scan_csv(state.raw_data_path).columns)
            test_cols = set(pl.scan_csv(state.test_data_path).columns)
            
            # Target must be in train but not test (usually)
            if state.target_col and state.target_col in test_cols:
                warnings.append(f"TARGET LEAK: '{state.target_col}' found in test data!")
                
    except Exception as e:
        warnings.append(f"Format verification failed: {e}")

    # Final verdict
    passed = len([w for w in warnings if "not found" in w or "empty" in w]) == 0
    
    return ProfessorState.validated_update(state, "preflight", {
        "preflight_passed": passed,
        "preflight_warnings": state.get("preflight_warnings", []) + warnings
    })
