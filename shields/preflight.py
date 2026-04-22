# shields/preflight.py

import os
import re
import logging
import psutil
import polars as pl
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from core.state import ProfessorState
from tools.operator_channel import emit_to_operator

logger = logging.getLogger(__name__)

AGENT_NAME = "preflight"

# ── Helpers ──────────────────────────────────────────────────────────────────

def _inventory_data_files(data_dir: str) -> Tuple[List[Dict], float, List[Dict]]:
    """Walk data dir, detect formats, check RAM limits."""
    files = []
    warnings = []
    if not os.path.exists(data_dir):
        return [], 0.0, [{"type": "missing_dir", "description": f"Directory not found: {data_dir}"}]

    format_map = {
        ".csv": "csv", ".tsv": "csv", ".json": "json", ".parquet": "parquet",
        ".npy": "numpy", ".npz": "numpy", ".jpg": "image", ".jpeg": "image",
        ".png": "image", ".bmp": "image", ".gif": "image", ".wav": "audio",
        ".mp3": "audio", ".flac": "audio", ".txt": "text", ".zip": "archive",
        ".tar": "archive", ".gz": "archive"
    }
    
    use_patterns = [
        r"train\.(csv|parquet|tsv)", r"test\.(csv|parquet|tsv)", 
        r"sample_submission\.(csv|parquet|tsv)"
    ]

    total_size_mb = 0.0
    for filename in os.listdir(data_dir):
        path = os.path.join(data_dir, filename)
        if not os.path.isfile(path): continue
        
        size_mb = os.path.getsize(path) / (1024 * 1024)
        total_size_mb += size_mb
        ext = os.path.splitext(filename)[1].lower()
        fmt = format_map.get(ext, "other")
        
        will_use = any(re.search(p, filename, re.I) for p in use_patterns)
        
        files.append({
            "name": filename,
            "size_mb": round(size_mb, 1),
            "format": fmt,
            "will_use": will_use
        })
        
        if size_mb > 2048:
            warnings.append({
                "type": "large_file",
                "description": f"File '{filename}' is {size_mb:.0f}MB. Will use pl.scan_csv() for lazy evaluation."
            })

    # RAM Check
    try:
        available_ram_mb = psutil.virtual_memory().total / (1024 * 1024)
    except:
        available_ram_mb = 16384
        
    if total_size_mb > available_ram_mb * 0.7:
        warnings.append({
            "type": "large_dataset",
            "description": f"Total data {total_size_mb:.0f}MB exceeds 70% of available RAM ({available_ram_mb:.0f}MB)."
        })

    return files, round(total_size_mb, 1), warnings

def _profile_columns(file_path: str, n_rows: int = 100) -> List[Dict]:
    """Load sample, detect NLP, image paths, JSON, etc."""
    if not os.path.exists(file_path): return []
    
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext in (".csv", ".tsv"):
            df = pl.read_csv(file_path, n_rows=n_rows, separator="\t" if ext == ".tsv" else ",")
        elif ext == ".parquet":
            df = pl.read_parquet(file_path, n_rows=n_rows)
        else:
            return []
    except Exception as e:
        logger.warning(f"Profiling failed for {file_path}: {e}")
        return []

    profiles = []
    for col in df.columns:
        n_unique = df[col].n_unique()
        null_pct = round(df[col].null_count() / len(df) * 100, 1)
        profile = {
            "name": col,
            "dtype": str(df[col].dtype),
            "n_unique_in_sample": n_unique,
            "null_pct_in_sample": null_pct,
            "flags": []
        }
        
        # Flags
        if df[col].dtype in (pl.Utf8, pl.String):
            avg_len = df[col].str.len_chars().mean()
            if avg_len and avg_len > 50:
                profile["flags"].append({"type": "possible_nlp", "description": f"Avg string length {avg_len:.0f} chars"})
            
            sample = df[col].drop_nulls().head(10).to_list()
            if sample:
                # Media/JSON patterns
                if sum(1 for s in sample if re.search(r"\.(jpg|jpeg|png|bmp|gif|webp)$", str(s), re.I)) > 5:
                    profile["flags"].append({"type": "image_paths", "description": "Column contains image file paths"})
                if sum(1 for s in sample if re.search(r"^[\[{]", str(s).strip())) > 5:
                    profile["flags"].append({"type": "nested_json", "description": "Column contains JSON structures"})
                if sum(1 for s in sample if "|" in str(s)) > 5:
                    profile["flags"].append({"type": "pipe_delimited", "description": "Column contains pipe-delimited lists"})
                if sum(1 for s in sample if re.search(r"^\d{4}-\d{2}-\d{2}", str(s))) > 5:
                    profile["flags"].append({"type": "datetime_candidate", "description": "Column contains ISO date strings"})
        
        if n_unique == 1:
            profile["flags"].append({"type": "constant", "description": "Only 1 unique value"})
        if null_pct > 80:
            profile["flags"].append({"type": "mostly_null", "description": f"{null_pct}% null"})
            
        profiles.append(profile)
        
    return profiles

def _verify_submission_format(data_dir: str) -> Dict:
    """Detect format and compatibility of sample submission."""
    for filename in os.listdir(data_dir):
        if re.search(r"sample_submission\.(csv|parquet|json)", filename, re.I):
            path = os.path.join(data_dir, filename)
            ext = os.path.splitext(filename)[1].lower()
            if ext == ".json":
                return {"format": "json", "compatible": False, "issues": ["JSON submission format not supported"]}
            
            try:
                df = pl.read_csv(path) if ext == ".csv" else pl.read_parquet(path)
                return {
                    "format": ext.strip("."),
                    "columns": df.columns,
                    "n_rows": len(df),
                    "compatible": True
                }
            except:
                pass
                
    return {"format": "unknown", "compatible": True, "issues": ["No sample submission found"]}

# ── Agent Node ───────────────────────────────────────────────────────────────

def run_preflight_checks(state: ProfessorState) -> ProfessorState:
    """
    Shield 6: Pre-flight Checks.
    Profiles data, detects modalities, and checks resource boundaries.
    """
    logger.info("[Shield 6] Running Pre-flight Checks...")
    
    raw_dir = os.path.dirname(state.get("raw_data_path", ".")) or "."
    
    # 1. Inventory
    inventory, total_mb, inv_warnings = _inventory_data_files(raw_dir)
    
    # 2. Profiling (Targeting train.csv)
    profiles = []
    if state.raw_data_path and os.path.exists(state.raw_data_path):
        profiles = _profile_columns(state.raw_data_path)
    
    # 3. Submission Format
    sub_format = _verify_submission_format(raw_dir)
    
    # 4. Capability Boundaries
    blocking = []
    if not sub_format["compatible"]: blocking.append("unsupported_submission_format")
    if total_mb > 10000: blocking.append("extreme_large_dataset")
    
    unsupported = []
    for p in profiles:
        for f in p["flags"]:
            if f["type"] == "image_paths": unsupported.append("image")
            if f["type"] == "audio_paths": unsupported.append("audio")

    # 5. Report & Emit
    all_warnings = inv_warnings + [f for p in profiles for f in p["flags"]]
    passed = len(blocking) == 0
    
    msg = f"🚀 PRE-FLIGHT REPORT\n\n📁 Total Size: {total_mb}MB\n✅ Passed: {passed}"
    emit_to_operator(msg, level="CHECKPOINT")

    return ProfessorState.validated_update(state, AGENT_NAME, {
        "preflight_passed": passed,
        "preflight_warnings": all_warnings,
        "preflight_data_size_mb": total_mb,
        "preflight_submission_format": sub_format,
        "preflight_unsupported_modalities": list(set(unsupported))
    })
