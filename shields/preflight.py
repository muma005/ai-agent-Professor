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

# ── Sub-function 1: _inventory_data_files ───────────────────────────────────

def _inventory_data_files(data_dir: str) -> Tuple[List[Dict], float, List[Dict]]:
    """Walk the data directory, detect formats, and check RAM boundaries."""
    files = []
    warnings = []
    if not os.path.exists(data_dir):
        return [], 0.0, [{"type": "missing_dir", "description": f"Data directory not found: {data_dir}"}]

    format_map = {
        ".csv": "csv", ".tsv": "csv", ".json": "json", ".parquet": "parquet",
        ".npy": "numpy", ".npz": "numpy", ".jpg": "image", ".jpeg": "image",
        ".png": "image", ".bmp": "image", ".gif": "image", ".wav": "audio",
        ".mp3": "audio", ".flac": "audio", ".txt": "text", ".zip": "archive",
        ".tar": "archive", ".gz": "archive"
    }
    
    use_keywords = [
        "train.csv", "train.parquet", "train.tsv",
        "test.csv", "test.parquet", "test.tsv",
        "sample_submission.csv", "sample_submission.parquet"
    ]

    total_size_mb = 0.0
    for filename in os.listdir(data_dir):
        path = os.path.join(data_dir, filename)
        if not os.path.isfile(path): continue
        
        size_mb = os.path.getsize(path) / (1024 * 1024)
        total_size_mb += size_mb
        ext = os.path.splitext(filename)[1].lower()
        fmt = format_map.get(ext, "other")
        
        will_use = any(kw == filename.lower() for kw in use_keywords)
        
        files.append({
            "name": filename,
            "size_mb": round(size_mb, 3),
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
        available_ram_mb = 16384 # Default 16GB
        
    if total_size_mb > available_ram_mb * 0.7:
        warnings.append({
            "type": "large_dataset",
            "description": f"Total data {total_size_mb:.0f}MB exceeds 70% of available RAM ({available_ram_mb:.0f}MB). Recommend chunked loading or cloud compute."
        })

    return files, max(0.001, round(total_size_mb, 3)), warnings

# ── Sub-function 2: _profile_columns ─────────────────────────────────────────

def _profile_columns(file_path: str, n_rows: int = 100) -> List[Dict]:
    """Profile first (and last) N rows to detect modalities and types."""
    if not os.path.exists(file_path): return []
    
    ext = os.path.splitext(file_path)[1].lower()
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    
    try:
        if ext in (".csv", ".tsv"):
            df_head = pl.read_csv(file_path, n_rows=n_rows, separator="\t" if ext == ".tsv" else ",")
            # Tail sampling for files < 1GB
            df_tail = pl.read_csv(file_path).tail(n_rows) if file_size_mb < 1024 else None
        elif ext == ".parquet":
            df_head = pl.read_parquet(file_path, n_rows=n_rows)
            # Tail sampling
            df_tail = pl.read_parquet(file_path).tail(n_rows) if file_size_mb < 1024 else None
        else:
            return []
    except Exception as e:
        logger.warning(f"Profiling failed for {file_path}: {e}")
        return []

    def get_column_profile(df, col):
        n_unique = df[col].n_unique()
        null_pct = round(df[col].null_count() / len(df) * 100, 1)
        flags = []
        
        if df[col].dtype in (pl.Utf8, pl.String):
            avg_len = df[col].str.len_chars().mean()
            if avg_len and avg_len > 50:
                flags.append({"type": "possible_nlp", "description": f"Avg string length {avg_len:.0f} chars — may require NLP processing"})
            
            sample = df[col].drop_nulls().head(10).to_list()
            if sample:
                if sum(1 for s in sample if re.search(r"\.(jpg|jpeg|png|bmp|gif|tiff|webp)$", str(s), re.I)) > 5:
                    flags.append({"type": "image_paths", "description": "Column contains image file paths"})
                if sum(1 for s in sample if re.search(r"\.(wav|mp3|flac|ogg|aac)$", str(s), re.I)) > 5:
                    flags.append({"type": "audio_paths", "description": "Column contains audio file paths"})
                if sum(1 for s in sample if re.search(r"^[\[{]", str(s).strip())) > 5:
                    flags.append({"type": "nested_json", "description": "Column contains JSON/nested structures"})
                if sum(1 for s in sample if "|" in str(s)) > 5:
                    flags.append({"type": "pipe_delimited", "description": "Column contains pipe-delimited lists — possible multi-label"})
                if sum(1 for s in sample if re.search(r"^\d{4}-\d{2}-\d{2}", str(s))) > 5:
                    flags.append({"type": "datetime_candidate", "description": "Column contains ISO date strings — parse as datetime"})
            
            if n_unique > 90: # High cardinality in 100 rows
                flags.append({"type": "high_cardinality", "description": f"{n_unique} unique values in {len(df)} rows — high cardinality categorical"})

        if n_unique == 1:
            flags.append({"type": "constant", "description": "Only 1 unique value — drop candidate"})
        if null_pct > 80:
            flags.append({"type": "mostly_null", "description": f"{null_pct}% null — consider dropping or engineering missingness feature"})
            
        return {
            "name": col,
            "dtype": str(df[col].dtype),
            "n_unique_in_sample": n_unique,
            "null_pct_in_sample": null_pct,
            "flags": flags
        }

    profiles = []
    for col in df_head.columns:
        head_prof = get_column_profile(df_head, col)
        if df_tail is not None:
            tail_prof = get_column_profile(df_tail, col)
            # Merge flags
            existing_flag_types = {f["type"] for f in head_prof["flags"]}
            for f in tail_prof["flags"]:
                if f["type"] not in existing_flag_types:
                    head_prof["flags"].append(f)
        profiles.append(head_prof)
        
    return profiles

# ── Sub-function 3: _verify_submission_format ────────────────────────────────

def _verify_submission_format(data_dir: str) -> Dict:
    """Detect and verify sample submission format."""
    patterns = [r"sample_submission\.*", r"sampleSubmission\.*", r"sample_sub\.*"]
    for filename in os.listdir(data_dir):
        if any(re.search(p, filename, re.I) for p in patterns):
            path = os.path.join(data_dir, filename)
            ext = os.path.splitext(filename)[1].lower()
            if ext == ".json":
                return {"format": "json", "compatible": False, "issues": ["JSON submission format — Professor v2 outputs CSV only"]}
            
            try:
                df = pl.read_csv(path) if ext in (".csv", ".tsv") else pl.read_parquet(path)
                # Value type inference
                val_types = {}
                for col in df.columns:
                    if col.lower() in ("id", "index", "row_id"): continue
                    s = df[col].drop_nulls()
                    if s.dtype in (pl.Float32, pl.Float64):
                        val_types[col] = "probability" if s.min() >= 0 and s.max() <= 1 else "continuous"
                    elif s.dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64):
                        val_types[col] = "binary_class" if s.n_unique() <= 2 else "multiclass"
                
                return {
                    "format": ext.strip("."),
                    "columns": df.columns,
                    "n_rows": len(df),
                    "value_types": val_types,
                    "compatible": True
                }
            except:
                pass
                
    return {"format": "unknown", "compatible": True, "issues": ["No sample submission found — cannot verify output format"]}

# ── Sub-function 4: _detect_target_type ──────────────────────────────────────

def _detect_target_type(file_path: str, target_col: str) -> str:
    """Determine target type based on sample distribution."""
    if not target_col or not os.path.exists(file_path): return "unknown"
    
    try:
        ext = os.path.splitext(file_path)[1].lower()
        if ext in (".csv", ".tsv"):
            df = pl.read_csv(file_path, n_rows=1000)
        else:
            df = pl.read_parquet(file_path, n_rows=1000)
            
        if target_col not in df.columns: return "unknown"
        
        s = df[target_col]
        n_unique = s.n_unique()
        
        if s.dtype in (pl.Float32, pl.Float64):
            return "binary" if n_unique == 2 else "regression"
        if s.dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
            if n_unique == 2: return "binary"
            if 3 <= n_unique <= 30: return "multiclass"
            return "regression"
        if s.dtype in (pl.Utf8, pl.String, pl.Boolean):
            return "binary" if n_unique == 2 else "multiclass"
            
        return "unknown"
    except:
        return "unknown"

# ── Agent Node ───────────────────────────────────────────────────────────────

def run_preflight_checks(state: ProfessorState) -> ProfessorState:
    """
    Shield 6: Comprehensive Pre-flight Audit.
    """
    logger.info("[Shield 6] Running Pre-flight Checks...")
    
    raw_data_path = state.get("raw_data_path")
    if not raw_data_path:
        return state.validated_update(state, AGENT_NAME, {"preflight_passed": False, "preflight_warnings": ["raw_data_path not set"]})

    raw_dir = os.path.dirname(raw_data_path) or "."
    
    # 1. Inventory
    inventory, total_mb, inv_warnings = _inventory_data_files(raw_dir)
    
    # 2. Profiling
    profiles = _profile_columns(raw_data_path)
    
    # 3. Submission Format
    sub_format = _verify_submission_format(raw_dir)
    
    # 4. Target Type
    target_type = _detect_target_type(raw_data_path, state.get("target_col", ""))
    
    # 5. Boundaries & Categorization
    unsupported = []
    blocking_warnings = []
    for p in profiles:
        for f in p["flags"]:
            if f["type"] == "image_paths": unsupported.append("image")
            if f["type"] == "audio_paths": unsupported.append("audio")
            if f["type"] == "possible_nlp" and len(profiles) == 1: unsupported.append("nlp")

    if not sub_format["compatible"]: 
        blocking_warnings.append(sub_format["issues"][0])
    if total_mb > 10000: 
        blocking_warnings.append("Dataset exceeds 10GB boundary for current compute.")

    all_warnings = inv_warnings + [f for p in profiles for f in p["flags"]]
    passed = len(blocking_warnings) == 0
    
    # 6. Report Assembly
    msg = f"🚀 PRE-FLIGHT REPORT\n\n📁 Total Size: {total_mb}MB\n📊 Columns: {len(profiles)}\n🎯 Target Type: {target_type}\n📤 Format: {sub_format['format']}\n✅ Passed: {passed}"
    if blocking_warnings:
        msg += f"\n\n🛑 BLOCKERS:\n- " + "\n- ".join(blocking_warnings)
        emit_to_operator(msg, level="GATE")
    else:
        emit_to_operator(msg, level="CHECKPOINT")

    return ProfessorState.validated_update(state, AGENT_NAME, {
        "preflight_passed": passed,
        "preflight_warnings": all_warnings,
        "preflight_data_files": inventory,
        "preflight_data_size_mb": total_mb,
        "preflight_submission_format": sub_format,
        "preflight_target_type": target_type,
        "preflight_unsupported_modalities": list(set(unsupported))
    })
