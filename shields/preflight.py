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
from graph.depth_router import classify_pipeline_depth
from guards.agent_retry import with_agent_retry
from tools.performance_monitor import timed_node

logger = logging.getLogger(__name__)

AGENT_NAME = "preflight"

# ── Helpers ──────────────────────────────────────────────────────────────────

def _estimate_row_count(file_path: str) -> int:
    """Estimate row count without loading the full file."""
    if not os.path.exists(file_path): return 0
    
    if file_path.endswith(".parquet"):
        try:
            return pl.scan_parquet(file_path).collect().height
        except:
            return 0
    
    # For CSV/TSV: sample-based estimation
    try:
        file_size = os.path.getsize(file_path)
        sample = pl.read_csv(file_path, n_rows=100, ignore_errors=True)
        if len(sample) == 0: return 0
        
        # Simple:
        sample_bytes = len(pl.read_csv(file_path, n_rows=500).write_csv().encode())
        avg_row_bytes = sample_bytes / 500
        return int(file_size / avg_row_bytes)
    except:
        return 1000 # Fallback

def _check_operator_depth_override(state: ProfessorState) -> Optional[str]:
    return None

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
                "description": f"File '{filename}' is {size_mb:.0f}MB. Will use lazy evaluation."
            })

    # RAM Check
    try:
        available_ram_mb = psutil.virtual_memory().total / (1024 * 1024)
    except:
        available_ram_mb = 16384 
        
    if total_size_mb > available_ram_mb * 0.7:
        warnings.append({
            "type": "large_dataset",
            "description": f"Total data {total_size_mb:.0f}MB exceeds 70% of available RAM."
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
            df_tail = pl.read_csv(file_path).tail(n_rows) if file_size_mb < 1024 else None
        elif ext == ".parquet":
            df_head = pl.read_parquet(file_path, n_rows=n_rows)
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
                flags.append({"type": "possible_nlp", "description": f"Avg string length {avg_len:.0f} chars"})
            
            sample = df[col].drop_nulls().head(10).to_list()
            if sample:
                if sum(1 for s in sample if re.search(r"\.(jpg|jpeg|png|bmp|gif|webp)$", str(s), re.I)) > 5:
                    flags.append({"type": "image_paths", "description": "Column contains image file paths"})
                if sum(1 for s in sample if re.search(r"^[\[{]", str(s).strip())) > 5:
                    flags.append({"type": "nested_json", "description": "Column contains JSON/nested structures"})
                if sum(1 for s in sample if "|" in str(s)) > 5:
                    flags.append({"type": "pipe_delimited", "description": "Column contains pipe-delimited lists"})
                if sum(1 for s in sample if re.search(r"^\d{4}-\d{2}-\d{2}", str(s))) > 5:
                    flags.append({"type": "datetime_candidate", "description": "Column contains ISO date strings"})
            
            if n_unique > 90:
                flags.append({"type": "high_cardinality", "description": f"{n_unique} unique values in sample"})

        if n_unique == 1:
            flags.append({"type": "constant", "description": "Only 1 unique value"})
        if null_pct > 80:
            flags.append({"type": "mostly_null", "description": f"{null_pct}% null"})
            
        return {
            "name": col, "dtype": str(df[col].dtype), "n_unique_in_sample": n_unique,
            "null_pct_in_sample": null_pct, "flags": flags
        }

    profiles = []
    for col in df_head.columns:
        head_prof = get_column_profile(df_head, col)
        if df_tail is not None:
            tail_prof = get_column_profile(df_tail, col)
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
                return {"format": "json", "compatible": False, "issues": ["JSON submission format not supported"]}
            try:
                df = pl.read_csv(path) if ext in (".csv", ".tsv") else pl.read_parquet(path)
                return {"format": ext.strip("."), "columns": df.columns, "n_rows": len(df), "compatible": True}
            except:
                pass
    return {"format": "unknown", "compatible": True, "issues": ["No sample submission found"]}

# ── Sub-function 4: _detect_target_type ──────────────────────────────────────

def _detect_target_type(file_path: str, target_col: str) -> str:
    """Determine target type based on sample distribution."""
    if not target_col or not os.path.exists(file_path): return "unknown"
    try:
        df = pl.read_parquet(file_path) if file_path.endswith(".parquet") else pl.read_csv(file_path, n_rows=1000)
        if target_col not in df.columns: return "unknown"
        s = df[target_col]
        n_unique = s.n_unique()
        if s.dtype.is_numeric():
            return "binary" if n_unique == 2 else ("regression" if n_unique > 30 else "multiclass")
        return "binary" if n_unique == 2 else "multiclass"
    except:
        return "unknown"

# ── Agent Node ───────────────────────────────────────────────────────────────

@timed_node
@with_agent_retry(AGENT_NAME)
def run_preflight_checks(state: ProfessorState) -> ProfessorState:
    """
    Shield 6: Comprehensive Pre-flight Audit with Depth Routing.
    """
    logger.info("[Shield 6] Running Pre-flight Checks...")
    
    raw_data_path = state.get("raw_data_path")
    if not raw_data_path:
        return state.validated_update(state, AGENT_NAME, {"preflight_passed": False, "preflight_warnings": ["raw_data_path not set"]})

    raw_dir = os.path.dirname(raw_data_path) or "."
    inventory, total_mb, inv_warnings = _inventory_data_files(raw_dir)
    profiles = _profile_columns(raw_data_path)
    sub_format = _verify_submission_format(raw_dir)
    target_type = _detect_target_type(raw_data_path, state.get("target_col", ""))
    n_rows = _estimate_row_count(raw_data_path)
    n_features = len(profiles)

    # 3. Depth Classification
    all_warnings = inv_warnings + [f for p in profiles for f in p["flags"]]
    contract = state.get("metric_contract")
    if contract:
        metric_name = contract.get("scorer_name", "unknown") if isinstance(contract, dict) else getattr(contract, "scorer_name", "unknown")
    else:
        metric_name = "unknown"

    depth_result = classify_pipeline_depth(
        preflight_data_files=inventory,
        preflight_warnings=all_warnings,
        preflight_target_type=target_type,
        preflight_data_size_mb=total_mb,
        n_rows=n_rows,
        n_features=n_features,
        metric_name=metric_name,
        operator_override=_check_operator_depth_override(state),
    )
    
    unsupported = []
    blocking_warnings = []
    for p in profiles:
        for f in p["flags"]:
            if f["type"] == "image_paths": unsupported.append("image")
            if f["type"] == "audio_paths": unsupported.append("audio")

    if not sub_format["compatible"]: blocking_warnings.append(sub_format["issues"][0])
    if total_mb > 10000: blocking_warnings.append("Dataset exceeds 10GB boundary.")

    passed = len(blocking_warnings) == 0
    msg = f"🚀 PRE-FLIGHT REPORT\n\n📁 Total Size: {total_mb}MB\n📊 Rows ≈ {n_rows}\n📈 Features: {n_features}\n🎯 Target Type: {target_type}\n📤 Format: {sub_format['format']}\n\n⚡ Pipeline Depth: {depth_result['depth'].upper()} ({'auto-detected' if depth_result['auto_detected'] else 'override'})\n   Reason: {depth_result['reason']}\n   Skipping: {', '.join(depth_result['agents_skipped']) if depth_result['agents_skipped'] else 'None'}\n\n✅ Passed: {passed}"
    emit_to_operator(msg, level="CHECKPOINT")

    return ProfessorState.validated_update(state, AGENT_NAME, {
        "preflight_passed": passed,
        "preflight_warnings": all_warnings,
        "preflight_data_files": inventory,
        "preflight_data_size_mb": total_mb,
        "preflight_submission_format": sub_format,
        "preflight_target_type": target_type,
        "preflight_unsupported_modalities": list(set(unsupported)),
        "pipeline_depth": depth_result["depth"],
        "pipeline_depth_auto_detected": depth_result["auto_detected"],
        "pipeline_depth_reason": depth_result["reason"],
        "agents_skipped": depth_result["agents_skipped"]
    })
