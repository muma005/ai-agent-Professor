# agents/data_engineer.py

import os
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from core.state import ProfessorState
from tools.data_tools import hash_dataset, ensure_session_dirs, profile_data, write_json
from core.preprocessor import TabularPreprocessor
from guards.agent_retry import with_agent_retry
from tools.performance_monitor import timed_node

logger = logging.getLogger(__name__)

AGENT_NAME = "data_engineer"

# ── Schema Authority — data-driven detection (zero hardcoding) ────

def _find_sibling_file(raw_path: str, candidate_names: List[str]) -> str:
    """Find a sibling file in the same directory. Returns path or ''."""
    parent = Path(raw_path).parent
    for name in candidate_names:
        candidate = parent / name
        if candidate.exists():
            return str(candidate)
    return ""


def _detect_target_col(df, state: ProfessorState, raw_path: str) -> str:
    """
    Detect target column using STRUCTURAL signals only.
    """
    import polars as pl

    # 1. Explicit from state (Check competition_intel first)
    explicit = state.get("target_col", "")
    if explicit and explicit in df.columns:
        return explicit

    # 2. From sample_submission.csv
    sample_path = _find_sibling_file(raw_path, [
        "sample_submission.csv", "sampleSubmission.csv",
        "sample_sub.csv", "submission_format.csv",
    ])
    if sample_path:
        try:
            sample_df = pl.read_csv(sample_path, n_rows=5)
            if len(sample_df.columns) >= 2:
                target = sample_df.columns[1]
                if target in df.columns:
                    return target
        except:
            pass

    raise ValueError(f"[{AGENT_NAME}] Cannot determine target column automatically.")


def _detect_id_columns(df) -> List[str]:
    """Detect ID columns using structural properties and common naming."""
    import polars as pl
    n_rows = len(df)
    id_cols = []
    for col in df.columns:
        lower = col.lower().strip()
        
        # 1. Naming heuristics (Priority)
        is_named_id = lower in {"id", "index", "row_id", "row_index"} or \
                      lower.endswith("id") or lower.startswith("id_") or \
                      "passengerid" in lower
        
        # 2. Structural heuristics
        unique_count = df[col].n_unique()
        is_structurally_unique = (unique_count == n_rows) and n_rows > 1
        
        if is_named_id and (is_structurally_unique or unique_count > n_rows * 0.95):
            id_cols.append(col)
            
    return id_cols


def _detect_task_type(df, target_col: str, state: ProfessorState) -> str:
    """Detect task type from target properties or state context."""
    import polars as pl
    
    # Check if competition_intel already detected it
    intel_task = state.get("task_type", "unknown")
    if intel_task != "unknown":
        return intel_task

    target = df[target_col]
    n_unique = target.drop_nulls().n_unique()

    if target.dtype in (pl.Utf8, pl.String, pl.Categorical):
        return "binary" if n_unique == 2 else "multiclass"
    
    if target.dtype == pl.Boolean:
        return "binary"

    # Numeric target
    if n_unique <= 2: return "binary"
    if n_unique <= 20: return "multiclass"
    return "regression"


# ── Agent Node ───────────────────────────────────────────────────────────────

@timed_node
@with_agent_retry(AGENT_NAME)
def run_data_engineer(state: ProfessorState) -> ProfessorState:
    """
    LangGraph node: Data Engineer — SCHEMA AUTHORITY.
    """
    import polars as pl

    session_id  = state.get("session_id", "default")
    raw_path    = state.get("raw_data_path", "")
    output_dir  = Path(f"outputs/{session_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not raw_path or not os.path.exists(raw_path):
        raise FileNotFoundError(f"raw_data_path not valid: {raw_path}")

    # 1. Load & Hash
    data_hash = hash_dataset(raw_path)
    
    # PEAK V2: We read the first few rows to find ID candidates and force them to string
    # to avoid float inference on alphanumeric IDs (like 0001_01)
    df_preview = pl.read_csv(raw_path, n_rows=100, ignore_errors=True)
    print(f"DEBUG: Preview columns: {df_preview.columns}")
    id_candidates = [c for c in df_preview.columns if "id" in c.lower()]
    schema_overrides = {c: pl.String for c in id_candidates}
    
    df_raw = pl.read_csv(
        raw_path, 
        infer_schema_length=10000, 
        ignore_errors=True,
        schema_overrides=schema_overrides
    )

    # 2. Detect Metadata
    target_col = _detect_target_col(df_raw, state, raw_path)
    id_columns = _detect_id_columns(df_raw)
    print(f"DEBUG: Detected ID columns: {id_columns}")
    id_columns = [c for c in id_columns if c != target_col]
    task_type = _detect_task_type(df_raw, target_col, state)

    # 3. Discover sibling files
    test_data_path = _find_sibling_file(raw_path, ["test.csv", "test_data.csv", "X_test.csv"])
    sample_sub_path = _find_sibling_file(raw_path, ["sample_submission.csv", "sample_sub.csv"])

    # 4. Clean & Profile
    raw_schema = profile_data(df_raw)
    preprocessor = TabularPreprocessor(target_col=target_col, id_cols=id_columns)
    df_clean = preprocessor.fit_transform(df_raw, raw_schema)

    # 5. Persist
    parquet_path = output_dir / "cleaned.parquet"
    preprocessor_path = output_dir / "preprocessor.pkl"
    
    df_clean.write_parquet(parquet_path, use_pyarrow=True)
    preprocessor.save(str(preprocessor_path))

    clean_schema = profile_data(df_clean)
    clean_schema.update({
        "cleaned_at": datetime.now(timezone.utc).isoformat(),
        "target_col": target_col,
        "id_columns": id_columns,
        "task_type":  task_type
    })
    schema_path = output_dir / "schema.json"
    write_json(clean_schema, str(schema_path))

    # 6. Update State via validated_update
    updates = {
        "clean_data_path":          str(parquet_path),
        "schema_path":              str(schema_path),
        "preprocessor_path":        str(preprocessor_path),
        "data_hash":                data_hash,
        "target_col":               target_col,
        "id_columns":               id_columns,
        "task_type":                task_type,
        "test_data_path":           test_data_path,
        "sample_submission_path":   sample_sub_path,
        "canonical_train_rows":     len(df_clean),
        "canonical_schema":         clean_schema,
    }

    return ProfessorState.validated_update(state, AGENT_NAME, updates)
