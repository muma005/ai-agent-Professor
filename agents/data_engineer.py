# agents/data_engineer.py

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from core.state import ProfessorState
from tools.e2b_sandbox import execute_code, SandboxExecutionError
from tools.llm_client import call_llm
from tools.data_tools import hash_dataset, ensure_session_dirs
from guards.agent_retry import with_agent_retry

logger = logging.getLogger(__name__)


# ── Schema Authority — data-driven detection (zero hardcoding) ────

def _find_sibling_file(raw_path: str, candidate_names: list[str]) -> str:
    """Find a sibling file in the same directory. Returns path or ''."""
    parent = Path(raw_path).parent
    for name in candidate_names:
        candidate = parent / name
        if candidate.exists():
            return str(candidate)
    return ""


def _detect_target_col(df, state: dict, raw_path: str) -> str:
    """
    Detect target column using STRUCTURAL signals only — never a hardcoded list.

    Priority:
      1. Explicit from state (user or competition_intel set it)
      2. From sample_submission.csv — the non-ID column(s) are the target(s)
      3. FAIL with clear error — never guess
    """
    import polars as pl

    # 1. Explicit from state
    explicit = state.get("target_col", "")
    if explicit and explicit != "" and explicit in df.columns:
        print(f"[DataEngineer] Target column from state: '{explicit}'")
        return explicit

    # 2. From sample_submission.csv
    sample_path = _find_sibling_file(raw_path, [
        "sample_submission.csv", "sampleSubmission.csv",
        "sample_sub.csv", "submission_format.csv",
    ])
    if sample_path:
        sample_df = pl.read_csv(sample_path, n_rows=5)
        # Sample submission columns: [id_col, target_col(s)]
        # The first column is always the ID/index; the rest are targets
        if len(sample_df.columns) >= 2:
            target = sample_df.columns[1]  # first non-ID column
            if target in df.columns:
                print(f"[DataEngineer] Target column from sample_submission: '{target}'")
                return target
            else:
                # Target in submission but not in train — multiclass with class names?
                print(f"[DataEngineer] WARNING: sample_submission target '{target}' not in train columns")

    # 3. Fail explicitly — never guess columns[-1]
    raise ValueError(
        f"[DataEngineer] Cannot determine target column.\n"
        f"  Train columns: {df.columns}\n"
        f"  Searched for sample_submission.csv in: {Path(raw_path).parent}\n"
        f"  Fix: Set state['target_col'] explicitly before running the pipeline."
    )


def _detect_id_columns(df) -> list[str]:
    """
    Detect ID columns using STRUCTURAL properties — never substring matching.

    A column is an ID if ALL of:
      1. Every value is unique (n_unique == n_rows)
      2. dtype is integer or string (not float/boolean)
      3. Column is NOT the target (checked by caller)

    Also catches: column named exactly 'id' or 'index' (case-insensitive)
    regardless of uniqueness, as these are never useful features.
    """
    import polars as pl

    n_rows = len(df)
    id_cols = []

    for col in df.columns:
        dtype = df[col].dtype
        lower = col.lower().strip()

        # Signal 1: Exact name match (very conservative — only exact names)
        is_exact_id_name = lower in {"id", "index", "row_id", "row_index"}

        # Signal 2: Structurally unique + integer/string type
        is_structurally_unique = (
            df[col].n_unique() == n_rows
            and n_rows > 1
            and dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, 
                         pl.UInt16, pl.UInt32, pl.UInt64, pl.Utf8, pl.String)
        )

        # Signal 3: Ends with exact "_id" suffix (not substring "id")
        has_id_suffix = lower.endswith("_id") or lower.startswith("id_")

        if is_exact_id_name or (is_structurally_unique and has_id_suffix):
            id_cols.append(col)
        elif is_structurally_unique:
            # Unique but no naming signal — log a warning, don't auto-drop
            print(f"[DataEngineer] NOTE: '{col}' has all unique values but no ID naming pattern. Keeping as feature.")

    return id_cols


def _detect_task_type(df, target_col: str) -> str:
    """
    Detect task type from the target column — pure data analysis.

    Rules:
      - String/categorical target → classification
      - Boolean target → binary classification
      - Integer with ≤2 unique → binary classification
      - Integer with 3-20 unique → multiclass classification
      - Float or integer with >20 unique → regression
    """
    import polars as pl

    target = df[target_col]
    dtype = target.dtype
    n_unique = target.drop_nulls().n_unique()

    if dtype in (pl.Utf8, pl.String, pl.Categorical):
        if n_unique == 2:
            return "binary"
        return "multiclass"

    if dtype == pl.Boolean:
        return "binary"

    if dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, 
                 pl.UInt16, pl.UInt32, pl.UInt64):
        if n_unique <= 2:
            return "binary"
        if n_unique <= 20:
            return "multiclass"
        return "regression"

    # Float
    if n_unique <= 2:
        return "binary"
    if n_unique <= 20:
        return "multiclass"
    return "regression"


# ── Day 15: External data check ───────────────────────────────────
def _check_external_data(state: ProfessorState) -> ProfessorState:
    """
    If external_data_manifest has recommended sources, log them for the engineer.
    Does NOT auto-download — that requires explicit user confirmation.
    """
    manifest = state.get("external_data_manifest", {})
    recommended = manifest.get("recommended_sources", [])
    sources = manifest.get("external_sources", [])

    if not recommended:
        return state

    high_relevance = [s for s in sources if float(s.get("relevance_score", 0)) >= 0.8]

    if high_relevance:
        from core.lineage import log_event
        logger.info(
            f"[data_engineer] {len(high_relevance)} high-relevance external source(s) available."
        )
        log_event(
            session_id=state["session_id"],
            agent="data_engineer",
            action="external_data_available",
            keys_read=["external_data_manifest"],
            keys_written=[],
            values_changed={"sources": [s["name"] for s in high_relevance]},
        )

    return state


# ── Main agent function (LangGraph node) ─────────────────────────
@with_agent_retry("DataEngineer")
def run_data_engineer(state: ProfessorState) -> ProfessorState:
    """
    LangGraph node: Data Engineer — SCHEMA AUTHORITY.

    This agent is the SINGLE SOURCE OF TRUTH for:
      - target_col:  which column is the prediction target
      - id_columns:  which columns are row identifiers (not features)
      - task_type:   binary / multiclass / regression

    All downstream agents read these from state — they NEVER re-detect.

    Reads:  state["raw_data_path"]
    Writes: state["clean_data_path"], state["schema_path"],
            state["preprocessor_path"], state["data_hash"],
            state["target_col"], state["id_columns"], state["task_type"],
            state["test_data_path"], state["sample_submission_path"]
    """
    import polars as pl
    from core.preprocessor import TabularPreprocessor
    from tools.data_tools import profile_data, write_json

    session_id  = state["session_id"]
    raw_path    = state["raw_data_path"]
    output_dir  = f"outputs/{session_id}"

    print(f"[DataEngineer] Starting — session: {session_id}")
    print(f"[DataEngineer] Input: {raw_path}")

    # ── Validate input ────────────────────────────────────────────
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"raw_data_path does not exist: {raw_path}")
        
    if os.path.isdir(raw_path):
        raw_path = os.path.join(raw_path, "train.csv")
        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"train.csv not found in directory: {state['raw_data_path']}")
        # Update state so downstream nodes know the actual file
        state["raw_data_path"] = raw_path

    # ── Ensure output directories ─────────────────────────────────
    ensure_session_dirs(session_id)

    # ── Hash the source file ──────────────────────────────────────
    data_hash = hash_dataset(raw_path)
    print(f"[DataEngineer] data_hash: {data_hash}")

    # ── 1. Load Raw Data ──────────────────────────────────────────
    df_raw = pl.read_csv(raw_path, infer_schema_length=10000)

    # ── 2. SCHEMA AUTHORITY — detect target, IDs, task type ───────
    target_col = _detect_target_col(df_raw, state, raw_path)
    id_columns = _detect_id_columns(df_raw)
    # Ensure target is not in id_columns
    id_columns = [c for c in id_columns if c != target_col]
    task_type = _detect_task_type(df_raw, target_col)

    print(f"[DataEngineer] [PASS] target_col:  '{target_col}'")
    print(f"[DataEngineer] [PASS] id_columns:  {id_columns}")
    print(f"[DataEngineer] [PASS] task_type:   '{task_type}'")

    # ── Discover sibling files (test.csv, sample_submission.csv) ──
    test_data_path = _find_sibling_file(raw_path, [
        "test.csv", "test_data.csv", "X_test.csv",
    ])
    sample_submission_path = _find_sibling_file(raw_path, [
        "sample_submission.csv", "sampleSubmission.csv",
        "sample_sub.csv", "submission_format.csv",
    ])

    # ── 3. Profile Raw Schema ─────────────────────────────────────
    raw_schema = profile_data(df_raw)

    # ── 4. Initialize and Fit Preprocessor ────────────────────────
    preprocessor = TabularPreprocessor(target_col=target_col, id_cols=id_columns)
    df_clean = preprocessor.fit_transform(df_raw, raw_schema)

    # ── 5. Serialize Outputs ──────────────────────────────────────
    parquet_path = f"{output_dir}/cleaned.parquet"
    preprocessor_path = f"{output_dir}/preprocessor.pkl"

    df_clean.write_parquet(parquet_path)
    preprocessor.save(preprocessor_path)

    # Profile the clean data to write the final schema.json
    clean_schema = profile_data(df_clean)
    clean_schema["cleaned_at"] = datetime.utcnow().isoformat()
    # Embed schema authority into the schema file
    clean_schema["target_col"] = target_col
    clean_schema["id_columns"] = id_columns
    clean_schema["task_type"] = task_type
    schema_path = f"{output_dir}/schema.json"
    write_json(clean_schema, schema_path)

    print(f"[DataEngineer] Imputed {len(preprocessor.numeric_imputes)} numeric cols")
    print(f"[DataEngineer] Imputed {len(preprocessor.string_imputes)} string cols")
    print(f"[DataEngineer] cleaned.parquet: {os.path.getsize(parquet_path):,} bytes")

    # ── Update cost tracker ───────────────────────────────────────
    cost_tracker = dict(state["cost_tracker"])
    cost_tracker["llm_calls"] += 0  # No LLM used for deterministic operations

    # ── Log lineage ──────────────────────────────────────────────
    from core.lineage import log_event
    log_event(
        session_id=session_id,
        agent="data_engineer",
        action="cleaned_and_profiled",
        keys_read=["raw_data_path"],
        keys_written=["clean_data_path", "schema_path", "data_hash",
                      "target_col", "id_columns", "task_type"],
        values_changed={
            "data_hash": data_hash,
            "target_col": target_col,
            "id_columns": id_columns,
            "task_type": task_type,
        },
    )

    # ── Return updated state ──────────────────────────────────────
    return {
        **state,
        "clean_data_path":         parquet_path,
        "schema_path":             schema_path,
        "preprocessor_path":       preprocessor_path,
        "data_hash":               data_hash,
        "target_col":              target_col,
        "id_columns":              id_columns,
        "task_type":               task_type,
        "test_data_path":          test_data_path,
        "sample_submission_path":  sample_submission_path,
        "cost_tracker":            cost_tracker,
    }

