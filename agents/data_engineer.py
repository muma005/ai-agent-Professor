# agents/data_engineer.py

import os
import json
from datetime import datetime
from core.state import ProfessorState
from tools.e2b_sandbox import execute_code, SandboxExecutionError
from tools.llm_client import call_llm
from tools.data_tools import hash_dataset, ensure_session_dirs


# ── LLM fix callback for sandbox retry loop ───────────────────────
def _make_fix_callback(session_id: str):
    """Returns a callback that asks the LLM to fix broken sandbox code."""
    def fix_callback(code: str, error: str, traceback_str: str) -> str:
        prompt = f"""
The following Python code failed with an error.
Fix ONLY the specific error. Do not restructure the code.
Return the complete corrected code and nothing else — no explanation,
no markdown fences, just raw Python.

FAILED CODE:
{code}

ERROR TYPE: {error}
TRACEBACK:
{traceback_str}
"""
        fixed = call_llm(
            prompt=prompt,
            system="You are a Python debugging assistant. Return only corrected code.",
            model="deepseek",
            is_coding_task=True
        )
        # Strip any accidental markdown fences
        fixed = fixed.strip()
        if fixed.startswith("```"):
            fixed = "\n".join(fixed.split("\n")[1:])
        if fixed.endswith("```"):
            fixed = "\n".join(fixed.split("\n")[:-1])
        return fixed.strip()
    return fix_callback


# ── Preprocessing code template ───────────────────────────────────
def _build_preprocessing_code(
    raw_data_path: str,
    output_dir: str,
    schema: dict
) -> str:
    """
    Builds the preprocessing script the sandbox will execute.
    Uses ONLY standard Polars calls — no project module imports.
    Sandbox writes cleaned.parquet. Profiling/schema done outside sandbox.
    """
    return f"""
import polars as pl
import polars.selectors as cs
import os

# ── Load raw data ─────────────────────────────────────────────────
df = pl.read_csv("{raw_data_path}", infer_schema_length=10000)
print(f"Loaded: {{df.shape[0]}} rows, {{df.shape[1]}} columns")

# ── Profile BEFORE cleaning ───────────────────────────────────────
missing_rates = {{
    col: round(df[col].null_count() / len(df), 4)
    for col in df.columns
}}
print(f"Missing rates: {{missing_rates}}")

# ── Basic cleaning ────────────────────────────────────────────────
# Fill numeric nulls with median
for col in df.select(cs.numeric()).columns:
    median_val = df[col].median()
    if median_val is not None:
        df = df.with_columns(pl.col(col).fill_null(median_val))

# Fill string nulls with "missing"
for col in df.select(cs.string()).columns:
    df = df.with_columns(pl.col(col).fill_null("missing"))

# Fill boolean nulls with False
for col in df.select(cs.boolean()).columns:
    df = df.with_columns(pl.col(col).fill_null(False))

print(f"After cleaning: {{df.shape[0]}} rows, {{df.null_count().sum_horizontal().item()}} nulls")

# ── Write cleaned parquet ─────────────────────────────────────────
os.makedirs("{output_dir}", exist_ok=True)
parquet_path = "{output_dir}/cleaned.parquet"
df.write_parquet(parquet_path)
print(f"Saved: {{parquet_path}}")
print("DATA_ENGINEER_COMPLETE")
"""


# ── Main agent function (LangGraph node) ─────────────────────────
def run_data_engineer(state: ProfessorState) -> ProfessorState:
    """
    LangGraph node: Data Engineer.

    Reads:  state["raw_data_path"]
    Writes: state["clean_data_path"]  — str pointer to cleaned.parquet
            state["schema_path"]      — str pointer to schema.json
            state["data_hash"]        — SHA-256 of source file (first 16 chars)
            state["cost_tracker"]     — incremented
    Never puts raw data in state.
    """
    session_id  = state["session_id"]
    raw_path    = state["raw_data_path"]
    output_dir  = f"outputs/{session_id}"

    print(f"[DataEngineer] Starting — session: {session_id}")
    print(f"[DataEngineer] Input: {raw_path}")

    # ── Validate input ────────────────────────────────────────────
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"raw_data_path does not exist: {raw_path}")

    # ── Ensure output directories ─────────────────────────────────
    ensure_session_dirs(session_id)

    # ── Hash the source file ──────────────────────────────────────
    data_hash = hash_dataset(raw_path)
    print(f"[DataEngineer] data_hash: {data_hash}")

    # ── Build preprocessing code ──────────────────────────────────
    code = _build_preprocessing_code(
        raw_data_path=raw_path,
        output_dir=output_dir,
        schema={}
    )

    # ── Execute in sandbox with retry loop ────────────────────────
    fix_callback = _make_fix_callback(session_id)

    try:
        result = execute_code(
            code=code,
            session_id=session_id,
            llm_fix_callback=fix_callback,
            max_attempts=3
        )
    except SandboxExecutionError as e:
        print(f"[DataEngineer] Sandbox failed after 3 attempts: {e}")
        raise

    if not result["success"]:
        raise RuntimeError(f"[DataEngineer] Unexpected failure: {result}")

    print(f"[DataEngineer] Sandbox output:\n{result['stdout']}")

    # ── Verify parquet exists ─────────────────────────────────────
    parquet_path = f"{output_dir}/cleaned.parquet"
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"cleaned.parquet not produced: {parquet_path}")

    # ── Profile cleaned data + write schema (outside sandbox) ─────
    from tools.data_tools import profile_data, write_json
    import polars as pl

    df_clean = pl.read_parquet(parquet_path)
    schema_data = profile_data(df_clean)
    schema_data["cleaned_at"] = datetime.utcnow().isoformat()

    schema_path = f"{output_dir}/schema.json"
    write_json(schema_data, schema_path)

    # ── Validate schema has required fields ───────────────────────
    required_schema_fields = ["columns", "types", "missing_rates"]
    for field in required_schema_fields:
        if field not in schema_data:
            raise ValueError(f"schema.json missing required field: '{field}'")

    print(f"[DataEngineer] cleaned.parquet: {os.path.getsize(parquet_path):,} bytes")
    print(f"[DataEngineer] schema.json:     {len(schema_data['columns'])} columns")

    # ── Update cost tracker ───────────────────────────────────────
    cost_tracker = dict(state["cost_tracker"])
    cost_tracker["llm_calls"] += result.get("attempts_used", 1)

    print(f"[DataEngineer] Complete. Attempts used: {result.get('attempts_used', 1)}")

    # ── Log lineage ──────────────────────────────────────────────
    from core.lineage import log_event
    log_event(
        session_id=session_id,
        agent="data_engineer",
        action="cleaned_and_profiled",
        keys_read=["raw_data_path"],
        keys_written=["clean_data_path", "schema_path", "data_hash"],
        values_changed={"data_hash": data_hash},
    )

    # ── Return updated state — ONLY pointers, never raw data ──────
    return {
        **state,
        "clean_data_path": parquet_path,
        "schema_path":     schema_path,
        "data_hash":       data_hash,
        "cost_tracker":    cost_tracker,
    }

