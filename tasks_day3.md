# Day 3 Tasks
Got everything. Here's the full Day 3 breakdown.

---

## Day 3 Tasks — From Notion

```
┌────┬──────────────────────────────────────────┬──────────────────────────┬──────────┬───────────┐
│ #  │ Task                                     │ Phase                    │ Priority │ Cuttable  │
├────┼──────────────────────────────────────────┼──────────────────────────┼──────────┼───────────┤
│ 1  │ Build agents/data_engineer.py            │ 🚀 Phase 1: Make It Run  │ Critical │ Never Cut │
│ 2  │ Write contract test — Data Engineer      │ 🚀 Phase 1: Make It Run  │ Critical │ Never Cut │
│ 3  │ Test Data Engineer on Spaceship Titanic  │ 🚀 Phase 1: Make It Run  │ Critical │ Never Cut │
└────┴──────────────────────────────────────────┴──────────────────────────┴──────────┴───────────┘
```

All three are Phase 1. All Critical. All Never Cut.

**The ONE thing that must work by end of today:**
Feed `train.csv` into `data_engineer.py` and get back `cleaned.parquet` + `schema.json` with only string pointers in state. No raw data in state. Ever.

---

## Task 1 — Build `agents/data_engineer.py`

The Data Engineer is a LangGraph node. It takes `raw_data_path` from state, runs preprocessing code inside the sandbox, and writes two files: `cleaned.parquet` and `schema.json`. It then updates state with the file paths — never the data itself.

```python
# agents/data_engineer.py

import os
import json
import hashlib
import polars as pl
from datetime import datetime
from core.state import ProfessorState
from tools.e2b_sandbox import execute_code, SandboxExecutionError
from tools.llm_client import call_llm

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
            model="fireworks-deepseek",
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
    The LLM customises this per-dataset. For now: deterministic template.
    Full LLM-driven preprocessing comes in Phase 2.
    """
    return f"""
import polars as pl
import polars.selectors as cs
import json
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

# ── Type inference ────────────────────────────────────────────────
column_types = {{col: str(df[col].dtype) for col in df.columns}}

# ── Basic cleaning ────────────────────────────────────────────────
# Fill numeric nulls with median
numeric_cols = df.select(cs.numeric()).columns
for col in numeric_cols:
    median_val = df[col].median()
    if median_val is not None:
        df = df.with_columns(pl.col(col).fill_null(median_val))

# Fill string nulls with "missing"
string_cols = df.select(cs.string()).columns
for col in string_cols:
    df = df.with_columns(pl.col(col).fill_null("missing"))

# Fill boolean nulls with False
bool_cols = df.select(cs.boolean()).columns
for col in bool_cols:
    df = df.with_columns(pl.col(col).fill_null(False))

print(f"After cleaning: {{df.shape[0]}} rows, {{df.shape[1]}} columns")
print(f"Remaining nulls: {{df.null_count().sum_horizontal().item()}}")

# ── Write outputs ─────────────────────────────────────────────────
os.makedirs("{output_dir}", exist_ok=True)

parquet_path = "{output_dir}/cleaned.parquet"
schema_path  = "{output_dir}/schema.json"

df.write_parquet(parquet_path)
print(f"Saved: {{parquet_path}}")

schema = {{
    "columns":       df.columns,
    "types":         {{col: str(df[col].dtype) for col in df.columns}},
    "missing_rates": missing_rates,
    "shape":         list(df.shape),
    "cleaned_at":    "{datetime.utcnow().isoformat()}"
}}

with open(schema_path, "w") as f:
    json.dump(schema, f, indent=2)
print(f"Saved: {{schema_path}}")

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

    # ── Hash the source file ──────────────────────────────────────
    with open(raw_path, "rb") as f:
        data_hash = hashlib.sha256(f.read()).hexdigest()[:16]
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

    # ── Verify outputs exist ──────────────────────────────────────
    parquet_path = f"{output_dir}/cleaned.parquet"
    schema_path  = f"{output_dir}/schema.json"

    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"cleaned.parquet not produced: {parquet_path}")
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"schema.json not produced: {schema_path}")

    # ── Validate schema.json has required fields ──────────────────
    with open(schema_path) as f:
        schema = json.load(f)

    required_schema_fields = ["columns", "types", "missing_rates"]
    for field in required_schema_fields:
        if field not in schema:
            raise ValueError(f"schema.json missing required field: '{field}'")

    print(f"[DataEngineer] cleaned.parquet: {os.path.getsize(parquet_path):,} bytes")
    print(f"[DataEngineer] schema.json:     {len(schema['columns'])} columns")

    # ── Update cost tracker ───────────────────────────────────────
    cost_tracker = dict(state["cost_tracker"])
    cost_tracker["llm_calls"] += result.get("attempts_used", 1)

    print(f"[DataEngineer] Complete. Attempts used: {result.get('attempts_used', 1)}")

    # ── Return updated state — ONLY pointers, never raw data ──────
    return {
        **state,
        "clean_data_path": parquet_path,
        "schema_path":     schema_path,
        "data_hash":       data_hash,
        "cost_tracker":    cost_tracker,
    }
```

---

## Task 2 — Write Contract Test (Immutable From Today)

```python
# tests/contracts/test_data_engineer_contract.py
# ─────────────────────────────────────────────────────────────────
# Written: Day 3
# Status:  IMMUTABLE — never edit this file after today
#
# CONTRACT: run_data_engineer()
#   INPUT:   state["raw_data_path"] — str, must exist on disk
#   OUTPUT:  outputs/{session_id}/cleaned.parquet — must exist
#            outputs/{session_id}/schema.json — must have:
#              columns (list), types (dict), missing_rates (dict)
#   STATE:   clean_data_path — str pointer (not DataFrame)
#            schema_path     — str pointer
#            data_hash       — 16-char hex string
#            cost_tracker    — llm_calls incremented
#   NEVER:   raw DataFrame in state
#            raw DataFrame in any state field
# ─────────────────────────────────────────────────────────────────
import pytest
import os
import json
import polars as pl
from pathlib import Path
from core.state import initial_state
from agents.data_engineer import run_data_engineer

# ── Fixture: minimal CSV the tests always use ─────────────────────
FIXTURE_CSV = "tests/fixtures/tiny_train.csv"

@pytest.fixture(scope="session", autouse=True)
def create_fixture_csv():
    """Create a minimal CSV fixture for contract tests."""
    os.makedirs("tests/fixtures", exist_ok=True)
    if not os.path.exists(FIXTURE_CSV):
        import polars as pl
        df = pl.DataFrame({
            "PassengerId": ["0001_01", "0002_01", "0003_01",
                            "0004_01", "0005_01"],
            "HomePlanet":  ["Europa", "Earth", None, "Mars", "Earth"],
            "Age":         [39.0, 24.0, None, 58.0, 33.0],
            "RoomService": [0.0, 109.0, None, 43.0, 0.0],
            "Transported": [False, True, True, False, True],
        })
        df.write_csv(FIXTURE_CSV)


@pytest.fixture
def base_state():
    return initial_state(
        competition="test-titanic",
        data_path=FIXTURE_CSV,
        budget_usd=2.0
    )


class TestDataEngineerContract:

    def test_accepts_valid_raw_data_path(self, base_state):
        result = run_data_engineer(base_state)
        assert result is not None

    def test_rejects_nonexistent_path(self, base_state):
        bad_state = {**base_state, "raw_data_path": "/nonexistent/train.csv"}
        with pytest.raises(FileNotFoundError):
            run_data_engineer(bad_state)

    def test_produces_cleaned_parquet(self, base_state):
        result = run_data_engineer(base_state)
        assert os.path.exists(result["clean_data_path"]), \
            "cleaned.parquet must exist after run"

    def test_produces_schema_json(self, base_state):
        result = run_data_engineer(base_state)
        assert os.path.exists(result["schema_path"]), \
            "schema.json must exist after run"

    def test_schema_has_columns_field(self, base_state):
        result = run_data_engineer(base_state)
        schema = json.loads(Path(result["schema_path"]).read_text())
        assert "columns" in schema, "schema.json must have 'columns'"
        assert isinstance(schema["columns"], list)
        assert len(schema["columns"]) > 0

    def test_schema_has_types_field(self, base_state):
        result = run_data_engineer(base_state)
        schema = json.loads(Path(result["schema_path"]).read_text())
        assert "types" in schema, "schema.json must have 'types'"
        assert isinstance(schema["types"], dict)

    def test_schema_has_missing_rates_field(self, base_state):
        result = run_data_engineer(base_state)
        schema = json.loads(Path(result["schema_path"]).read_text())
        assert "missing_rates" in schema, "schema.json must have 'missing_rates'"
        assert isinstance(schema["missing_rates"], dict)

    def test_clean_data_path_is_string_not_dataframe(self, base_state):
        result = run_data_engineer(base_state)
        assert isinstance(result["clean_data_path"], str), \
            "clean_data_path must be a str pointer — never a DataFrame"

    def test_no_raw_data_in_state(self, base_state):
        result = run_data_engineer(base_state)
        for key, value in result.items():
            assert not isinstance(value, pl.DataFrame), \
                f"DataFrame found in state['{key}'] — only pointers allowed"

    def test_data_hash_set_in_state(self, base_state):
        result = run_data_engineer(base_state)
        assert "data_hash" in result
        assert isinstance(result["data_hash"], str)
        assert len(result["data_hash"]) == 16, \
            "data_hash must be 16-char hex string"

    def test_cost_tracker_llm_calls_incremented(self, base_state):
        before = base_state["cost_tracker"]["llm_calls"]
        result = run_data_engineer(base_state)
        after  = result["cost_tracker"]["llm_calls"]
        assert after >= before, \
            "cost_tracker.llm_calls must be incremented after run"

    def test_parquet_is_polars_readable(self, base_state):
        result = run_data_engineer(base_state)
        df = pl.read_parquet(result["clean_data_path"])
        assert isinstance(df, pl.DataFrame), \
            "cleaned.parquet must be readable as Polars DataFrame"

    def test_parquet_has_no_object_dtype(self, base_state):
        result = run_data_engineer(base_state)
        df = pl.read_parquet(result["clean_data_path"])
        object_cols = [c for c in df.columns if df[c].dtype == pl.Object]
        assert len(object_cols) == 0, \
            f"Object dtype columns detected (Pandas contamination): {object_cols}"

    def test_no_nulls_in_cleaned_parquet(self, base_state):
        result = run_data_engineer(base_state)
        df = pl.read_parquet(result["clean_data_path"])
        total_nulls = df.null_count().sum_horizontal().item()
        assert total_nulls == 0, \
            f"cleaned.parquet should have 0 nulls after cleaning, found {total_nulls}"

    def test_session_id_namespacing(self, base_state):
        """Output files must live under outputs/{session_id}/"""
        result = run_data_engineer(base_state)
        session_id = base_state["session_id"]
        assert session_id in result["clean_data_path"], \
            "clean_data_path must be namespaced under session_id"
        assert session_id in result["schema_path"], \
            "schema_path must be namespaced under session_id"
```

---

## Task 3 — Test Data Engineer on Spaceship Titanic

Run it against the real dataset, not just the fixture.

```python
# Run this in a terminal — not in the notebook
from core.state import initial_state
from agents.data_engineer import run_data_engineer
import json

state = initial_state(
    competition="spaceship-titanic",
    data_path="data/spaceship_titanic/train.csv",
    budget_usd=2.0
)

print("Running Data Engineer on Spaceship Titanic...")
result = run_data_engineer(state)

# ── Verify outputs ────────────────────────────────────────────────
import polars as pl

df = pl.read_parquet(result["clean_data_path"])
schema = json.loads(open(result["schema_path"]).read())

print(f"\n✓ cleaned.parquet shape:  {df.shape}")
print(f"✓ Columns:                {df.columns}")
print(f"✓ Null count:             {df.null_count().sum_horizontal().item()}")
print(f"✓ data_hash:              {result['data_hash']}")
print(f"✓ Schema columns:         {schema['columns']}")
print(f"✓ Missing rates:          {schema['missing_rates']}")
print(f"✓ State clean_data_path:  {result['clean_data_path']}")
print(f"✓ State schema_path:      {result['schema_path']}")
print(f"✓ No DataFrame in state:  TRUE")
print(f"✓ Cost tracker calls:     {result['cost_tracker']['llm_calls']}")
```

Expected output:

```
Running Data Engineer on Spaceship Titanic...
[DataEngineer] Starting — session: spaceship_abc123de
[DataEngineer] data_hash: a3f9c21d4b8e7f01
[DataEngineer] Sandbox output:
  Loaded: 8693 rows, 14 columns
  Missing rates: {'HomePlanet': 0.0201, 'CryoSleep': 0.0247, ...}
  After cleaning: 8693 rows, 14 columns
  Remaining nulls: 0
  Saved: outputs/spaceship_abc123de/cleaned.parquet
  Saved: outputs/spaceship_abc123de/schema.json
  DATA_ENGINEER_COMPLETE

✓ cleaned.parquet shape:  (8693, 14)
✓ Null count:             0
✓ data_hash:              a3f9c21d4b8e7f01
✓ State clean_data_path:  outputs/spaceship_abc123de/cleaned.parquet
✓ No DataFrame in state:  TRUE
```

---

## End of Day 3 Checklist

```bash
# 1. Run contract tests
pytest tests/contracts/test_data_engineer_contract.py -v
# All 15 tests must be green

# 2. Run all contract tests together (sandbox + data engineer)
pytest tests/contracts/ -v
# All tests from both Day 2 and Day 3 must pass

# 3. Confirm real dataset run produced outputs
ls outputs/spaceship-titanic*/
# Must show: cleaned.parquet  schema.json

# 4. Commit
git add .
git commit -m "Day 3: Build Data Engineer + contract test — all tests pass"
git push origin phase-1
```


┌────┬──────────────────────────────────────────────┬─────────────────────────┬──────────┬────────────────┐
│ #  │ Task                                         │ Phase                   │ Priority │ Cuttable       │
├────┼──────────────────────────────────────────────┼─────────────────────────┼──────────┼────────────────┤
│ 1  │ Build tools/data_tools.py — Polars helpers   │ 🚀 Phase 1: Make It Run │ High     │ Safe to Stub   │
│ 2  │ Build agents/data_engineer.py                │ 🚀 Phase 1: Make It Run │ Critical │ Never Cut      │
│ 3  │ Write contract test — Data Engineer          │ 🚀 Phase 1: Make It Run │ Critical │ Never Cut      │
│ 4  │ Test Data Engineer on Spaceship Titanic      │ 🚀 Phase 1: Make It Run │ Critical │ Never Cut      │
└────┴──────────────────────────────────────────────┴─────────────────────────┴──────────┴────────────────┘
```

**Important note on order:** `data_tools.py` must be built first. The Data Engineer imports from it. Build Task 1 before Task 2 or the import will fail.

---

## Task 1 — Build `tools/data_tools.py`

This is the utility layer that all agents use for data I/O. Three core functions: `read_csv`, `write_parquet`, `profile_data`. The Data Engineer calls `profile_data` to produce `schema.json`. Every agent that reads or writes data calls this instead of calling Polars directly — so if the I/O layer ever needs to change, you change it in one place.

```python
# tools/data_tools.py

import os
import json
import hashlib
import polars as pl
import polars.selectors as cs
from datetime import datetime
from pathlib import Path


# ── Read ──────────────────────────────────────────────────────────

def read_csv(path: str, infer_schema_length: int = 10000) -> pl.DataFrame:
    """
    Read a CSV file into a Polars DataFrame.
    Always use this — never pl.read_csv() directly in agents.
    Validates the file exists before reading.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    return pl.read_csv(path, infer_schema_length=infer_schema_length)


def read_parquet(path: str) -> pl.DataFrame:
    """
    Read a Parquet file into a Polars DataFrame.
    Always use this — never pl.read_parquet() directly in agents.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Parquet not found: {path}")
    df = pl.read_parquet(path)
    if not isinstance(df, pl.DataFrame):
        raise TypeError(f"Expected Polars DataFrame, got {type(df)}")
    return df


# ── Write ─────────────────────────────────────────────────────────

def write_parquet(df: pl.DataFrame, path: str) -> str:
    """
    Write a Polars DataFrame to Parquet.
    Creates parent directories if they don't exist.
    Returns the path written.
    """
    if not isinstance(df, pl.DataFrame):
        raise TypeError(
            f"write_parquet expects a Polars DataFrame, got {type(df)}. "
            "If you have a Pandas DataFrame, convert with pl.from_pandas(df) first."
        )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.write_parquet(path)
    return path


def write_json(data: dict, path: str) -> str:
    """
    Write a dict to JSON. Creates parent directories if needed.
    Returns the path written.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def read_json(path: str) -> dict:
    """Read a JSON file and return as dict."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON not found: {path}")
    with open(path) as f:
        return json.load(f)


# ── Profile ───────────────────────────────────────────────────────

def profile_data(df: pl.DataFrame) -> dict:
    """
    Profile a Polars DataFrame and return a schema dict.
    This is what gets written to schema.json by the Data Engineer.

    Returns:
        {
            columns:       [list of column names],
            types:         {col: dtype_str},
            missing_rates: {col: float 0-1},
            missing_counts:{col: int},
            shape:         [rows, cols],
            numeric_cols:  [list],
            categorical_cols: [list],
            boolean_cols:  [list],
            cardinality:   {col: n_unique} for categorical cols,
            profiled_at:   ISO timestamp
        }
    """
    if not isinstance(df, pl.DataFrame):
        raise TypeError(f"profile_data expects Polars DataFrame, got {type(df)}")

    n_rows = len(df)

    # Column types
    column_types = {col: str(df[col].dtype) for col in df.columns}

    # Missing rates and counts
    missing_counts = {col: int(df[col].null_count()) for col in df.columns}
    missing_rates  = {
        col: round(missing_counts[col] / n_rows, 4) if n_rows > 0 else 0.0
        for col in df.columns
    }

    # Categorise columns by type
    numeric_cols     = df.select(cs.numeric()).columns
    categorical_cols = df.select(cs.string()).columns
    boolean_cols     = df.select(cs.boolean()).columns

    # Cardinality for categoricals (useful for encoding decisions)
    cardinality = {
        col: int(df[col].n_unique())
        for col in categorical_cols
    }

    return {
        "columns":          df.columns,
        "types":            column_types,
        "missing_rates":    missing_rates,
        "missing_counts":   missing_counts,
        "shape":            list(df.shape),
        "numeric_cols":     numeric_cols,
        "categorical_cols": categorical_cols,
        "boolean_cols":     boolean_cols,
        "cardinality":      cardinality,
        "profiled_at":      datetime.utcnow().isoformat(),
    }


# ── Hash ──────────────────────────────────────────────────────────

def hash_file(path: str, length: int = 16) -> str:
    """
    SHA-256 hash of a file. Returns first `length` hex chars.
    Used to detect if the dataset changes mid-competition.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot hash — file not found: {path}")
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:length]


def hash_dataframe(df: pl.DataFrame, length: int = 16) -> str:
    """
    SHA-256 hash of a Polars DataFrame's content.
    Deterministic — same data always produces same hash.
    """
    if not isinstance(df, pl.DataFrame):
        raise TypeError(f"hash_dataframe expects Polars DataFrame, got {type(df)}")
    content = df.write_csv().encode("utf-8")
    return hashlib.sha256(content).hexdigest()[:length]


# ── Validate ──────────────────────────────────────────────────────

def validate_submission(
    submission: pl.DataFrame,
    sample_submission: pl.DataFrame
) -> dict:
    """
    Validate a submission DataFrame against the sample submission format.

    Returns:
        {"valid": bool, "errors": [list of error strings]}
    """
    errors = []

    # Column names must match exactly
    if set(submission.columns) != set(sample_submission.columns):
        errors.append(
            f"Column mismatch. Expected: {sample_submission.columns}. "
            f"Got: {submission.columns}"
        )

    # Row count must match
    if len(submission) != len(sample_submission):
        errors.append(
            f"Row count mismatch. Expected: {len(sample_submission)}. "
            f"Got: {len(submission)}"
        )

    # No nulls in submission
    null_count = submission.null_count().sum_horizontal().item()
    if null_count > 0:
        errors.append(f"Submission contains {null_count} null values")

    return {
        "valid":  len(errors) == 0,
        "errors": errors
    }


# ── Ensure output dir ─────────────────────────────────────────────

def ensure_session_dirs(session_id: str) -> dict:
    """
    Create all output subdirectories for a session.
    Returns dict of all paths.
    """
    base = f"outputs/{session_id}"
    dirs = {
        "base":        base,
        "models":      f"{base}/models",
        "predictions": f"{base}/predictions",
        "charts":      f"{base}/charts",
        "logs":        f"{base}/logs",
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs
```

Now update `agents/data_engineer.py` to use `data_tools` instead of calling Polars directly:

```python
# In agents/data_engineer.py — update the imports at the top
# Replace the direct polars import with data_tools

from tools.data_tools import (
    read_csv,
    write_parquet,
    write_json,
    profile_data,
    hash_file,
    ensure_session_dirs
)
```

And update `_build_preprocessing_code` to use `data_tools` functions inside the sandbox:

```python
# The sandbox code now imports and uses data_tools too
def _build_preprocessing_code(raw_data_path, output_dir, schema):
    return f"""
import polars as pl
import polars.selectors as cs
import json, os, sys
sys.path.insert(0, '.')          # so sandbox can import from project root
from tools.data_tools import profile_data, write_parquet, write_json

df = pl.read_csv("{raw_data_path}", infer_schema_length=10000)
print(f"Loaded: {{df.shape[0]}} rows, {{df.shape[1]}} columns")

# Profile BEFORE cleaning
schema = profile_data(df)
print(f"Missing rates: {{schema['missing_rates']}}")

# Clean
import polars.selectors as cs
for col in df.select(cs.numeric()).columns:
    median_val = df[col].median()
    if median_val is not None:
        df = df.with_columns(pl.col(col).fill_null(median_val))
for col in df.select(cs.string()).columns:
    df = df.with_columns(pl.col(col).fill_null("missing"))
for col in df.select(cs.boolean()).columns:
    df = df.with_columns(pl.col(col).fill_null(False))

print(f"After cleaning: {{df.shape[0]}} rows, {{df.null_count().sum_horizontal().item()}} nulls")

# Write outputs
os.makedirs("{output_dir}", exist_ok=True)
write_parquet(df,    "{output_dir}/cleaned.parquet")
write_json(schema,   "{output_dir}/schema.json")
print("DATA_ENGINEER_COMPLETE")
"""
```

---

## Quick Verification

```bash
# Test data_tools in isolation before wiring into Data Engineer
python -c "
import polars as pl
from tools.data_tools import read_csv, profile_data, hash_file

df = read_csv('data/spaceship_titanic/train.csv')
schema = profile_data(df)
h = hash_file('data/spaceship_titanic/train.csv')

print('Shape:',    df.shape)
print('Hash:',     h)
print('Columns:',  schema['columns'])
print('Numerics:', schema['numeric_cols'])
print('Cats:',     schema['categorical_cols'])
print('Missing:',  schema['missing_rates'])
print('data_tools: WORKING')
"
```

---

The reason this task is **Safe to Stub** (unlike the others) is that in an emergency you could inline the Polars calls directly in the Data Engineer. But you should never do that — `data_tools.py` is the single place all I/O lives, which means Polars version changes, format changes, or validation logic only ever need one fix in one file. Build it properly today.

## Day 4 Preview

Tomorrow's ONE thing: build `agents/ml_optimizer.py` — a single LightGBM model, no Optuna yet, just StratifiedKFold CV, reads `cleaned.parquet` from state, writes OOF predictions and test predictions back as file pointers. First time the pipeline produces a real CV score.