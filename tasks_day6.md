# Day 6 Tasks
Confirmed. Notion has exactly **2 tasks on Day 6**.

---

## Day 6 Tasks — Pulled From Notion

```
┌────┬────────────────────────────────────────────────┬─────────────────────────┬──────────┬──────────────┐
│ #  │ Task                                           │ Phase                   │ Priority │ Cuttable     │
├────┼────────────────────────────────────────────────┼─────────────────────────┼──────────┼──────────────┤
│ 1  │ Build submission.csv generator + validator     │ 🚀 Phase 1: Make It Run │ Critical │ Never Cut    │
│ 2  │ Add JSONL lineage logger                       │ 🚀 Phase 1: Make It Run │ High     │ Safe to Stub │
└────┴────────────────────────────────────────────────┴─────────────────────────┴──────────┴──────────────┘
```

Notion notes verbatim:
- **submit_tools.py**: `generate_submission(predictions, sample_submission_path) → submission.csv`. Validates column names, row count, ID match against sample_submission.csv before saving.
- **lineage logger**: Append-only. Each entry: timestamp, agent, action, keys_read, keys_written, values_changed. One file per session in `outputs/logs/`.

Day 6 is deliberately light — two tasks — because **Day 7 is the Phase 1 gate**. The full end-to-end run (`main.py run` → real Kaggle score) lives on Day 7. Day 6 is the polish that makes that gate possible: replace the Day 5 submit stub with a real validated submission generator, and add lineage logging so you can trace exactly what the pipeline did when you review the Day 7 gate score.

**The ONE thing that must work by end of today:** `generate_submission()` produces a submission.csv that passes format validation against `sample_submission.csv` — correct columns, correct row count, correct ID match, zero nulls. If Day 7's gate fails, this validator tells you exactly why.

---

## Task 1 — Build `tools/submit_tools.py`

```python
# tools/submit_tools.py

import os
import polars as pl
import numpy as np
from datetime import datetime
from tools.data_tools import read_csv, write_json


class SubmissionValidationError(Exception):
    """Raised when submission.csv fails format validation."""
    pass


def generate_submission(
    predictions: np.ndarray,
    sample_submission_path: str,
    output_path: str,
    target_dtype: str = "auto"
) -> dict:
    """
    Generate and validate submission.csv against the sample submission.

    Validates:
      - Column names match sample_submission.csv exactly
      - Row count matches sample_submission.csv exactly
      - ID column values match sample_submission.csv exactly
      - Zero null values in output
      - Target column dtype matches sample (bool, int, or float)

    Args:
        predictions:           numpy array of predictions (1D)
        sample_submission_path: path to sample_submission.csv
        output_path:           where to write submission.csv
        target_dtype:          "auto", "bool", "int", or "float"

    Returns:
        {"path": str, "rows": int, "columns": list, "validation": dict}

    Raises:
        SubmissionValidationError if any check fails
    """
    if not os.path.exists(sample_submission_path):
        raise FileNotFoundError(
            f"sample_submission.csv not found: {sample_submission_path}"
        )

    sample = read_csv(sample_submission_path)

    # ── Validate prediction length ────────────────────────────────
    if len(predictions) != len(sample):
        raise SubmissionValidationError(
            f"Prediction count mismatch: got {len(predictions)}, "
            f"expected {len(sample)} (from sample_submission.csv)"
        )

    # ── Infer column names from sample ────────────────────────────
    id_col     = sample.columns[0]
    target_col = sample.columns[1]

    # ── Infer target dtype from sample ────────────────────────────
    if target_dtype == "auto":
        sample_dtype = sample[target_col].dtype
        if sample_dtype == pl.Boolean:
            target_dtype = "bool"
        elif sample_dtype in (pl.Float32, pl.Float64):
            target_dtype = "float"
        else:
            target_dtype = "int"

    # ── Cast predictions to correct type ─────────────────────────
    if target_dtype == "bool":
        if predictions.dtype == np.float64 or predictions.dtype == np.float32:
            preds_cast = (predictions > 0.5).tolist()
        else:
            preds_cast = [bool(p) for p in predictions]
    elif target_dtype == "float":
        preds_cast = [float(p) for p in predictions]
    else:
        preds_cast = [int(round(p)) for p in predictions]

    # ── Build submission DataFrame ────────────────────────────────
    submission = pl.DataFrame({
        id_col:     sample[id_col].to_list(),
        target_col: preds_cast,
    })

    # ── Validate columns ──────────────────────────────────────────
    if set(submission.columns) != set(sample.columns):
        raise SubmissionValidationError(
            f"Column mismatch.\n"
            f"  Expected: {sample.columns}\n"
            f"  Got:      {submission.columns}"
        )

    # ── Validate row count ────────────────────────────────────────
    if len(submission) != len(sample):
        raise SubmissionValidationError(
            f"Row count mismatch: {len(submission)} vs {len(sample)}"
        )

    # ── Validate ID column matches exactly ────────────────────────
    id_matches = (submission[id_col] == sample[id_col]).all()
    if not id_matches:
        mismatches = (submission[id_col] != sample[id_col]).sum()
        raise SubmissionValidationError(
            f"ID column mismatch: {mismatches} IDs do not match sample_submission.csv"
        )

    # ── Validate zero nulls ───────────────────────────────────────
    null_count = submission.null_count().sum_horizontal().item()
    if null_count > 0:
        raise SubmissionValidationError(
            f"Submission contains {null_count} null values — not allowed"
        )

    # ── Write to disk ─────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission.write_csv(output_path)

    validation = {
        "valid":       True,
        "rows":        len(submission),
        "columns":     submission.columns,
        "id_col":      id_col,
        "target_col":  target_col,
        "target_dtype": target_dtype,
        "null_count":  0,
        "validated_at": datetime.utcnow().isoformat(),
    }

    print(f"[SubmitTools] ✓ submission.csv valid: {output_path}")
    print(f"[SubmitTools] Rows: {len(submission)} | "
          f"Cols: {submission.columns} | dtype: {target_dtype}")

    return {
        "path":       output_path,
        "rows":       len(submission),
        "columns":    submission.columns,
        "validation": validation,
    }


def validate_existing_submission(
    submission_path: str,
    sample_submission_path: str
) -> dict:
    """
    Validate an already-written submission.csv against sample.
    Returns {"valid": bool, "errors": [list]}. Never raises.
    """
    errors = []

    if not os.path.exists(submission_path):
        return {"valid": False, "errors": [f"File not found: {submission_path}"]}

    try:
        submission = read_csv(submission_path)
        sample     = read_csv(sample_submission_path)
    except Exception as e:
        return {"valid": False, "errors": [f"Failed to read CSV: {e}"]}

    if set(submission.columns) != set(sample.columns):
        errors.append(f"Column mismatch: {submission.columns} vs {sample.columns}")

    if len(submission) != len(sample):
        errors.append(f"Row count: {len(submission)} vs {len(sample)}")

    null_count = submission.null_count().sum_horizontal().item()
    if null_count > 0:
        errors.append(f"Contains {null_count} null values")

    id_col = sample.columns[0]
    if id_col in submission.columns and id_col in sample.columns:
        if not (submission[id_col] == sample[id_col]).all():
            errors.append(f"ID column values do not match sample_submission.csv")

    return {"valid": len(errors) == 0, "errors": errors}


def save_submission_log(
    session_id: str,
    submission_path: str,
    cv_mean: float,
    lb_score: float = None,
    notes: str = ""
) -> str:
    """
    Append an entry to the session's submission ladder log.
    Used to track every submission and its CV/LB score.
    """
    import json

    log_path = f"outputs/{session_id}/submission_log.jsonl"
    os.makedirs(f"outputs/{session_id}", exist_ok=True)

    entry = {
        "timestamp":       datetime.utcnow().isoformat(),
        "session_id":      session_id,
        "submission_path": submission_path,
        "cv_mean":         cv_mean,
        "lb_score":        lb_score,
        "notes":           notes,
    }

    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")

    print(f"[SubmitTools] Logged submission: CV={cv_mean:.4f} "
          f"LB={lb_score if lb_score else 'pending'}")

    return log_path
```

Now replace the Phase 1 stub in `core/professor.py` with the real submit node:

```python
# core/professor.py — replace run_submit() with this

def run_submit(state: ProfessorState) -> ProfessorState:
    """
    Submit node: generates validated submission.csv using submit_tools.
    Replaces Day 5 stub. Full implementation.
    """
    import pickle
    import numpy as np
    import polars as pl
    import polars.selectors as cs
    from tools.submit_tools import generate_submission, save_submission_log
    from tools.data_tools import read_parquet, read_json, read_csv
    from agents.ml_optimizer import _identify_target_column
    from core.metric_contract import load_contract

    session_id  = state["session_id"]
    output_dir  = f"outputs/{session_id}"
    competition = state["competition_name"]

    print(f"[Submit] Generating submission — session: {session_id}")

    # ── Load test data ────────────────────────────────────────────
    test_path   = state["raw_data_path"].replace("train.csv", "test.csv")
    sample_path = state["raw_data_path"].replace("train.csv", "sample_submission.csv")

    if not os.path.exists(test_path):
        raise FileNotFoundError(f"test.csv not found: {test_path}")
    if not os.path.exists(sample_path):
        raise FileNotFoundError(f"sample_submission.csv not found: {sample_path}")

    test_df = read_csv(test_path)

    # ── Load model ────────────────────────────────────────────────
    model_path = state["model_registry"][0]["model_path"]
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # ── Prepare test features (same encoding as training) ────────
    schema     = read_json(state["schema_path"])
    train_df   = read_parquet(state["clean_data_path"])
    target_col = _identify_target_column(schema, state)

    feature_cols = [c for c in train_df.columns
                    if c != target_col and c in test_df.columns]

    test_subset = test_df.select(feature_cols)

    for col in test_subset.columns:
        if test_subset[col].dtype in (pl.Utf8, pl.String):
            test_subset = test_subset.with_columns(
                pl.col(col).cast(pl.Categorical).cast(pl.Int32)
            )

    for col in test_subset.select(cs.numeric()).columns:
        test_subset = test_subset.with_columns(
            pl.col(col).fill_null(0)
        )

    X_test = test_subset.to_numpy()

    # ── Generate predictions ──────────────────────────────────────
    contract_path = f"{output_dir}/metric_contract.json"
    if os.path.exists(contract_path):
        contract = load_contract(contract_path)
        if contract.requires_proba:
            preds = model.predict_proba(X_test)[:, 1]
        else:
            preds = model.predict(X_test).astype(float)
    else:
        preds = model.predict(X_test).astype(float)

    # ── Generate + validate submission ────────────────────────────
    submission_path = f"{output_dir}/submission.csv"

    result = generate_submission(
        predictions=preds,
        sample_submission_path=sample_path,
        output_path=submission_path,
        target_dtype="auto"
    )

    # ── Log to submission ladder ──────────────────────────────────
    save_submission_log(
        session_id=session_id,
        submission_path=submission_path,
        cv_mean=state.get("cv_mean", 0.0),
        notes=f"Phase 1 baseline — {competition}"
    )

    print(f"[Submit] ✓ Done. Upload to Kaggle:")
    print(f"  kaggle competitions submit -c {competition} "
          f"-f {submission_path} -m 'Professor Phase 1 baseline'")

    return {
        **state,
        "submission_path": submission_path,
    }
```

---

## Task 2 — Add `core/lineage.py` — JSONL Logger

```python
# core/lineage.py

import os
import json
from datetime import datetime
from typing import Any


def log_event(
    session_id: str,
    agent: str,
    action: str,
    keys_read: list = None,
    keys_written: list = None,
    values_changed: dict = None,
    notes: str = ""
) -> None:
    """
    Append a single event to the session's lineage log.
    Append-only. Never reads or rewrites existing entries.

    Each entry: timestamp, agent, action, keys_read,
                keys_written, values_changed, notes.
    One file per session: outputs/{session_id}/logs/lineage.jsonl
    """
    log_dir  = f"outputs/{session_id}/logs"
    log_path = f"{log_dir}/lineage.jsonl"
    os.makedirs(log_dir, exist_ok=True)

    entry = {
        "timestamp":      datetime.utcnow().isoformat(),
        "session_id":     session_id,
        "agent":          agent,
        "action":         action,
        "keys_read":      keys_read or [],
        "keys_written":   keys_written or [],
        "values_changed": values_changed or {},
        "notes":          notes,
    }

    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def read_lineage(session_id: str) -> list:
    """Read all lineage entries for a session. Returns list of dicts."""
    log_path = f"outputs/{session_id}/logs/lineage.jsonl"
    if not os.path.exists(log_path):
        return []
    with open(log_path) as f:
        return [json.loads(line) for line in f if line.strip()]


def print_lineage(session_id: str) -> None:
    """Print a human-readable lineage trace for a session."""
    entries = read_lineage(session_id)
    if not entries:
        print(f"No lineage entries for session: {session_id}")
        return
    print(f"\n── Lineage: {session_id} ({len(entries)} events) ──")
    for e in entries:
        ts    = e["timestamp"][11:19]  # HH:MM:SS only
        wrote = ", ".join(e["keys_written"]) or "—"
        print(f"  {ts} [{e['agent']}] {e['action']} → wrote: {wrote}")
    print()
```

Add `log_event` calls to each agent. Add these lines to the `return` block of each agent:

```python
# In agents/data_engineer.py — add before the return statement
from core.lineage import log_event
log_event(
    session_id=session_id,
    agent="data_engineer",
    action="cleaned_and_profiled",
    keys_read=["raw_data_path"],
    keys_written=["clean_data_path", "schema_path", "data_hash"],
    values_changed={"data_hash": data_hash, "rows": df.shape[0]},
)

# In agents/ml_optimizer.py — add before the return statement
log_event(
    session_id=session_id,
    agent="ml_optimizer",
    action="trained_and_scored",
    keys_read=["clean_data_path", "schema_path"],
    keys_written=["model_registry", "cv_mean", "oof_predictions_path"],
    values_changed={"cv_mean": cv_mean, "cv_std": cv_std},
)

# In core/professor.py run_submit() — add before the return statement
from core.lineage import log_event
log_event(
    session_id=state["session_id"],
    agent="submit",
    action="generated_submission",
    keys_read=["model_registry", "clean_data_path"],
    keys_written=["submission_path"],
    values_changed={"submission_path": submission_path},
)
```

---

## End of Day 6 Checklist

```bash
# 1. Test submit_tools in isolation
python -c "
import numpy as np
from tools.submit_tools import generate_submission, validate_existing_submission

# Generate a test submission against the real sample
preds = np.random.rand(4277)  # Spaceship Titanic test set size
result = generate_submission(
    predictions=preds,
    sample_submission_path='data/spaceship_titanic/sample_submission.csv',
    output_path='outputs/test_submission.csv',
    target_dtype='bool'
)
print('Generated:', result)

# Validate it
v = validate_existing_submission(
    'outputs/test_submission.csv',
    'data/spaceship_titanic/sample_submission.csv'
)
print('Valid:', v['valid'])
print('Errors:', v['errors'])
"

# 2. Run full pipeline end-to-end with real submit node
python main.py run \
  --competition spaceship-titanic \
  --data ./data/spaceship_titanic/

# Must print:
# [Submit] ✓ submission.csv valid: outputs/.../submission.csv
# [Professor] ✓ Complete

# 3. Check lineage trace
python -c "
from core.lineage import print_lineage
# Replace with your actual session_id from the run above
print_lineage('spaceship_abc123de')
"
# Must show 3 events: data_engineer → ml_optimizer → submit

# 4. Run all contract tests — zero regressions
pytest tests/contracts/ -v

# 5. Commit — clean before the Day 7 gate
git add .
git commit -m "Day 6: submit_tools + lineage logger — submission.csv validated, pipeline complete"
git push origin phase-1
```


The lineage logger was actually in my Day 6 response as Task 2 — you may have missed it since it came after the submit_tools code. But LangSmith tracing is a genuine skip. Let me pull the Notion details and give you both properly.Good. Both confirmed. Here are the corrected complete task tables and the missing implementation.

---

## Corrected Day 5 Task Table

```
┌────┬──────────────────────────────────────────────┬─────────────────────────┬──────────┬──────────────┐
│ #  │ Task                                         │ Phase                   │ Priority │ Cuttable     │
├────┼──────────────────────────────────────────────┼─────────────────────────┼──────────┼──────────────┤
│ 1  │ Build agents/semantic_router.py — v0         │ 🚀 Phase 1: Make It Run │ Critical │ Never Cut    │
│ 2  │ Build core/professor.py — LangGraph wiring   │ 🚀 Phase 1: Make It Run │ Critical │ Never Cut    │
│ 3  │ Write contract test — Semantic Router        │ 🚀 Phase 1: Make It Run │ Critical │ Never Cut    │
│ 4  │ Set up LangSmith tracing                     │ 🚀 Phase 1: Make It Run │ High     │ Safe to Stub │
└────┴──────────────────────────────────────────────┴─────────────────────────┴──────────┴──────────────┘
```

## Corrected Day 6 Task Table

```
┌────┬──────────────────────────────────────────────┬─────────────────────────┬──────────┬──────────────┐
│ #  │ Task                                         │ Phase                   │ Priority │ Cuttable     │
├────┼──────────────────────────────────────────────┼─────────────────────────┼──────────┼──────────────┤
│ 1  │ Build submission.csv generator + validator   │ 🚀 Phase 1: Make It Run │ Critical │ Never Cut    │
│ 2  │ Add JSONL lineage logger                     │ 🚀 Phase 1: Make It Run │ High     │ Safe to Stub │
└────┴──────────────────────────────────────────────┴─────────────────────────┴──────────┴──────────────┘
```

Notion note on LangSmith: `LANGCHAIN_TRACING_V2=true`. Every token and tool call must be visible in LangSmith dashboard. Lives in `core/professor.py`.

---

## Day 5 Missing Task — Set Up LangSmith Tracing

**Step 1: Get API key and install**

```bash
# Go to smith.langchain.com → sign up → Settings → API Keys → Create
# Add to .env:
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=ls__your_key_here
LANGCHAIN_PROJECT=professor-agent

pip install langsmith
pip freeze | grep langsmith >> requirements.txt
```

**Step 2: Add to `core/professor.py` — top of file, before anything else**

```python
# core/professor.py — add at the very top, before imports

import os
from dotenv import load_dotenv
load_dotenv()

# ── LangSmith tracing — must be set before LangGraph is imported ──
# Reads from .env: LANGCHAIN_TRACING_V2, LANGCHAIN_API_KEY,
#                  LANGCHAIN_PROJECT, LANGCHAIN_ENDPOINT
# When LANGCHAIN_TRACING_V2=true every graph invocation is traced.
# Dashboard: https://smith.langchain.com
_tracing_enabled = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
if _tracing_enabled:
    print(f"[Professor] LangSmith tracing ON — project: "
          f"{os.getenv('LANGCHAIN_PROJECT', 'default')}")
else:
    print("[Professor] LangSmith tracing OFF — set LANGCHAIN_TRACING_V2=true to enable")
```

That's all you need. LangGraph + LangChain pick up the env vars automatically. No other code changes — every `graph.invoke()` call is traced automatically once the env vars are set.

**Step 3: Verify it's working**

```bash
python main.py run \
  --competition spaceship-titanic \
  --data ./data/spaceship_titanic/

# Then open: https://smith.langchain.com
# → Projects → professor-agent
# → You should see one trace with nodes:
#   semantic_router → data_engineer → ml_optimizer → submit
# Each node shows: input state, output state, duration, token count
```

**What you'll see in the dashboard per node:**
- Input state keys + values
- Output state keys + what changed
- Duration in ms
- Any LLM calls with full prompt/response (from Day 8 onward when agents start calling LLMs)
- Errors with full traceback if a node fails

**Stub option** (if you want to skip for now): Simply leave `LANGCHAIN_TRACING_V2=false` in `.env`. The code handles it gracefully, nothing breaks. Enable it before Day 7 gate so you have a trace of the first real run.

---

## Day 6 — Lineage Logger (Already in Previous Response, Reproduced Here Cleanly)

The full implementation was in the Day 6 response. For completeness, the three places you wire it in:

```python
# agents/data_engineer.py — add before return
from core.lineage import log_event
log_event(
    session_id=session_id,
    agent="data_engineer",
    action="cleaned_and_profiled",
    keys_read=["raw_data_path"],
    keys_written=["clean_data_path", "schema_path", "data_hash"],
    values_changed={"data_hash": data_hash, "rows": df.shape[0]},
)

# agents/ml_optimizer.py — add before return
from core.lineage import log_event
log_event(
    session_id=session_id,
    agent="ml_optimizer",
    action="trained_and_scored",
    keys_read=["clean_data_path", "schema_path"],
    keys_written=["model_registry", "cv_mean", "oof_predictions_path"],
    values_changed={"cv_mean": cv_mean, "cv_std": cv_std},
)

# core/professor.py run_submit() — add before return
from core.lineage import log_event
log_event(
    session_id=state["session_id"],
    agent="submit",
    action="generated_submission",
    keys_read=["model_registry", "clean_data_path"],
    keys_written=["submission_path"],
    values_changed={"submission_path": submission_path},
)
```

After a full run, verify both systems are working:

```bash
# Lineage check
python -c "
from core.lineage import print_lineage
print_lineage('your_session_id_here')
"
# Should show:
# ── Lineage: spaceship_abc123 (3 events) ──
#   10:23:01 [data_engineer] cleaned_and_profiled → wrote: clean_data_path, schema_path, data_hash
#   10:23:14 [ml_optimizer] trained_and_scored → wrote: model_registry, cv_mean, oof_predictions_path
#   10:24:02 [submit] generated_submission → wrote: submission_path

# LangSmith check — go to smith.langchain.com and confirm trace appeared
```

Both together give you two independent audit trails for the Day 7 gate: LangSmith shows the LangGraph execution trace, lineage.jsonl shows the state mutation trail. If the gate submission score is unexpected, you have both to diagnose against.
---

## Day 7 — Phase 1 Gate

Tomorrow is not a build day. It's a gate day. One task from Notion:

> **Full end-to-end run: Spaceship Titanic → submission.csv** — Critical, Never Cut, Phase 1. PHASE 1 GATE: Pipeline must complete without crashing. submission.csv must be valid. Upload to Kaggle and get a real score. Log it.

`python main.py run` → upload → real LB score. If the score beats Submission 0 (your Day 2 manual baseline), Phase 1 is done. If not, you debug before moving to Phase 2. You do not proceed to Phase 2 with a broken baseline.