# Professor Agent — Day 8 Implementation Guide
**For: Claude Code**
**Status: Day 7 COMPLETE — Phase 1 gate passed. All 5 regression tests frozen and green.**
**Mission: Begin Phase 2. Every task here moves Professor from "runs" to "decides".**

---

## ⚠️ NON-NEGOTIABLE RULES BEFORE YOU WRITE A SINGLE LINE

1. **Read the existing codebase first.** Run `find . -name "*.py" | head -60` and read `core/state.py`, `core/professor.py`, `agents/semantic_router.py`, `tools/data_tools.py` in full before touching anything. You must understand existing patterns before extending them.
2. **No boilerplate.** Every class, function, and test must do real work. No `# TODO`, no `pass`, no stub that returns `{}` and calls it done. If a function exists it must be correct and complete.
3. **No silent failures.** Every error must raise explicitly with a message that tells the engineer exactly what broke and how to fix it.
4. **Regression suite must stay green.** After every task run: `pytest tests/regression/test_phase1_regression.py -v`. If it breaks, fix it before moving to the next task.
5. **Build order is mandatory.** Tasks 2+7+6 all touch `core/state.py` — do them in a single editing session. Do not create state.py conflicts by editing it across multiple tasks.

---

## BUILD ORDER

```
Task 1  →  FIX: ChromaDB embedding verification
Task 2  →  core/state.py: add task_type field
Task 7  →  core/state.py: add data_hash + version guard  (same file, same session)
Task 6  →  core/state.py + semantic_router.py: add competition_context  (same file, same session)
            ── commit: "Day 8: state.py Phase 2 fields" ──
Task 3  →  Build agents/validation_architect.py
Task 4  →  Write contract test — Validation Architect
            ── commit: "Day 8: validation_architect + contract test" ──
Task 5  →  Build agents/eda_agent.py
            ── commit: "Day 8: eda_agent" ──
Task 8  →  GM-CAP 1: agents/competition_intel.py scraper upgrade
            ── commit: "Day 8: competition_intel GM-CAP 1 forum scraper" ──
```

---

## TASK 1 — FIX: ChromaDB Silent Fallback to Random Embeddings

**File:** `memory/chroma_client.py`
**Priority:** CRITICAL — must be done first. Every memory operation in all subsequent tasks depends on this being correct.
**Why it matters:** ChromaDB silently falls back to random embeddings if `all-MiniLM-L6-v2` fails to load. Random embeddings = every memory query returns random results. Optuna warm-start becomes pure noise. The failure is invisible — the system runs normally but produces garbage memory retrievals.

### Pre-flight
```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```
If this fails, the model is not downloaded. Run it and wait for the download before proceeding.

### Implementation

Rewrite `build_chroma_client()` in its entirety. The existing version must be replaced — not patched.

```python
# memory/chroma_client.py

import chromadb
from chromadb.utils import embedding_functions
from typing import Optional

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM   = 384
CHROMA_PATH     = "memory/chroma"


def _build_embedding_function() -> embedding_functions.SentenceTransformerEmbeddingFunction:
    """
    Explicitly constructs and validates the SentenceTransformer embedding function.
    Never relies on ChromaDB's default embedding — that path leads to silent random embeddings.
    Raises RuntimeError with actionable instructions if the model is not available.
    """
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    # Validate the model actually loaded by running a test embedding
    try:
        test_result = ef(["embedding validation probe"])
        assert len(test_result) == 1, "Expected exactly 1 embedding vector"
        assert len(test_result[0]) == EMBEDDING_DIM, (
            f"Expected embedding dim {EMBEDDING_DIM}, got {len(test_result[0])}. "
            f"Wrong model loaded or model is corrupted."
        )
    except Exception as e:
        raise RuntimeError(
            f"\n[ChromaDB] Embedding model '{EMBEDDING_MODEL}' failed validation.\n"
            f"Error: {e}\n\n"
            f"Fix: Pre-download the model before running Professor:\n"
            f"  python -c \"from sentence_transformers import SentenceTransformer; "
            f"SentenceTransformer('{EMBEDDING_MODEL}')\"\n"
            f"This downloads ~80MB from HuggingFace. Run once per machine."
        ) from e

    print(f"[ChromaDB] Embedding model verified: {EMBEDDING_MODEL} ({EMBEDDING_DIM}-dim)")
    return ef


def build_chroma_client(persist_dir: str = CHROMA_PATH) -> chromadb.ClientAPI:
    """
    Returns a PersistentClient with a validated embedding function.
    Call this once at startup. Store the result — do not call on every query.
    """
    ef = _build_embedding_function()
    client = chromadb.PersistentClient(path=persist_dir)

    # Attach the validated embedding function to the client for downstream use
    client._professor_ef = ef

    return client


def get_or_create_collection(
    client: chromadb.ClientAPI,
    name: str,
) -> chromadb.Collection:
    """
    Gets or creates a named collection using the validated embedding function.
    Always call this instead of client.get_or_create_collection() directly —
    that path does not guarantee the correct embedding function is used.
    """
    ef = getattr(client, "_professor_ef", None)
    if ef is None:
        raise RuntimeError(
            "[ChromaDB] Client was not created via build_chroma_client(). "
            "Embedding function is missing. Fix: use build_chroma_client()."
        )
    return client.get_or_create_collection(name=name, embedding_function=ef)
```

### Verification
```bash
python -c "
from memory.chroma_client import build_chroma_client, get_or_create_collection
client = build_chroma_client()
col = get_or_create_collection(client, 'test_verify')
col.add(documents=['tabular classification lightgbm'], ids=['v1'])
results = col.query(query_texts=['gradient boosting tabular'], n_results=1)
print('Query result:', results['documents'])
print('[PASS] ChromaDB embedding verified end-to-end')
"
```
Expected output: the query returns the document. If it returns nothing or errors, do not proceed.

---

## TASK 2+7+6 — core/state.py: Phase 2 Fields

**Files:** `core/state.py`, `tools/data_tools.py`, `agents/semantic_router.py`
**Do all three in a single editing session. They all touch the same TypedDict.**

### 2a — Add `task_type` to ProfessorState

`task_type` gates which execution path every Phase 2 agent takes. Without it, agents cannot know if they're working on tabular, time-series, or NLP data.

Add to `ProfessorState`:
```python
from typing import Literal

# In ProfessorState TypedDict:
task_type: Literal["tabular", "timeseries", "nlp", "image", "unknown"]
```

Add to `initial_state()`:
```python
"task_type": "unknown",  # overwritten by Semantic Router on first node
```

### 2b — Add `data_hash` to ProfessorState (GAP 13)

Kaggle hosts release corrected data mid-competition. Without version tracking, the Ensemble Architect silently mixes models trained on different data versions.

**In `tools/data_tools.py`**, add:
```python
import hashlib

def hash_dataset(file_path: str) -> str:
    """
    SHA-256 of raw file bytes, truncated to 16 hex chars.
    Called by Data Engineer on every data load.
    Stored in ProfessorState and in every model_registry entry.
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()[:16]
```

**In `ProfessorState`**, add:
```python
data_hash: str  # SHA-256[:16] of train.csv — set by Data Engineer
```

**In `initial_state()`**, add:
```python
"data_hash": "",
```

**In `agents/data_engineer.py`**, at the point where raw data is loaded, add:
```python
from tools.data_tools import hash_dataset
state["data_hash"] = hash_dataset(state["raw_data_path"])
```

**In `agents/ml_optimizer.py`**, in the model_registry entry construction, add:
```python
"data_hash": state.get("data_hash", ""),
```

### 2c — Add `competition_context` to ProfessorState (GAP 14)

The Supervisor has no world model. It makes routing decisions without knowing if it should conserve a top rank or swing aggressively for a better one. This field is the fix.

**In `ProfessorState`**, add:
```python
competition_context: dict  # {days_remaining, submissions_remaining,
                           #  current_public_rank, total_competitors,
                           #  current_percentile, shakeup_risk, strategy}
```

**In `initial_state()`**, add:
```python
"competition_context": {
    "days_remaining":        None,
    "hours_remaining":       None,
    "submissions_used":      0,
    "submissions_remaining": None,
    "current_public_rank":   None,
    "total_competitors":     None,
    "current_percentile":    None,
    "shakeup_risk":          "unknown",  # "low" | "medium" | "high" | "unknown"
    "strategy":              "balanced", # "conservative" | "balanced" | "aggressive"
    "last_updated":          None,
},
```

**In `agents/semantic_router.py`**, add strategy determination logic:

```python
def _determine_strategy(context: dict) -> str:
    """
    Called by router after every submission or at pipeline start.
    Returns the strategy the Supervisor should use for routing.
    """
    percentile    = context.get("current_percentile")
    days_remaining = context.get("days_remaining")

    if percentile is None or days_remaining is None:
        return "balanced"  # not enough data yet

    if days_remaining <= 2 and percentile <= 0.10:
        return "conservative"   # top 10%, almost done — lock in the rank
    if days_remaining <= 2 and percentile > 0.10:
        return "aggressive"     # running out of time, need big moves
    if days_remaining > 7 and percentile > 0.40:
        return "aggressive"     # plenty of time, far from goal
    return "balanced"
```

Wire `_determine_strategy` into the router: after the Intel Agent populates `competition_context`, call `_determine_strategy(state["competition_context"])` and write the result back to `state["competition_context"]["strategy"]`.

### Verification
```bash
python -c "
from core.state import initial_state
s = initial_state('test', 'data/spaceship_titanic/train.csv')
assert s['task_type'] == 'unknown'
assert s['data_hash'] == ''
assert s['competition_context']['strategy'] == 'balanced'
print('[PASS] All Phase 2 state fields present and initialised correctly')
"
pytest tests/regression/test_phase1_regression.py -v
# Must stay green. The new fields have defaults — nothing should break.
```

---

## TASK 3 — Build `agents/validation_architect.py`

**File:** `agents/validation_architect.py`
**Priority:** CRITICAL — this is the agent that makes the CV strategy decision. Without it, Professor uses StratifiedKFold blindly on every competition regardless of structure.

### What This Agent Does

The Validation Architect runs after the Data Engineer and before the ML Optimizer. It:
1. Reads `schema.json` and `competition_brief.json`
2. Detects which CV strategy is correct for this competition
3. Writes `validation_strategy.json` and `metric_contract.json`
4. If a CV/LB mismatch signature is detected, sets `hitl_required=True` and halts

### CV Strategy Detection Logic

This is the core intelligence of the agent. Do not use an LLM for this — use deterministic rules:

```
Has group column in schema?      → GroupKFold(groups=group_col)
Has datetime column in schema?   → TimeSeriesSplit(n_splits=5)
target is binary/multiclass?     → StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
target is continuous?            → KFold(n_splits=5, shuffle=True, random_state=42)
```

### CV/LB Mismatch Detection

A mismatch is likely when:
- Competition has time-based structure but StratifiedKFold was selected (splits leak future data)
- Competition has a group ID column (rows from same group appear in both train and val)
- Target distribution differs significantly between train and test set (detected from competition_brief)

When mismatch is detected:
```python
state["hitl_required"] = True
state["hitl_reason"]   = "CV/LB mismatch risk: [specific reason]. Manual review required before optimizer runs."
# Do NOT proceed to optimizer. Return state here.
```

### Full Implementation

```python
# agents/validation_architect.py

import os
import json
import polars as pl
from typing import Optional
from core.state import ProfessorState
from core.metric_contract import build_metric_contract, save_contract
from tools.data_tools import read_json
from core.lineage import log_event


_CV_STRATEGY_RULES = {
    "group":       "GroupKFold",
    "timeseries":  "TimeSeriesSplit",
    "stratified":  "StratifiedKFold",
    "kfold":       "KFold",
}

_DATETIME_DTYPES = {
    "Date", "Datetime", "Time", "Duration",
    pl.Date, pl.Datetime, pl.Time, pl.Duration,
}


def _detect_group_column(schema: dict) -> Optional[str]:
    """Return the name of a group/ID column if one exists in schema."""
    group_keywords = ["group", "patient", "user_id", "customer_id",
                      "store_id", "site_id", "subject", "household"]
    for col in schema.get("columns", []):
        if any(kw in col.lower() for kw in group_keywords):
            return col
    return None


def _detect_datetime_column(schema: dict) -> Optional[str]:
    """Return the name of a datetime column if one exists."""
    time_keywords = ["date", "time", "timestamp", "year", "month", "week"]
    types = schema.get("types", {})
    for col, dtype in types.items():
        if any(kw in col.lower() for kw in time_keywords):
            return col
        if str(dtype) in {"Date", "Datetime", "Time"}:
            return col
    return None


def _detect_target_type(schema: dict, target_col: str) -> str:
    """Returns 'binary', 'multiclass', or 'continuous'."""
    types = schema.get("types", {})
    dtype = str(types.get(target_col, ""))

    if dtype in {"Boolean", "bool"}:
        return "binary"

    n_unique = schema.get("n_unique", {}).get(target_col)
    if n_unique is not None:
        if n_unique == 2:
            return "binary"
        if 2 < n_unique <= 20:
            return "multiclass"
        return "continuous"

    # Fall back to dtype heuristics
    if "Int" in dtype or "UInt" in dtype:
        return "multiclass"
    if "Float" in dtype:
        return "continuous"
    return "binary"  # safe default for unknown


def _detect_cv_mismatch_risk(
    cv_type: str,
    datetime_col: Optional[str],
    group_col: Optional[str],
    brief: dict,
) -> Optional[str]:
    """
    Returns a mismatch reason string if risk is detected, else None.
    These are the patterns that produce inflated CV scores that collapse on LB.
    """
    if datetime_col and cv_type == "StratifiedKFold":
        return (
            f"Datetime column '{datetime_col}' detected but CV strategy is StratifiedKFold. "
            f"This splits time-ordered data randomly, leaking future information into training folds. "
            f"Use TimeSeriesSplit instead."
        )

    if group_col and cv_type in ("StratifiedKFold", "KFold"):
        return (
            f"Group column '{group_col}' detected but CV strategy does not respect groups. "
            f"Rows from the same group will appear in both train and validation, inflating CV. "
            f"Use GroupKFold instead."
        )

    # Check if brief explicitly mentions LB shakeup risk
    if brief.get("known_pitfalls"):
        pitfalls = str(brief["known_pitfalls"]).lower()
        if "shake" in pitfalls or "lb gap" in pitfalls or "public lb" in pitfalls:
            return (
                "Competition brief flags known public/private LB gap risk. "
                "Validate CV strategy carefully before trusting public scores."
            )

    return None


def run_validation_architect(state: ProfessorState) -> ProfessorState:
    """
    LangGraph node: Validation Architect.

    Reads:  state["schema_path"]          — schema.json from Data Engineer
            state["competition_brief_path"] — competition_brief.json from Intel Agent (optional)
    Writes: validation_strategy.json      — cv_type, n_splits, group_col, datetime_col
            metric_contract.json          — scorer_fn, direction, forbidden_metrics
            state["validation_strategy"]  — dict
            state["hitl_required"]        — True if CV/LB mismatch detected
    """
    session_id = state["session_id"]
    output_dir = f"outputs/{session_id}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"[ValidationArchitect] Starting — session: {session_id}")

    # ── Load schema ────────────────────────────────────────────────────────────
    if not state.get("schema_path") or not os.path.exists(state["schema_path"]):
        raise ValueError(
            "[ValidationArchitect] schema_path missing or file not found. "
            "Run Data Engineer first."
        )
    schema = read_json(state["schema_path"])

    # ── Load competition brief (optional — may not exist in Phase 1) ──────────
    brief = {}
    brief_path = state.get("competition_brief_path", "")
    if brief_path and os.path.exists(brief_path):
        brief = read_json(brief_path)
        print(f"[ValidationArchitect] Competition brief loaded: {brief_path}")
    else:
        print("[ValidationArchitect] No competition brief — using schema only")

    # ── Determine target column ────────────────────────────────────────────────
    target_col = state.get("target_col") or schema.get("target_col")
    if not target_col:
        # Fall back: last column in schema
        cols = schema.get("columns", [])
        target_col = cols[-1] if cols else None
    if not target_col:
        raise ValueError("[ValidationArchitect] Cannot determine target column.")

    # ── CV strategy detection ──────────────────────────────────────────────────
    group_col    = _detect_group_column(schema)
    datetime_col = _detect_datetime_column(schema)
    target_type  = _detect_target_type(schema, target_col)

    if group_col:
        cv_type  = "GroupKFold"
        n_splits = 5
    elif datetime_col:
        cv_type  = "TimeSeriesSplit"
        n_splits = 5
    elif target_type in ("binary", "multiclass"):
        cv_type  = "StratifiedKFold"
        n_splits = 5
    else:
        cv_type  = "KFold"
        n_splits = 5

    print(f"[ValidationArchitect] CV strategy: {cv_type}(n_splits={n_splits})")
    if group_col:
        print(f"[ValidationArchitect] Group column: {group_col}")
    if datetime_col:
        print(f"[ValidationArchitect] Datetime column: {datetime_col}")

    # ── CV/LB mismatch detection ───────────────────────────────────────────────
    mismatch_reason = _detect_cv_mismatch_risk(cv_type, datetime_col, group_col, brief)
    if mismatch_reason:
        print(f"[ValidationArchitect] ⚠️  CV/LB MISMATCH RISK: {mismatch_reason}")
        validation_strategy = {
            "cv_type":          cv_type,
            "n_splits":         n_splits,
            "group_col":        group_col,
            "datetime_col":     datetime_col,
            "target_col":       target_col,
            "target_type":      target_type,
            "mismatch_risk":    mismatch_reason,
            "hitl_required":    True,
        }
        strategy_path = f"{output_dir}/validation_strategy.json"
        with open(strategy_path, "w") as f:
            json.dump(validation_strategy, f, indent=2)

        log_event(
            session_id=session_id,
            agent="validation_architect",
            action="halted_cv_mismatch",
            keys_read=["schema_path"],
            keys_written=["validation_strategy"],
            values_changed={"mismatch_reason": mismatch_reason},
        )

        return {
            **state,
            "validation_strategy":      validation_strategy,
            "validation_strategy_path": strategy_path,
            "hitl_required":            True,
            "hitl_reason":              f"CV/LB mismatch risk: {mismatch_reason}",
        }

    # ── Determine metric ───────────────────────────────────────────────────────
    # Priority: competition_brief > task_type in state > target_type heuristic
    scorer_name = brief.get("evaluation_metric", "").lower().strip()
    task_type   = state.get("task_type", "unknown")

    if not scorer_name:
        # Heuristic fallback
        if target_type == "binary":
            scorer_name = "auc"
        elif target_type == "multiclass":
            scorer_name = "f1_weighted"
        else:
            scorer_name = "rmse"

    if task_type == "unknown":
        task_type = "classification" if target_type in ("binary", "multiclass") else "regression"

    print(f"[ValidationArchitect] Metric: {scorer_name} | Task: {task_type}")

    # ── Build and save MetricContract ──────────────────────────────────────────
    contract      = build_metric_contract(scorer_name, task_type, state["competition_name"])
    contract_path = f"{output_dir}/metric_contract.json"
    save_contract(contract, contract_path)
    print(f"[ValidationArchitect] MetricContract saved: {contract_path}")

    # ── Build and save validation strategy ────────────────────────────────────
    validation_strategy = {
        "cv_type":          cv_type,
        "n_splits":         n_splits,
        "group_col":        group_col,
        "datetime_col":     datetime_col,
        "target_col":       target_col,
        "target_type":      target_type,
        "scorer_name":      scorer_name,
        "task_type":        task_type,
        "mismatch_risk":    None,
        "hitl_required":    False,
    }
    strategy_path = f"{output_dir}/validation_strategy.json"
    with open(strategy_path, "w") as f:
        json.dump(validation_strategy, f, indent=2)

    log_event(
        session_id=session_id,
        agent="validation_architect",
        action="strategy_decided",
        keys_read=["schema_path"],
        keys_written=["validation_strategy", "metric_contract"],
        values_changed={
            "cv_type": cv_type,
            "scorer_name": scorer_name,
            "task_type": task_type,
        },
    )

    print(f"[ValidationArchitect] Complete.")

    return {
        **state,
        "validation_strategy":      validation_strategy,
        "validation_strategy_path": strategy_path,
        "metric_contract_path":     contract_path,
        "task_type":                task_type,
        "hitl_required":            False,
    }
```

### Wire into LangGraph

In `core/professor.py`:
1. Import `run_validation_architect`
2. Add node: `graph.add_node("validation_architect", run_validation_architect)`
3. Update edges: `data_engineer → validation_architect → ml_optimizer`
4. Add conditional edge on `hitl_required`: if True, route to HITL handler, not ml_optimizer

---

## TASK 4 — Contract Test: Validation Architect

**File:** `tests/contracts/test_validation_architect_contract.py`
**Status after writing: IMMUTABLE — do not edit this file after Day 8**

```python
# tests/contracts/test_validation_architect_contract.py
# ─────────────────────────────────────────────────────────────────────────────
# Written: Day 8
# Status:  IMMUTABLE — never edit this file after today
#
# CONTRACT: run_validation_architect()
#   INPUT:   state["schema_path"]            — must exist
#   OUTPUT:  validation_strategy.json        — must have cv_type/n_splits/group_col
#            metric_contract.json            — must have scorer_fn/direction/forbidden_metrics
#   BLOCKER: CV/LB mismatch must set hitl_required=True and NOT proceed to optimizer
#   NEVER:   Return unknown scorer name. Use forbidden metrics. Proceed past mismatch.
# ─────────────────────────────────────────────────────────────────────────────
import pytest
import os
import json
from core.state import initial_state
from agents.data_engineer import run_data_engineer
from agents.validation_architect import run_validation_architect

FIXTURE_CSV = "tests/fixtures/tiny_train.csv"


@pytest.fixture(scope="module")
def validated_state():
    state = initial_state("test-validation", FIXTURE_CSV, budget_usd=2.0)
    state = run_data_engineer(state)
    state = run_validation_architect(state)
    return state


class TestValidationArchitectContract:

    def test_runs_without_error(self, validated_state):
        assert validated_state is not None

    def test_validation_strategy_json_exists(self, validated_state):
        path = validated_state.get("validation_strategy_path")
        assert path is not None, "validation_strategy_path must be in state"
        assert os.path.exists(path), f"validation_strategy.json not found at {path}"

    def test_validation_strategy_has_cv_type(self, validated_state):
        vs = validated_state["validation_strategy"]
        assert "cv_type" in vs
        assert vs["cv_type"] in (
            "StratifiedKFold", "KFold", "GroupKFold", "TimeSeriesSplit"
        ), f"cv_type '{vs['cv_type']}' is not a recognised CV strategy"

    def test_validation_strategy_has_n_splits(self, validated_state):
        vs = validated_state["validation_strategy"]
        assert "n_splits" in vs
        assert isinstance(vs["n_splits"], int)
        assert 2 <= vs["n_splits"] <= 10

    def test_validation_strategy_has_group_col_key(self, validated_state):
        vs = validated_state["validation_strategy"]
        assert "group_col" in vs  # may be None — that's fine

    def test_metric_contract_json_exists(self, validated_state):
        path = validated_state.get("metric_contract_path")
        assert path is not None
        assert os.path.exists(path)

    def test_metric_contract_has_scorer_name(self, validated_state):
        path = validated_state["metric_contract_path"]
        mc   = json.load(open(path))
        assert "scorer_name" in mc
        assert isinstance(mc["scorer_name"], str)
        assert len(mc["scorer_name"]) > 0

    def test_metric_contract_has_direction(self, validated_state):
        path = validated_state["metric_contract_path"]
        mc   = json.load(open(path))
        assert mc["direction"] in ("maximize", "minimize")

    def test_metric_contract_has_forbidden_metrics(self, validated_state):
        path = validated_state["metric_contract_path"]
        mc   = json.load(open(path))
        assert "forbidden_metrics" in mc
        assert isinstance(mc["forbidden_metrics"], list)

    def test_task_type_written_to_state(self, validated_state):
        assert "task_type" in validated_state
        assert validated_state["task_type"] in (
            "tabular", "timeseries", "nlp", "image", "classification", "regression"
        )

    def test_no_hitl_on_clean_tabular_data(self, validated_state):
        # The fixture is clean tabular data — should not trigger HITL
        assert validated_state.get("hitl_required") is False, (
            "HITL triggered on clean tabular fixture — check mismatch detection logic"
        )

    def test_mismatch_triggers_hitl(self):
        """Inject a datetime column into schema and verify HITL triggers."""
        import tempfile, copy
        from core.state import initial_state
        from agents.data_engineer import run_data_engineer

        state = initial_state("test-mismatch", FIXTURE_CSV)
        state = run_data_engineer(state)

        # Patch the schema to include a datetime column
        schema = json.load(open(state["schema_path"]))
        schema["columns"].append("transaction_date")
        schema["types"]["transaction_date"] = "Date"

        patched_path = state["schema_path"].replace(".json", "_patched.json")
        with open(patched_path, "w") as f:
            json.dump(schema, f)

        state["schema_path"] = patched_path
        # Force StratifiedKFold to be selected by removing any hints
        result = run_validation_architect(state)

        assert result["hitl_required"] is True, (
            "Datetime column should trigger CV/LB mismatch detection and set hitl_required=True"
        )
        assert "hitl_reason" in result
        assert len(result["hitl_reason"]) > 0

    def test_mismatch_halts_before_writing_metric_contract(self):
        """When mismatch is detected, validation_strategy.json must be written but
        the pipeline must not continue to ML Optimizer."""
        import tempfile
        from core.state import initial_state
        from agents.data_engineer import run_data_engineer

        state = initial_state("test-halt", FIXTURE_CSV)
        state = run_data_engineer(state)

        schema = json.load(open(state["schema_path"]))
        schema["columns"].append("patient_id")
        schema["types"]["patient_id"] = "Utf8"
        patched_path = state["schema_path"].replace(".json", "_group_patched.json")
        with open(patched_path, "w") as f:
            json.dump(schema, f)

        state["schema_path"] = patched_path
        result = run_validation_architect(state)

        assert result["hitl_required"] is True
        # validation_strategy.json must exist (for debugging)
        assert os.path.exists(result["validation_strategy_path"])
        # metric_contract_path should NOT be in state when halted
        assert result.get("metric_contract_path") is None or not os.path.exists(
            result.get("metric_contract_path", "")
        ), "metric_contract.json must not be written when mismatch halts the pipeline"
```

### Run
```bash
pytest tests/contracts/test_validation_architect_contract.py -v
pytest tests/contracts/ -v  # full suite must stay green
pytest tests/regression/test_phase1_regression.py -v
```

---

## TASK 5 — Build `agents/eda_agent.py` (GAP 1)

**File:** `agents/eda_agent.py`
**Priority:** HIGH — without this, the Feature Factory builds features blind. This agent is inserted between Data Engineer and Feature Factory and produces `eda_report.json` that both downstream agents read.

### What It Must Produce

`eda_report.json` with exactly these keys — no more, no less:

```json
{
  "target_distribution": {
    "skew": float,
    "kurtosis": float,
    "is_multimodal": bool,
    "recommended_transform": "none" | "log" | "sqrt" | "boxcox"
  },
  "feature_correlations": [
    {"feature": str, "correlation": float, "relationship_type": "linear" | "nonlinear" | "none"}
  ],
  "outlier_profile": [
    {"column": str, "strategy": "winsorize" | "cap" | "remove" | "keep", "n_outliers": int, "pct_outliers": float}
  ],
  "duplicate_analysis": {
    "exact_count": int,
    "near_duplicate_count": int,
    "id_conflict_count": int,
    "id_conflict_columns": [str]
  },
  "temporal_profile": {
    "has_dates": bool,
    "date_columns": [str],
    "seasonality_detected": bool,
    "train_test_drift_risk": bool
  },
  "leakage_fingerprint": [
    {"feature": str, "target_correlation": float, "verdict": "FLAG" | "WATCH" | "OK"}
  ],
  "drop_candidates": [str],
  "summary": str
}
```

### Implementation Rules

- Use Polars throughout — no Pandas.
- `skew` and `kurtosis` apply only to continuous targets. For boolean/categorical: set to 0.0.
- Leakage fingerprint: any feature with absolute correlation to target > 0.95 = `"FLAG"`, 0.80–0.95 = `"WATCH"`, < 0.80 = `"OK"`.
- Outlier strategy per column:
  - `pct_outliers < 1%` → `"keep"`
  - `pct_outliers 1–5%` → `"winsorize"`
  - `pct_outliers 5–10%` → `"cap"`
  - `pct_outliers > 10%` → `"remove"` (flag for human review)
- ID conflict: find columns where the same ID appears with different target values. This is the most dangerous form of label noise.
- `drop_candidates`: list of column names that are either flagged leakage or have zero variance.
- `summary`: a single human-readable paragraph the Critic and Feature Factory will read in their system prompts.

### Pipeline Position

```python
# In core/professor.py:
# Old: data_engineer → validation_architect → ml_optimizer
# New: data_engineer → eda_agent → validation_architect → ml_optimizer
```

Wire `run_eda_agent` as a LangGraph node between data_engineer and validation_architect. Both the Validation Architect and the Feature Factory will read `state["eda_report_path"]`.

Add to `ProfessorState`:
```python
eda_report_path: str  # path to eda_report.json
eda_report: dict      # the parsed eda_report contents
```

Add to `initial_state()`:
```python
"eda_report_path": "",
"eda_report": {},
```

### Verification
```bash
python -c "
from core.state import initial_state
from agents.data_engineer import run_data_engineer
from agents.eda_agent import run_eda_agent
import json

state = initial_state('test-eda', 'data/spaceship_titanic/train.csv')
state = run_data_engineer(state)
state = run_eda_agent(state)

report = state['eda_report']
required_keys = ['target_distribution','feature_correlations','outlier_profile',
                 'duplicate_analysis','temporal_profile','leakage_fingerprint',
                 'drop_candidates','summary']
for k in required_keys:
    assert k in report, f'Missing key: {k}'
print(json.dumps(report['target_distribution'], indent=2))
print(f'Drop candidates: {report[\"drop_candidates\"]}')
print(f'Leakage flags: {[f for f in report[\"leakage_fingerprint\"] if f[\"verdict\"]==\"FLAG\"]}')
print('[PASS] EDA report complete')
"
```

---

## TASK 8 — GM-CAP 1: Kaggle Forum + Notebook Intelligence Scraper

**File:** `agents/competition_intel.py`
**Priority:** CRITICAL — this is the highest-leverage Phase 2 upgrade. Every GM wins in the forums first. Without this, Professor is blind to 80% of competitive signal.

### What This Agent Must Scrape

Using the Kaggle API (`kaggle.api`), scrape for a given competition:

1. **Discussion posts** — sorted by votes, top 20. Extract: title, vote count, author, date, full text.
2. **Host Q&A responses** — posts by the competition organiser (filter by `is_host_author`). These are gold — they clarify metric rules, data quirks, and allowed techniques.
3. **Public notebooks** — by vote rank, top 10. Extract: title, votes, author, first 500 chars of markdown summary.
4. **Data tab comments** — if accessible via API.

### LLM Synthesis

After scraping, call LLM present here (deepseek or LGM 5 ) with all scraped text and produce `intel_brief.json`:

```json
{
  "critical_findings": [str],    // things that will make or break the score
  "proven_features": [str],      // feature engineering that works — from public notebooks
  "known_leaks": [str],          // data leakage others have found
  "external_datasets": [str],    // external data sources proven to help
  "dominant_approach": str,      // the model/approach winning the LB
  "cv_strategy_hint": str,       // what CV approach the community uses
  "forbidden_techniques": [str], // what the host has explicitly banned
  "shakeup_risk": "low" | "medium" | "high",  // based on competition type and history
  "source_post_count": int,
  "scraped_at": str              // ISO timestamp
}
```

### System Prompt for LLM Synthesis

```
You are a Kaggle Grandmaster reading competition forum posts.
Extract only actionable competitive intelligence — not general ML advice.
For critical_findings: only include things that are non-obvious and will affect the final score.
For known_leaks: only include leaks that have been confirmed by multiple sources or by the host.
For proven_features: only include features that improved scores in shared notebooks.
Be specific. No vague statements like "feature engineering is important".
Output ONLY valid JSON matching the schema. No preamble, no explanation.
```

### Rate Limiting and Graceful Degradation

The Kaggle API has rate limits. Handle them:
- Wrap all API calls in `try/except` with exponential backoff (3 retries, 60s base delay)
- If the competition does not have forum data (private/unreleased competitions), return a minimal brief with `"source_post_count": 0` and continue — do not crash
- Log every scrape attempt and result to lineage

### Injection Into Downstream Agents

After `intel_brief.json` is written, its `summary` must be injected into:
- `agents/validation_architect.py` system prompt (as `cv_strategy_hint`)
- `agents/eda_agent.py` system prompt (as `critical_findings` and `known_leaks`)
- `agents/semantic_router.py` (to override `task_type` if `dominant_approach` indicates a different modality)

This injection happens via `state["competition_brief_path"]` which all agents already read. The scraper writes `competition_brief.json` in the output directory.

### Wire into LangGraph

```python
# In core/professor.py:
# New pipeline start:
# semantic_router → competition_intel → data_engineer → eda_agent → validation_architect → ml_optimizer
```

`competition_intel` runs before the data pipeline so its findings can influence EDA and validation strategy.

Add to `ProfessorState`:
```python
competition_brief_path: str
competition_brief: dict
intel_brief_path: str
```

### Verification
```bash
python -c "
from agents.competition_intel import run_competition_intel
from core.state import initial_state

state = initial_state('spaceship-titanic', 'data/spaceship_titanic/train.csv')
state['competition_name'] = 'spaceship-titanic'
state = run_competition_intel(state)

import json
brief = state['competition_brief']
print(json.dumps(brief, indent=2))

required = ['critical_findings','proven_features','known_leaks',
            'dominant_approach','shakeup_risk','source_post_count']
for k in required:
    assert k in brief, f'Missing key: {k}'
print('[PASS] Intel brief complete — source posts:', brief['source_post_count'])
"
```

---

## END OF DAY CHECKLIST

Run all of these in order. Every command must be green before the next.

```bash
# 1. Regression suite — must not break
pytest tests/regression/test_phase1_regression.py -v

# 2. Contract tests — all must pass
pytest tests/contracts/ -v

# 3. Full end-to-end smoke test
python -c "
from core.state import initial_state
from agents.competition_intel import run_competition_intel
from agents.data_engineer import run_data_engineer
from agents.eda_agent import run_eda_agent
from agents.validation_architect import run_validation_architect
from agents.ml_optimizer import run_ml_optimizer

state = initial_state('spaceship-titanic', 'data/spaceship_titanic/train.csv')
state = run_competition_intel(state)
state = run_data_engineer(state)
state = run_eda_agent(state)
state = run_validation_architect(state)
state = run_ml_optimizer(state)

print(f'CV: {state[\"cv_mean\"]:.4f}')
print(f'Task type: {state[\"task_type\"]}')
print(f'CV strategy: {state[\"validation_strategy\"][\"cv_type\"]}')
print(f'Drop candidates: {state[\"eda_report\"][\"drop_candidates\"]}')
print(f'Intel sources: {state[\"competition_brief\"][\"source_post_count\"]}')
print('[PASS] Day 8 pipeline complete end-to-end')
"

# 4. Commit
git add .
git commit -m "Day 8: ChromaDB fix, Phase 2 state fields, validation_architect, eda_agent, competition_intel GM-CAP 1 — all contracts green"
git push origin phase-2
```

### Definition of Done for Day 8

- [ ] ChromaDB embedding model validated at startup — runtime error with fix instructions if missing
- [ ] `task_type` field in `ProfessorState`, set by Semantic Router
- [ ] `data_hash` in `ProfessorState` and in every `model_registry` entry
- [ ] `competition_context` in `ProfessorState` with strategy logic in Semantic Router
- [ ] `validation_architect.py` running, detecting CV strategy from schema, halting on mismatch
- [ ] Contract test for Validation Architect: all tests green, file is immutable
- [ ] `eda_agent.py` running, producing all 8 required keys in `eda_report.json`
- [ ] `competition_intel.py` scraping forum + notebooks, writing `intel_brief.json`, injecting into downstream agents
- [ ] `pytest tests/regression/test_phase1_regression.py` — all green (Phase 1 baseline unchanged)
- [ ] `pytest tests/contracts/` — all green
- [ ] `python main.py run` completes end-to-end on Spaceship Titanic without error

---

## WHAT PODIUM WORK LOOKS LIKE ON THIS DAY

Day 8 is the day Professor stops being a script that runs ML and starts being an agent that makes decisions. By end of day, Professor should be able to tell you:

- *"This competition has a group structure — I'm using GroupKFold so your CV correlates with the LB"*
- *"The forums say log-transform on the target is critical — I've flagged this in the EDA report"*
- *"There's a known data leak in the `customer_id` column — I've added it to drop_candidates"*
- *"I'm 8 days out and in the top 35% — strategy is balanced, continuing feature iteration"*

If Professor cannot say any of those things after Day 8, something is missing.