# Day 2 Tasks
Got everything. Day 2 has exactly 4 tasks — all Critical, all Never Cut.

```
Task 1:  Build tools/e2b_sandbox.py — full RestrictedPython sandbox
Task 2:  Add 3-attempt inner retry loop to sandbox
Task 3:  Write contract test for the sandbox (immutable from today)
Task 4:  Manual Submission 0 — Spaceship Titanic by hand
```

**The ONE thing that must work by end of today:**
The sandbox executes code reliably with retries, AND you have a real Kaggle score on Spaceship Titanic to beat. That score is the floor everything Professor builds toward.

---

## Task 1 + 2 — Build `tools/e2b_sandbox.py` With Retry Loop

Both tasks live in the same file. Build them together.

```python
# tools/e2b_sandbox.py

import os
import sys
import signal
import traceback
from typing import Optional
from RestrictedPython import compile_restricted, safe_globals, safe_builtins
from RestrictedPython.Guards import safe_iter_unpack_sequence, guarded_iter_unpack_sequence
from dotenv import load_dotenv

load_dotenv()

# ── Polars preamble injected before every generated script ─────────
SANDBOX_PREAMBLE = """\
import polars as pl
import polars.selectors as cs
import numpy as np
import json
import os
# ── Library standard: Polars not Pandas ───────────────────────────
# CORRECT:   pl.read_csv()  df.write_parquet()  df.fill_null()
# INCORRECT: pd.read_csv()  df.to_parquet()     df.fillna()
# If pandas required: convert with pl.from_pandas(df) before returning
# ──────────────────────────────────────────────────────────────────
"""

# ── Allowed imports inside sandbox ────────────────────────────────
ALLOWED_MODULES = {
    "polars", "numpy", "json", "os", "math",
    "sklearn", "lightgbm", "xgboost", "catboost",
    "optuna", "scipy", "statistics", "itertools",
    "collections", "functools", "datetime", "pathlib"
}

class SandboxExecutionError(Exception):
    """Raised when code fails all 3 retry attempts."""
    pass


class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("Code execution exceeded 10 minute limit")


def _make_safe_globals(session_id: str) -> dict:
    """Build a restricted global namespace for the sandbox."""
    import polars as pl
    import numpy as np
    import json
    import math

    glb = dict(safe_globals)
    glb["__builtins__"] = dict(safe_builtins)

    # Inject allowed libraries directly
    glb["pl"] = pl
    glb["np"] = np
    glb["json"] = json
    glb["math"] = math
    glb["os"] = os

    # Allow print for debugging output
    glb["__builtins__"]["print"] = print
    glb["__builtins__"]["len"] = len
    glb["__builtins__"]["range"] = range
    glb["__builtins__"]["enumerate"] = enumerate
    glb["__builtins__"]["zip"] = zip
    glb["__builtins__"]["list"] = list
    glb["__builtins__"]["dict"] = dict
    glb["__builtins__"]["str"] = str
    glb["__builtins__"]["int"] = int
    glb["__builtins__"]["float"] = float
    glb["__builtins__"]["bool"] = bool
    glb["__builtins__"]["type"] = type
    glb["__builtins__"]["isinstance"] = isinstance
    glb["__builtins__"]["hasattr"] = hasattr
    glb["__builtins__"]["getattr"] = getattr
    glb["__builtins__"]["open"] = open  # needed for file I/O
    glb["__builtins__"]["Exception"] = Exception
    glb["__builtins__"]["ValueError"] = ValueError
    glb["__builtins__"]["TypeError"] = TypeError

    # Session output path — sandbox writes here
    glb["SESSION_OUTPUT_DIR"] = f"outputs/{session_id}"
    os.makedirs(f"outputs/{session_id}", exist_ok=True)

    return glb


def _execute_once(code: str, session_id: str, timeout_seconds: int = 600) -> dict:
    """
    Single execution attempt — no retry logic here.
    Returns: {success, stdout, stderr, result}
    """
    full_code = SANDBOX_PREAMBLE + code

    # Capture stdout
    import io
    from contextlib import redirect_stdout, redirect_stderr

    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    # Set timeout (Unix only — Windows uses threading approach)
    if sys.platform != "win32":
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout_seconds)

    try:
        compiled = compile_restricted(full_code, "<sandbox>", "exec")
        glb = _make_safe_globals(session_id)

        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(compiled, glb)

        if sys.platform != "win32":
            signal.alarm(0)  # cancel timeout

        return {
            "success": True,
            "stdout": stdout_capture.getvalue(),
            "stderr": stderr_capture.getvalue(),
            "result": glb.get("result"),  # scripts can set result = value
            "globals": glb
        }

    except TimeoutError:
        return {
            "success": False,
            "stdout": stdout_capture.getvalue(),
            "stderr": "TIMEOUT: Code exceeded 10 minute execution limit",
            "error": "TimeoutError",
            "traceback": "Execution timeout"
        }
    except Exception as e:
        if sys.platform != "win32":
            signal.alarm(0)
        return {
            "success": False,
            "stdout": stdout_capture.getvalue(),
            "stderr": stderr_capture.getvalue(),
            "error": type(e).__name__,
            "traceback": traceback.format_exc()
        }


def execute_code(
    code: str,
    session_id: str,
    llm_fix_callback=None,
    max_attempts: int = 3,
    timeout_seconds: int = 600
) -> dict:
    """
    Execute code in RestrictedPython sandbox with 3-attempt retry loop.

    On failure: feeds full traceback back to LLM (via llm_fix_callback)
    which returns corrected code. Retries up to max_attempts times.
    After 3 failures: raises SandboxExecutionError (never hangs).

    Args:
        code:              Python code string to execute
        session_id:        Session namespace for file I/O
        llm_fix_callback:  fn(code, error, traceback) -> fixed_code
                           If None: retries same code (for testing)
        max_attempts:      Maximum retry attempts (default: 3)
        timeout_seconds:   Timeout per attempt in seconds (default: 600)

    Returns:
        {success, stdout, stderr, result, attempts_used}

    Raises:
        SandboxExecutionError: after max_attempts failures
    """
    current_code = code
    last_result = None

    for attempt in range(1, max_attempts + 1):
        print(f"[sandbox] Attempt {attempt}/{max_attempts}...")
        result = _execute_once(current_code, session_id, timeout_seconds)

        if result["success"]:
            result["attempts_used"] = attempt
            print(f"[sandbox] Success on attempt {attempt}.")
            return result

        # ── Failure — log and prepare retry ───────────────────────
        last_result = result
        error_info = f"""
EXECUTION FAILED (Attempt {attempt}/{max_attempts})
Error type:  {result.get('error', 'Unknown')}
Traceback:
{result.get('traceback', 'No traceback available')}
Stdout before failure:
{result.get('stdout', '')}
"""
        print(f"[sandbox] {error_info}")

        # If we have more attempts AND a fix callback, get corrected code
        if attempt < max_attempts and llm_fix_callback is not None:
            print(f"[sandbox] Requesting LLM fix for attempt {attempt + 1}...")
            try:
                current_code = llm_fix_callback(
                    code=current_code,
                    error=result.get("error", ""),
                    traceback_str=result.get("traceback", "")
                )
            except Exception as callback_error:
                print(f"[sandbox] LLM fix callback failed: {callback_error}")
                # Continue with same code if callback fails

        elif attempt < max_attempts:
            print(f"[sandbox] No fix callback. Retrying same code...")

    # ── All attempts exhausted ─────────────────────────────────────
    raise SandboxExecutionError(
        f"Code failed after {max_attempts} attempts.\n"
        f"Final error: {last_result.get('error')}\n"
        f"Final traceback:\n{last_result.get('traceback')}"
    )
```

---

## Task 3 — Write Contract Test (Immutable From Today)

```python
# tests/contracts/test_e2b_sandbox_contract.py
# ─────────────────────────────────────────────────────────────────
# Written: Day 2
# Status:  IMMUTABLE — never edit this file after today
#
# CONTRACT: execute_code()
#   INPUT:  code (str), session_id (str)
#   OUTPUT: dict with keys: success (bool), stdout (str), stderr (str)
#   ERRORS: raises SandboxExecutionError after 3 failed attempts
#           never hangs — always returns or raises within timeout
# ─────────────────────────────────────────────────────────────────
import pytest
import time
from tools.e2b_sandbox import execute_code, SandboxExecutionError

SESSION = "test_session_sandbox"


class TestSandboxContract:

    def test_successful_execution_returns_success_true(self):
        result = execute_code("result = 2 + 2", session_id=SESSION)
        assert result["success"] is True

    def test_output_has_required_keys(self):
        result = execute_code("x = 1", session_id=SESSION)
        assert "success" in result
        assert "stdout" in result
        assert "stderr" in result

    def test_stdout_captured(self):
        result = execute_code("print('hello_sandbox')", session_id=SESSION)
        assert result["success"] is True
        assert "hello_sandbox" in result["stdout"]

    def test_polars_available_in_sandbox(self):
        code = """
import polars as pl
df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
result = df.shape
print(f"shape: {df.shape}")
"""
        result = execute_code(code, session_id=SESSION)
        assert result["success"] is True
        assert "shape" in result["stdout"]

    def test_numpy_available_in_sandbox(self):
        code = "import numpy as np\nresult = np.mean([1, 2, 3])\nprint(result)"
        result = execute_code(code, session_id=SESSION)
        assert result["success"] is True

    def test_syntax_error_raises_sandbox_error(self):
        bad_code = "def broken( :"  # intentional syntax error
        with pytest.raises(SandboxExecutionError):
            execute_code(bad_code, session_id=SESSION, max_attempts=1)

    def test_runtime_error_raises_sandbox_error_after_max_attempts(self):
        bad_code = "result = 1 / 0"  # ZeroDivisionError
        with pytest.raises(SandboxExecutionError) as exc_info:
            execute_code(bad_code, session_id=SESSION, max_attempts=3)
        assert "3 attempts" in str(exc_info.value)

    def test_retry_loop_uses_fix_callback(self):
        """LLM fix callback is called on failure and fixed code succeeds."""
        call_count = {"n": 0}

        def mock_fix(code, error, traceback_str):
            call_count["n"] += 1
            return "result = 42  # fixed"  # always returns working code

        bad_code = "result = 1 / 0"
        result = execute_code(
            bad_code,
            session_id=SESSION,
            llm_fix_callback=mock_fix,
            max_attempts=3
        )
        assert result["success"] is True
        assert call_count["n"] == 1  # called once on first failure

    def test_never_allows_dangerous_imports(self):
        """Sandbox must block filesystem and system access."""
        dangerous_code = "import subprocess\nsubprocess.run(['ls'])"
        with pytest.raises((SandboxExecutionError, Exception)):
            execute_code(dangerous_code, session_id=SESSION, max_attempts=1)

    def test_attempts_used_recorded_in_result(self):
        result = execute_code("x = 1", session_id=SESSION)
        assert "attempts_used" in result
        assert result["attempts_used"] == 1

    def test_output_dir_created_for_session(self):
        import os
        execute_code("x = 1", session_id="output_test_session")
        assert os.path.exists("outputs/output_test_session")
```

Run it immediately:

```bash
pytest tests/contracts/test_e2b_sandbox_contract.py -v
# All tests must pass before moving to Task 4
```

---

## Task 4 — Manual Submission 0 (Spaceship Titanic)

This is built **by you in a notebook** — not by Professor. The purpose is to confirm the dataset format, metric, and submission structure before Professor writes a single line of agent code.

First, download the data:

```bash
# Make sure kaggle CLI is set up
# ~/.kaggle/kaggle.json must have your username and key

kaggle competitions download -c spaceship-titanic -p data/spaceship_titanic/
cd data/spaceship_titanic/
unzip spaceship-titanic.zip
ls
# Should show: train.csv  test.csv  sample_submission.csv
```

Now open `notebooks/sanity_check.ipynb` and run this — no feature engineering, no CV, default everything:

```python
# notebooks/sanity_check.ipynb
# Manual Submission 0 — built by hand, not by Professor
# Purpose: confirm format, metric, and establish baseline score

import polars as pl
import pandas as pd  # sklearn needs pandas for now — fine in notebook
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# ── Load data ─────────────────────────────────────────────────────
train = pl.read_csv("../data/spaceship_titanic/train.csv")
test  = pl.read_csv("../data/spaceship_titanic/test.csv")
sample = pl.read_csv("../data/spaceship_titanic/sample_submission.csv")

print("Train shape:", train.shape)
print("Test shape: ", test.shape)
print("\nColumns:", train.columns)
print("\nTarget distribution:")
print(train["Transported"].value_counts())
print("\nSample submission format:")
print(sample.head(3))

# ── Minimal preprocessing ─────────────────────────────────────────
# Convert to pandas for sklearn
train_pd = train.to_pandas()
test_pd  = test.to_pandas()

# Drop high-cardinality and complex columns for this baseline
drop_cols = ["PassengerId", "Name", "Cabin"]
train_pd = train_pd.drop(columns=drop_cols, errors="ignore")
test_pd  = test_pd.drop(columns=drop_cols, errors="ignore")

# Encode target
y = train_pd["Transported"].astype(int)
train_pd = train_pd.drop(columns=["Transported"])

# Label encode categoricals
categorical_cols = train_pd.select_dtypes(include="object").columns.tolist()
le = LabelEncoder()
for col in categorical_cols:
    train_pd[col] = train_pd[col].fillna("missing")
    test_pd[col]  = test_pd[col].fillna("missing")
    # Fit on combined to avoid unseen labels
    combined = pd.concat([train_pd[col], test_pd[col]])
    le.fit(combined)
    train_pd[col] = le.transform(train_pd[col])
    test_pd[col]  = le.transform(test_pd[col])

# Fill numeric nulls
train_pd = train_pd.fillna(train_pd.median(numeric_only=True))
test_pd  = test_pd.fillna(test_pd.median(numeric_only=True))

print("\nFeatures used:", list(train_pd.columns))
print("Training shape:", train_pd.shape)

# ── Train single model — default params, no tuning ────────────────
from lightgbm import LGBMClassifier

model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    random_state=42,
    verbose=-1
)
model.fit(train_pd, y)

# ── Quick local CV estimate ────────────────────────────────────────
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(
    LGBMClassifier(n_estimators=500, learning_rate=0.05,
                   random_state=42, verbose=-1),
    train_pd, y,
    cv=5,
    scoring="accuracy"
)
print(f"\nLocal CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ── Generate submission ────────────────────────────────────────────
preds = model.predict(test_pd)

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"].to_list(),
    "Transported": preds.astype(bool)
})

submission.to_csv("../outputs/submission_0_manual.csv", index=False)
print("\nSubmission saved: outputs/submission_0_manual.csv")
print("\nSubmission format check:")
print(submission.head(5))
print(f"\nShape: {submission.shape}")
print(f"Expected: ({len(test)}, 2)")

# ── Format validation ──────────────────────────────────────────────
assert set(submission.columns) == {"PassengerId", "Transported"}, \
    "Wrong columns in submission"
assert len(submission) == len(test), \
    f"Wrong row count: {len(submission)} vs {len(test)}"
assert submission["Transported"].dtype == bool or \
    submission["Transported"].isin([True, False]).all(), \
    "Transported must be boolean"

print("\n✓ Submission format valid")
print("✓ Ready to submit to Kaggle")
```

Submit it:

```bash
kaggle competitions submit \
  -c spaceship-titanic \
  -f outputs/submission_0_manual.csv \
  -m "Submission 0 — manual baseline, default LightGBM"

# Check your score
kaggle competitions submissions -c spaceship-titanic
```

---

## End of Day 2 — What Must Be True

```
Before committing and closing:

□ pytest tests/contracts/test_e2b_sandbox_contract.py -v
  → All tests green

□ python -c "from tools.e2b_sandbox import execute_code;
             r = execute_code('print(42)', 'test');
             print(r['success'])"
  → True

□ Submission 0 submitted to Kaggle
  → Public LB score recorded in DAILY_LOG.md
  → Must beat 0.50 (random baseline for binary classification)
  → Spaceship Titanic random baseline ≈ 0.50, good baseline ≈ 0.77-0.79

□ DAILY_LOG.md updated:
  → CV score from notebook
  → Public LB score from Kaggle
  → Both recorded as the floors Professor must beat
```

**Record the LB score carefully.** That number is Submission 0. Every Professor-built submission from Day 7 onward must beat it. If you cannot beat a manual default LightGBM, something in the pipeline is wrong.

---

## Day 3 Preview

Tomorrow's ONE thing: build `agents/data_engineer.py` — takes `raw_data_path` from state, produces `cleaned.parquet` + `schema.json`, writes both pointers back to state. First real agent. First time the state schema gets exercised.