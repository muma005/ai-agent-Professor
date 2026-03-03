# tools/e2b_sandbox.py

import os
import sys
import signal
import traceback
from typing import Optional
from RestrictedPython import compile_restricted, safe_globals, safe_builtins
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
    """Raised when code fails all retry attempts."""
    pass


class SandboxTimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise SandboxTimeoutError("Code execution exceeded 10 minute limit")


def _safe_import(name, *args, **kwargs):
    """Controlled import — only allows modules in ALLOWED_MODULES."""
    # Check top-level module name (e.g. "sklearn" from "sklearn.ensemble")
    top_level = name.split(".")[0]
    if top_level not in ALLOWED_MODULES:
        raise ImportError(
            f"Import of '{name}' is not allowed in sandbox. "
            f"Allowed: {', '.join(sorted(ALLOWED_MODULES))}"
        )
    return __builtins__["__import__"](name, *args, **kwargs) if isinstance(__builtins__, dict) \
        else __import__(name, *args, **kwargs)


def _make_safe_globals(session_id: str) -> dict:
    """Build a restricted global namespace for the sandbox."""
    import polars as pl
    import polars.selectors as cs
    import numpy as np
    import json
    import math

    glb = dict(safe_globals)
    glb["__builtins__"] = dict(safe_builtins)

    # Controlled import — only ALLOWED_MODULES
    glb["__builtins__"]["__import__"] = _safe_import

    # ── RestrictedPython guard functions ───────────────────────────
    # RestrictedPython transforms print() -> _print_(), x.attr -> _getattr_(x, 'attr'), etc.
    # We must provide these guard functions for the transformed code to run.
    from RestrictedPython import PrintCollector

    glb["_print_"] = PrintCollector
    glb["_getattr_"] = getattr
    glb["_getitem_"] = lambda obj, key: obj[key]
    glb["_getiter_"] = iter
    glb["_write_"] = lambda x: x  # allow attribute assignment on objects
    glb["_inplacevar_"] = lambda op, x, y: op(x, y)  # +=, -=, etc.
    glb["pl"] = pl
    glb["cs"] = cs
    glb["np"] = np
    glb["json"] = json
    glb["math"] = math
    glb["os"] = os
    glb["polars"] = pl
    glb["numpy"] = np

    # Allow print for debugging output
    glb["__builtins__"]["print"] = print
    glb["__builtins__"]["len"] = len
    glb["__builtins__"]["range"] = range
    glb["__builtins__"]["enumerate"] = enumerate
    glb["__builtins__"]["zip"] = zip
    glb["__builtins__"]["sorted"] = sorted
    glb["__builtins__"]["min"] = min
    glb["__builtins__"]["max"] = max
    glb["__builtins__"]["sum"] = sum
    glb["__builtins__"]["abs"] = abs
    glb["__builtins__"]["round"] = round
    glb["__builtins__"]["map"] = map
    glb["__builtins__"]["filter"] = filter
    glb["__builtins__"]["any"] = any
    glb["__builtins__"]["all"] = all
    glb["__builtins__"]["list"] = list
    glb["__builtins__"]["dict"] = dict
    glb["__builtins__"]["tuple"] = tuple
    glb["__builtins__"]["set"] = set
    glb["__builtins__"]["str"] = str
    glb["__builtins__"]["int"] = int
    glb["__builtins__"]["float"] = float
    glb["__builtins__"]["bool"] = bool
    glb["__builtins__"]["type"] = type
    glb["__builtins__"]["isinstance"] = isinstance
    glb["__builtins__"]["issubclass"] = issubclass
    glb["__builtins__"]["hasattr"] = hasattr
    glb["__builtins__"]["getattr"] = getattr
    glb["__builtins__"]["setattr"] = setattr
    glb["__builtins__"]["open"] = open  # needed for file I/O
    glb["__builtins__"]["Exception"] = Exception
    glb["__builtins__"]["ValueError"] = ValueError
    glb["__builtins__"]["TypeError"] = TypeError
    glb["__builtins__"]["KeyError"] = KeyError
    glb["__builtins__"]["IndexError"] = IndexError
    glb["__builtins__"]["RuntimeError"] = RuntimeError
    glb["__builtins__"]["StopIteration"] = StopIteration
    glb["__builtins__"]["NotImplementedError"] = NotImplementedError

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

        # Extract PrintCollector output (RestrictedPython transforms
        # print() -> _print_(), which stores output in 'printed' variable)
        printed_output = ""
        if "_print" in glb:
            collector = glb["_print"]
            if hasattr(collector, "__call__"):
                # PrintCollector instance — call it to get the output
                try:
                    printed_output = str(collector())
                except Exception:
                    pass
        elif "printed" in glb:
            printed_output = str(glb["printed"])

        # Combine redirect_stdout capture with PrintCollector output
        combined_stdout = stdout_capture.getvalue()
        if printed_output:
            combined_stdout = printed_output + combined_stdout

        return {
            "success": True,
            "stdout": combined_stdout,
            "stderr": stderr_capture.getvalue(),
            "result": glb.get("result"),  # scripts can set result = value
            "globals": glb
        }

    except SandboxTimeoutError:
        return {
            "success": False,
            "stdout": stdout_capture.getvalue(),
            "stderr": "TIMEOUT: Code exceeded 10 minute execution limit",
            "error": "SandboxTimeoutError",
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
    After all failures: raises SandboxExecutionError (never hangs).

    Args:
        code:              Python code string to execute
        session_id:        Session namespace for file I/O
        llm_fix_callback:  fn(code, error, traceback_str) -> fixed_code
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
