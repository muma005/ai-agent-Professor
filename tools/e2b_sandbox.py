# tools/e2b_sandbox.py

import os
import sys
import subprocess
import tempfile
import traceback
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Polars preamble injected before every generated script ─────────
SANDBOX_PREAMBLE = """\
import polars as pl
import polars.selectors as cs
import numpy as np
import json
import os
# -- Library standard: Polars not Pandas --
# CORRECT:   pl.read_csv()  df.write_parquet()  df.fill_null()
# INCORRECT: pd.read_csv()  df.to_parquet()     df.fillna()
# If pandas required: convert with pl.from_pandas(df) before returning
# -----------------------------------------
"""

# ── Allowed imports inside sandbox ────────────────────────────────
ALLOWED_MODULES = {
    "polars", "numpy", "json", "os", "math",
    "sklearn", "lightgbm", "xgboost", "catboost",
    "optuna", "scipy", "statistics", "itertools",
    "collections", "functools", "datetime", "pathlib"
}

# ── Blocked modules — never allow these from generated code ───────
BLOCKED_MODULES = {
    "subprocess", "shutil", "socket", "http", "urllib",
    "ftplib", "smtplib", "ctypes", "multiprocessing",
    "signal", "pty", "resource", "sys",
}

SANDBOX_TIMEOUT_S = 600  # 10 minutes


class SandboxExecutionError(Exception):
    """Raised when code fails all retry attempts."""
    pass


class SandboxTimeoutError(Exception):
    pass


def _validate_imports(code: str) -> Optional[str]:
    """
    Static check: scans code for blocked import statements.
    Returns error message if blocked import found, else None.
    """
    for line in code.split("\n"):
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            # Extract module name
            if stripped.startswith("from "):
                parts = stripped.split()
                if len(parts) >= 2:
                    module = parts[1].split(".")[0]
            else:
                parts = stripped.split()
                if len(parts) >= 2:
                    module = parts[1].split(".")[0].split(",")[0]
                else:
                    continue
            if module in BLOCKED_MODULES:
                return (
                    f"Import of '{module}' is not allowed in sandbox. "
                    f"Blocked modules: {', '.join(sorted(BLOCKED_MODULES))}"
                )
    return None


def _execute_once(code: str, session_id: str, timeout_seconds: int = 600) -> dict:
    """
    Single execution attempt via subprocess.
    Returns: {success, stdout, stderr, result}
    """
    full_code = SANDBOX_PREAMBLE + code

    # Validate imports before running
    import_error = _validate_imports(code)
    if import_error:
        return {
            "success": False,
            "stdout": "",
            "stderr": import_error,
            "error": "ImportError",
            "traceback": import_error,
        }

    # Ensure output directory exists
    output_dir = f"outputs/{session_id}"
    os.makedirs(output_dir, exist_ok=True)

    # Inject the session output dir as a variable
    header = f'SESSION_OUTPUT_DIR = "{output_dir.replace(chr(92), "/")}"\n'
    full_code = header + full_code

    # Write script to a temp file and execute via subprocess
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", prefix="professor_sandbox_",
            delete=False, dir=output_dir, encoding="utf-8"
        ) as f:
            f.write(full_code)
            script_path = f.name

        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=os.getcwd(),
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )

        # Clean up temp script
        try:
            os.unlink(script_path)
        except OSError:
            pass

        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "result": None,  # subprocess can't return Python objects
            "globals": {},
        }

    except subprocess.TimeoutExpired as e:
        try:
            os.unlink(script_path)
        except (OSError, NameError):
            pass
        return {
            "success": False,
            "stdout": e.stdout or "",
            "stderr": f"TIMEOUT: Code exceeded {timeout_seconds}s execution limit",
            "error": "SandboxTimeoutError",
            "traceback": "Execution timeout",
        }

    except Exception as e:
        try:
            os.unlink(script_path)
        except (OSError, NameError):
            pass
        return {
            "success": False,
            "stdout": "",
            "stderr": str(e),
            "error": type(e).__name__,
            "traceback": traceback.format_exc(),
        }


def execute_code(
    code: str,
    session_id: str,
    llm_fix_callback=None,
    max_attempts: int = 3,
    timeout_seconds: int = 600
) -> dict:
    """
    Execute code in subprocess sandbox with retry loop.

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
{result.get('traceback', result.get('stderr', 'No traceback available'))}
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
                    error=result.get("error", result.get("stderr", "")),
                    traceback_str=result.get("traceback", result.get("stderr", ""))
                )
            except Exception as callback_error:
                print(f"[sandbox] LLM fix callback failed: {callback_error}")
                # Continue with same code if callback fails

        elif attempt < max_attempts:
            print(f"[sandbox] No fix callback. Retrying same code...")

    # ── All attempts exhausted ─────────────────────────────────────
    raise SandboxExecutionError(
        f"Code failed after {max_attempts} attempts.\n"
        f"Final error: {last_result.get('error', last_result.get('stderr', ''))}\n"
        f"Final traceback:\n{last_result.get('traceback', last_result.get('stderr', ''))}"
    )


def run_in_sandbox(code: str, timeout: int = SANDBOX_TIMEOUT_S, **kwargs) -> dict:
    """Standalone sandbox execution — used by service_health.py fallback."""
    return _execute_once(code, session_id="standalone", timeout_seconds=timeout)


def run_in_subprocess_sandbox(code: str, timeout: int = SANDBOX_TIMEOUT_S, **kwargs) -> dict:
    """Alias for run_in_sandbox — used as fallback target in service_health.py."""
    return run_in_sandbox(code, timeout=timeout, **kwargs)
