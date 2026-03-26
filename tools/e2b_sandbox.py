# tools/e2b_sandbox.py

import os
import sys
import shutil
import subprocess
import tempfile
import traceback
import uuid
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Day 15: Docker sandbox constants ──────────────────────────────
MAX_OUTPUT_BYTES = 10 * 1024 * 1024   # 10 MB stdout cap
DOCKER_IMAGE = "python:3.11-slim"
MEMORY_LIMIT = "8g"
CPU_LIMIT = "2"


def _docker_available() -> bool:
    """Returns True if Docker CLI is installed and daemon is running."""
    if shutil.which("docker") is None:
        return False
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True, timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


_USE_DOCKER = (
    _docker_available()
    and os.getenv("PROFESSOR_USE_DOCKER_SANDBOX", "1") != "0"
)
if not _USE_DOCKER:
    logger.warning(
        "[sandbox] Docker not available — falling back to subprocess sandbox. "
        "Install Docker Desktop for full isolation."
    )

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

# ── Safe environment for subprocess sandbox (strips API keys) ────
_ENV_WHITELIST = {
    "PATH", "PYTHONPATH", "PYTHONHOME", "HOME", "USERPROFILE",
    "SYSTEMROOT", "TEMP", "TMP", "VIRTUAL_ENV", "CONDA_PREFIX",
    "LANG", "LC_ALL",
}


def _safe_env() -> dict:
    """Build a safe env dict — only whitelisted vars, no API keys."""
    env = {k: v for k, v in os.environ.items() if k in _ENV_WHITELIST}
    env["PYTHONUNBUFFERED"] = "1"
    return env


class SandboxExecutionError(Exception):
    """Raised when code fails all retry attempts."""
    pass


class SandboxTimeoutError(Exception):
    pass


def _validate_imports(code: str) -> Optional[str]:
    """
    Static check: scans code for blocked import statements AND bypass patterns.
    Returns error message if blocked import found, else None.
    """
    for line in code.split("\n"):
        stripped = line.strip()
        # Check standard import statements
        if stripped.startswith("import ") or stripped.startswith("from "):
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
        # Check bypass patterns: __import__(), importlib
        if "__import__" in stripped:
            return "Use of __import__() is not allowed in sandbox."
        if "importlib" in stripped:
            return "Use of importlib is not allowed in sandbox."
    return None


def _execute_once(code: str, session_id: str, timeout_seconds: int = 600,
                  extra_files: dict = None) -> dict:
    """
    Single execution attempt. Routes to Docker if available, else subprocess.
    Returns: {success, stdout, stderr, result, timed_out, returncode, backend}
    """
    # Validate imports before running (both backends)
    import_error = _validate_imports(code)
    if import_error:
        return {
            "success": False,
            "stdout": "",
            "stderr": import_error,
            "error": "ImportError",
            "traceback": import_error,
            "timed_out": False,
            "returncode": 1,
            "backend": "docker" if _USE_DOCKER else "subprocess",
        }

    if _USE_DOCKER:
        return _execute_docker(code, session_id, timeout_seconds, extra_files)
    return _execute_subprocess(code, session_id, timeout_seconds, extra_files)


def _execute_docker(code: str, session_id: str, timeout_seconds: int = 600,
                    extra_files: dict = None) -> dict:
    """Execute code inside a Docker container with full isolation."""
    container_name = f"professor-sandbox-{uuid.uuid4().hex[:12]}"
    full_code = SANDBOX_PREAMBLE + code

    with tempfile.TemporaryDirectory() as tmpdir:
        code_path = Path(tmpdir) / "code.py"
        code_path.write_text(full_code, encoding="utf-8")

        # Write extra files into tmpdir so they are available in the container
        if extra_files:
            for fname, content in extra_files.items():
                (Path(tmpdir) / fname).write_text(content, encoding="utf-8")

        cmd = [
            "docker", "run",
            "--rm",
            "--name", container_name,
            "--network", "none",
            "--memory", MEMORY_LIMIT,
            "--memory-swap", MEMORY_LIMIT,
            "--cpus", CPU_LIMIT,
            "--read-only",
            "--tmpfs", "/tmp:rw,size=512m",
            "--security-opt", "no-new-privileges",
            "-v", f"{tmpdir}:/code:ro",
            DOCKER_IMAGE,
            "python", "/code/code.py",
        ]

        timed_out = False
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=timeout_seconds,
                text=True,
            )
            stdout = result.stdout[:MAX_OUTPUT_BYTES]
            stderr = result.stderr[:MAX_OUTPUT_BYTES]
            returncode = result.returncode

        except subprocess.TimeoutExpired:
            timed_out = True
            stdout = ""
            stderr = f"Execution timed out after {timeout_seconds} seconds."
            returncode = -1
            _force_remove_container(container_name)

        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Docker execution failed: {e}",
                "returncode": -1,
                "timed_out": False,
                "backend": "docker",
            }

    return {
        "success": returncode == 0,
        "stdout": stdout,
        "stderr": stderr,
        "result": None,
        "globals": {},
        "timed_out": timed_out,
        "returncode": returncode,
        "backend": "docker",
    }


def _force_remove_container(name: str) -> None:
    """Kills and removes a container by name. Silent on failure."""
    try:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=10)
    except Exception:
        pass


def _execute_subprocess(code: str, session_id: str, timeout_seconds: int = 600,
                        extra_files: dict = None) -> dict:
    """Fallback: Day 9 subprocess sandbox."""
    full_code = SANDBOX_PREAMBLE + code

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

        # Write extra files into the output dir (cwd for subprocess)
        if extra_files:
            for fname, content in extra_files.items():
                fpath = os.path.join(os.getcwd(), fname)
                with open(fpath, "w", encoding="utf-8") as ef:
                    ef.write(content)

        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=os.getcwd(),
            env=_safe_env(),
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
            "result": None,
            "globals": {},
            "timed_out": False,
            "returncode": result.returncode,
            "backend": "subprocess",
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
            "timed_out": True,
            "returncode": -1,
            "backend": "subprocess",
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
            "timed_out": False,
            "returncode": 1,
            "backend": "subprocess",
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


def run_in_sandbox(code: str, timeout: int = SANDBOX_TIMEOUT_S,
                   extra_files: dict = None, **kwargs) -> dict:
    """Standalone sandbox execution -- used by service_health.py fallback."""
    return _execute_once(code, session_id="standalone", timeout_seconds=timeout,
                         extra_files=extra_files)


def run_in_subprocess_sandbox(code: str, timeout: int = SANDBOX_TIMEOUT_S, **kwargs) -> dict:
    """Alias for run_in_sandbox — used as fallback target in service_health.py."""
    return run_in_sandbox(code, timeout=timeout, **kwargs)
