# tools/sandbox.py

import os
import sys
import re
import json
import time
import logging
import hashlib
import traceback
import subprocess
import difflib
import textwrap
from dataclasses import dataclass, asdict
from typing import Optional, Any, Dict, List, Union, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

# ── Code Ledger Schema ───────────────────────────────────────────────────────

@dataclass
class CodeLedgerEntry:
    entry_id: str
    timestamp: str
    agent: str
    purpose: str
    round: int
    attempt: int
    code: str
    code_hash: str
    inputs: List[str]
    outputs: List[str]
    dependencies: List[str]
    success: bool
    stdout: str
    stderr: str
    runtime_seconds: float
    llm_prompt: str = ""
    llm_reasoning: str = ""
    kept: bool = True
    rejection_reason: str = ""

# ── Layer 1: Diagnostic Injection ────────────────────────────────────────────

def _inject_diagnostics(code: str) -> str:
    """Wraps user code in a diagnostic capture block."""
    
    wrapper_lines = [
        "import os",
        "import sys",
        "import json",
        "import pickle",
        "import traceback",
        "import polars as pl",
        "import numpy as np",
        "",
        "def __get_diagnostics(error=None):",
        "    try:",
        "        diag = {",
        "            'error': None,",
        "            'locals': {},",
        "            'dataframes': {},",
        "            'files': []",
        "        }",
        "        if error:",
        "            diag['error'] = {",
        "                'type': type(error).__name__,",
        "                'message': str(error),",
        "                'traceback': traceback.format_exc(),",
        "                'line': traceback.extract_tb(error.__traceback__)[-1].lineno",
        "            }",
        "        import inspect",
        "        frame = inspect.currentframe().f_back",
        "        if error: frame = frame.f_back",
        "        if frame:",
        "            for name, val in frame.f_locals.items():",
        "                if name.startswith('_'): continue",
        "                if hasattr(val, 'shape') and hasattr(val, 'columns'):",
        "                    try:",
        "                        cols = list(val.columns)[:50]",
        "                        diag['dataframes'][name] = {",
        "                            'shape': list(val.shape),",
        "                            'columns': cols,",
        "                            'null_counts': {c: int(val[c].null_count()) if hasattr(val, 'null_count') else 0 for c in cols[:20]},",
        "                            'head': str(val.head(3))",
        "                        }",
        "                    except: pass",
        "                elif isinstance(val, (int, float, str, bool)):",
        "                    diag['locals'][name] = repr(val)[:200]",
        "        print(f'\\n__DIAGNOSTICS__\\n{json.dumps(diag)}\\n__DIAGNOSTICS__')",
        "    except Exception as e_inner:",
        "        print(f'\\nDEBUG: __get_diagnostics failed: {e_inner}')",
        "",
        "try:",
        "    # === USER CODE START ==="
    ]
    
    for line in code.strip().split("\n"):
        wrapper_lines.append("    " + line)
        
    wrapper_lines.extend([
        "    # === USER CODE END ===",
        "    __get_diagnostics(None)",
        "except Exception as e:",
        "    __get_diagnostics(e)",
        "    raise e"
    ])
    
    return "\n".join(wrapper_lines)

# ── Layer 2: Error Classification ────────────────────────────────────────────

def _classify_error(diagnostics: dict) -> Tuple[str, str]:
    """Classifies error and generates fix instructions."""
    error_info = diagnostics.get("error")
    if not error_info:
        return "logic_error", "Code succeeded but output validation failed."
        
    tb = error_info.get("traceback", "")
    msg = error_info.get("message", "")
    err_type = error_info.get("type", "")
    
    if "KeyError" in err_type or "ColumnNotFoundError" in tb:
        cols = []
        for df in diagnostics.get("dataframes", {}).values():
            cols.extend(df.get("columns", []))
        return "column_missing", f"Available columns: {list(set(cols))}"

    if any(p in tb.lower() or p in msg.lower() for p in ["shape", "length", "dimension", "broadcast"]):
        return "shape_mismatch", "Shape conflict detected between dataframes."

    if "TypeError" in err_type or "InvalidOperationError" in tb:
        return "type_error", f"Type mismatch: {msg}"

    if any(p in tb.lower() or p in msg.lower() for p in ["null", "nan", "none"]):
        return "null_values", "Unchecked nulls found in input dataframes."

    if "ModuleNotFoundError" in err_type or "ImportError" in err_type:
        return "import_missing", "Missing library. Ensure all imports are standard."

    return "unknown", f"Error: {err_type}: {msg}"

# ── Layer 4: Decomposition ───────────────────────────────────────────────────

def _decompose_code(code: str) -> List[str]:
    """Splits code into logical blocks at double blank lines."""
    blocks = re.split(r"\n\s*\n", code.strip())
    return [b.strip() for b in blocks if b.strip()]

# ── Core Sandbox Function ───────────────────────────────────────────────────

def run_in_sandbox(
    code: str,
    timeout: int = 300,
    working_dir: str = None,
    agent_name: str = "unknown",
    purpose: str = "",
    round_num: int = 0,
    attempt: int = 1,
    llm_prompt: str = "",
    llm_reasoning: str = "",
    expected_row_change: str = "none",
) -> dict:
    """Executes code with 4-layer self-debugging."""
    
    # ── Commit 1: Pre-Execution Leakage Check ──
    LEAKAGE_CHECK_AGENTS = {"data_engineer", "feature_factory", "ml_optimizer", 
                             "creative_hypothesis", "post_processor", "freeform_sandbox"}

    if agent_name in LEAKAGE_CHECK_AGENTS:
        from guards.leakage_precheck import check_code_for_leakage
        leakage = check_code_for_leakage(code)
        if leakage["leakage_detected"]:
            # Do NOT execute the code. Return as failure.
            stderr = (
                f"PRE-EXECUTION LEAKAGE DETECTED: {leakage['description']}\n"
                f"Line {leakage['line']}: {leakage['code_line']}\n"
                f"Fix: {leakage['fix_suggestion']}\n"
                f"Code was NOT executed to prevent wasted compute."
            )
            return {
                "success": False,
                "stdout": "",
                "stderr": stderr,
                "runtime": 0.0,
                "entry_id": f"blocked_{int(time.time())}",
                "diagnostics": {"leakage_precheck": leakage},
                "integrity_ok": True,
                "pre_execution_blocked": True,
            }

    if working_dir:
        os.makedirs(working_dir, exist_ok=True)
    
    wrapped_code = _inject_diagnostics(code)
    temp_file = f"sandbox_{int(time.time())}.py"
    with open(temp_file, "w") as f:
        f.write(wrapped_code)
        
    try:
        proc = subprocess.run(
            [sys.executable, temp_file],
            capture_output=True, text=True, timeout=timeout, cwd=working_dir
        )
        stdout, stderr = proc.stdout, proc.stderr
        success = proc.returncode == 0
    except subprocess.TimeoutExpired:
        stdout, stderr = "", "TimeoutExpired"
        success = False
    finally:
        if os.path.exists(temp_file):
            try: os.remove(temp_file)
            except: pass

    diagnostics = {}
    if "__DIAGNOSTICS__" in stdout:
        parts = stdout.split("__DIAGNOSTICS__")
        try:
            diagnostics = json.loads(parts[1].strip())
            stdout = parts[0] + parts[2]
        except: pass

    err_class = "none"
    fix_instr = ""
    if not success:
        err_class, fix_instr = _classify_error(diagnostics)

    entry = CodeLedgerEntry(
        entry_id=f"sb_{int(time.time())}",
        timestamp=datetime.now().isoformat(),
        agent=agent_name,
        purpose=purpose,
        round=round_num,
        attempt=attempt,
        code=code,
        code_hash=hashlib.sha256(code.encode()).hexdigest(),
        inputs=[], outputs=[], dependencies=[],
        success=success,
        stdout=stdout.strip(),
        stderr=stderr.strip(),
        runtime_seconds=0.0
    )

    return {
        "success": success,
        "stdout": stdout.strip(),
        "stderr": stderr.strip(),
        "entry": asdict(entry),
        "diagnostics": diagnostics,
        "error_class": err_class,
        "fix_instructions": fix_instr
    }
