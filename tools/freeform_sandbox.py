# tools/freeform_sandbox.py

import os
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from tools.llm_provider import llm_call, _safe_json_loads
from tools.sandbox import run_in_sandbox
from tools.operator_channel import emit_to_operator

logger = logging.getLogger(__name__)

# ── Prompt Templates ──────────────────────────────────────────────────────────

FREEFORM_SYSTEM_PROMPT = """You are a Kaggle Grandmaster tasked with writing a standalone Python ML script.
The user will provide a high-level request (e.g., 'try a neural network', 'use PCA').

RULES:
1. Output ONLY a valid Python code block.
2. The script must be STANDALONE: zero dependencies on the 'professor' package.
3. The script must load data from the provided paths (passed as constants).
4. The script must output results to 'stdout' using print().
5. Use Polars for data processing. No Pandas.
6. The script should perform a complete ML task: loading, preprocessing, model fitting, and validation.
"""

# ── Core Logic ──────────────────────────────────────────────────────────────

def run_freeform_execution(
    prompt: str,
    session_id: str,
    train_path: str,
    test_path: str,
    target_col: str,
    task_type: str = "classification"
) -> Dict[str, Any]:
    """
    Generates and executes a standalone ML script in the sandbox.
    """
    output_dir = Path(f"outputs/{session_id}/freeform")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"ff_{timestamp}"

    emit_to_operator(f"🔨 Freeform: Generating script for prompt: '{prompt}'...", level="STATUS")

    # 1. Generate Script
    llm_prompt = f"""REQUEST: {prompt}

DATA PATHS (Use these constants in your script):
TRAIN_PATH = "{train_path}"
TEST_PATH = "{test_path}"
TARGET_COL = "{target_col}"
TASK_TYPE = "{task_type}"

Implement the request in a robust Polars-based ML script.
"""
    try:
        code_raw = llm_call(llm_prompt, system_prompt=FREEFORM_SYSTEM_PROMPT)
        # Extract code
        if "```python" in code_raw:
            code = code_raw.split("```python")[1].split("```")[0].strip()
        else:
            code = code_raw.strip()
    except Exception as e:
        msg = f"❌ Freeform script generation failed: {e}"
        emit_to_operator(msg, level="ERROR")
        return {"success": False, "error": str(e)}

    # 2. Persist Code
    code_path = output_dir / f"{run_id}.py"
    with open(code_path, "w") as f:
        f.write(code)

    emit_to_operator(f"🚀 Freeform: Executing standalone script...", level="STATUS")

    # 3. Execute in Sandbox (with safety guards)
    # Using run_in_sandbox ensures leakage check and output validation are applied
    res = run_in_sandbox(
        code, 
        agent_name="freeform_sandbox", 
        purpose=f"Freeform execution: {prompt}",
        working_dir=str(output_dir)
    )

    # 4. Result Reporting
    if res["success"]:
        emit_to_operator(f"✅ Freeform SUCCESS: {run_id}\nStdout:\n{res['stdout'][:500]}...", level="RESULT")
    else:
        emit_to_operator(f"❌ Freeform FAILED: {run_id}\nError:\n{res['stderr']}", level="ERROR")

    # 5. Persist Results
    result_path = output_dir / f"{run_id}_result.json"
    with open(result_path, "w") as f:
        json.dump(res, f, indent=2)

    return res
