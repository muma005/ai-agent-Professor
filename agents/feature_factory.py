# agents/feature_factory.py

import os
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from core.state import ProfessorState
from tools.llm_provider import llm_call, _safe_json_loads
from tools.sandbox import run_in_sandbox
from core.lineage import log_event
from guards.agent_retry import with_agent_retry
from tools.performance_monitor import timed_node

logger = logging.getLogger(__name__)

AGENT_NAME = "feature_factory"

# ── Feature Engineering System Prompt ────────────────────────────────────────

FACTORY_SYSTEM_PROMPT = """You are a Kaggle Grandmaster specializing in Python/Polars feature engineering.
Generate high-quality feature code that transforms raw data into strong signals.

RULES:
1. Output ONLY a valid Python code block using Polars (pl).
2. Use the variable 'df' as the input and output (df = df.with_columns(...)).
3. Do NOT include imports — they are handled by the sandbox.
4. Keep logic efficient. Use vectorized Polars operations.
5. Focus on the hypotheses provided in the context.
"""

# ── Core Logic ──────────────────────────────────────────────────────────────

def _run_feature_round(
    state: ProfessorState, 
    round_num: int, 
    hypotheses: List[Dict],
    ledger: List[Dict]
) -> Tuple[bool, str, Dict]:
    """Single round of feature generation and sandbox validation."""
    
    # 1. LLM Generation
    prompt = f"""ROUND {round_num} of Feature Engineering.
Hypotheses to implement:
{json.dumps(hypotheses, indent=2)}

Previous Code Ledger (Successes and Failures):
{json.dumps(ledger[-5:], indent=2)}

Generate a single Python code block using Polars to implement these features.
"""
    try:
        code_raw = llm_call(prompt, system_prompt=FACTORY_SYSTEM_PROMPT)
        # Extract code from markdown
        if "```python" in code_raw:
            code = code_raw.split("```python")[1].split("```")[0].strip()
        else:
            code = code_raw.strip()
    except Exception as e:
        return False, f"LLM Generation failed: {e}", {}

    # 2. Sandbox Execution
    # Note: In real run, we'd use the clean_data_path. For now, we mock success.
    res = run_in_sandbox(
        code, 
        agent_name=AGENT_NAME,
        purpose=f"Round {round_num} engineering",
        round_num=round_num
    )
    
    return res["success"], code, res

# ── Agent Node ───────────────────────────────────────────────────────────────

@timed_node
@with_agent_retry(AGENT_NAME)
def run_feature_factory(state: ProfessorState) -> ProfessorState:
    """
    Intelligence Layer: Iterative 5-Round Feature Factory.
    """
    session_id = state.get("session_id", "default")
    
    logger.info(f"[{AGENT_NAME}] Starting iterative refinement...")

    # 1. Gather Hypotheses
    hypotheses = state.get("feature_candidates", [])
    if not hypotheses:
        logger.warning(f"[{AGENT_NAME}] No hypotheses found. Skipping.")
        return state

    # 2. Iterative Refinement Loop (Mandated 5 Rounds)
    # Note: We iterate but use current state to keep loop logic clean in this node
    ledger = []
    rounds_completed = 0
    max_rounds = 5 # As per v2_layer2.md
    
    # Check if pipeline_depth reduces rounds
    depth = state.get("pipeline_depth", "standard")
    if depth == "sprint": max_rounds = 2

    for r in range(1, max_rounds + 1):
        logger.info(f"[{AGENT_NAME}] Round {r}/{max_rounds} starting...")
        success, code, result = _run_feature_round(state, r, hypotheses, ledger)
        
        entry = result.get("entry", {})
        if entry: ledger.append(entry)
        
        if success:
            rounds_completed += 1
            logger.info(f"[{AGENT_NAME}] Round {r} SUCCESS.")
        else:
            logger.warning(f"[{AGENT_NAME}] Round {r} FAILED: {result.get('stderr')}")

    # 3. Update State
    updates = {
        "round1_features": ledger[0].get("code") if len(ledger) > 0 else None,
        "round2_features": ledger[1].get("code") if len(ledger) > 1 else None,
        "round3_features": ledger[2].get("code") if len(ledger) > 2 else None,
        "round4_features": ledger[3].get("code") if len(ledger) > 3 else None,
        "round5_features": ledger[4].get("code") if len(ledger) > 4 else None,
        "feature_order": [e.get("entry_id") for e in ledger if e.get("success")],
    }

    log_event(
        session_id=session_id,
        agent=AGENT_NAME,
        action="factory_rounds_complete",
        values_changed={"rounds": rounds_completed}
    )

    return ProfessorState.validated_update(state, AGENT_NAME, updates)
