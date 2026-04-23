# agents/feature_hypothesis.py

import os
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from core.state import ProfessorState
from tools.llm_provider import llm_call, _safe_json_loads
from guards.agent_retry import with_agent_retry
from tools.performance_monitor import timed_node
from core.lineage import log_event

logger = logging.getLogger(__name__)

AGENT_NAME = "creative_hypothesis"

# ── Prompt Templates ──────────────────────────────────────────────────────────

HYPOTHESIS_SYSTEM_PROMPT = """You are a Kaggle Grandmaster specializing in Creative Feature Engineering.
Your goal is to hypothesize novel features that capture hidden signals in the data.

You have access to:
1. Data Profiling (EDA)
2. Domain Research
3. Code Ledger (Previous feature attempts)
4. Distribution Shift Report

STRATEGY:
- Combine features in non-obvious ways (ratios, differences, counts).
- Use domain-specific transformations (e.g., log-financial, medical buckets).
- Think about temporal signals even in non-time-series data.
- Avoid obvious features already implemented.
"""

# ── Core Logic ──────────────────────────────────────────────────────────────

@timed_node
@with_agent_retry(AGENT_NAME)
def run_feature_hypothesis(state: ProfessorState) -> ProfessorState:
    """
    Intelligence Layer: Generates creative feature engineering ideas.
    """
    session_id = state.get("session_id", "default")
    
    logger.info(f"[{AGENT_NAME}] Generating hypotheses...")

    # 1. Gather Context
    eda = state.get("eda_report", {})
    domain = state.get("competition_context", {})
    ledger = state.get("state_mutations_log", [])[-20:] # Recent history
    shift = state.get("shift_report", {})

    # 2. LLM Reasoning
    prompt = f"""Generate a structured feature hypothesis brief.
DATA CONTEXT:
Target: {state.get('target_col')}
Task: {state.get('task_type')}
EDA Insights: {state.get('eda_insights_summary')}

DOMAIN CONTEXT:
{json.dumps(domain, indent=2)}

PREVIOUS ATTEMPTS:
{json.dumps(ledger, indent=2)}

Provide a JSON with:
1. "hypotheses": List of dicts with:
   - "name": Unique short name
   - "logic": Human-readable explanation
   - "signal_type": Why this should work? (interaction, transformation, etc.)
   - "complexity": 1-10
2. "creative_direction": A single paragraph explaining the overall strategy.
"""
    try:
        response = llm_call(prompt, system_prompt=HYPOTHESIS_SYSTEM_PROMPT)
        brief = _safe_json_loads(response)
    except:
        brief = {
            "hypotheses": [
                {"name": "interaction_top_2", "logic": "Multiply top 2 correlated features", "signal_type": "interaction", "complexity": 2}
            ],
            "creative_direction": "Fallback: Standard interaction baseline."
        }

    # 3. Update State
    updates = {
        "feature_candidates": brief.get("hypotheses", []),
        "feature_manifest": brief # Top-level brief
    }

    log_event(
        session_id=session_id,
        agent=AGENT_NAME,
        action="hypothesis_generated",
        keys_written=list(updates.keys())
    )

    return ProfessorState.validated_update(state, AGENT_NAME, updates)
