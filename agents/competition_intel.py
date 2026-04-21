# agents/competition_intel.py

import os
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from core.state import ProfessorState
from tools.performance_monitor import timed_node
from tools.llm_provider import llm_call, _safe_json_loads
from guards.agent_retry import with_agent_retry

logger = logging.getLogger(__name__)

AGENT_NAME = "competition_intel"

# ── Prompts ──────────────────────────────────────────────────────────────────

INTEL_PROMPT = """
Analyze the following Kaggle competition details and provide a JSON brief.
Competition: {competition_name}
Notebooks/Discussions Context: {context}

Return a JSON with:
- "problem_summary": string
- "target_analysis": string
- "evaluation_metric": string
- "shakeup_risk": "low" | "medium" | "high"
- "suggested_validation": string
- "source_post_count": integer
- "relevance_score": float (0.0 to 1.0)
"""

# ── Agent Node ───────────────────────────────────────────────────────────────

@with_agent_retry(AGENT_NAME)
def run_competition_intel(state: ProfessorState) -> Dict[str, Any]:
    """
    Intelligence Layer: Scrapes Kaggle and synthesizes a competition brief.
    """
    session_id = state.get("session_id", "default")
    comp_name = state.get("competition_name", "unknown")
    
    # 1. Skip logic from config
    config = state.get("config")
    if config and config.agents.skip_competition_intel:
        logger.info(f"[{AGENT_NAME}] Skipping per config.")
        return state

    logger.info(f"[{AGENT_NAME}] Gathering intel for: {comp_name}")

    # 2. Fetch context (Stub for Kaggle API logic)
    # In a real run, this calls kaggle_scraper.py
    context = "Tabular binary classification competition. Focused on stable validation."
    
    # 3. LLM Synthesis
    prompt = INTEL_PROMPT.format(competition_name=comp_name, context=context)
    try:
        response = llm_call(prompt, system_prompt="You are a Competition Expert.")
        brief = _safe_json_loads(response)
    except Exception as e:
        logger.warning(f"[{AGENT_NAME}] LLM synthesis failed, using default brief. Error: {e}")
        brief = {
            "problem_summary": "General competition",
            "target_analysis": "Standard target",
            "evaluation_metric": "unknown",
            "shakeup_risk": "medium",
            "suggested_validation": "KFold",
            "source_post_count": 0,
            "relevance_score": 0.5
        }

    # 4. Handle External Data Scout (Stub)
    manifest = {}
    if state.get("external_data_allowed", False):
        manifest = {"suggested_datasets": []}
    
    # 5. Persist artifacts
    output_dir = Path(f"outputs/{session_id}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    brief_path = output_dir / "competition_brief.json"
    brief_path.write_text(json.dumps(brief, indent=2))
    
    manifest_path = output_dir / "external_data_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # 6. Update State
    updates = {
        "competition_brief": brief,
        "competition_brief_path": str(brief_path),
        "intel_brief_path": str(brief_path),
        "external_data_manifest": manifest,
        "task_type": "binary" if "binary" in brief.get("problem_summary", "").lower() else "unknown"
    }

    # Use validated_update to enforce ownership
    return ProfessorState.validated_update(state, AGENT_NAME, updates)
