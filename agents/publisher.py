# agents/publisher.py

import os
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from core.state import ProfessorState
from core.lineage import log_event
from guards.agent_retry import with_agent_retry
from tools.performance_monitor import timed_node
from tools.operator_channel import emit_to_operator

logger = logging.getLogger(__name__)

AGENT_NAME = "publisher"

@timed_node
@with_agent_retry(AGENT_NAME)
def run_publisher(state: ProfessorState) -> ProfessorState:
    """
    Final node of the Professor pipeline. Summarizes the run and verifies artifacts.
    """
    session_id = state.get("session_id", "default")
    output_dir = Path(f"outputs/{session_id}")
    
    emit_to_operator("🏁 Pipeline Finalizing... Generating Publisher Report.", level="STATUS")

    # 1. Gather Summary Data
    cv_mean = state.get("cv_mean")
    cv_str = f"{cv_mean:.4f}" if cv_mean is not None else "N/A"
    best_model = state.get("best_model_type", "None")
    n_features = len(state.get("feature_order", []))
    pseudo_active = state.get("pseudo_label_activated", False)
    depth = str(state.get("pipeline_depth", "unknown")).upper()
    
    # ── Commit 3: Solution Provenance ──
    from tools.solution_assembler import assemble_standalone_solution
    from tools.code_ledger import CodeLedger
    try:
        ledger = CodeLedger(session_id)
        winning_sequence = ledger.get_winning_sequence()
        
        assemble_standalone_solution(
            session_id=session_id,
            winning_sequence=winning_sequence,
            train_path=state.get("feature_data_path") or state.get("clean_data_path") or "",
            test_path=state.get("test_data_path") or "",
            target_col=state.get("target_col", "")
        )
    except Exception as e:
        logger.warning(f"[{AGENT_NAME}] Solution assembly failed: {e}")

    # 2. Artifact Verification
    sub_path = state.get("submission_path")
    sub_exists = sub_path and os.path.exists(sub_path)
    
    # 3. Cost Report
    cost_data = state.get("cost_tracker") or {}
    total_cost = cost_data.get("api_cost_usd", 0.0)
    llm_calls = cost_data.get("llm_calls", 0)
    
    # 4. Provenance Links
    solution_dir = output_dir / "solution"
    script_exists = os.path.exists(solution_dir / "solution_notebook.py")
    writeup_exists = os.path.exists(solution_dir / "solution_writeup.md")

    # 5. Build Milestone 4 Report
    report = f"""🏆 PROFESSOR v2.0 FINAL REPORT — {session_id}

📊 PIPELINE SUMMARY:
- Task: {state.get('task_type')}
- Best Model: {best_model}
- CV Score: {cv_str}
- Features: {n_features} implemented
- Pseudo-labeling: {'✅ Active' if pseudo_active else '❌ Inactive'}
- Depth: {depth}

💰 RESOURCE REPORT:
- LLM Calls: {llm_calls}
- API Cost: ${total_cost:.4f}
- Peak RAM: {state.get('memory_peak_gb', 0.0)} GB

📦 ARTIFACTS:
- Submission: {'✅ VALIDATED' if sub_exists else '❌ MISSING'}
- Standalone Notebook: {'✅ CREATED' if script_exists else '❌ MISSING'}
- Documentation: {'✅ CREATED' if writeup_exists else '❌ MISSING'}

Session complete. Standalone solution is ready for reproduction.
"""
    emit_to_operator(report, level="RESULT")

    # 6. Final State Update
    report_file = output_dir / "final_report.txt"
    try:
        os.makedirs(output_dir, exist_ok=True)
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report)
    except Exception as e:
        logger.error(f"[{AGENT_NAME}] Could not write report file: {e}")

    updates = {
        "post_mortem_completed": True,
        "report_path": str(report_file)
    }

    log_event(
        session_id=session_id,
        agent=AGENT_NAME,
        action="pipeline_finalized",
        keys_written=list(updates.keys())
    )

    return ProfessorState.validated_update(state, AGENT_NAME, updates)
