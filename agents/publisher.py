# agents/publisher.py
#
# Day 23 — Publisher
# Structured HTML report with programmatic slot injection + LLM narrative.
#
# Produces a structured HTML report after every pipeline run.
# Numbers come from metrics.json — LLM writes narrative only.

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from core.state import ProfessorState
from core.lineage import log_event

logger = logging.getLogger(__name__)

# ── HTML Template ─────────────────────────────────────────────────

TEMPLATE = """
<!DOCTYPE html>
<html>
<head><title>Professor Agent — Competition Report</title></head>
<body>

<h1>{{COMPETITION_NAME}}</h1>
<p class="timestamp">Generated: {{TIMESTAMP}}</p>

<section id="performance">
  <h2>Performance</h2>
  <table>
    <tr><td>CV Score</td><td>{{CV_SCORE}}</td></tr>
    <tr><td>CV Std</td><td>{{CV_STD}}</td></tr>
    <tr><td>Winning Model</td><td>{{WINNING_MODEL_TYPE}}</td></tr>
    <tr><td>Features Used</td><td>{{N_FEATURES_FINAL}}</td></tr>
    <tr><td>Features Dropped (Null Importance)</td><td>{{N_FEATURES_DROPPED}}</td></tr>
    <tr><td>Ensemble Accepted</td><td>{{ENSEMBLE_ACCEPTED}}</td></tr>
    <tr><td>Pseudo-labels Applied</td><td>{{PSEUDO_LABELS_APPLIED}}</td></tr>
    <tr><td>Runtime</td><td>{{RUNTIME_SECONDS}}s</td></tr>
  </table>
</section>

<section id="narrative">
  <h2>Analysis</h2>
  <p>{{NARRATIVE_PERFORMANCE}}</p>
</section>

<section id="critic">
  <h2>Critic Findings</h2>
  <p>Overall severity: {{CRITIC_SEVERITY}}</p>
  <p>{{NARRATIVE_CRITIC}}</p>
</section>

<section id="next">
  <h2>Recommended Next Steps</h2>
  <p>{{NARRATIVE_NEXT_STEPS}}</p>
</section>

</body>
</html>
"""

NUMERIC_SLOTS = {
    "CV_SCORE", "CV_STD", "N_FEATURES_FINAL", "N_FEATURES_DROPPED",
    "RUNTIME_SECONDS", "N_SUBMISSIONS_WITH_LB",
}

NARRATIVE_SLOTS = {
    "NARRATIVE_PERFORMANCE", "NARRATIVE_CRITIC", "NARRATIVE_NEXT_STEPS",
}


# ── Numeric slot injection ────────────────────────────────────────

def fill_numeric_slots(template: str, metrics: dict, state: dict) -> str:
    """
    Injects all non-narrative values from metrics.json and state.
    Every NUMERIC_SLOT must be filled. If a value is missing, inject "N/A".
    Never calls an LLM for this step.
    """
    n_features_dropped = (
        state.get("stage1_drop_count", 0) + state.get("stage2_drop_count", 0)
    )

    replacements = {
        "COMPETITION_NAME":       state.get("competition_name", "Unknown"),
        "TIMESTAMP":              datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "CV_SCORE":               f"{metrics.get('cv_mean', 'N/A'):.5f}" if isinstance(metrics.get('cv_mean'), (int, float)) else "N/A",
        "CV_STD":                 f"{metrics.get('cv_std', 'N/A'):.5f}" if isinstance(metrics.get('cv_std'), (int, float)) else "N/A",
        "WINNING_MODEL_TYPE":     state.get("winning_model_type", "N/A"),
        "N_FEATURES_FINAL":       str(state.get("n_features_final", "N/A")),
        "N_FEATURES_DROPPED":     str(n_features_dropped),
        "ENSEMBLE_ACCEPTED":      str(state.get("ensemble_accepted", False)),
        "PSEUDO_LABELS_APPLIED":  str(state.get("pseudo_labels_applied", False)),
        "RUNTIME_SECONDS":        str(state.get("total_runtime_seconds", "N/A")),
        "CRITIC_SEVERITY":        state.get("critic_severity", "unchecked"),
        "N_SUBMISSIONS_WITH_LB":  str(state.get("n_submissions_with_lb", "N/A")),
    }
    for slot, value in replacements.items():
        template = template.replace("{{" + slot + "}}", value)
    return template


# ── LLM narrative generation ──────────────────────────────────────

NARRATIVE_PROMPT = """
You are writing one section of a Kaggle competition analysis report.
The report already contains all statistics — they were injected programmatically.
Your job is to write 2-3 sentences of narrative for the {slot_name} section.

DO NOT invent or repeat any numbers. The numbers are already in the report.
DO NOT write phrases like "with a CV score of X" — the CV score is already displayed in the table.
Write only observations, interpretations, and recommendations in plain English.

Context from this run:
- Competition: {competition_name}
- CV score: {cv_score}
- Winning model: {winning_model_type}
- Features used: {n_features_final}
- Critic severity: {critic_severity}
- Ensemble accepted: {ensemble_accepted}

Write the narrative for: {slot_name}
"""


def _call_llm_for_narrative(slot_name: str, state: dict) -> str:
    """
    Call the LLM to generate narrative for one slot.
    Returns the narrative text, or '[Narrative unavailable]' on failure.
    """
    try:
        from core.llm_client import call_llm
    except ImportError:
        logger.warning("[publisher] LLM client not available — using placeholder narrative.")
        return "[Narrative unavailable]"

    prompt = NARRATIVE_PROMPT.format(
        slot_name=slot_name,
        competition_name=state.get("competition_name", "Unknown"),
        cv_score=state.get("cv_mean", "N/A"),
        winning_model_type=state.get("winning_model_type", "N/A"),
        n_features_final=state.get("n_features_final", "N/A"),
        critic_severity=state.get("critic_severity", "unchecked"),
        ensemble_accepted=state.get("ensemble_accepted", False),
    )

    try:
        response = call_llm(
            prompt=prompt,
            system_prompt="You are a data science report writer. Write concise, factual narrative.",
            max_tokens=200,
            temperature=0.3,
        )
        return response.strip()
    except Exception as e:
        logger.warning(f"[publisher] LLM call failed for {slot_name}: {e}")
        return "[Narrative unavailable]"


def _fill_narrative_slots(template: str, state: dict) -> str:
    """
    Fill all narrative slots using LLM calls. If any call fails,
    inject '[Narrative unavailable]' instead. Never crash.
    """
    for slot in NARRATIVE_SLOTS:
        narrative = _call_llm_for_narrative(slot, state)
        template = template.replace("{{" + slot + "}}", narrative)
    return template


# ── Main entry point ──────────────────────────────────────────────

def run_publisher(state: ProfessorState) -> ProfessorState:
    """
    Publisher pipeline:
    1. Load metrics.json
    2. Fill numeric slots programmatically
    3. Fill narrative slots via LLM
    4. Write HTML report to disk
    5. Set state outputs
    6. Log lineage event
    """
    session_id = state.get("session_id", "unknown")
    output_dir = Path(f"outputs/{session_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"[publisher] Starting — session: {session_id}")

    # ── Step 1: Load metrics ──────────────────────────────────────
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        import json
        metrics = json.loads(metrics_path.read_text())
    else:
        logger.warning(f"[publisher] metrics.json not found at {metrics_path}. Using empty metrics.")
        metrics = {}

    # ── Step 2: Fill numeric slots ────────────────────────────────
    html = fill_numeric_slots(TEMPLATE, metrics, state)

    # ── Step 3: Fill narrative slots ──────────────────────────────
    html = _fill_narrative_slots(html, state)

    # ── Step 4: Write report ──────────────────────────────────────
    report_path = str(output_dir / "report.html")
    Path(report_path).write_text(html)

    # ── Step 5: Set state outputs ─────────────────────────────────
    result = {
        **state,
        "report_path": report_path,
        "report_written": True,
    }

    # ── Step 6: Lineage event ─────────────────────────────────────
    log_event(
        session_id=session_id,
        agent="publisher",
        action="publisher_complete",
        keys_read=["metrics.json", "competition_name", "cv_mean", "cv_std",
                    "critic_severity", "ensemble_accepted"],
        keys_written=["report_path", "report_written"],
        values_changed={
            "report_path": report_path,
            "report_written": True,
        },
    )

    logger.info(f"[publisher] Report written to {report_path}")
    return result
