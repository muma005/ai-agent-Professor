# agents/qa_gate.py
#
# Day 23 — QA Gate
# Deterministic format, slot, and orphan number checks.
#
# Runs after publisher. Checks three things using only deterministic code —
# no LLM judgment anywhere in this file.

import logging
import re
from pathlib import Path
import polars as pl
from core.state import ProfessorState
from core.lineage import log_event

logger = logging.getLogger(__name__)


# ── Check 1: Unfilled template slots ──────────────────────────────

def check_no_unfilled_slots(report_html: str) -> list[str]:
    """
    Returns list of unfilled slot names. Empty list = all filled.
    Searches for any remaining {{SLOT_NAME}} patterns in the HTML.
    """
    pattern = re.compile(r"\{\{([A-Z_]+)\}\}")
    return pattern.findall(report_html)


# ── Check 2: Orphan numbers in narrative ──────────────────────────

def check_no_orphan_numbers_in_narrative(report_html: str) -> list[str]:
    """
    Extracts text from narrative <p> tags (those not inside <table>).
    Returns list of decimal numbers found in narrative text.
    Integers like "5" or "10" are allowed — only decimals (containing ".") are flagged.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        logger.warning(
            "[qa_gate] beautifulsoup4 not installed. "
            "Skipping orphan number check. Install with: pip install beautifulsoup4"
        )
        return []

    soup = BeautifulSoup(report_html, "html.parser")

    # Remove all table content
    for table in soup.find_all("table"):
        table.decompose()

    narrative_text = " ".join(p.get_text() for p in soup.find_all("p"))
    decimal_pattern = re.compile(r"\b\d+\.\d+\b")
    return decimal_pattern.findall(narrative_text)


# ── Check 3: Submission CSV format validation ─────────────────────

def check_submission_format(submission_path: str, sample_path: str) -> list[str]:
    """
    Returns list of format violations. Empty list = format correct.
    Checks: column names, column count, row count, id column order, target dtype.
    """
    errors = []

    if not Path(submission_path).exists():
        return [f"Submission file not found: {submission_path}"]
    if not Path(sample_path).exists():
        return [f"Sample submission not found: {sample_path}"]

    sub = pl.read_csv(submission_path)
    sample = pl.read_csv(sample_path)

    if list(sub.columns) != list(sample.columns):
        errors.append(
            f"Column names mismatch: expected {list(sample.columns)}, "
            f"got {list(sub.columns)}"
        )
    if len(sub) != len(sample):
        errors.append(
            f"Row count mismatch: expected {len(sample)}, got {len(sub)}"
        )
    if not errors:  # only check dtypes if column names match
        for col in sample.columns:
            if sub[col].dtype != sample[col].dtype:
                errors.append(
                    f"Column '{col}' dtype mismatch: "
                    f"expected {sample[col].dtype}, got {sub[col].dtype}"
                )
    return errors


# ── Main entry point ──────────────────────────────────────────────

AGENT_NAME = "qa_gate"

def run_qa_gate(state: ProfessorState) -> ProfessorState:
    """
    QA Gate pipeline:
    1. Check for unfilled template slots in HTML report
    2. Check for orphan decimal numbers in narrative
    3. Check submission CSV format against sample_submission
    4. Set qa_passed and qa_failures in state
    5. Log lineage event

    Never raises. Always returns state with qa_passed set.
    """
    failures = []
    session_id = state.get("session_id", "unknown")

    # ── Check 1 & 2: Report quality ───────────────────────────────
    report_path = state.get("report_path")
    if report_path and Path(report_path).exists():
        try:
            html = Path(report_path).read_text(encoding="utf-8")

            unfilled = check_no_unfilled_slots(html)
            if unfilled:
                failures.append(f"Unfilled template slots: {unfilled}")

            orphans = check_no_orphan_numbers_in_narrative(html)
            if orphans:
                failures.append(f"Orphan numbers in narrative: {orphans}")
        except Exception as e:
            failures.append(f"Error reading report file: {e}")
    else:
        failures.append("Report file not found or not written.")

    # ── Check 3: Submission format ────────────────────────────────
    submission_path = state.get("submission_path")
    # Default path resolution logic
    sample_path = state.get("sample_submission_path")
    if not sample_path:
        sample_path = f"data/{state.get('competition_name', 'unknown')}/sample_submission.csv"

    if submission_path and Path(submission_path).exists():
        fmt_errors = check_submission_format(submission_path, sample_path)
        failures.extend(fmt_errors)
    else:
        failures.append("submission.csv not found.")

    # ── Verdict ───────────────────────────────────────────────────
    passed = len(failures) == 0

    updates = {
        "qa_passed": passed,
        "qa_failures": failures,
    }

    if not passed:
        logger.warning(
            f"[qa_gate] QA FAILED — {len(failures)} issue(s):\n" +
            "\n".join(f"  - {f}" for f in failures)
        )
    else:
        logger.info("[qa_gate] QA PASSED — all checks green.")

    # ── Lineage event ─────────────────────────────────────────────
    log_event(
        session_id=session_id,
        agent=AGENT_NAME,
        action="qa_gate_complete",
        keys_read=["report_path", "submission_path", "sample_submission_path"],
        keys_written=["qa_passed", "qa_failures"],
        values_changed={
            "passed": passed,
            "failures": failures,
            "n_checks": 3,
        },
    )

    return ProfessorState.validated_update(state, AGENT_NAME, updates)
