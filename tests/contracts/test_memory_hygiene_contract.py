# tests/contracts/test_memory_hygiene_contract.py

import pytest
from datetime import datetime, timezone, timedelta
from core.state import ProfessorState, initial_state
from shields.memory_hygiene import check_memory_hygiene, detect_semantic_contradiction

# ── Tests ───────────────────────────────────────────────────────────────────

class TestMemoryHygieneContract:
    """
    Contract: Memory Hygiene Guard (Component 5)
    """

    def test_contradiction_detection_triggered(self):
        """Verify logic flags obvious contradictions."""
        eda = "The data shows no temporal signal."
        logic = "Create time-based features."
        is_bad, reason = detect_semantic_contradiction(eda, logic)
        assert is_bad is True
        assert "no temporal signal" in reason

    def test_confidence_decay_age_based(self):
        """Verify confidence drops over time (1% per day)."""
        state = ProfessorState(**initial_state())
        # Pattern from 10 days ago
        old_date = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        patterns = [{"name": "old_p", "logic": "x", "confidence": 1.0, "created_at": old_date}]
        
        res = check_memory_hygiene(state, patterns)
        decayed = res["memory_hygiene_report"]["pattern_details"][0]["decayed_confidence"]
        # 0.99^10 is approx 0.904
        assert 0.90 < decayed < 0.91

    def test_hygiene_report_structure(self):
        state = ProfessorState(**initial_state())
        patterns = [{"name": "p1", "logic": "x"}]
        res = check_memory_hygiene(state, patterns)
        report = res["memory_hygiene_report"]
        assert "n_retrieved" in report
        assert "n_accepted" in report
        assert "average_confidence" in report
        assert "pattern_details" in report

    def test_all_patterns_accepted_on_clean_match(self):
        state = ProfessorState(**initial_state())
        state.eda_insights_summary = "Linear relationship detected."
        patterns = [{"name": "p1", "logic": "df*2"}] # No contradiction
        res = check_memory_hygiene(state, patterns)
        assert res["memory_hygiene_report"]["n_accepted"] == 1
        assert res["memory_hygiene_report"]["pattern_details"][0]["status"] == "ACCEPTED"

    def test_rejected_on_contradiction(self):
        state = ProfessorState(**initial_state())
        state.eda_insights_summary = "The target is not correlated with target." # Trigger 'not correlated' keyword
        patterns = [{"name": "p1", "logic": "multiply features"}]
        res = check_memory_hygiene(state, patterns)
        assert res["memory_hygiene_report"]["n_accepted"] == 0
        assert res["memory_hygiene_report"]["pattern_details"][0]["status"] == "REJECTED"

    def test_never_halts_pipeline(self):
        state = ProfessorState(**initial_state())
        # Passing garbage as patterns
        res = check_memory_hygiene(state, [None]) 
        assert "memory_hygiene_report" in res
        assert res.pipeline_halted is False
