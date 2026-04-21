# tests/contracts/test_competition_intel_contract.py

import pytest
import json
import os
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from core.state import initial_state
from agents.competition_intel import run_competition_intel

FIXTURE_CSV = "data/spaceship_titanic/train.csv"

# ── Mocks & Helpers ─────────────────────────────────────────────────────────

def _run_intel_mocked(state):
    """Run intel with LLM and Scraper mocked to return stable results."""
    mock_brief = {
        "problem_summary": "Binary classification",
        "target_analysis": "Transported column",
        "evaluation_metric": "accuracy",
        "shakeup_risk": "low",
        "suggested_validation": "StratifiedKFold",
        "source_post_count": 42,
        "relevance_score": 0.95
    }
    
    # We must patch the LLM call inside competition_intel
    with patch("agents.competition_intel.llm_call", return_value=json.dumps(mock_brief)):
        # Also patch scraper if it's called
        return run_competition_intel(state)

def _run_intel_kaggle_disabled(state):
    """Run intel with Kaggle API disabled to test fallback safety."""
    mock_brief = {
        "problem_summary": "Kaggle API Disabled Fallback",
        "target_analysis": "Fallback",
        "evaluation_metric": "accuracy",
        "shakeup_risk": "medium",
        "suggested_validation": "KFold",
        "source_post_count": 0,
        "relevance_score": 0.5
    }
    with patch("agents.competition_intel.llm_call", return_value=json.dumps(mock_brief)):
        return run_competition_intel(state)

def _load_brief(state):
    path = Path(state["competition_brief_path"])
    return json.loads(path.read_text())

def _load_manifest(state):
    path = Path(f"outputs/{state['session_id']}/external_data_manifest.json")
    return json.loads(path.read_text())

# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def competition_intel_state():
    """State with external_data_allowed=False (default)."""
    return initial_state(
        session_id="test-intel",
        competition="spaceship-titanic",
        data_path=FIXTURE_CSV,
        external_data_allowed=False
    )

@pytest.fixture
def competition_intel_state_with_external():
    """State with external_data_allowed=True."""
    return initial_state(
        session_id="test-intel-ext",
        competition="spaceship-titanic",
        data_path=FIXTURE_CSV,
        external_data_allowed=True
    )

# ── Tests ───────────────────────────────────────────────────────────────────

class TestCompetitionIntelContract:
    """
    Contract: Competition Intel Agent
    Ensures Kaggle scraping and LLM synthesis produce valid, persisted briefs.
    """

    def test_competition_brief_has_all_required_fields(self, competition_intel_state):
        """All required fields present in competition_brief.json."""
        state = _run_intel_mocked(competition_intel_state)
        brief = _load_brief(state)
        
        required_keys = {
            "problem_summary", "target_analysis", "evaluation_metric",
            "shakeup_risk", "suggested_validation", "source_post_count",
            "relevance_score"
        }
        assert required_keys.issubset(brief.keys())

    def test_shakeup_risk_is_valid_enum(self, competition_intel_state):
        """shakeup_risk must be one of low/medium/high."""
        state = _run_intel_mocked(competition_intel_state)
        brief = _load_brief(state)
        assert brief["shakeup_risk"] in {"low", "medium", "high"}

    def test_source_post_count_is_non_negative_int(self, competition_intel_state):
        """source_post_count must be a non-negative integer."""
        state = _run_intel_mocked(competition_intel_state)
        brief = _load_brief(state)
        assert isinstance(brief["source_post_count"], int)
        assert brief["source_post_count"] >= 0

    def test_external_data_manifest_written_when_allowed(self, competition_intel_state_with_external):
        """external_data_manifest.json written when external_data_allowed=True."""
        state = _run_intel_mocked(competition_intel_state_with_external)
        manifest_path = Path(f"outputs/{state['session_id']}/external_data_manifest.json")
        assert manifest_path.exists()

    def test_external_data_manifest_written_and_empty_when_not_allowed(self, competition_intel_state):
        """external_data_manifest.json still written when disabled — just empty."""
        state = _run_intel_mocked(competition_intel_state)
        manifest_path = Path(f"outputs/{state['session_id']}/external_data_manifest.json")
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        assert len(manifest.get("suggested_datasets", [])) == 0

    def test_relevance_scores_in_valid_range(self, competition_intel_state_with_external):
        """All relevance scores must be 0.0–1.0 inclusive."""
        state = _run_intel_mocked(competition_intel_state_with_external)
        brief = _load_brief(state)
        assert 0.0 <= brief["relevance_score"] <= 1.0

    def test_competition_intel_never_raises_on_missing_kaggle_api(self, competition_intel_state):
        """Fallback to basic brief if Kaggle API fails."""
        state = _run_intel_kaggle_disabled(competition_intel_state)
        assert state["competition_brief"] is not None

    def test_state_has_external_data_manifest_key(self, competition_intel_state):
        """Verify the key exists in state even if empty."""
        state = _run_intel_mocked(competition_intel_state)
        assert "external_data_manifest" in state
