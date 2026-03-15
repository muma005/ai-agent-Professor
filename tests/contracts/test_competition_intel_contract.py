# tests/contracts/test_competition_intel_contract.py
# ─────────────────────────────────────────────────────────────────
# Written: Day 15
# Status:  IMMUTABLE — never edit this file after today
#
# CONTRACT: agents/competition_intel.py
#
# INPUT:  competition name
# OUTPUT: competition_brief.json with validated schema
#         external_data_manifest.json (may be empty)
#
# INVARIANTS:
#   - All required fields present in competition_brief.json (safe defaults if missing)
#   - Forum insights tagged with validated=False until tested
#   - No insight is ever marked validated=True by competition_intel
#   - external_data_manifest.json written regardless of external_data_allowed
#   - Relevance scores 0.0–1.0 only
#   - Scout gate: no sources returned if external_data_allowed=False
# ─────────────────────────────────────────────────────────────────

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from core.state import initial_state


# ── Fixtures ──────────────────────────────────────────────────────

MOCK_BRIEF_JSON = json.dumps({
    "critical_findings": ["Group-based CV important to prevent shakeup"],
    "proven_features": ["target_encoded_HomePlanet"],
    "known_leaks": [],
    "external_datasets": [],
    "dominant_approach": "LightGBM",
    "cv_strategy_hint": "StratifiedKFold",
    "forbidden_techniques": [],
    "shakeup_risk": "medium",
    "source_post_count": 3,
    "scraped_at": "2025-01-01T00:00:00",
})

MOCK_MANIFEST_JSON = json.dumps({
    "external_sources": [
        {
            "name": "WorldBank GDP Data",
            "type": "public_dataset",
            "description": "GDP per capita by country",
            "source_url": "https://data.worldbank.org",
            "relevance_score": 0.85,
            "join_strategy": "country column",
            "acquisition_method": "wget URL",
            "competition_precedent": None,
        }
    ],
    "recommended_sources": ["WorldBank GDP Data"],
    "total_sources_found": 1,
    "scout_notes": "GDP data may correlate with spending patterns.",
})


@pytest.fixture
def competition_intel_state():
    """State with external_data_allowed=False (default)."""
    state = initial_state(
        competition="spaceship-titanic",
        data_path="data/spaceship_titanic/train.csv",
        budget_usd=2.0,
    )
    return state


@pytest.fixture
def competition_intel_state_with_external():
    """State with external_data_allowed=True."""
    state = initial_state(
        competition="spaceship-titanic",
        data_path="data/spaceship_titanic/train.csv",
        budget_usd=2.0,
    )
    state = {**state, "external_data_allowed": True}
    return state


def _run_intel_mocked(state):
    """Run competition_intel with mocked Kaggle API and LLM."""
    with patch("agents.competition_intel._fetch_notebooks", return_value=[
        {"title": "Top Notebook", "author": "user1", "votes": 100, "ref": "user1/nb1"},
        {"title": "Second NB", "author": "user2", "votes": 50, "ref": "user2/nb2"},
        {"title": "Third NB", "author": "user3", "votes": 30, "ref": "user3/nb3"},
    ]):
        with patch("agents.competition_intel.call_llm") as mock_llm:
            # First call = _synthesize_brief, second call = scout (if enabled)
            mock_llm.side_effect = [MOCK_BRIEF_JSON, MOCK_MANIFEST_JSON]
            from agents.competition_intel import run_competition_intel
            return run_competition_intel(state)


def _run_intel_kaggle_disabled(state):
    """Run with Kaggle API raising an error — tests graceful degradation."""
    with patch("agents.competition_intel._fetch_notebooks", side_effect=Exception("Kaggle unavailable")):
        with patch("agents.competition_intel.call_llm", return_value=MOCK_BRIEF_JSON):
            from agents.competition_intel import run_competition_intel
            return run_competition_intel(state)


def _load_brief(state) -> dict:
    path = Path(f"outputs/{state['session_id']}/competition_brief.json")
    return json.loads(path.read_text())


def _load_manifest(state) -> dict:
    path = Path(f"outputs/{state['session_id']}/external_data_manifest.json")
    return json.loads(path.read_text())


class TestCompetitionIntelContract:
    """Contract tests — immutable after Day 15."""

    REQUIRED_BRIEF_FIELDS = {
        "critical_findings",
        "proven_features",
        "known_leaks",
        "external_datasets",
        "dominant_approach",
        "cv_strategy_hint",
        "forbidden_techniques",
        "shakeup_risk",
        "source_post_count",
        "scraped_at",
    }

    def test_competition_brief_has_all_required_fields(self, competition_intel_state):
        """All required fields present in competition_brief.json."""
        state = _run_intel_mocked(competition_intel_state)
        brief_path = Path(f"outputs/{state['session_id']}/competition_brief.json")
        assert brief_path.exists(), "competition_brief.json not written"

        brief = json.loads(brief_path.read_text())
        missing = self.REQUIRED_BRIEF_FIELDS - set(brief.keys())
        assert not missing, f"competition_brief.json missing required fields: {missing}"

    def test_shakeup_risk_is_valid_enum(self, competition_intel_state):
        """shakeup_risk must be one of low/medium/high — not a free-form string."""
        VALID_RISK = {"low", "medium", "high"}
        state = _run_intel_mocked(competition_intel_state)
        brief = _load_brief(state)
        assert brief["shakeup_risk"] in VALID_RISK, (
            f"shakeup_risk '{brief['shakeup_risk']}' is not valid. Must be one of: {VALID_RISK}"
        )

    def test_source_post_count_is_non_negative_int(self, competition_intel_state):
        """source_post_count must be a non-negative integer."""
        state = _run_intel_mocked(competition_intel_state)
        brief = _load_brief(state)
        assert isinstance(brief["source_post_count"], int), "source_post_count must be int"
        assert brief["source_post_count"] >= 0, "source_post_count must be >= 0"

    def test_scraped_at_is_non_empty(self, competition_intel_state):
        """scraped_at must not be empty string or None."""
        state = _run_intel_mocked(competition_intel_state)
        brief = _load_brief(state)
        assert brief.get("scraped_at"), "scraped_at is empty."

    def test_external_data_manifest_written_when_allowed(self, competition_intel_state_with_external):
        """external_data_manifest.json written when external_data_allowed=True."""
        state = _run_intel_mocked(competition_intel_state_with_external)
        manifest_path = Path(f"outputs/{state['session_id']}/external_data_manifest.json")
        assert manifest_path.exists(), (
            "external_data_manifest.json not written despite external_data_allowed=True."
        )

    def test_external_data_manifest_written_and_empty_when_not_allowed(self, competition_intel_state):
        """external_data_manifest.json still written when external_data_allowed=False — just empty."""
        state = _run_intel_mocked(competition_intel_state)
        manifest_path = Path(f"outputs/{state['session_id']}/external_data_manifest.json")
        assert manifest_path.exists(), (
            "external_data_manifest.json not written even when scout is disabled."
        )
        manifest = json.loads(manifest_path.read_text())
        assert manifest.get("external_sources") == [], (
            "external_sources must be empty list when external_data_allowed=False."
        )

    def test_relevance_scores_in_valid_range(self, competition_intel_state_with_external):
        """All relevance scores must be 0.0–1.0 inclusive."""
        state = _run_intel_mocked(competition_intel_state_with_external)
        manifest = _load_manifest(state)

        for source in manifest.get("external_sources", []):
            score = float(source.get("relevance_score", -1))
            assert 0.0 <= score <= 1.0, (
                f"Source '{source.get('name', '?')}' has relevance_score={score}. "
                "Must be 0.0–1.0."
            )

    def test_competition_intel_never_raises_on_missing_kaggle_api(self, competition_intel_state):
        """
        If Kaggle API is unavailable, competition_intel must still produce a brief
        (with safe defaults) rather than propagating the API error.
        """
        state = _run_intel_kaggle_disabled(competition_intel_state)
        brief_path = Path(f"outputs/{state['session_id']}/competition_brief.json")
        assert brief_path.exists(), (
            "competition_brief.json not written when Kaggle API unavailable."
        )

    def test_state_has_external_data_manifest_key(self, competition_intel_state):
        """After run_competition_intel, state must contain external_data_manifest dict."""
        state = _run_intel_mocked(competition_intel_state)
        assert "external_data_manifest" in state, "external_data_manifest missing from state"
        assert isinstance(state["external_data_manifest"], dict), "external_data_manifest must be dict"
