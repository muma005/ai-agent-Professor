# tests/contracts/test_critic_contract.py

import pytest
import os
import json
import polars as pl
from pathlib import Path
from core.state import initial_state
from agents.red_team_critic import run_red_team_critic

FIXTURE_PARQUET = "data/spaceship_titanic/train.parquet"

# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def critic_state():
    """State after EDA."""
    # Ensure parquet exists
    if not os.path.exists(FIXTURE_PARQUET):
        df = pl.read_csv("data/spaceship_titanic/train.csv")
        df.write_parquet(FIXTURE_PARQUET)
        
    return initial_state(
        session_id="test-critic",
        clean_data_path=FIXTURE_PARQUET,
        target_col="Transported",
        task_type="binary"
    )

# ── Tests ───────────────────────────────────────────────────────────────────

class TestCriticContract:
    """
    Contract: Red Team Critic Agent
    Ensures 7-vector quality audit and verdict persistence.
    """

    def test_critic_verdict_written_and_valid(self, critic_state):
        """Verify critic_verdict.json exists and contains overall_severity."""
        state = run_red_team_critic(critic_state)
        path = Path(state["critic_verdict_path"])
        assert path.exists()
        
        verdict = json.loads(path.read_text())
        assert "overall_severity" in verdict
        assert "vectors_checked" in verdict
        assert "clean" in verdict

    def test_critic_severity_in_state(self, critic_state):
        """Verify critic_severity key is written to state."""
        state = run_red_team_critic(critic_state)
        assert state["critic_severity"] in {"OK", "MEDIUM", "HIGH", "CRITICAL"}

    def test_replan_requested_on_critical(self, critic_state):
        """
        Verify that a CRITICAL verdict triggers replan_requested.
        We mock _check_shuffled_target to return CRITICAL.
        """
        with pytest.MonkeyPatch.context() as m:
            m.setattr("agents.red_team_critic._check_shuffled_target", 
                      lambda *args: {"verdict": "CRITICAL", "evidence": "Mocked leak"})
            
            state = run_red_team_critic(critic_state)
            assert state["critic_severity"] == "CRITICAL"
            assert state["replan_requested"] is True
            # In Day 11 logic, critical audit doesn't require HITL immediately
            # because supervisor tries to auto-fix first.
            assert state["hitl_required"] is False

    def test_skip_critic_logic(self, critic_state):
        """Verify Critic can be skipped via config."""
        from core.config import ProfessorConfig
        config = ProfessorConfig()
        config.agents.skip_red_team_critic = True
        
        state_with_skip = {**critic_state, "config": config}
        result = run_red_team_critic(state_with_skip)
        
        assert result["critic_severity"] == "OK"
        assert result.get("critic_verdict") is None
