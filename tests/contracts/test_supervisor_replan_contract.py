# tests/contracts/test_supervisor_replan_contract.py
# -------------------------------------------------------------------------
# Written: Day 11   Status: IMMUTABLE
#
# CONTRACT: run_supervisor_replan()
#   INPUT:   state with replan_requested=True, replan_remove_features, replan_rerun_nodes
#   OUTPUT:  dag_version incremented, features_dropped accumulated,
#            replan_requested cleared, critic_severity reset
#   MUST:    Escalate to HITL after MAX_REPLAN_ATTEMPTS
#   NEVER:   Loop infinitely (dag_version gate), lose previously dropped features
# -------------------------------------------------------------------------
import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from core.state import initial_state
from agents.supervisor import run_supervisor_replan, get_replan_target, MAX_REPLAN_ATTEMPTS, NODE_PRIORITY


@pytest.fixture
def replan_state():
    """State simulating a CRITICAL verdict with replan instructions."""
    s = initial_state("test-supervisor", "tests/fixtures/tiny_train.csv")
    s["replan_requested"] = True
    s["replan_remove_features"] = ["leaked_feature_A"]
    s["replan_rerun_nodes"] = ["feature_factory", "ml_optimizer"]
    s["critic_severity"] = "CRITICAL"
    s["dag_version"] = 1
    return s


class TestSupervisorReplanContract:

    def test_replan_increments_dag_version(self, replan_state):
        result = run_supervisor_replan(replan_state)
        assert result["dag_version"] == 2, (
            f"dag_version should be 2 after first replan, got {result['dag_version']}"
        )

    def test_replan_increments_dag_version_from_current_value(self):
        s = initial_state("test-sv-v3", "tests/fixtures/tiny_train.csv")
        s["dag_version"] = 2
        s["replan_requested"] = True
        s["replan_remove_features"] = ["feat_B"]
        s["replan_rerun_nodes"] = ["ml_optimizer"]
        result = run_supervisor_replan(s)
        assert result["dag_version"] == 3, "dag_version should increment from 2 to 3"

    def test_replan_adds_to_features_dropped(self, replan_state):
        result = run_supervisor_replan(replan_state)
        assert "leaked_feature_A" in result["features_dropped"]

    def test_replan_accumulates_dropped_features_across_cycles(self):
        s = initial_state("test-sv-acc", "tests/fixtures/tiny_train.csv")
        s["dag_version"] = 1
        s["features_dropped"] = ["old_feature"]
        s["replan_requested"] = True
        s["replan_remove_features"] = ["new_feature"]
        s["replan_rerun_nodes"] = ["ml_optimizer"]
        result = run_supervisor_replan(s)
        assert "old_feature" in result["features_dropped"], "Must keep previously dropped features"
        assert "new_feature" in result["features_dropped"], "Must add new dropped features"

    def test_replan_clears_replan_requested(self, replan_state):
        result = run_supervisor_replan(replan_state)
        assert result["replan_requested"] is False

    def test_replan_clears_replan_remove_features(self, replan_state):
        result = run_supervisor_replan(replan_state)
        assert result["replan_remove_features"] == []

    def test_replan_clears_hitl_required(self, replan_state):
        replan_state["hitl_required"] = True
        result = run_supervisor_replan(replan_state)
        assert result["hitl_required"] is False

    def test_replan_resets_critic_severity(self, replan_state):
        result = run_supervisor_replan(replan_state)
        assert result["critic_severity"] == "unchecked"

    def test_max_replan_attempts_triggers_hitl(self):
        s = initial_state("test-sv-max", "tests/fixtures/tiny_train.csv")
        s["dag_version"] = MAX_REPLAN_ATTEMPTS
        s["replan_requested"] = True
        s["replan_remove_features"] = ["x"]
        s["replan_rerun_nodes"] = ["ml_optimizer"]
        result = run_supervisor_replan(s)
        assert result["hitl_required"] is True, "Must escalate to HITL after max attempts"
        assert result["pipeline_halted"] is True

    def test_max_replan_hitl_reason_mentions_attempt_count(self):
        s = initial_state("test-sv-reason", "tests/fixtures/tiny_train.csv")
        s["dag_version"] = MAX_REPLAN_ATTEMPTS
        s["replan_requested"] = True
        s["replan_remove_features"] = []
        s["replan_rerun_nodes"] = []
        result = run_supervisor_replan(s)
        assert str(MAX_REPLAN_ATTEMPTS) in result.get("hitl_reason", ""), (
            f"HITL reason must mention {MAX_REPLAN_ATTEMPTS}. Got: {result.get('hitl_reason')}"
        )

    def test_replan_does_not_drop_features_not_in_remove_list(self, replan_state):
        replan_state["features_dropped"] = ["keep_me"]
        replan_state["replan_remove_features"] = ["drop_me"]
        result = run_supervisor_replan(replan_state)
        assert "keep_me" in result["features_dropped"]
        assert "drop_me" in result["features_dropped"]

    def test_replan_routes_to_earliest_affected_node(self, replan_state):
        replan_state["replan_rerun_nodes"] = ["feature_factory", "ml_optimizer"]
        result = run_supervisor_replan(replan_state)
        target = get_replan_target(result)
        assert target == "feature_factory", (
            f"Should route to feature_factory (earliest). Got: {target}"
        )

    def test_replan_routes_to_data_engineer_when_in_rerun_nodes(self):
        s = initial_state("test-sv-de", "tests/fixtures/tiny_train.csv")
        s["replan_rerun_nodes"] = ["data_engineer", "ml_optimizer"]
        target = get_replan_target(s)
        assert target == "data_engineer"
