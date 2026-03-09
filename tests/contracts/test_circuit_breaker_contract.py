# tests/contracts/test_circuit_breaker_contract.py
# ─────────────────────────────────────────────────────────────────────────────
# Written: Day 9   Status: IMMUTABLE
#
# CONTRACT: guards/circuit_breaker.py
#   failure_count=1 → MICRO (patch node, append error context)
#   failure_count=2 → MACRO (rewrite DAG, dag_version increments)
#   failure_count=3 → HITL  (hitl_required=True, pipeline_halted=True)
#   budget exhausted → TRIAGE (triage_mode=True, pipeline_halted=True)
#   success           → reset_failure_count returns count to 0
# ─────────────────────────────────────────────────────────────────────────────
import pytest
from core.state import initial_state
from guards.circuit_breaker import (
    get_escalation_level,
    handle_escalation,
    reset_failure_count,
    EscalationLevel,
)

FIXTURE_CSV = "tests/fixtures/tiny_train.csv"


@pytest.fixture
def base_state():
    return initial_state("test-cb-contract", FIXTURE_CSV)


class TestCircuitBreakerContract:

    # ── Level detection ──────────────────────────────────────────────────────

    def test_first_failure_is_micro(self, base_state):
        state = {**base_state, "current_node_failure_count": 1}
        assert get_escalation_level(state) == EscalationLevel.MICRO

    def test_second_failure_is_macro(self, base_state):
        state = {**base_state, "current_node_failure_count": 2}
        assert get_escalation_level(state) == EscalationLevel.MACRO

    def test_third_failure_is_hitl(self, base_state):
        state = {**base_state, "current_node_failure_count": 3}
        assert get_escalation_level(state) == EscalationLevel.HITL

    def test_budget_exhaustion_is_triage(self, base_state):
        state = {**base_state, "budget_remaining_usd": 0.01, "budget_limit_usd": 5.0}
        assert get_escalation_level(state) == EscalationLevel.TRIAGE

    def test_triage_overrides_failure_count(self, base_state):
        """TRIAGE takes priority even if failure_count is only 1."""
        state = {
            **base_state,
            "current_node_failure_count": 1,
            "budget_remaining_usd": 0.01,
            "budget_limit_usd": 5.0,
        }
        assert get_escalation_level(state) == EscalationLevel.TRIAGE

    def test_time_exhaustion_is_triage(self, base_state):
        state = {
            **base_state,
            "competition_context": {
                **base_state.get("competition_context", {}),
                "hours_remaining": 1,
            },
        }
        assert get_escalation_level(state) == EscalationLevel.TRIAGE

    def test_zero_failures_defaults_to_micro(self, base_state):
        """When no failures have occurred, first escalation should be MICRO."""
        state = {**base_state, "current_node_failure_count": 0}
        assert get_escalation_level(state) == EscalationLevel.MICRO

    # ── MICRO behaviour ──────────────────────────────────────────────────────

    def test_micro_appends_error_context(self, base_state):
        state = {**base_state, "current_node_failure_count": 1}
        result = handle_escalation(
            state=state,
            level=EscalationLevel.MICRO,
            agent_name="test_agent",
            error=ValueError("test error"),
            traceback_str="Traceback (most recent call last): ...",
        )
        assert len(result["error_context"]) == 1
        assert result["error_context"][0]["agent"] == "test_agent"
        assert "test error" in result["error_context"][0]["error"]

    def test_micro_increments_failure_count(self, base_state):
        state = {**base_state, "current_node_failure_count": 1}
        result = handle_escalation(
            state=state,
            level=EscalationLevel.MICRO,
            agent_name="test_agent",
            error=ValueError("micro error"),
            traceback_str="tb",
        )
        assert result["current_node_failure_count"] == 2

    def test_micro_does_not_set_hitl(self, base_state):
        state = {**base_state, "current_node_failure_count": 1}
        result = handle_escalation(
            state=state, level=EscalationLevel.MICRO,
            agent_name="a", error=ValueError("e"), traceback_str="t"
        )
        assert result.get("hitl_required") is not True

    def test_micro_preserves_existing_error_context(self, base_state):
        """Error context should accumulate across MICRO escalations."""
        existing = [{"agent": "old", "attempt": 0, "error": "old err", "traceback": "old tb"}]
        state = {**base_state, "current_node_failure_count": 1, "error_context": existing}
        result = handle_escalation(
            state=state, level=EscalationLevel.MICRO,
            agent_name="new_agent", error=ValueError("new"), traceback_str="new tb"
        )
        assert len(result["error_context"]) == 2
        assert result["error_context"][0]["agent"] == "old"
        assert result["error_context"][1]["agent"] == "new_agent"

    # ── MACRO behaviour ──────────────────────────────────────────────────────

    def test_macro_increments_dag_version(self, base_state):
        state = {**base_state, "current_node_failure_count": 2, "dag_version": 1}
        result = handle_escalation(
            state=state,
            level=EscalationLevel.MACRO,
            agent_name="test_agent",
            error=RuntimeError("macro error"),
            traceback_str="tb",
        )
        assert result["dag_version"] == 2, (
            f"dag_version should have incremented to 2, got {result['dag_version']}"
        )

    def test_macro_sets_replan_flag(self, base_state):
        state = {**base_state, "current_node_failure_count": 2}
        result = handle_escalation(
            state=state, level=EscalationLevel.MACRO,
            agent_name="a", error=RuntimeError("e"), traceback_str="t"
        )
        assert result["macro_replan_requested"] is True

    def test_macro_replan_reason_names_agent(self, base_state):
        state = {**base_state, "current_node_failure_count": 2}
        result = handle_escalation(
            state=state, level=EscalationLevel.MACRO,
            agent_name="ml_optimizer", error=RuntimeError("OOM"), traceback_str="t"
        )
        assert "ml_optimizer" in result["macro_replan_reason"]

    def test_macro_does_not_halt_pipeline(self, base_state):
        """MACRO replans but does not halt — the Supervisor rewrites and continues."""
        state = {**base_state, "current_node_failure_count": 2}
        result = handle_escalation(
            state=state, level=EscalationLevel.MACRO,
            agent_name="a", error=RuntimeError("e"), traceback_str="t"
        )
        assert result.get("pipeline_halted") is not True

    # ── HITL behaviour ───────────────────────────────────────────────────────

    def test_hitl_sets_hitl_required_true(self, base_state):
        state = {**base_state, "current_node_failure_count": 3}
        result = handle_escalation(
            state=state,
            level=EscalationLevel.HITL,
            agent_name="feature_factory",
            error=Exception("persistent failure"),
            traceback_str="tb",
        )
        assert result["hitl_required"] is True

    def test_hitl_halts_pipeline(self, base_state):
        state = {**base_state, "current_node_failure_count": 3}
        result = handle_escalation(
            state=state, level=EscalationLevel.HITL,
            agent_name="a", error=Exception("e"), traceback_str="t"
        )
        assert result["pipeline_halted"] is True

    def test_hitl_reason_names_agent_and_session(self, base_state):
        state = {**base_state, "current_node_failure_count": 3}
        result = handle_escalation(
            state=state, level=EscalationLevel.HITL,
            agent_name="red_team_critic", error=Exception("critical fail"), traceback_str="t"
        )
        reason = result.get("hitl_reason", "")
        assert "red_team_critic" in reason, f"Agent name missing from hitl_reason: {reason}"
        assert base_state["session_id"] in reason, f"Session ID missing from hitl_reason: {reason}"

    # ── TRIAGE behaviour ─────────────────────────────────────────────────────

    def test_triage_sets_triage_mode(self, base_state):
        state = {**base_state, "budget_remaining_usd": 0.001, "budget_limit_usd": 5.0}
        result = handle_escalation(
            state=state, level=EscalationLevel.TRIAGE,
            agent_name="a", error=Exception("budget"), traceback_str="t"
        )
        assert result["triage_mode"] is True

    def test_triage_halts_pipeline(self, base_state):
        state = {**base_state, "budget_remaining_usd": 0.001, "budget_limit_usd": 5.0}
        result = handle_escalation(
            state=state, level=EscalationLevel.TRIAGE,
            agent_name="a", error=Exception("e"), traceback_str="t"
        )
        assert result["pipeline_halted"] is True

    # ── Reset ────────────────────────────────────────────────────────────────

    def test_reset_sets_failure_count_to_zero(self, base_state):
        state = {**base_state, "current_node_failure_count": 2, "error_context": [{"x": 1}]}
        result = reset_failure_count(state)
        assert result["current_node_failure_count"] == 0

    def test_reset_clears_error_context(self, base_state):
        state = {**base_state, "error_context": [{"agent": "x", "error": "y"}]}
        result = reset_failure_count(state)
        assert result["error_context"] == []

    def test_handle_escalation_never_raises(self, base_state):
        """The circuit breaker must never crash — it is the last line of defence."""
        for level in EscalationLevel:
            try:
                handle_escalation(
                    state=base_state,
                    level=level,
                    agent_name="test",
                    error=Exception("test"),
                    traceback_str="test traceback",
                )
            except Exception as e:
                pytest.fail(
                    f"handle_escalation raised an exception for level {level}: {e}. "
                    "The circuit breaker must never raise — it is the last line of defence."
                )
