# tests/regression/test_phase2_regression.py
#
# Written: Day 13 (stabilisation gate)
# Gate baseline: 105 tests passing across 14 files (21 from day12, 84 from contracts)
# Commit hash: b9e13c8292e2a6b9e3626093af960a2d8c1ba276
#
# IMMUTABLE: NEVER edit this file after the Phase 2 gate is locked.
# This is the permanent floor that protects everything built in Phase 2.
# If any of these tests fail after a code change, the change introduced
# a regression and must be reverted or fixed before merging.

import os
import sys
import subprocess
import inspect

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.state import initial_state, ProfessorState
from guards.circuit_breaker import (
    EscalationLevel,
    get_escalation_level,
    handle_escalation,
    reset_failure_count,
    generate_hitl_prompt,
    resume_from_checkpoint,
)


# ── Freeze 1: Graph Compiles ─────────────────────────────────────
class TestGraphCompiles:
    """The LangGraph StateGraph must compile without errors."""

    def test_graph_compiles(self):
        from core.professor import build_graph
        graph = build_graph()
        assert graph is not None, "build_graph() returned None"

    def test_feature_factory_node_registered(self):
        from core.professor import build_graph
        graph = build_graph()
        node_names = [n for n in graph.nodes]
        assert "feature_factory" in node_names, (
            "feature_factory node missing from graph"
        )


# ── Freeze 2: Circuit Breaker Contract ───────────────────────────
class TestCircuitBreakerContract:
    """Core escalation logic must behave correctly at each level."""

    def test_four_escalation_levels_exist(self):
        levels = list(EscalationLevel)
        assert len(levels) == 4
        assert set(l.value for l in levels) == {"micro", "macro", "hitl", "triage"}

    def test_micro_appends_error_context(self):
        from unittest.mock import patch
        state = initial_state("comp", "data")
        with patch("guards.circuit_breaker.log_event"):
            result = handle_escalation(
                state, EscalationLevel.MICRO, "test_agent",
                ValueError("test"), "traceback"
            )
        assert len(result.get("error_context", [])) >= 1

    def test_macro_increments_dag_version(self):
        from unittest.mock import patch
        state = initial_state("comp", "data")
        state["dag_version"] = 1
        with patch("guards.circuit_breaker.log_event"):
            result = handle_escalation(
                state, EscalationLevel.MACRO, "test_agent",
                ValueError("test"), "traceback"
            )
        assert result["dag_version"] == 2
        assert result["macro_replan_requested"] is True

    def test_hitl_sets_required_flag(self):
        from unittest.mock import patch
        state = initial_state("comp", "data")
        state["current_node_failure_count"] = 2
        with patch("guards.circuit_breaker._checkpoint_state_to_redis"):
            with patch("guards.circuit_breaker.log_event"):
                with patch("guards.circuit_breaker.generate_hitl_prompt",
                           return_value={"interventions": [], "checkpoint_key": "ck"}):
                    result = handle_escalation(
                        state, EscalationLevel.HITL, "test_agent",
                        KeyError("missing"), "traceback"
                    )
        assert result["hitl_required"] is True
        assert result["pipeline_halted"] is True

    def test_escalation_level_progression(self):
        state = initial_state("comp", "data")
        state["current_node_failure_count"] = 0
        assert get_escalation_level(state) == EscalationLevel.MICRO
        state["current_node_failure_count"] = 1
        assert get_escalation_level(state) == EscalationLevel.MICRO
        state["current_node_failure_count"] = 2
        assert get_escalation_level(state) == EscalationLevel.MACRO
        state["current_node_failure_count"] = 3
        assert get_escalation_level(state) == EscalationLevel.HITL

    def test_reset_failure_count(self):
        state = initial_state("comp", "data")
        state["current_node_failure_count"] = 5
        result = reset_failure_count(state)
        assert result["current_node_failure_count"] == 0


# ── Freeze 3: HITL Prompt Generation ─────────────────────────────
class TestHITLPromptGeneration:
    """generate_hitl_prompt must produce a valid prompt dict."""

    def test_prompt_has_required_keys(self):
        from unittest.mock import patch
        state = initial_state("comp", "data")
        state["session_id"] = "test_session"
        with patch("guards.circuit_breaker._write_hitl_prompt"):
            with patch("guards.circuit_breaker._print_hitl_banner"):
                prompt = generate_hitl_prompt(state, "data_engineer", ValueError("err"))
        assert "failed_agent" in prompt
        assert "checkpoint_key" in prompt
        assert "interventions" in prompt
        assert prompt["failed_agent"] == "data_engineer"


# ── Freeze 4: Safe Contract Tests Still Pass ─────────────────────
class TestSafeContractTestsPass:
    """
    Meta-test: contract test files that passed at gate time must still pass.
    Excludes files known to hang without API keys or fixtures.
    """

    SAFE_CONTRACT_FILES = [
        "tests/contracts/test_circuit_breaker_contract.py",
        "tests/contracts/test_e2b_sandbox_contract.py",
        "tests/contracts/test_hitl_prompt_contract.py",
        "tests/contracts/test_placeholder.py",
        "tests/contracts/test_resume_checkpoint_contract.py",
        "tests/contracts/test_semantic_router_contract.py",
        "tests/contracts/test_supervisor_replan_contract.py",
        "tests/contracts/test_validation_architect_contract.py",
    ]

    @pytest.mark.parametrize("test_file", SAFE_CONTRACT_FILES)
    def test_contract_file_passes(self, test_file):
        result = subprocess.run(
            [sys.executable, "-m", "pytest", test_file, "-q", "--tb=short"],
            capture_output=True, text=True, timeout=120
        )
        assert result.returncode == 0, (
            f"{test_file} failed (exit {result.returncode}):\n"
            f"{result.stdout[-300:]}\n{result.stderr[-200:]}"
        )


# ── Freeze 5: Day 12 Quality Tests Pass ──────────────────────────
class TestDay12QualityPasses:
    """All 21 day12 quality tests must pass — covers HITL + OOM guardrails."""

    def test_day12_all_pass(self):
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/test_day12_quality.py",
             "-q", "--tb=short"],
            capture_output=True, text=True, timeout=120
        )
        assert result.returncode == 0, (
            f"test_day12_quality.py failed (exit {result.returncode}):\n"
            f"{result.stdout[-400:]}\n{result.stderr[-200:]}"
        )


# ── Freeze 6: Key Module Imports Resolve ──────────────────────────
class TestModuleImports:
    """All Phase 2 modules must import without error."""

    def test_import_circuit_breaker(self):
        import guards.circuit_breaker

    def test_import_agent_retry(self):
        import guards.agent_retry

    def test_import_validation_architect(self):
        import agents.validation_architect

    def test_import_feature_factory(self):
        import agents.feature_factory

    def test_import_eda_agent(self):
        import agents.eda_agent

    def test_import_competition_intel(self):
        import agents.competition_intel

    def test_import_semantic_router(self):
        import agents.semantic_router

    def test_import_supervisor(self):
        import agents.supervisor

    def test_import_submission_strategist(self):
        import agents.submission_strategist
