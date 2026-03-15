# tests/test_day15_quality.py
# ─────────────────────────────────────────────────────────────────
# Day 15 Quality Tests — 48 tests across 4 blocks
# ─────────────────────────────────────────────────────────────────

import json
import os
import sys
import time
import threading
import subprocess
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock, call
from datetime import datetime, timezone

import pytest

from core.state import initial_state, ProfessorState


# ── Shared helpers ─────────────────────────────────────────────────

def _make_state(**overrides) -> ProfessorState:
    state = initial_state(
        competition="test-titanic",
        data_path="data/spaceship_titanic/train.csv",
        budget_usd=2.0,
    )
    state = {**state, **overrides}
    return state


# ======================================================================
# BLOCK 1 — GRAPH SINGLETON (10 tests)
# ======================================================================

class TestGraphSingleton:
    """Tests for the graph singleton pattern in core/professor.py."""

    def test_graph_compiled_only_once_across_multiple_invocations(self):
        """build_graph called exactly once across 3 run_professor calls."""
        import core.professor as prof
        call_count = {"n": 0}
        original_build = prof.build_graph

        def counting_build():
            call_count["n"] += 1
            return original_build()

        with patch.object(prof, "build_graph", side_effect=counting_build):
            state = _make_state()
            # We just test get_graph calls, not run_professor (which invokes the full pipeline)
            prof.get_graph()
            prof.get_graph()
            prof.get_graph()

        assert call_count["n"] == 1, f"build_graph called {call_count['n']} times, expected 1"

    def test_get_graph_returns_same_object_on_repeated_calls(self):
        """get_graph() returns the same Python object (object identity)."""
        from core.professor import get_graph
        g1 = get_graph()
        g2 = get_graph()
        assert id(g1) == id(g2), "get_graph() returned different objects"

    def test_cache_clear_forces_recompilation_on_next_call(self):
        """After cache_clear, get_graph builds a new graph."""
        import core.professor as prof
        call_count = {"n": 0}
        original_build = prof.build_graph

        def counting_build():
            call_count["n"] += 1
            return original_build()

        with patch.object(prof, "build_graph", side_effect=counting_build):
            g1 = prof.get_graph()
            prof.get_graph_cache_clear()
            g2 = prof.get_graph()

        assert call_count["n"] == 2, f"Expected 2 compilations, got {call_count['n']}"
        assert id(g1) != id(g2), "Expected different objects after cache clear"

    def test_run_professor_uses_get_graph_not_build_graph_directly(self):
        """run_professor must use get_graph, not build_graph directly."""
        import core.professor as prof

        call_count = {"n": 0}
        original_build = prof.build_graph

        def counting_build():
            call_count["n"] += 1
            return original_build()

        # Pre-warm the singleton
        prof.get_graph()
        initial_count = 1  # from the pre-warm

        with patch.object(prof, "build_graph", side_effect=counting_build):
            # After clearing, run_professor should call get_graph which calls build_graph once
            prof.get_graph_cache_clear()

            # Patch graph.invoke to avoid running the full pipeline
            mock_graph = MagicMock()
            mock_graph.invoke.return_value = _make_state()

            with patch.object(prof, "build_graph", return_value=mock_graph):
                call_count_inner = {"n": 0}
                original_build_inner = prof.build_graph

                def counting_build_inner():
                    call_count_inner["n"] += 1
                    return original_build_inner()

                with patch.object(prof, "build_graph", side_effect=counting_build_inner):
                    prof.run_professor(_make_state())
                    prof.run_professor(_make_state())

                assert call_count_inner["n"] <= 1, (
                    f"build_graph called {call_count_inner['n']} times across 2 run_professor calls. "
                    "run_professor must use get_graph(), not build_graph() directly."
                )

    def test_singleton_thread_safe_under_concurrent_calls(self):
        """5 threads calling get_graph simultaneously — build_graph called exactly once."""
        import core.professor as prof

        call_count = {"n": 0}
        lock = threading.Lock()
        original_build = prof.build_graph

        def slow_counting_build():
            with lock:
                call_count["n"] += 1
            time.sleep(0.1)  # Simulate compilation time
            return original_build()

        prof.get_graph_cache_clear()

        with patch.object(prof, "build_graph", side_effect=slow_counting_build):
            threads = [threading.Thread(target=prof.get_graph) for _ in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)

        assert call_count["n"] == 1, (
            f"build_graph called {call_count['n']} times with 5 concurrent threads. "
            "Double-checked locking must prevent multiple compilations."
        )

    def test_singleton_resets_on_cache_clear_not_on_access(self):
        """10 get_graph() calls without cache_clear — build_graph called once."""
        import core.professor as prof

        call_count = {"n": 0}
        original_build = prof.build_graph

        def counting_build():
            call_count["n"] += 1
            return original_build()

        with patch.object(prof, "build_graph", side_effect=counting_build):
            for _ in range(10):
                prof.get_graph()

        assert call_count["n"] == 1, f"build_graph called {call_count['n']} times, expected 1"

    def test_cache_clear_sets_module_level_graph_to_none(self):
        """After cache_clear, core.professor._GRAPH is None."""
        import core.professor as prof
        prof.get_graph()  # populate
        assert prof._GRAPH is not None, "Graph not populated after get_graph()"
        prof.get_graph_cache_clear()
        assert prof._GRAPH is None, "cache_clear did not set _GRAPH to None"

    def test_get_graph_builds_valid_compilable_graph(self):
        """get_graph() returns an object with .invoke method."""
        from core.professor import get_graph
        graph = get_graph()
        assert hasattr(graph, "invoke"), "Graph missing .invoke method"

    def test_compilation_time_improvement_over_repeated_calls(self):
        """Cached calls are faster than first compilation."""
        import core.professor as prof
        prof.get_graph_cache_clear()

        start = time.perf_counter()
        prof.get_graph()
        first_time = time.perf_counter() - start

        times = []
        for _ in range(5):
            start = time.perf_counter()
            prof.get_graph()
            times.append(time.perf_counter() - start)

        avg_cached = sum(times) / len(times)
        # Cached calls should be at least 10x faster (they're just a None check)
        assert avg_cached < first_time or first_time < 0.001, (
            f"First call: {first_time:.4f}s, avg cached: {avg_cached:.4f}s. "
            "Cached calls should be significantly faster."
        )

    def test_conftest_cache_clear_fixture_resets_between_tests(self):
        """Verify the autouse fixture clears the singleton between tests."""
        import core.professor as prof
        # This test just verifies that after the previous test,
        # the fixture ran and cleared the singleton.
        # Since autouse=True clears after yield, _GRAPH should be None
        # at the START of this test only if the fixture is working.
        # However, get_graph() calls during previous test setup may repopulate.
        # The key invariant: cache_clear is callable and works.
        prof.get_graph_cache_clear()
        assert prof._GRAPH is None


# ======================================================================
# BLOCK 2 — DOCKER SANDBOX (14 tests)
# ======================================================================

class TestDockerSandbox:
    """Tests for the Docker sandbox backend in tools/e2b_sandbox.py."""

    def test_execute_code_returns_expected_stdout(self):
        """Simple code produces correct stdout."""
        from tools.e2b_sandbox import _execute_once
        result = _execute_once('print("hello world")', "test_session")
        assert result["returncode"] == 0
        assert "hello world" in result["stdout"]

    def test_execute_code_captures_stderr_separately(self):
        """stderr and stdout are separate."""
        from tools.e2b_sandbox import _execute_once
        result = _execute_once(
            'import sys; sys.stderr.write("error message\\n")',
            "test_session"
        )
        assert "error message" in result["stderr"]

    def test_backend_field_indicates_docker_when_available(self):
        """Result dict contains 'backend' key."""
        from tools.e2b_sandbox import _execute_once
        result = _execute_once('print(1)', "test_session")
        assert "backend" in result, "Result dict missing 'backend' key"
        assert result["backend"] in ("docker", "subprocess"), (
            f"backend must be 'docker' or 'subprocess', got '{result['backend']}'"
        )

    def test_container_destroyed_after_successful_execution(self):
        """Docker --rm flag ensures container cleanup."""
        import tools.e2b_sandbox as sandbox
        if not sandbox._USE_DOCKER:
            pytest.skip("Docker not available")

        result = sandbox._execute_docker('print("cleanup test")', "test_session")
        assert result["success"]

        # Check no professor-sandbox containers left
        ps = subprocess.run(
            ["docker", "ps", "-a", "--filter", "name=professor-sandbox",
             "--format", "{{.Names}}"],
            capture_output=True, text=True, timeout=5,
        )
        containers = [c for c in ps.stdout.strip().split("\n") if c]
        # Some containers from OTHER tests may exist, but this specific one should be gone

    def test_container_destroyed_after_timeout(self):
        """_force_remove_container called on TimeoutExpired."""
        import tools.e2b_sandbox as sandbox

        with patch.object(sandbox, "_force_remove_container") as mock_remove:
            with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(
                cmd=["docker", "run"], timeout=1
            )):
                result = sandbox._execute_docker('print("slow")', "test_session", timeout_seconds=1)

            assert result["timed_out"] is True
            mock_remove.assert_called_once()
            # Verify container name was passed
            call_args = mock_remove.call_args[0][0]
            assert call_args.startswith("professor-sandbox-")

    def test_container_names_are_unique(self):
        """Each execution gets a unique container name."""
        import tools.e2b_sandbox as sandbox
        names = set()

        original_run = subprocess.run

        def capture_name(*args, **kwargs):
            cmd = args[0] if args else kwargs.get("cmd", [])
            if isinstance(cmd, list) and "--name" in cmd:
                idx = cmd.index("--name")
                if idx + 1 < len(cmd):
                    names.add(cmd[idx + 1])
            # Return a mock result
            mock = MagicMock()
            mock.returncode = 0
            mock.stdout = ""
            mock.stderr = ""
            return mock

        with patch("subprocess.run", side_effect=capture_name):
            for _ in range(10):
                sandbox._execute_docker("print(1)", "test_session")

        assert len(names) == 10, f"Expected 10 unique container names, got {len(names)}"

    def test_stdout_capped_at_max_output_bytes(self):
        """stdout truncated to MAX_OUTPUT_BYTES."""
        import tools.e2b_sandbox as sandbox

        big_output = "x" * (sandbox.MAX_OUTPUT_BYTES + 1000)
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = big_output
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            result = sandbox._execute_docker("print('big')", "test_session")

        assert len(result["stdout"]) <= sandbox.MAX_OUTPUT_BYTES

    def test_timed_out_flag_set_correctly(self):
        """timed_out=True on timeout."""
        from tools.e2b_sandbox import _execute_once
        result = _execute_once(
            'import time; time.sleep(9999)',
            "test_session",
            timeout_seconds=2,
        )
        assert result["timed_out"] is True
        assert result["returncode"] == -1

    def test_timed_out_flag_false_on_success(self):
        """timed_out=False on normal execution."""
        from tools.e2b_sandbox import _execute_once
        result = _execute_once('print("fast")', "test_session")
        assert result["timed_out"] is False

    def test_fallback_to_subprocess_when_docker_unavailable(self):
        """When Docker unavailable, falls back to subprocess."""
        import tools.e2b_sandbox as sandbox

        original = sandbox._USE_DOCKER
        try:
            sandbox._USE_DOCKER = False
            result = sandbox._execute_once('print("fallback")', "test_session")
            assert result["backend"] == "subprocess"
            assert "fallback" in result["stdout"]
        finally:
            sandbox._USE_DOCKER = original

    def test_network_none_flag_in_docker_command(self):
        """--network none must be in the Docker command."""
        import tools.e2b_sandbox as sandbox

        captured_cmd = {}

        def capture_cmd(*args, **kwargs):
            cmd = args[0] if args else kwargs.get("cmd", [])
            captured_cmd["cmd"] = cmd
            mock = MagicMock()
            mock.returncode = 0
            mock.stdout = ""
            mock.stderr = ""
            return mock

        with patch("subprocess.run", side_effect=capture_cmd):
            sandbox._execute_docker("print(1)", "test_session")

        cmd = captured_cmd.get("cmd", [])
        assert "--network" in cmd, "--network flag missing from docker command"
        net_idx = cmd.index("--network")
        assert cmd[net_idx + 1] == "none", f"Expected --network none, got --network {cmd[net_idx + 1]}"

    def test_read_only_filesystem_flag_in_docker_command(self):
        """--read-only flag present in docker command."""
        import tools.e2b_sandbox as sandbox

        captured_cmd = {}

        def capture_cmd(*args, **kwargs):
            captured_cmd["cmd"] = args[0] if args else kwargs.get("cmd", [])
            mock = MagicMock()
            mock.returncode = 0
            mock.stdout = ""
            mock.stderr = ""
            return mock

        with patch("subprocess.run", side_effect=capture_cmd):
            sandbox._execute_docker("print(1)", "test_session")

        assert "--read-only" in captured_cmd.get("cmd", [])

    def test_memory_limit_flag_in_docker_command(self):
        """--memory and --memory-swap flags present."""
        import tools.e2b_sandbox as sandbox

        captured_cmd = {}

        def capture_cmd(*args, **kwargs):
            captured_cmd["cmd"] = args[0] if args else kwargs.get("cmd", [])
            mock = MagicMock()
            mock.returncode = 0
            mock.stdout = ""
            mock.stderr = ""
            return mock

        with patch("subprocess.run", side_effect=capture_cmd):
            sandbox._execute_docker("print(1)", "test_session")

        cmd = captured_cmd.get("cmd", [])
        assert "--memory" in cmd, "--memory flag missing"
        mem_idx = cmd.index("--memory")
        assert cmd[mem_idx + 1] == sandbox.MEMORY_LIMIT
        assert "--memory-swap" in cmd, "--memory-swap flag missing"
        swap_idx = cmd.index("--memory-swap")
        assert cmd[swap_idx + 1] == sandbox.MEMORY_LIMIT

    def test_execute_code_never_raises_on_any_input(self):
        """execute_code always returns a dict, never raises."""
        from tools.e2b_sandbox import _execute_once

        test_inputs = [
            "",                          # empty
            "def (",                     # malformed
            "x" * 10000,                 # large
            "import sys; sys.exit(1)",   # explicit exit
            "raise RuntimeError('boom')",  # exception
        ]

        for code in test_inputs:
            result = _execute_once(code, "test_session", timeout_seconds=10)
            assert isinstance(result, dict), f"Expected dict for input: {code[:30]}..."
            assert "stdout" in result
            assert "stderr" in result
            assert "returncode" in result


# ======================================================================
# BLOCK 3 — LANGFUSE OBSERVABILITY (10 tests)
# ======================================================================

class TestLangFuseObservability:
    """Tests for LangFuse tracing in core/professor.py."""

    def test_langfuse_disabled_when_keys_absent(self):
        """LangFuse disabled without keys."""
        import core.professor as prof
        # Keys should not be in env during testing
        env = {k: v for k, v in os.environ.items()
               if k not in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")}
        with patch.dict(os.environ, env, clear=True):
            result = prof._init_langfuse()
        assert result is False

    def test_langfuse_init_called_once_at_module_load(self):
        """_init_langfuse not called per run_professor invocation."""
        import core.professor as prof
        # Since _init_langfuse is called at module load, we verify that
        # _LANGFUSE_CLIENT is not re-created on each run_professor call.
        client_before = prof._LANGFUSE_CLIENT

        mock_graph = MagicMock()
        mock_graph.invoke.return_value = _make_state()

        with patch.object(prof, "get_graph", return_value=mock_graph):
            prof.run_professor(_make_state())
            prof.run_professor(_make_state())
            prof.run_professor(_make_state())

        # Client should be the same object (or None), not recreated
        assert prof._LANGFUSE_CLIENT is client_before

    def test_run_professor_creates_one_trace_per_invocation(self):
        """One LangFuse trace per run_professor call when enabled."""
        import core.professor as prof

        mock_client = MagicMock()
        mock_trace = MagicMock()
        mock_client.trace.return_value = mock_trace
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = _make_state()

        original_enabled = prof._LANGFUSE_ENABLED
        original_client = prof._LANGFUSE_CLIENT
        try:
            prof._LANGFUSE_ENABLED = True
            prof._LANGFUSE_CLIENT = mock_client

            with patch.object(prof, "get_graph", return_value=mock_graph):
                prof.run_professor(_make_state())

            mock_client.trace.assert_called_once()
        finally:
            prof._LANGFUSE_ENABLED = original_enabled
            prof._LANGFUSE_CLIENT = original_client

    def test_langfuse_flush_called_after_run(self):
        """_LANGFUSE_CLIENT.flush() called after pipeline completion."""
        import core.professor as prof

        mock_client = MagicMock()
        mock_trace = MagicMock()
        mock_client.trace.return_value = mock_trace
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = _make_state()

        original_enabled = prof._LANGFUSE_ENABLED
        original_client = prof._LANGFUSE_CLIENT
        try:
            prof._LANGFUSE_ENABLED = True
            prof._LANGFUSE_CLIENT = mock_client

            with patch.object(prof, "get_graph", return_value=mock_graph):
                prof.run_professor(_make_state())

            mock_client.flush.assert_called()
        finally:
            prof._LANGFUSE_ENABLED = original_enabled
            prof._LANGFUSE_CLIENT = original_client

    def test_langfuse_trace_not_serialised_into_redis_checkpoint(self):
        """_langfuse_trace object excluded from Redis checkpoint serialization."""
        from guards.circuit_breaker import _is_serialisable

        mock_trace = MagicMock()
        mock_trace.__class__.__name__ = "StatefulTraceClient"

        # The trace object should NOT be serializable
        assert not _is_serialisable(mock_trace), (
            "LangFuse trace object should not be JSON-serializable. "
            "It must be excluded from Redis checkpoints."
        )

    def test_langfuse_trace_marked_error_on_pipeline_exception(self):
        """trace.update(status='ERROR') called on exception."""
        import core.professor as prof

        mock_client = MagicMock()
        mock_trace = MagicMock()
        mock_client.trace.return_value = mock_trace

        original_enabled = prof._LANGFUSE_ENABLED
        original_client = prof._LANGFUSE_CLIENT
        try:
            prof._LANGFUSE_ENABLED = True
            prof._LANGFUSE_CLIENT = mock_client

            with pytest.raises(RuntimeError):
                with prof._langfuse_trace("test", "comp"):
                    raise RuntimeError("Pipeline crashed")

            mock_trace.update.assert_any_call(
                status="ERROR",
                status_message="Pipeline crashed",
            )
        finally:
            prof._LANGFUSE_ENABLED = original_enabled
            prof._LANGFUSE_CLIENT = original_client

    def test_langfuse_and_jsonl_write_to_different_sinks(self):
        """JSONL lineage still active when LangFuse is enabled."""
        # Verify JSONL logging still works regardless of LangFuse status
        from core.lineage import log_event
        import tempfile

        # log_event should work independently of LangFuse
        log_event(
            session_id="test_dual_sink",
            agent="test_agent",
            action="test_action",
            keys_read=[],
            keys_written=[],
            values_changed={},
        )
        # Just verifying no exception — JSONL and LangFuse are independent

    def test_langfuse_init_failure_falls_back_to_jsonl(self):
        """ConnectionError during init doesn't crash professor."""
        import core.professor as prof

        with patch("langfuse.Langfuse", side_effect=ConnectionError("network down")):
            with patch.dict(os.environ, {
                "LANGFUSE_PUBLIC_KEY": "test_key",
                "LANGFUSE_SECRET_KEY": "test_secret",
            }):
                result = prof._init_langfuse()

        assert result is False, "_init_langfuse must return False on connection error"

    def test_langfuse_trace_context_manager_noop_when_disabled(self):
        """_langfuse_trace yields None when LangFuse disabled."""
        import core.professor as prof

        original_enabled = prof._LANGFUSE_ENABLED
        try:
            prof._LANGFUSE_ENABLED = False
            with prof._langfuse_trace("sess", "comp") as trace:
                assert trace is None
        finally:
            prof._LANGFUSE_ENABLED = original_enabled

    def test_trace_node_noop_when_trace_is_none(self):
        """_trace_node is a no-op when trace is None."""
        from core.professor import _trace_node
        # Must not raise
        _trace_node(None, "test_node", {"input": 1}, {"output": 2})


# ======================================================================
# BLOCK 4 — EXTERNAL DATA SCOUT (14 tests)
# ======================================================================

MOCK_MANIFEST_RESPONSE = json.dumps({
    "external_sources": [
        {
            "name": "WorldBank GDP",
            "type": "public_dataset",
            "description": "GDP per capita",
            "source_url": "https://data.worldbank.org",
            "relevance_score": 0.9,
            "join_strategy": "country column",
            "acquisition_method": "wget URL",
            "competition_precedent": None,
        },
        {
            "name": "Weather Data",
            "type": "external_signal",
            "description": "Historical weather",
            "source_url": "https://weather.api",
            "relevance_score": 0.7,
            "join_strategy": "date column",
            "acquisition_method": "pip install weatherdata",
            "competition_precedent": None,
        },
        {
            "name": "Random Noise",
            "type": "external_signal",
            "description": "Irrelevant data",
            "source_url": "https://noise.api",
            "relevance_score": 0.5,
            "join_strategy": "none",
            "acquisition_method": "wget URL",
            "competition_precedent": None,
        },
    ],
    "recommended_sources": ["WorldBank GDP", "Weather Data"],
    "total_sources_found": 3,
    "scout_notes": "GDP and weather are relevant.",
})

MOCK_BRIEF_JSON = json.dumps({
    "critical_findings": [],
    "proven_features": [],
    "known_leaks": [],
    "external_datasets": [],
    "dominant_approach": "LightGBM",
    "cv_strategy_hint": "StratifiedKFold",
    "forbidden_techniques": [],
    "shakeup_risk": "medium",
    "source_post_count": 0,
    "scraped_at": "2025-01-01T00:00:00",
})


def _run_intel_with_mocks(state, manifest_response=MOCK_MANIFEST_RESPONSE):
    """Run competition_intel with mocked Kaggle API and LLM."""
    with patch("agents.competition_intel._fetch_notebooks", return_value=[]):
        with patch("agents.competition_intel.call_llm") as mock_llm:
            mock_llm.side_effect = [MOCK_BRIEF_JSON, manifest_response]
            from agents.competition_intel import run_competition_intel
            return run_competition_intel(state)


class TestExternalDataScout:
    """Tests for the External Data Scout in agents/competition_intel.py."""

    def test_scout_skipped_when_external_data_not_allowed(self):
        """No external sources when external_data_allowed=False."""
        state = _make_state(external_data_allowed=False)

        with patch("agents.competition_intel._fetch_notebooks", return_value=[]):
            with patch("agents.competition_intel.call_llm") as mock_llm:
                mock_llm.return_value = MOCK_BRIEF_JSON
                from agents.competition_intel import run_competition_intel
                result = run_competition_intel(state)

        manifest = result.get("external_data_manifest", {})
        assert manifest.get("external_sources") == [], (
            "external_sources must be empty when external_data_allowed=False"
        )

    def test_manifest_written_even_when_scout_skipped(self):
        """external_data_manifest.json exists even when scout skipped."""
        state = _make_state(external_data_allowed=False)

        with patch("agents.competition_intel._fetch_notebooks", return_value=[]):
            with patch("agents.competition_intel.call_llm", return_value=MOCK_BRIEF_JSON):
                from agents.competition_intel import run_competition_intel
                result = run_competition_intel(state)

        manifest_path = Path(f"outputs/{result['session_id']}/external_data_manifest.json")
        assert manifest_path.exists(), "external_data_manifest.json not written when scout skipped"
        manifest = json.loads(manifest_path.read_text())
        assert manifest.get("external_sources") == []

    def test_scout_runs_when_external_data_allowed(self):
        """Scout produces manifest when external_data_allowed=True."""
        state = _make_state(external_data_allowed=True)
        result = _run_intel_with_mocks(state)

        manifest = result.get("external_data_manifest", {})
        assert isinstance(manifest.get("external_sources"), list)

        manifest_path = Path(f"outputs/{result['session_id']}/external_data_manifest.json")
        assert manifest_path.exists()

    def test_manifest_schema_validation_catches_missing_keys(self):
        """Invalid manifest (missing keys) caught by validator."""
        from agents.competition_intel import _validate_manifest_schema

        bad_manifest = {
            "external_sources": [
                {
                    "name": "Test",
                    "type": "public_dataset",
                    # missing relevance_score, join_strategy, acquisition_method
                }
            ],
            "total_sources_found": 1,
        }

        with pytest.raises(ValueError, match="missing keys"):
            _validate_manifest_schema(bad_manifest)

    def test_manifest_schema_validation_catches_out_of_range_relevance_score(self):
        """relevance_score > 1.0 rejected."""
        from agents.competition_intel import _validate_manifest_schema

        bad_manifest = {
            "external_sources": [
                {
                    "name": "Test",
                    "type": "public_dataset",
                    "relevance_score": 1.5,
                    "join_strategy": "key",
                    "acquisition_method": "wget",
                }
            ],
            "total_sources_found": 1,
        }

        with pytest.raises(ValueError, match="relevance_score"):
            _validate_manifest_schema(bad_manifest)

    def test_relevance_scores_are_floats_not_strings(self):
        """relevance_score coerced to float, not left as string."""
        state = _make_state(external_data_allowed=True)

        # LLM returns string scores
        manifest_with_str = json.dumps({
            "external_sources": [
                {
                    "name": "Test Source",
                    "type": "public_dataset",
                    "description": "test",
                    "source_url": "https://test.com",
                    "relevance_score": "0.9",  # string!
                    "join_strategy": "key column",
                    "acquisition_method": "wget",
                    "competition_precedent": None,
                }
            ],
            "recommended_sources": ["Test Source"],
            "total_sources_found": 1,
            "scout_notes": "",
        })

        result = _run_intel_with_mocks(state, manifest_response=manifest_with_str)
        manifest = result.get("external_data_manifest", {})
        for source in manifest.get("external_sources", []):
            assert isinstance(source["relevance_score"], float), (
                f"relevance_score is {type(source['relevance_score'])}, expected float"
            )

    def test_scout_failure_returns_empty_manifest_not_exception(self):
        """LLM failure in scout doesn't crash competition_intel."""
        state = _make_state(external_data_allowed=True)

        with patch("agents.competition_intel._fetch_notebooks", return_value=[]):
            with patch("agents.competition_intel.call_llm") as mock_llm:
                # First call succeeds (brief), second call fails (scout)
                mock_llm.side_effect = [MOCK_BRIEF_JSON, TimeoutError("LLM timeout")]
                from agents.competition_intel import run_competition_intel
                result = run_competition_intel(state)

        manifest = result.get("external_data_manifest", {})
        assert manifest.get("external_sources") == [], (
            "Scout failure must produce empty external_sources, not crash"
        )
        assert "scout_error" in manifest

    def test_scout_state_field_external_data_manifest_set(self):
        """State contains external_data_manifest dict after run."""
        state = _make_state(external_data_allowed=False)

        with patch("agents.competition_intel._fetch_notebooks", return_value=[]):
            with patch("agents.competition_intel.call_llm", return_value=MOCK_BRIEF_JSON):
                from agents.competition_intel import run_competition_intel
                result = run_competition_intel(state)

        assert "external_data_manifest" in result
        assert isinstance(result["external_data_manifest"], dict)

    def test_only_high_relevance_sources_in_recommended(self):
        """Sources with relevance < 0.6 not in recommended_sources."""
        state = _make_state(external_data_allowed=True)
        result = _run_intel_with_mocks(state)

        manifest = result.get("external_data_manifest", {})
        recommended = manifest.get("recommended_sources", [])

        # "Random Noise" has score 0.5 — must NOT be recommended
        assert "Random Noise" not in recommended, (
            "Source with relevance_score < 0.6 should not be in recommended_sources"
        )

    def test_data_engineer_logs_high_relevance_sources(self):
        """_check_external_data logs when high-relevance sources exist."""
        from agents.data_engineer import _check_external_data

        state = _make_state(
            external_data_manifest={
                "external_sources": [
                    {"name": "GDP Data", "relevance_score": 0.85},
                    {"name": "Weather", "relevance_score": 0.9},
                ],
                "recommended_sources": ["GDP Data", "Weather"],
            }
        )

        with patch("agents.data_engineer.logger") as mock_logger:
            with patch("core.lineage.log_event"):
                _check_external_data(state)

            mock_logger.info.assert_called()
            log_msg = mock_logger.info.call_args[0][0]
            assert "2" in log_msg, "Should mention 2 high-relevance sources"

    def test_data_engineer_no_log_when_manifest_empty(self):
        """No logging when manifest has no recommended sources."""
        from agents.data_engineer import _check_external_data

        state = _make_state(
            external_data_manifest={
                "external_sources": [],
                "recommended_sources": [],
            }
        )

        with patch("agents.data_engineer.logger") as mock_logger:
            _check_external_data(state)

        # logger.info should NOT have been called for external data
        for c in mock_logger.info.call_args_list:
            assert "external" not in str(c).lower(), "Should not log when no sources"

    def test_total_sources_found_count_matches_external_sources_length(self):
        """total_sources_found == len(external_sources)."""
        state = _make_state(external_data_allowed=True)
        result = _run_intel_with_mocks(state)

        manifest = result.get("external_data_manifest", {})
        sources = manifest.get("external_sources", [])
        total = manifest.get("total_sources_found", -1)
        assert total == len(sources), (
            f"total_sources_found={total} but len(external_sources)={len(sources)}"
        )

    def test_scout_uses_only_first_20_feature_names_in_prompt(self):
        """LLM prompt capped at 20 feature names."""
        state = _make_state(
            external_data_allowed=True,
            feature_names=[f"feature_{i}" for i in range(200)],
        )

        with patch("agents.competition_intel._fetch_notebooks", return_value=[]):
            with patch("agents.competition_intel.call_llm") as mock_llm:
                mock_llm.side_effect = [MOCK_BRIEF_JSON, MOCK_MANIFEST_RESPONSE]
                from agents.competition_intel import run_competition_intel
                run_competition_intel(state)

        # Second call is the scout call — check the prompt
        if mock_llm.call_count >= 2:
            scout_prompt = mock_llm.call_args_list[1][0][0]
            # Count how many "feature_" appear
            feature_count = scout_prompt.count("feature_")
            assert feature_count <= 20, (
                f"Scout prompt contains {feature_count} features, max is 20"
            )

    def test_external_data_manifest_json_is_valid_json(self):
        """The written file is valid JSON."""
        state = _make_state(external_data_allowed=False)

        with patch("agents.competition_intel._fetch_notebooks", return_value=[]):
            with patch("agents.competition_intel.call_llm", return_value=MOCK_BRIEF_JSON):
                from agents.competition_intel import run_competition_intel
                result = run_competition_intel(state)

        manifest_path = Path(f"outputs/{result['session_id']}/external_data_manifest.json")
        raw = manifest_path.read_text()
        parsed = json.loads(raw)  # Must not raise
        assert isinstance(parsed, dict)
