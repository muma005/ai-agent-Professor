# tests/contracts/test_e2b_sandbox_contract.py
# ─────────────────────────────────────────────────────────────────
# Written: Day 2
# Status:  IMMUTABLE — never edit this file after today
#
# CONTRACT: execute_code()
#   INPUT:  code (str), session_id (str)
#   OUTPUT: dict with keys: success (bool), stdout (str), stderr (str)
#   ERRORS: raises SandboxExecutionError after max failed attempts
#           never hangs — always returns or raises within timeout
# ─────────────────────────────────────────────────────────────────
import pytest
import time
from tools.e2b_sandbox import execute_code, SandboxExecutionError

SESSION = "test_session_sandbox"


class TestSandboxContract:

    def test_successful_execution_returns_success_true(self):
        result = execute_code("result = 2 + 2", session_id=SESSION)
        assert result["success"] is True

    def test_output_has_required_keys(self):
        result = execute_code("x = 1", session_id=SESSION)
        assert "success" in result
        assert "stdout" in result
        assert "stderr" in result

    def test_stdout_captured(self):
        result = execute_code("print('hello_sandbox')", session_id=SESSION)
        assert result["success"] is True
        assert "hello_sandbox" in result["stdout"]

    def test_polars_available_in_sandbox(self):
        code = """
import polars as pl
df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
result = df.shape
print(f"shape: {df.shape}")
"""
        result = execute_code(code, session_id=SESSION)
        assert result["success"] is True
        assert "shape" in result["stdout"]

    def test_numpy_available_in_sandbox(self):
        code = "import numpy as np\nresult = np.mean([1, 2, 3])\nprint(result)"
        result = execute_code(code, session_id=SESSION)
        assert result["success"] is True

    def test_syntax_error_raises_sandbox_error(self):
        bad_code = "def broken( :"  # intentional syntax error
        with pytest.raises(SandboxExecutionError):
            execute_code(bad_code, session_id=SESSION, max_attempts=1)

    def test_runtime_error_raises_sandbox_error_after_max_attempts(self):
        bad_code = "result = 1 / 0"  # ZeroDivisionError
        with pytest.raises(SandboxExecutionError) as exc_info:
            execute_code(bad_code, session_id=SESSION, max_attempts=3)
        assert "3 attempts" in str(exc_info.value)

    def test_retry_loop_uses_fix_callback(self):
        """LLM fix callback is called on failure and fixed code succeeds."""
        call_count = {"n": 0}

        def mock_fix(code, error, traceback_str):
            call_count["n"] += 1
            return "result = 42  # fixed"  # always returns working code

        bad_code = "result = 1 / 0"
        result = execute_code(
            bad_code,
            session_id=SESSION,
            llm_fix_callback=mock_fix,
            max_attempts=3
        )
        assert result["success"] is True
        assert call_count["n"] == 1  # called once on first failure

    def test_never_allows_dangerous_imports(self):
        """Sandbox must block filesystem and system access."""
        dangerous_code = "import subprocess\nsubprocess.run(['ls'])"
        with pytest.raises((SandboxExecutionError, Exception)):
            execute_code(dangerous_code, session_id=SESSION, max_attempts=1)

    def test_attempts_used_recorded_in_result(self):
        result = execute_code("x = 1", session_id=SESSION)
        assert "attempts_used" in result
        assert result["attempts_used"] == 1

    def test_output_dir_created_for_session(self):
        import os
        execute_code("x = 1", session_id="output_test_session")
        assert os.path.exists("outputs/output_test_session")
