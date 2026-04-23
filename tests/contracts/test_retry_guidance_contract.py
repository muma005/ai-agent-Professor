# tests/contracts/test_retry_guidance_contract.py

import pytest
from agents._retry_utils import build_retry_prompt

# ── Tests ───────────────────────────────────────────────────────────────────

class TestRetryGuidanceContract:
    """
    Contract: Error-Classified Retry Guidance (Component 3)
    """

    def test_keyerror_classified_as_data_quality(self):
        """Verify KeyError triggers Data Quality guidance."""
        err = KeyError("target")
        res = build_retry_prompt(err, "Traceback...", 1, "data_engineer")
        assert "Data quality issue" in res
        assert "Missing column" in res

    def test_memory_error_classified_as_memory(self):
        """Verify MemoryError triggers Memory guidance."""
        err = MemoryError("Out of memory")
        res = build_retry_prompt(err, "Traceback...", 1, "ml_optimizer")
        assert "Memory exhaustion" in res
        assert "n_jobs=1" in res

    def test_timeout_classified_as_api_timeout(self):
        """Verify TimeoutError triggers API Timeout guidance."""
        err = TimeoutError("API timeout")
        res = build_retry_prompt(err, "Traceback...", 1, "domain_researcher")
        assert "External API timeout" in res
        assert "NOT a code bug" in res

    def test_unknown_error_gets_generic(self):
        """Verify generic errors get generic guidance."""
        err = ZeroDivisionError("division by zero")
        res = build_retry_prompt(err, "Traceback...", 1, "any_agent")
        assert "Unclassified error" in res
        assert "Read the traceback" in res

    def test_traceback_truncated(self):
        """Verify a 500-line traceback is correctly truncated."""
        long_tb = "\n".join(["line %d" % i for i in range(500)])
        res = build_retry_prompt(Exception("Test"), long_tb, 1, "test_agent")
        lines = res.split("\n")
        # Header (5) + Guidance (~10) + Truncated TB (30+marker) + Footer (2)
        # We just check the truncation marker exists
        assert "truncated 470 lines" in res

    def test_short_traceback_preserved(self):
        """Verify a short traceback is kept in full."""
        short_tb = "Line 1\nLine 2\nLine 3"
        res = build_retry_prompt(Exception("Test"), short_tb, 1, "test_agent")
        assert "Line 1" in res
        assert "Line 3" in res
        assert "truncated" not in res

    def test_attempt_number_in_output(self):
        res = build_retry_prompt(Exception("Test"), "TB", 3, "test_agent")
        assert "ATTEMPT 3 FAILED" in res

    def test_original_error_text_present(self):
        err_msg = "Critical database failure"
        res = build_retry_prompt(RuntimeError(err_msg), "TB", 1, "test_agent")
        assert err_msg in res
