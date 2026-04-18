# tests/test_lightning_integration.py
"""
Tests that verify Lightning integration does not break local execution.
These tests run WITHOUT Lightning credentials — they test the fallback path.
"""

import os
import pytest


class TestLightningFallback:

    def test_is_lightning_configured_false_without_credentials(self, monkeypatch):
        monkeypatch.delenv("LIGHTNING_API_KEY", raising=False)
        monkeypatch.delenv("LIGHTNING_USER_ID", raising=False)
        monkeypatch.delenv("LIGHTNING_USERNAME", raising=False)
        monkeypatch.delenv("LIGHTNING_STUDIO_NAME", raising=False)
        from tools.lightning_runner import is_lightning_configured
        assert is_lightning_configured() is False

    def test_run_on_lightning_returns_failure_without_credentials(self, monkeypatch):
        monkeypatch.delenv("LIGHTNING_API_KEY", raising=False)
        monkeypatch.delenv("LIGHTNING_USER_ID", raising=False)
        monkeypatch.delenv("LIGHTNING_USERNAME", raising=False)
        monkeypatch.delenv("LIGHTNING_STUDIO_NAME", raising=False)
        from tools.lightning_runner import run_on_lightning
        result = run_on_lightning(
            script="tools/lightning_jobs/run_optuna.py",
            args={"session_id": "test"},
            job_name="test_job",
        )
        assert result["success"] is False
        assert "error" in result
        assert result["result"] == {}

    def test_run_on_lightning_never_raises(self, monkeypatch):
        monkeypatch.setenv("LIGHTNING_API_KEY", "bad_key")
        monkeypatch.setenv("LIGHTNING_USER_ID", "bad_id")
        monkeypatch.setenv("LIGHTNING_USERNAME", "bad_user")
        monkeypatch.setenv("LIGHTNING_STUDIO_NAME", "bad_studio")
        from tools.lightning_runner import run_on_lightning
        # Must not raise even with bad credentials
        result = run_on_lightning(
            script="any_script.py",
            args={},
            job_name="test",
        )
        assert "success" in result

    def test_sync_files_returns_false_without_credentials(self, monkeypatch):
        monkeypatch.delenv("LIGHTNING_API_KEY", raising=False)
        monkeypatch.delenv("LIGHTNING_USER_ID", raising=False)
        monkeypatch.delenv("LIGHTNING_USERNAME", raising=False)
        monkeypatch.delenv("LIGHTNING_STUDIO_NAME", raising=False)
        from tools.lightning_runner import sync_files_to_lightning
        result = sync_files_to_lightning(
            session_id="test",
            files={"fake_path.csv": "train.csv"},
        )
        assert result is False

    def test_all_lightning_flags_default_to_0(self):
        """All LIGHTNING_OFFLOAD_* flags must default to disabled (0)."""
        flags = [
            "LIGHTNING_OFFLOAD_OPTUNA",
            "LIGHTNING_OFFLOAD_NULL_IMPORTANCE",
            "LIGHTNING_OFFLOAD_EDA",
            "LIGHTNING_OFFLOAD_FEATURE_TESTING",
            "LIGHTNING_OFFLOAD_STABILITY",
            "LIGHTNING_OFFLOAD_CRITIC",
            "LIGHTNING_OFFLOAD_PSEUDO_LABEL",
            "LIGHTNING_OFFLOAD_ENSEMBLE",
            "LIGHTNING_OFFLOAD_HARNESS",
        ]
        for flag in flags:
            val = os.getenv(flag, "0")
            assert val == "0", (
                f"{flag} default is '{val}', expected '0'. "
                "All Lightning flags must default to disabled."
            )

    def test_lightning_runner_module_imports(self):
        """Lightning runner module must import cleanly even without SDK."""
        from tools.lightning_runner import (
            is_lightning_configured,
            run_on_lightning,
            sync_files_to_lightning,
        )
        assert callable(is_lightning_configured)
        assert callable(run_on_lightning)
        assert callable(sync_files_to_lightning)

    def test_lightning_job_scripts_are_valid_python(self):
        """All lightning job scripts must parse as valid Python."""
        import ast
        job_dir = os.path.join(os.path.dirname(__file__), "..", "tools", "lightning_jobs")
        job_dir = os.path.normpath(job_dir)
        
        if not os.path.isdir(job_dir):
            pytest.skip("tools/lightning_jobs/ directory not found")

        for fname in os.listdir(job_dir):
            if fname.endswith(".py") and fname != "__init__.py":
                fpath = os.path.join(job_dir, fname)
                with open(fpath, "r") as f:
                    source = f.read()
                try:
                    ast.parse(source)
                except SyntaxError as e:
                    pytest.fail(f"{fname} has syntax error: {e}")

    def test_result_dict_shape(self, monkeypatch):
        """run_on_lightning must always return dict with expected keys."""
        monkeypatch.delenv("LIGHTNING_API_KEY", raising=False)
        from tools.lightning_runner import run_on_lightning
        result = run_on_lightning(
            script="any.py",
            args={"session_id": "test"},
            job_name="shape_test",
        )
        required_keys = {"success", "result", "job_link", "runtime_s", "error"}
        assert required_keys.issubset(set(result.keys())), (
            f"Missing keys: {required_keys - set(result.keys())}"
        )
