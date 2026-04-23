# tests/contracts/test_data_usage_checker_contract.py

import pytest
import os
import polars as pl
from guards.data_usage_checker import check_data_usage

# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_data_dir(tmp_path):
    """Creates a directory with multiple data and non-data files."""
    (tmp_path / "train.csv").write_text("id,val\n1,10")
    (tmp_path / "metadata.csv").write_text("id,info\n1,meta")
    (tmp_path / "extra_data.parquet").write_text("stub")
    (tmp_path / "README.md").write_text("doc")
    (tmp_path / "sample_submission.csv").write_text("id,val\n1,0.5")
    return tmp_path

# ── Tests ───────────────────────────────────────────────────────────────────

class TestDataUsageCheckerContract:
    """
    Contract: Data Usage Checker (Component 2)
    """

    def test_all_files_used(self, mock_data_dir):
        """Verify code referencing all files results in all_data_used=True."""
        code = 'df = pl.read_csv("train.csv"); meta = pl.read_csv("metadata.csv"); extra = pl.read_parquet("extra_data.parquet")'
        res = check_data_usage(str(mock_data_dir), code)
        assert res["all_data_used"] is True
        assert res["total_data_files"] == 3

    def test_unused_file_flagged(self, mock_data_dir):
        """Verify missing references are flagged in unused_files."""
        code = 'df = pl.read_csv("train.csv")'
        res = check_data_usage(str(mock_data_dir), code)
        assert res["all_data_used"] is False
        assert "metadata.csv" in res["unused_files"]
        assert "extra_data.parquet" in res["unused_files"]

    def test_sample_submission_ignored(self, mock_data_dir):
        """Verify sample_submission.csv is never counted as a data file."""
        code = ""
        res = check_data_usage(str(mock_data_dir), code)
        assert "sample_submission.csv" not in res["used_files"]
        assert "sample_submission.csv" not in res["unused_files"]

    def test_non_data_extensions_ignored(self, mock_data_dir):
        """Verify .md files are not counted as data files."""
        code = ""
        res = check_data_usage(str(mock_data_dir), code)
        assert "README.md" not in res["used_files"]
        assert "README.md" not in res["unused_files"]

    def test_stem_matching_works(self, mock_data_dir):
        """Verify 'train' matches 'train.csv'."""
        code = 'df = pl.read_csv("train")'
        res = check_data_usage(str(mock_data_dir), code)
        assert "train.csv" in res["used_files"]

    def test_empty_directory(self, tmp_path):
        """Verify graceful handling of an empty data directory."""
        res = check_data_usage(str(tmp_path), "code")
        assert res["total_data_files"] == 0
        assert res["all_data_used"] is True

    def test_report_has_required_keys(self, mock_data_dir):
        res = check_data_usage(str(mock_data_dir), "code")
        keys = ["total_data_files", "used_files", "unused_files", "all_data_used"]
        for k in keys:
            assert k in res
