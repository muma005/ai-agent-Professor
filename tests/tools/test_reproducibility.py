"""
Tests for reproducibility checks.

FLAW-10.2: Reproducibility Checks
"""
import pytest
import os
import json
from tools.reproducibility import (
    get_python_info,
    get_package_versions,
    get_git_info,
    get_environment_info,
    compute_file_hash,
    get_data_version,
    validate_reproducibility_prerequisites,
    generate_reproducibility_report,
    log_reproducibility_summary,
)


class TestPythonInfo:
    """Test Python info collection."""

    def test_get_python_info(self):
        """Test Python info is collected."""
        info = get_python_info()
        
        assert "version" in info
        assert "version_info" in info
        assert "executable" in info
        assert "platform" in info
        assert info["version_info"]["major"] >= 3

    def test_python_version_info(self):
        """Test version info structure."""
        info = get_python_info()
        version_info = info["version_info"]
        
        assert "major" in version_info
        assert "minor" in version_info
        assert "micro" in version_info
        assert isinstance(version_info["major"], int)


class TestPackageVersions:
    """Test package version collection."""

    def test_get_package_versions(self):
        """Test package versions are collected."""
        versions = get_package_versions()
        
        assert isinstance(versions, dict)
        assert len(versions) > 0

    def test_critical_packages_present(self):
        """Test critical packages are tracked."""
        versions = get_package_versions()
        
        # At least some packages should be installed
        installed = [k for k, v in versions.items() if v != "not_installed"]
        assert len(installed) > 0


class TestGitInfo:
    """Test git info collection."""

    def test_get_git_info(self):
        """Test git info structure."""
        info = get_git_info()
        
        assert "available" in info
        assert "commit" in info
        assert "branch" in info
        assert "dirty" in info
        assert "remote" in info

    def test_git_info_available(self):
        """Test git info is available in repo."""
        info = get_git_info()
        
        # Should be available in a git repo
        assert info["available"] is True
        assert info["commit"] is not None
        assert len(info["commit"]) == 40  # SHA-1 hash length


class TestEnvironmentInfo:
    """Test environment info collection."""

    def test_get_environment_info(self):
        """Test environment info is collected."""
        info = get_environment_info()
        
        assert "timestamp" in info
        assert "python" in info
        assert "packages" in info
        assert "git" in info
        assert "env_vars" in info

    def test_env_vars_tracked(self):
        """Test environment variables are tracked."""
        info = get_environment_info()
        env_vars = info["env_vars"]
        
        assert "PROFESSOR_SEED" in env_vars
        assert "LANGCHAIN_TRACING_V2" in env_vars


class TestFileHash:
    """Test file hash computation."""

    def test_compute_file_hash(self, tmp_path):
        """Test file hash is computed."""
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")
        
        hash1 = compute_file_hash(str(test_file))
        
        assert isinstance(hash1, str)
        assert len(hash1) == 64  # SHA-256 hex length

    def test_file_hash_deterministic(self, tmp_path):
        """Test file hash is deterministic."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")
        
        hash1 = compute_file_hash(str(test_file))
        hash2 = compute_file_hash(str(test_file))
        
        assert hash1 == hash2

    def test_different_files_different_hashes(self, tmp_path):
        """Test different files have different hashes."""
        file1 = tmp_path / "file1.txt"
        file1.write_text("Content 1")
        
        file2 = tmp_path / "file2.txt"
        file2.write_text("Content 2")
        
        hash1 = compute_file_hash(str(file1))
        hash2 = compute_file_hash(str(file2))
        
        assert hash1 != hash2

    def test_md5_hash(self, tmp_path):
        """Test MD5 hash computation."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test")
        
        hash_md5 = compute_file_hash(str(test_file), algorithm="md5")
        
        assert len(hash_md5) == 32  # MD5 hex length


class TestDataVersion:
    """Test data versioning."""

    def test_get_data_version(self, tmp_path):
        """Test data version info."""
        test_file = tmp_path / "data.csv"
        test_file.write_text("col1,col2\n1,2\n3,4")
        
        version = get_data_version(str(test_file))
        
        assert version["available"] is True
        assert "path" in version
        assert "size_bytes" in version
        assert "modified_at" in version
        assert "hash_sha256" in version
        assert "hash_md5" in version

    def test_get_data_version_not_found(self, tmp_path):
        """Test data version for non-existent file."""
        version = get_data_version(str(tmp_path / "nonexistent.csv"))
        
        assert version["available"] is False
        assert "path" in version


class TestReproducibilityValidation:
    """Test reproducibility validation."""

    def test_validate_prerequisites(self):
        """Test validation runs."""
        result = validate_reproducibility_prerequisites()
        
        assert "valid" in result
        assert "issues" in result
        assert "warnings" in result
        assert "timestamp" in result
        assert isinstance(result["issues"], list)
        assert isinstance(result["warnings"], list)

    def test_validation_result_structure(self):
        """Test validation result structure."""
        result = validate_reproducibility_prerequisites()
        
        assert isinstance(result["valid"], bool)


class TestReproducibilityReport:
    """Test reproducibility report generation."""

    def test_generate_report(self, tmp_path):
        """Test report generation."""
        state = {
            "session_id": "test_session",
            "competition_name": "test_competition",
        }
        output_dir = tmp_path / "reports"
        
        report_path = generate_reproducibility_report(state, str(output_dir))
        
        assert os.path.exists(report_path)
        assert report_path.endswith("reproducibility_report.json")

    def test_report_content(self, tmp_path):
        """Test report content."""
        state = {
            "session_id": "test_session",
            "competition_name": "test_competition",
        }
        output_dir = tmp_path / "reports"
        
        report_path = generate_reproducibility_report(state, str(output_dir))
        
        with open(report_path) as f:
            report = json.load(f)
        
        assert report["report_type"] == "reproducibility_report"
        assert report["session_id"] == "test_session"
        assert report["competition"] == "test_competition"
        assert "environment" in report
        assert "data_versions" in report
        assert "validation" in report
        assert "seed_info" in report

    def test_report_includes_data_versions(self, tmp_path):
        """Test report includes data file versions."""
        # Create test data file
        data_file = tmp_path / "train.csv"
        data_file.write_text("col1,col2\n1,2\n3,4")
        
        state = {
            "session_id": "test_session",
            "competition_name": "test_competition",
            "clean_data_path": str(data_file),
        }
        output_dir = tmp_path / "reports"
        
        report_path = generate_reproducibility_report(state, str(output_dir))
        
        with open(report_path) as f:
            report = json.load(f)
        
        data_versions = report["data_versions"]
        assert "clean_data_path" in data_versions
        assert data_versions["clean_data_path"]["available"] is True
