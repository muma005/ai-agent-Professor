"""
Tests for dependency checker.

FLAW-2.6: Dependency Version Pinning
"""
import pytest
from pathlib import Path
from tools.dependency_checker import (
    DependencyChecker,
    DependencyStatus,
    DependencyReport,
    get_dependency_checker,
    validate_dependencies,
    validate_critical_dependencies,
)


class TestDependencyStatus:
    """Test DependencyStatus dataclass."""

    def test_status_creation(self):
        """Test dependency status creation."""
        status = DependencyStatus(
            name="test_package",
            required_version="1.0.0",
            installed_version="1.0.0",
            is_installed=True,
            version_matches=True,
        )
        
        assert status.name == "test_package"
        assert status.is_installed is True
        assert status.version_matches is True

    def test_status_to_dict(self):
        """Test conversion to dict."""
        status = DependencyStatus(
            name="test_package",
            required_version="1.0.0",
            installed_version="1.0.0",
            is_installed=True,
            version_matches=True,
        )
        
        result = status.to_dict()
        
        assert result["name"] == "test_package"
        assert result["required_version"] == "1.0.0"
        assert result["installed_version"] == "1.0.0"
        assert result["is_installed"] is True
        assert result["version_matches"] is True


class TestDependencyChecker:
    """Test DependencyChecker class."""

    def test_checker_creation(self, tmp_path):
        """Test checker initialization."""
        # Create empty requirements file
        req_file = tmp_path / "requirements.txt"
        req_file.write_text("# Empty requirements\n")
        
        checker = DependencyChecker(str(req_file))
        
        assert len(checker.required_versions) == 0

    def test_parse_requirements(self, tmp_path):
        """Test requirements parsing."""
        req_file = tmp_path / "requirements.txt"
        req_file.write_text(
            "package1==1.0.0\n"
            "package2==2.0.0\n"
            "# Comment\n"
            "package3==3.0.0\n"
        )
        
        checker = DependencyChecker(str(req_file))
        
        assert len(checker.required_versions) == 3
        assert checker.required_versions["package1"] == "1.0.0"
        assert checker.required_versions["package2"] == "2.0.0"
        assert checker.required_versions["package3"] == "3.0.0"

    def test_check_installed_package(self, tmp_path):
        """Test checking an installed package."""
        req_file = tmp_path / "requirements.txt"
        req_file.write_text("pytest==9.0.2\n")
        
        checker = DependencyChecker(str(req_file))
        status = checker._check_package("pytest", "9.0.2")
        
        assert status.is_installed is True
        assert status.version_matches is True
        assert status.installed_version is not None

    def test_check_missing_package(self, tmp_path):
        """Test checking a missing package."""
        req_file = tmp_path / "requirements.txt"
        req_file.write_text("nonexistent_package_xyz123==1.0.0\n")
        
        checker = DependencyChecker(str(req_file))
        status = checker._check_package("nonexistent_package_xyz123", "1.0.0")
        
        assert status.is_installed is False
        assert status.version_matches is False
        assert status.installed_version is None

    def test_check_version_mismatch(self, tmp_path):
        """Test checking version mismatch."""
        req_file = tmp_path / "requirements.txt"
        # pytest is installed, but with different version
        req_file.write_text("pytest==1.0.0\n")
        
        checker = DependencyChecker(str(req_file))
        status = checker._check_package("pytest", "1.0.0")
        
        assert status.is_installed is True
        assert status.version_matches is False  # Version won't match

    def test_check_all(self, tmp_path):
        """Test checking all dependencies."""
        req_file = tmp_path / "requirements.txt"
        req_file.write_text(
            "pytest==9.0.2\n"
            "nonexistent_xyz==1.0.0\n"
        )
        
        checker = DependencyChecker(str(req_file))
        report = checker.check_all()
        
        assert report.total_dependencies == 2
        assert report.installed_count >= 1
        assert report.missing_count >= 1
        assert isinstance(report.dependencies, list)

    def test_validate_critical(self):
        """Test critical package validation."""
        checker = DependencyChecker()
        all_valid, missing = checker.validate_critical()
        
        # Should return tuple
        assert isinstance(all_valid, bool)
        assert isinstance(missing, list)
        
        # Most critical packages should be installed in test env
        # (test might fail if running in minimal env)

    def test_log_report(self, tmp_path, caplog):
        """Test report logging."""
        import logging
        
        req_file = tmp_path / "requirements.txt"
        req_file.write_text("pytest==9.0.2\n")
        
        checker = DependencyChecker(str(req_file))
        report = checker.check_all()
        
        with caplog.at_level(logging.INFO):
            checker.log_report(report)
        
        assert "DEPENDENCY VALIDATION REPORT" in caplog.text
        assert "Total dependencies:" in caplog.text


class TestDependencyReport:
    """Test DependencyReport dataclass."""

    def test_report_to_dict(self):
        """Test report conversion to dict."""
        report = DependencyReport(
            total_dependencies=10,
            installed_count=8,
            missing_count=2,
            mismatched_count=1,
            all_valid=False,
            dependencies=[],
            recommendations=["Install missing packages"],
        )
        
        result = report.to_dict()
        
        assert result["total_dependencies"] == 10
        assert result["installed_count"] == 8
        assert result["missing_count"] == 2
        assert result["all_valid"] is False
        assert "valid_percent" in result
        assert result["valid_percent"] == 80.0
        assert "recommendations" in result


class TestGlobalFunctions:
    """Test global convenience functions."""

    def test_get_dependency_checker_singleton(self):
        """Test get_dependency_checker returns same instance."""
        # Clear global state
        from tools import dependency_checker
        dependency_checker._checker = None
        
        checker1 = get_dependency_checker()
        checker2 = get_dependency_checker()
        
        assert checker1 is checker2

    def test_validate_dependencies(self):
        """Test validate_dependencies function."""
        # Clear global state
        from tools import dependency_checker
        dependency_checker._checker = None
        
        report = validate_dependencies()
        
        assert isinstance(report, DependencyReport)
        assert "total_dependencies" in report.to_dict()

    def test_validate_critical_dependencies(self):
        """Test validate_critical_dependencies function."""
        # Clear global state
        from tools import dependency_checker
        dependency_checker._checker = None
        
        all_valid, missing = validate_critical_dependencies()
        
        assert isinstance(all_valid, bool)
        assert isinstance(missing, list)
