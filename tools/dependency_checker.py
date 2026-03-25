# tools/dependency_checker.py

"""
Dependency version checker for reproducible environments.

FLAW-2.6 FIX: Dependency Version Pinning
- Validates installed versions match requirements.txt
- Checks for known incompatible versions
- Reports missing dependencies
- Ensures reproducible environments
"""

import os
import re
import logging
import importlib.metadata
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DependencyStatus:
    """Status of a single dependency."""
    
    name: str
    required_version: str
    installed_version: Optional[str]
    is_installed: bool
    version_matches: bool
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to serializable dict."""
        return {
            "name": self.name,
            "required_version": self.required_version,
            "installed_version": self.installed_version,
            "is_installed": self.is_installed,
            "version_matches": self.version_matches,
            "error": self.error,
        }


@dataclass
class DependencyReport:
    """Full dependency validation report."""
    
    total_dependencies: int
    installed_count: int
    missing_count: int
    mismatched_count: int
    all_valid: bool
    dependencies: list
    recommendations: list
    
    @property
    def valid_percent(self) -> float:
        """Calculate validation percentage."""
        if self.total_dependencies == 0:
            return 100.0
        return round(self.installed_count / self.total_dependencies * 100, 1)
    
    def to_dict(self) -> dict:
        """Convert to serializable dict."""
        return {
            "total_dependencies": self.total_dependencies,
            "installed_count": self.installed_count,
            "missing_count": self.missing_count,
            "mismatched_count": self.mismatched_count,
            "all_valid": self.all_valid,
            "valid_percent": self.valid_percent,
            "dependencies": [d.to_dict() for d in self.dependencies],
            "recommendations": self.recommendations,
        }


class DependencyChecker:
    """
    Checks dependency versions against requirements.txt.
    
    Features:
    - Parse requirements.txt
    - Check installed versions
    - Validate version compatibility
    - Generate reports
    """
    
    # Known incompatible versions
    INCOMPATIBLE_VERSIONS = {
        # Example: "numpy": ["1.24.0"],  # Has known bugs
    }
    
    # Critical dependencies that must be present
    CRITICAL_PACKAGES = {
        "polars", "numpy", "scikit-learn", "lightgbm",
        "langgraph", "langchain-core", "optuna",
    }
    
    def __init__(self, requirements_path: str = "requirements.txt"):
        """
        Initialize dependency checker.
        
        Args:
            requirements_path: Path to requirements.txt
        """
        self.requirements_path = Path(requirements_path)
        self.required_versions: Dict[str, str] = {}
        
        if self.requirements_path.exists():
            self._parse_requirements()
            logger.info(
                f"[DependencyChecker] Loaded {len(self.required_versions)} "
                f"dependencies from {requirements_path}"
            )
        else:
            logger.warning(
                f"[DependencyChecker] Requirements file not found: {requirements_path}"
            )
    
    def _parse_requirements(self) -> None:
        """Parse requirements.txt file."""
        if not self.requirements_path.exists():
            return
        
        with open(self.requirements_path, "r") as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            
            # Skip comments and empty lines
            if not line or line.startswith("#"):
                continue
            
            # Parse package==version format
            match = re.match(r'^([a-zA-Z0-9_-]+)==([^\s#]+)', line)
            if match:
                package_name = match.group(1).lower()
                version = match.group(2)
                self.required_versions[package_name] = version
    
    def check_all(self) -> DependencyReport:
        """
        Check all dependencies.
        
        Returns:
            Full dependency report
        """
        dependencies = []
        missing = []
        mismatched = []
        recommendations = []
        
        for package_name, required_version in self.required_versions.items():
            status = self._check_package(package_name, required_version)
            dependencies.append(status)
            
            if not status.is_installed:
                missing.append(package_name)
            elif not status.version_matches:
                mismatched.append(package_name)
        
        # Generate recommendations
        if missing:
            recommendations.append(
                f"Install missing packages: pip install -r {self.requirements_path}"
            )
        
        if mismatched:
            recommendations.append(
                f"Reinstall mismatched packages: pip install --force-reinstall -r {self.requirements_path}"
            )
        
        # Check for critical packages
        missing_critical = [
            pkg for pkg in self.CRITICAL_PACKAGES
            if pkg.lower() in [m.lower() for m in missing]
        ]
        
        if missing_critical:
            recommendations.insert(0, f"CRITICAL: Missing critical packages: {missing_critical}")
        
        return DependencyReport(
            total_dependencies=len(self.required_versions),
            installed_count=len(self.required_versions) - len(missing),
            missing_count=len(missing),
            mismatched_count=len(mismatched),
            all_valid=len(missing) == 0 and len(mismatched) == 0,
            dependencies=dependencies,
            recommendations=recommendations,
        )
    
    def _check_package(
        self,
        name: str,
        required_version: str,
    ) -> DependencyStatus:
        """
        Check a single package.
        
        Args:
            name: Package name
            required_version: Required version
        
        Returns:
            Dependency status
        """
        try:
            installed_version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return DependencyStatus(
                name=name,
                required_version=required_version,
                installed_version=None,
                is_installed=False,
                version_matches=False,
                error=f"Package '{name}' not found",
            )
        
        # Check version match
        version_matches = installed_version == required_version
        
        # Check for incompatible versions
        is_incompatible = self._check_incompatible(name, installed_version)
        
        if is_incompatible and not version_matches:
            version_matches = False
        
        return DependencyStatus(
            name=name,
            required_version=required_version,
            installed_version=installed_version,
            is_installed=True,
            version_matches=version_matches,
        )
    
    def _check_incompatible(self, name: str, version: str) -> bool:
        """
        Check if version is known to be incompatible.
        
        Args:
            name: Package name
            version: Installed version
        
        Returns:
            True if incompatible
        """
        incompatible = self.INCOMPATIBLE_VERSIONS.get(name, [])
        return version in incompatible
    
    def validate_critical(self) -> Tuple[bool, List[str]]:
        """
        Validate critical packages are installed.
        
        Returns:
            (all_valid, missing_critical)
        """
        missing = []
        
        for package in self.CRITICAL_PACKAGES:
            try:
                importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                missing.append(package)
        
        return len(missing) == 0, missing
    
    def log_report(self, report: DependencyReport) -> None:
        """Log dependency report."""
        logger.info("=" * 70)
        logger.info("DEPENDENCY VALIDATION REPORT")
        logger.info("=" * 70)
        logger.info(f"Total dependencies: {report.total_dependencies}")
        logger.info(f"Installed: {report.installed_count}")
        logger.info(f"Missing: {report.missing_count}")
        logger.info(f"Mismatched: {report.mismatched_count}")
        logger.info(f"Valid: {report.valid_percent}%")
        logger.info(f"All valid: {report.all_valid}")
        
        if report.recommendations:
            logger.warning("Recommendations:")
            for rec in report.recommendations:
                logger.warning(f"  - {rec}")
        
        if report.mismatched_count > 0:
            logger.warning("Mismatched packages:")
            for dep in report.dependencies:
                if not dep.version_matches and dep.is_installed:
                    logger.warning(
                        f"  - {dep.name}: installed={dep.installed_version}, "
                        f"required={dep.required_version}"
                    )
        
        logger.info("=" * 70)


# Global checker instance
_checker: Optional[DependencyChecker] = None


def get_dependency_checker() -> DependencyChecker:
    """Get or create global dependency checker."""
    global _checker
    
    if _checker is None:
        _checker = DependencyChecker()
    
    return _checker


def validate_dependencies() -> DependencyReport:
    """
    Validate all dependencies.
    
    Returns:
        Dependency report
    """
    checker = get_dependency_checker()
    return checker.check_all()


def validate_critical_dependencies() -> Tuple[bool, List[str]]:
    """
    Validate critical dependencies.
    
    Returns:
        (all_valid, missing_critical)
    """
    checker = get_dependency_checker()
    return checker.validate_critical()


def log_dependency_report() -> None:
    """Log dependency validation report."""
    checker = get_dependency_checker()
    report = checker.check_all()
    checker.log_report(report)
