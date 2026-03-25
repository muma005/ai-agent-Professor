# tools/reproducibility.py

"""
Reproducibility checks and environment documentation.

FLAW-10.2 FIX: Reproducibility Checks
- Captures environment metadata (Python, packages, git commit)
- Documents data versions (hashes, timestamps)
- Validates reproducibility prerequisites
- Generates reproducibility report
"""

import os
import sys
import json
import hashlib
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


def get_python_info() -> dict:
    """Get Python environment information."""
    return {
        "version": sys.version,
        "version_info": {
            "major": sys.version_info.major,
            "minor": sys.version_info.minor,
            "micro": sys.version_info.micro,
        },
        "executable": sys.executable,
        "platform": sys.platform,
    }


def get_package_versions() -> dict:
    """Get versions of key packages."""
    packages = {}
    
    package_names = [
        "numpy", "pandas", "polars", "scikit-learn",
        "lightgbm", "xgboost", "catboost",
        "optuna", "langgraph", "langchain",
        "pytest", "torch",
    ]
    
    for pkg_name in package_names:
        try:
            pkg = __import__(pkg_name)
            version = getattr(pkg, "__version__", "unknown")
            packages[pkg_name] = version
        except ImportError:
            packages[pkg_name] = "not_installed"
    
    return packages


def get_git_info() -> dict:
    """Get git repository information."""
    info = {
        "available": False,
        "commit": None,
        "branch": None,
        "dirty": None,
        "remote": None,
    }
    
    try:
        # Get current commit
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()
        info["commit"] = commit
        info["available"] = True
        
        # Get branch
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()
        info["branch"] = branch
        
        # Check if working tree is dirty
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()
        info["dirty"] = bool(status)
        
        # Get remote URL
        try:
            remote = subprocess.check_output(
                ["git", "config", "--get", "remote.origin.url"],
                stderr=subprocess.DEVNULL,
                text=True
            ).strip()
            info["remote"] = remote
        except subprocess.CalledProcessError:
            pass
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.debug("[Reproducibility] Git info not available")
    
    return info


def get_environment_info() -> dict:
    """Get full environment information."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python": get_python_info(),
        "packages": get_package_versions(),
        "git": get_git_info(),
        "env_vars": {
            "PROFESSOR_SEED": os.environ.get("PROFESSOR_SEED", "42"),
            "PROFESSOR_MAX_MEMORY_GB": os.environ.get("PROFESSOR_MAX_MEMORY_GB", "6.0"),
            "LANGCHAIN_TRACING_V2": os.environ.get("LANGCHAIN_TRACING_V2", "false"),
        },
    }


def compute_file_hash(file_path: str, algorithm: str = "sha256") -> str:
    """
    Compute hash of a file.
    
    Args:
        file_path: Path to file
        algorithm: Hash algorithm (default: sha256)
    
    Returns:
        Hex digest of file hash
    """
    hash_func = hashlib.new(algorithm)
    
    with open(file_path, "rb") as f:
        # Read in chunks for large files
        for chunk in iter(lambda: f.read(8192), b""):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()


def get_data_version(data_path: str) -> dict:
    """
    Get data versioning information.
    
    Args:
        data_path: Path to data file
    
    Returns:
        Data version info dict
    """
    path = Path(data_path)
    
    if not path.exists():
        return {
            "available": False,
            "path": str(data_path),
        }
    
    stat = path.stat()
    
    return {
        "available": True,
        "path": str(path.absolute()),
        "size_bytes": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "hash_sha256": compute_file_hash(str(path)),
        "hash_md5": compute_file_hash(str(path), "md5"),
    }


def get_competition_data_versions(state: dict) -> dict:
    """
    Get versions of all competition data files from state.
    
    Args:
        state: Professor state dict
    
    Returns:
        Dict of data file versions
    """
    data_keys = [
        "raw_data_path",
        "test_data_path",
        "sample_submission_path",
        "clean_data_path",
        "feature_data_path",
        "schema_path",
        "preprocessor_path",
    ]
    
    versions = {}
    
    for key in data_keys:
        path = state.get(key)
        if path:
            versions[key] = get_data_version(path)
    
    return versions


def validate_reproducibility_prerequisites() -> dict:
    """
    Validate that reproducibility prerequisites are met.
    
    Returns:
        Validation result dict
    """
    issues = []
    warnings = []
    
    # Check git availability
    git_info = get_git_info()
    if not git_info["available"]:
        warnings.append("Git not available - code versioning disabled")
    elif git_info["dirty"]:
        warnings.append("Working tree is dirty - changes not committed")
    
    # Check critical packages
    packages = get_package_versions()
    critical_packages = ["numpy", "polars", "scikit-learn", "lightgbm"]
    
    for pkg in critical_packages:
        if packages.get(pkg) == "not_installed":
            issues.append(f"Critical package not installed: {pkg}")
    
    # Check seed configuration
    seed = os.environ.get("PROFESSOR_SEED")
    if seed is None:
        warnings.append("PROFESSOR_SEED not set - using default (42)")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def generate_reproducibility_report(
    state: dict,
    output_dir: str,
) -> str:
    """
    Generate comprehensive reproducibility report.
    
    Args:
        state: Professor state dict
        output_dir: Directory to save report
    
    Returns:
        Path to saved report
    """
    os.makedirs(output_dir, exist_ok=True)
    
    report = {
        "report_type": "reproducibility_report",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "session_id": state.get("session_id", "unknown"),
        "competition": state.get("competition_name", "unknown"),
        # Environment
        "environment": get_environment_info(),
        # Data versions
        "data_versions": get_competition_data_versions(state),
        # Validation
        "validation": validate_reproducibility_prerequisites(),
        # Seed info
        "seed_info": {
            "base_seed": int(os.environ.get("PROFESSOR_SEED", "42")),
            "configurable_via": "PROFESSOR_SEED env var",
        },
    }
    
    # Save report
    report_path = os.path.join(output_dir, "reproducibility_report.json")
    
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"[Reproducibility] Report saved to: {report_path}")
    
    return report_path


def log_reproducibility_summary(state: dict) -> None:
    """
    Log a concise reproducibility summary.
    
    Args:
        state: Professor state dict
    """
    env_info = get_environment_info()
    validation = validate_reproducibility_prerequisites()
    
    logger.info("=" * 70)
    logger.info("REPRODUCIBILITY SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Python: {env_info['python']['version_info']['major']}.{env_info['python']['version_info']['minor']}")
    logger.info(f"Git commit: {env_info['git'].get('commit', 'N/A')[:8] if env_info['git']['commit'] else 'N/A'}")
    logger.info(f"Git branch: {env_info['git'].get('branch', 'N/A')}")
    logger.info(f"Working tree dirty: {env_info['git'].get('dirty', 'N/A')}")
    logger.info(f"Seed: {os.environ.get('PROFESSOR_SEED', '42')}")
    
    if validation["issues"]:
        logger.error(f"Issues: {validation['issues']}")
    
    if validation["warnings"]:
        logger.warning(f"Warnings: {validation['warnings']}")
    
    logger.info("=" * 70)
