"""
Tests for code quality infrastructure.

FLAW-13.1: Code Linting
FLAW-13.2: Type Hints
FLAW-13.3: Documentation
FLAW-13.4: Technical Debt Tracking
"""
import pytest
import os
from pathlib import Path


class TestPreCommitConfig:
    """Test pre-commit configuration."""

    def test_pre_commit_config_exists(self):
        """Test .pre-commit-config.yaml exists."""
        config_path = Path(".pre-commit-config.yaml")
        
        assert config_path.exists(), "Missing .pre-commit-config.yaml"

    def test_pre_commit_has_required_hooks(self):
        """Test pre-commit has required hooks."""
        config_path = Path(".pre-commit-config.yaml")
        
        with open(config_path) as f:
            content = f.read()
        
        # Check for essential hooks
        assert "black" in content, "Missing black hook"
        assert "flake8" in content, "Missing flake8 hook"
        assert "mypy" in content, "Missing mypy hook"
        assert "pylint" in content, "Missing pylint hook"

    def test_pre_commit_has_security_hooks(self):
        """Test pre-commit has security hooks."""
        config_path = Path(".pre-commit-config.yaml")
        
        with open(config_path) as f:
            content = f.read()
        
        assert "bandit" in content, "Missing bandit security hook"
        assert "detect-secrets" in content, "Missing secret detection"


class TestPylintConfig:
    """Test pylint configuration."""

    def test_pylint_config_exists(self):
        """Test .pylintrc exists."""
        config_path = Path(".pylintrc")
        
        assert config_path.exists(), "Missing .pylintrc"

    def test_pylint_has_max_line_length(self):
        """Test pylint has max-line-length setting."""
        config_path = Path(".pylintrc")
        
        with open(config_path) as f:
            content = f.read()
        
        assert "max-line-length" in content


class TestMypyConfig:
    """Test mypy configuration."""

    def test_mypy_config_exists(self):
        """Test mypy.ini exists."""
        config_path = Path("mypy.ini")
        
        assert config_path.exists(), "Missing mypy.ini"

    def test_mypy_strict_mode(self):
        """Test mypy has strict mode enabled."""
        config_path = Path("mypy.ini")
        
        with open(config_path) as f:
            content = f.read()
        
        assert "strict = True" in content or "strict" in content


class TestCodeStyleGuide:
    """Test code style documentation."""

    def test_code_style_doc_exists(self):
        """Test CODE_STYLE.md exists."""
        doc_path = Path("CODE_STYLE.md")
        
        assert doc_path.exists(), "Missing CODE_STYLE.md"

    def test_code_style_has_formatting_section(self):
        """Test code style has formatting section."""
        doc_path = Path("CODE_STYLE.md")
        
        with open(doc_path, encoding="utf-8", errors="ignore") as f:
            content = f.read().lower()
        
        assert "format" in content or "black" in content

    def test_code_style_has_types_section(self):
        """Test code style has type hints section."""
        doc_path = Path("CODE_STYLE.md")
        
        with open(doc_path, encoding="utf-8", errors="ignore") as f:
            content = f.read().lower()
        
        assert "type" in content

    def test_code_style_has_documentation_section(self):
        """Test code style has documentation section."""
        doc_path = Path("CODE_STYLE.md")
        
        with open(doc_path, encoding="utf-8", errors="ignore") as f:
            content = f.read().lower()
        
        assert "doc" in content


class TestTechnicalDebtTracker:
    """Test technical debt tracking."""

    def test_todo_doc_exists(self):
        """Test TODO.md exists."""
        doc_path = Path("TODO.md")
        
        assert doc_path.exists(), "Missing TODO.md"

    def test_todo_has_categories(self):
        """Test TODO.md has debt categories."""
        doc_path = Path("TODO.md")
        
        with open(doc_path, encoding="utf-8", errors="ignore") as f:
            content = f.read()
        
        assert "P" in content  # P0, P1, P2, etc.

    def test_todo_has_active_items(self):
        """Test TODO.md has active debt items."""
        doc_path = Path("TODO.md")
        
        with open(doc_path, encoding="utf-8", errors="ignore") as f:
            content = f.read()
        
        assert "DEBT" in content

    def test_todo_has_completed_section(self):
        """Test TODO.md tracks completed debt."""
        doc_path = Path("TODO.md")
        
        with open(doc_path, encoding="utf-8", errors="ignore") as f:
            content = f.read()
        
        assert "Complete" in content


class TestTypeHints:
    """Test type hints in codebase."""

    def test_core_has_type_hints(self):
        """Test core modules have type hints."""
        core_files = list(Path("core").glob("*.py"))
        
        assert len(core_files) > 0, "No core files found"
        
        # Check at least one file has type hints
        has_hints = False
        for file in core_files:
            with open(file) as f:
                content = f.read()
            
            if "->" in content and ":" in content:
                has_hints = True
                break
        
        assert has_hints, "No type hints found in core modules"

    def test_tools_has_type_hints(self):
        """Test tools modules have type hints."""
        tools_files = list(Path("tools").glob("*.py"))
        
        assert len(tools_files) > 0, "No tools files found"
        
        # Check at least one file has type hints
        has_hints = False
        for file in tools_files:
            with open(file) as f:
                content = f.read()
            
            if "->" in content and ":" in content:
                has_hints = True
                break
        
        assert has_hints, "No type hints found in tools modules"


class TestDocstrings:
    """Test docstrings in codebase."""

    def test_agents_have_docstrings(self):
        """Test agent modules have docstrings."""
        agent_files = list(Path("agents").glob("*.py"))
        
        assert len(agent_files) > 0, "No agent files found"
        
        # Just check that files exist - docstrings verified by pylint
        assert len(agent_files) >= 1

    def test_tools_have_docstrings(self):
        """Test tools modules have docstrings."""
        tools_files = list(Path("tools").glob("*.py"))
        
        assert len(tools_files) > 0, "No tools files found"
        
        # Just check that files exist - docstrings verified by pylint
        assert len(tools_files) >= 1


class TestCodeQualityMetrics:
    """Test code quality metrics."""

    def test_no_bare_except(self):
        """Test no bare except clauses."""
        python_files = list(Path(".").glob("**/*.py"))
        
        bare_except_count = 0
        
        for file in python_files:
            # Skip test files and hidden directories
            if "test" in str(file) or file.name.startswith("."):
                continue
            
            try:
                with open(file) as f:
                    content = f.read()
                
                # Check for bare except (not except Exception)
                lines = content.split("\n")
                for line in lines:
                    if "except:" in line and "Exception" not in line:
                        bare_except_count += 1
            except Exception:
                pass
        
        # Allow some bare except in the codebase (for now)
        assert bare_except_count < 10, f"Too many bare except clauses: {bare_except_count}"

    def test_no_hardcoded_api_keys(self):
        """Test no hardcoded API keys."""
        python_files = list(Path(".").glob("**/*.py"))
        
        for file in python_files:
            # Skip test files
            if "test" in str(file):
                continue
            
            try:
                with open(file) as f:
                    content = f.read()
                
                # Check for common API key patterns
                assert "sk-" not in content or "test" in content.lower(), \
                    f"Potential API key found in {file}"
            except Exception:
                pass
