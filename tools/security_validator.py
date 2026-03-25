# tools/security_validator.py

"""
Security validation framework.

FLAW-5.7 FIX: Security Tests
- Sandbox escape detection
- Code injection prevention
- API key leakage detection
- Dangerous import blocking
- Eval/exec protection
"""

import os
import re
import ast
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SecurityIssue:
    """Represents a security issue."""
    
    issue_type: str
    severity: str  # "critical", "high", "medium", "low"
    description: str
    code_snippet: Optional[str] = None
    recommendation: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict."""
        return {
            "issue_type": self.issue_type,
            "severity": self.severity,
            "description": self.description,
            "code_snippet": self.code_snippet,
            "recommendation": self.recommendation,
        }


@dataclass
class SecurityReport:
    """Complete security validation report."""
    
    code_analyzed: str
    passed: bool
    critical_issues: int
    high_issues: int
    medium_issues: int
    low_issues: int
    issues: List[SecurityIssue]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict."""
        return {
            "code_analyzed": self.code_analyzed[:100] + "..." if len(self.code_analyzed) > 100 else self.code_analyzed,
            "passed": self.passed,
            "critical_issues": self.critical_issues,
            "high_issues": self.high_issues,
            "medium_issues": self.medium_issues,
            "low_issues": self.low_issues,
            "total_issues": len(self.issues),
            "issues": [i.to_dict() for i in self.issues],
        }


class SecurityValidator:
    """
    Security validation for code and data.
    
    Features:
    - Code injection detection
    - Sandbox escape prevention
    - API key leakage detection
    - Dangerous import blocking
    - Eval/exec protection
    
    Usage:
        validator = SecurityValidator()
        
        report = validator.validate_code(
            code=user_generated_code,
            allowed_imports=["polars", "numpy"],
        )
        
        if not report.passed:
            print(f"Security issues found: {report.issues}")
    """
    
    # Dangerous imports that should be blocked
    DANGEROUS_IMPORTS = {
        "os", "sys", "subprocess", "multiprocessing",
        "socket", "http", "urllib", "requests",
        "pickle", "marshal", "shelve",
        "importlib", "pkgutil", "compileall",
        "ctypes", "cffi",
        "eval", "exec", "compile",
        "getattr", "setattr", "delattr",
        "__import__",
    }
    
    # Dangerous patterns in code
    DANGEROUS_PATTERNS = {
        "eval": r"\beval\s*\(",
        "exec": r"\bexec\s*\(",
        "compile": r"\bcompile\s*\(",
        "__import__": r"__import__\s*\(",
        "getattr": r"\bgetattr\s*\(",
        "setattr": r"\bsetattr\s*\(",
        "open_file": r"\bopen\s*\([^)]*[\"'][^\"']*[\"']",
        "shell_command": r"os\.system\s*\(|subprocess\.",
        "network": r"socket\.|requests\.|urllib\.",
        "pickle": r"pickle\.(load|loads|dump|dumps)",
    }
    
    # API key patterns to detect leakage
    API_KEY_PATTERNS = {
        "aws_key": r"AKIA[0-9A-Z]{16}",
        "gcp_key": r"AIza[0-9A-Za-z\-_]{35}",
        "azure_key": r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        "generic_secret": r"(?i)(secret|api_key|apikey|password|token)\s*[=:]\s*[\"'][^\"']{20,}[\"']",
        "private_key": r"-----BEGIN (RSA |EC )?PRIVATE KEY-----",
        "github_token": r"gh[pousr]_[A-Za-z0-9_]{36}",
    }
    
    def __init__(
        self,
        block_dangerous_imports: bool = True,
        block_eval_exec: bool = True,
        detect_api_keys: bool = True,
    ):
        """
        Initialize security validator.
        
        Args:
            block_dangerous_imports: Block dangerous imports
            block_eval_exec: Block eval/exec usage
            detect_api_keys: Detect API key leakage
        """
        self.block_dangerous_imports = block_dangerous_imports
        self.block_eval_exec = block_eval_exec
        self.detect_api_keys = detect_api_keys
        
        logger.info("[SecurityValidator] Initialized")
    
    def validate_code(
        self,
        code: str,
        allowed_imports: Optional[List[str]] = None,
    ) -> SecurityReport:
        """
        Validate code for security issues.
        
        Args:
            code: Code to validate
            allowed_imports: List of allowed import names
        
        Returns:
            SecurityReport with all findings
        """
        issues = []
        
        # Run all security checks
        issues.extend(self._check_dangerous_imports(code, allowed_imports))
        issues.extend(self._check_dangerous_patterns(code))
        issues.extend(self._check_api_key_leakage(code))
        issues.extend(self._check_code_injection(code))
        
        # Count by severity
        critical = sum(1 for i in issues if i.severity == "critical")
        high = sum(1 for i in issues if i.severity == "high")
        medium = sum(1 for i in issues if i.severity == "medium")
        low = sum(1 for i in issues if i.severity == "low")
        
        # Determine if passed (no critical or high issues)
        passed = critical == 0 and high == 0
        
        return SecurityReport(
            code_analyzed=code,
            passed=passed,
            critical_issues=critical,
            high_issues=high,
            medium_issues=medium,
            low_issues=low,
            issues=issues,
        )
    
    def validate_sandbox_escape(self, code: str) -> SecurityReport:
        """
        Validate code for sandbox escape attempts.
        
        Args:
            code: Code to validate
        
        Returns:
            SecurityReport
        """
        issues = []
        
        # Check for common sandbox escape patterns
        escape_patterns = [
            (r"\.__class__", "Accessing __class__"),
            (r"\.__mro__", "Accessing __mro__"),
            (r"\.__subclasses__", "Accessing __subclasses__"),
            (r"\.__globals__", "Accessing __globals__"),
            (r"\.__builtins__", "Accessing __builtins__"),
            (r"\.__import__", "Using __import__"),
            (r"globals\s*\(", "Accessing globals()"),
            (r"locals\s*\(", "Accessing locals()"),
            (r"vars\s*\(", "Accessing vars()"),
            (r"dir\s*\(", "Accessing dir()"),
        ]
        
        for pattern, description in escape_patterns:
            if re.search(pattern, code):
                issues.append(SecurityIssue(
                    issue_type="sandbox_escape",
                    severity="critical",
                    description=f"Potential sandbox escape: {description}",
                    code_snippet=self._extract_snippet(code, pattern),
                    recommendation="Remove code that attempts to access Python internals",
                ))
        
        passed = len(issues) == 0
        
        return SecurityReport(
            code_analyzed=code,
            passed=passed,
            critical_issues=len(issues),
            high_issues=0,
            medium_issues=0,
            low_issues=0,
            issues=issues,
        )
    
    def _check_dangerous_imports(
        self,
        code: str,
        allowed_imports: Optional[List[str]],
    ) -> List[SecurityIssue]:
        """Check for dangerous imports."""
        issues = []
        
        if not self.block_dangerous_imports:
            return issues
        
        # Parse imports from code
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return issues
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in self.DANGEROUS_IMPORTS:
                        issues.append(SecurityIssue(
                            issue_type="dangerous_import",
                            severity="critical",
                            description=f"Dangerous import blocked: {alias.name}",
                            code_snippet=f"import {alias.name}",
                            recommendation=f"Remove 'import {alias.name}' - not allowed in sandbox",
                        ))
            
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module in self.DANGEROUS_IMPORTS:
                    issues.append(SecurityIssue(
                        issue_type="dangerous_import",
                        severity="critical",
                        description=f"Dangerous import blocked: {node.module}",
                        code_snippet=f"from {node.module} import ...",
                        recommendation=f"Remove 'from {node.module} import' - not allowed",
                    ))
        
        return issues
    
    def _check_dangerous_patterns(self, code: str) -> List[SecurityIssue]:
        """Check for dangerous code patterns."""
        issues = []
        
        if not self.block_eval_exec:
            return issues
        
        for pattern_name, pattern in self.DANGEROUS_PATTERNS.items():
            if re.search(pattern, code):
                severity = "critical" if pattern_name in ["eval", "exec", "compile"] else "high"
                
                issues.append(SecurityIssue(
                    issue_type="dangerous_pattern",
                    severity=severity,
                    description=f"Dangerous pattern detected: {pattern_name}",
                    code_snippet=self._extract_snippet(code, pattern),
                    recommendation=f"Remove usage of {pattern_name} - security risk",
                ))
        
        return issues
    
    def _check_api_key_leakage(self, code: str) -> List[SecurityIssue]:
        """Check for API key leakage in code."""
        issues = []
        
        if not self.detect_api_keys:
            return issues
        
        for key_type, pattern in self.API_KEY_PATTERNS.items():
            matches = re.findall(pattern, code)
            
            if matches:
                issues.append(SecurityIssue(
                    issue_type="api_key_leakage",
                    severity="critical",
                    description=f"Potential {key_type} detected in code",
                    code_snippet="[REDACTED]",
                    recommendation="Remove API keys from code - use environment variables",
                ))
        
        return issues
    
    def _check_code_injection(self, code: str) -> List[SecurityIssue]:
        """Check for code injection vulnerabilities."""
        issues = []
        
        # Check for string formatting that could be injection
        injection_patterns = [
            (r"f\s*[\"'][^\"']*\{[^}]*\([^)]*\)[^\"']*[\"']", "f-string with dict access"),
            (r"\"?\s*%\s*\(.*\)", "String formatting with dict"),
            (r"\.format\s*\(\s*\*\*", "format with **kwargs"),
        ]
        
        for pattern, description in injection_patterns:
            if re.search(pattern, code):
                issues.append(SecurityIssue(
                    issue_type="potential_injection",
                    severity="medium",
                    description=f"Potential code injection: {description}",
                    code_snippet=self._extract_snippet(code, pattern),
                    recommendation="Review string formatting - ensure no user input reaches code",
                ))
        
        return issues
    
    def _extract_snippet(self, code: str, pattern: str, max_length: int = 100) -> Optional[str]:
        """Extract code snippet matching pattern."""
        match = re.search(pattern, code)
        
        if match:
            start = max(0, match.start() - 20)
            end = min(len(code), match.end() + 20)
            snippet = code[start:end].strip()
            
            if len(snippet) > max_length:
                snippet = snippet[:max_length] + "..."
            
            return snippet
        
        return None


def validate_security(
    code: str,
    allowed_imports: Optional[List[str]] = None,
    **kwargs,
) -> SecurityReport:
    """
    Convenience function for security validation.
    
    Args:
        code: Code to validate
        allowed_imports: Allowed import names
        **kwargs: Passed to SecurityValidator
    
    Returns:
        SecurityReport
    """
    validator = SecurityValidator(**kwargs)
    return validator.validate_code(code, allowed_imports)


def validate_sandbox_escape(code: str) -> SecurityReport:
    """
    Convenience function for sandbox escape validation.
    
    Args:
        code: Code to validate
    
    Returns:
        SecurityReport
    """
    validator = SecurityValidator()
    return validator.validate_sandbox_escape(code)
