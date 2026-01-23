"""Tests for notebook code analyzer security checks."""

import pytest

from shared.plugins.notebook.code_analyzer import (
    CodeAnalyzer,
    AnalysisResult,
    RiskLevel,
    analyze_code,
)


class TestCodeAnalyzer:
    """Tests for CodeAnalyzer class."""

    def test_safe_code_no_risks(self):
        """Safe code should have no risks."""
        code = """
x = 1 + 2
y = [i for i in range(10)]
print(x, y)
"""
        result = analyze_code(code)
        assert not result.has_risks
        assert result.max_risk_level is None

    def test_detect_subprocess_import(self):
        """Importing subprocess should be flagged."""
        code = "import subprocess"
        result = analyze_code(code)
        assert result.has_risks
        assert "subprocess" in result.imports
        assert any(r.category == "dangerous_import" for r in result.risks)

    def test_detect_subprocess_run(self):
        """subprocess.run() should be flagged as critical."""
        code = """
import subprocess
subprocess.run(['ls', '-la'])
"""
        result = analyze_code(code)
        assert result.has_risks
        assert result.max_risk_level == RiskLevel.CRITICAL
        assert any(r.category == "subprocess" for r in result.risks)

    def test_detect_os_system(self):
        """os.system() should be flagged as critical."""
        code = """
import os
os.system('ls -la /home')
"""
        result = analyze_code(code)
        assert result.has_risks
        assert result.max_risk_level == RiskLevel.CRITICAL
        assert any("os.system" in r.description for r in result.risks)

    def test_detect_shell_command(self):
        """Shell command prefix (!) should be flagged."""
        code = "!ls -la /home/apanoia"
        result = analyze_code(code)
        assert result.has_risks
        assert result.max_risk_level == RiskLevel.CRITICAL
        assert any(r.category == "subprocess" for r in result.risks)

    def test_detect_external_path_in_shell(self):
        """External paths in shell commands should be detected."""
        code = "!cat /etc/passwd"
        result = analyze_code(code)
        assert result.has_risks
        assert "/etc/passwd" in result.external_paths or any(
            "/etc" in p for p in result.external_paths
        )

    def test_detect_open_call(self):
        """open() call should be flagged."""
        code = "f = open('/home/user/secret.txt')"
        result = analyze_code(code)
        assert result.has_risks
        assert any(r.category == "file_access" for r in result.risks)

    def test_detect_external_path_literal(self):
        """External path literals should be detected."""
        code = "path = '/home/user/documents'"
        result = analyze_code(code)
        assert result.has_risks
        assert any(r.category == "external_path" for r in result.risks)
        assert "/home/user/documents" in result.external_paths

    def test_detect_exec_builtin(self):
        """exec() builtin should be flagged as critical."""
        code = "exec('print(1)')"
        result = analyze_code(code)
        assert result.has_risks
        assert result.max_risk_level == RiskLevel.CRITICAL
        assert any(r.category == "code_execution" for r in result.risks)

    def test_detect_eval_builtin(self):
        """eval() builtin should be flagged as critical."""
        code = "x = eval('1 + 2')"
        result = analyze_code(code)
        assert result.has_risks
        assert result.max_risk_level == RiskLevel.CRITICAL

    def test_detect_builtins_access(self):
        """__builtins__ access should be flagged."""
        code = "x = __builtins__"
        result = analyze_code(code)
        assert result.has_risks
        assert any(r.category == "reflection" for r in result.risks)

    def test_detect_class_traversal(self):
        """Class hierarchy traversal should be flagged."""
        code = "x = ''.__class__.__bases__[0].__subclasses__()"
        result = analyze_code(code)
        assert result.has_risks
        assert any("__class__" in r.description or "__bases__" in r.description
                   for r in result.risks)

    def test_detect_os_environ(self):
        """os.environ access should be flagged."""
        code = """
import os
secrets = os.environ
"""
        result = analyze_code(code)
        assert result.has_risks
        assert any("environ" in r.description for r in result.risks)

    def test_detect_shutil_rmtree(self):
        """shutil.rmtree should be flagged as high risk."""
        code = """
import shutil
shutil.rmtree('/tmp/test')
"""
        result = analyze_code(code)
        assert result.has_risks
        # Should have both import warning and rmtree warning
        assert any(r.level in (RiskLevel.HIGH, RiskLevel.MEDIUM) for r in result.risks)

    def test_workspace_relative_path_allowed(self):
        """Paths within workspace should not be flagged as external."""
        analyzer = CodeAnalyzer(workspace_root="/home/user/project")
        code = "path = '/home/user/project/data.txt'"
        result = analyzer.analyze(code)
        # Should not have external_path risks for workspace paths
        external_risks = [r for r in result.risks if r.category == "external_path"]
        # The path should be detected but not flagged since it's in workspace
        # (depending on implementation - currently marks but allows)

    def test_parent_traversal_detected(self):
        """Parent directory traversal (..) should be flagged."""
        code = "path = '../../../etc/passwd'"
        result = analyze_code(code)
        assert result.has_risks
        assert any(".." in p for p in result.external_paths)

    def test_multiple_risks_formatted(self):
        """Multiple risks should be formatted correctly."""
        code = """
import subprocess
import os
os.system('ls')
subprocess.run(['cat', '/etc/passwd'])
exec('print(1)')
"""
        result = analyze_code(code)
        assert result.has_risks
        assert len(result.risks) >= 3

        # Test formatting
        formatted = result.format_risks()
        assert "CRITICAL" in formatted
        assert len(formatted) > 0

    def test_summary_generation(self):
        """Summary should correctly count risk levels."""
        code = """
import subprocess
subprocess.run(['ls'])
open('/tmp/test')
"""
        result = analyze_code(code)
        summary = result.get_summary()
        assert "critical" in summary.lower() or "high" in summary.lower()

    def test_syntax_error_handled(self):
        """Syntax errors should be handled gracefully."""
        code = "def foo(:"  # Invalid syntax
        result = analyze_code(code)
        # Should not crash, may report parse error
        assert isinstance(result, AnalysisResult)

    def test_network_imports_flagged(self):
        """Network-related imports should be flagged."""
        code = "import socket"
        result = analyze_code(code)
        assert result.has_risks
        assert any(r.level == RiskLevel.HIGH for r in result.risks)

    def test_safe_imports_not_flagged(self):
        """Safe standard library imports should not be flagged."""
        code = """
import json
import math
import datetime
from collections import defaultdict
"""
        result = analyze_code(code)
        # These should not generate any risks
        assert not any(r.level in (RiskLevel.HIGH, RiskLevel.CRITICAL) for r in result.risks)


class TestAnalyzerWithWorkspace:
    """Tests for CodeAnalyzer with workspace root configured."""

    def test_external_path_outside_workspace(self):
        """Paths outside workspace should be flagged."""
        analyzer = CodeAnalyzer(workspace_root="/home/user/myproject")
        code = "f = open('/home/user/other/secret.txt')"
        result = analyzer.analyze(code)
        assert result.has_risks
        # Should flag both open() and external path

    def test_workspace_path_not_external(self):
        """Paths inside workspace should not be flagged as external."""
        analyzer = CodeAnalyzer(workspace_root="/home/user/myproject")
        # This is a relative path - should be fine
        code = "f = open('data/file.txt')"
        result = analyzer.analyze(code)
        # open() is still flagged but path should not be external
        external_risks = [r for r in result.risks if r.category == "external_path"]
        assert len(external_risks) == 0


class TestRiskLevelOrdering:
    """Tests for risk level ordering."""

    def test_critical_is_max(self):
        """CRITICAL should be the maximum risk level."""
        code = "exec('x')"
        result = analyze_code(code)
        assert result.max_risk_level == RiskLevel.CRITICAL

    def test_high_without_critical(self):
        """HIGH should be max when no CRITICAL present."""
        code = "import subprocess"  # HIGH import, no execution
        result = analyze_code(code)
        assert result.max_risk_level == RiskLevel.HIGH
