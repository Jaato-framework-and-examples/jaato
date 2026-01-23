"""AST-based code analyzer for notebook sandbox security.

This module analyzes Python code before execution to detect potentially
dangerous patterns that could bypass workspace sandboxing:

- File operations (open, pathlib, io)
- Subprocess/shell execution (subprocess, os.system, os.popen)
- Dangerous builtins (exec, eval, compile, __import__)
- Network operations (socket, urllib, requests)
- Path references outside allowed directories

Usage:
    analyzer = CodeAnalyzer(workspace_root="/path/to/workspace")
    result = analyzer.analyze(code)
    if result.has_risks:
        print(result.format_risks())
"""

import ast
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional, Set, Tuple

try:
    from ..sandbox_utils import check_path_with_jaato_containment
except ImportError:
    # Fallback for direct module execution (testing)
    from shared.plugins.sandbox_utils import check_path_with_jaato_containment


class RiskLevel(Enum):
    """Risk level for detected patterns."""
    LOW = "low"          # Informational, probably safe
    MEDIUM = "medium"    # Potentially dangerous, needs review
    HIGH = "high"        # Very likely to bypass sandbox
    CRITICAL = "critical"  # Definitely bypasses sandbox


@dataclass
class DetectedRisk:
    """A detected security risk in code."""
    category: str           # e.g., "file_access", "subprocess", "dangerous_builtin"
    description: str        # Human-readable description
    level: RiskLevel        # Severity level
    line_number: int        # Line in source code
    code_snippet: str       # The relevant code fragment
    details: Optional[str] = None  # Additional context


@dataclass
class AnalysisResult:
    """Result of code analysis."""
    risks: List[DetectedRisk] = field(default_factory=list)
    imports: Set[str] = field(default_factory=set)
    external_paths: List[str] = field(default_factory=list)

    @property
    def has_risks(self) -> bool:
        """Check if any risks were detected."""
        return len(self.risks) > 0

    @property
    def max_risk_level(self) -> Optional[RiskLevel]:
        """Get the highest risk level found."""
        if not self.risks:
            return None
        # Order: CRITICAL > HIGH > MEDIUM > LOW
        levels = [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
        for level in reversed(levels):
            if any(r.level == level for r in self.risks):
                return level
        return None

    def format_risks(self, max_items: int = 5) -> str:
        """Format risks for display in permission prompt."""
        if not self.risks:
            return "No risks detected"

        lines = []
        # Group by level
        by_level = {}
        for risk in self.risks:
            by_level.setdefault(risk.level, []).append(risk)

        # Show highest severity first
        for level in [RiskLevel.CRITICAL, RiskLevel.HIGH, RiskLevel.MEDIUM, RiskLevel.LOW]:
            if level not in by_level:
                continue
            level_risks = by_level[level]
            level_name = level.value.upper()
            for risk in level_risks[:max_items]:
                lines.append(f"[{level_name}] {risk.description} (line {risk.line_number})")
                if risk.details:
                    lines.append(f"         {risk.details}")

        if len(self.risks) > max_items:
            lines.append(f"... and {len(self.risks) - max_items} more issues")

        return "\n".join(lines)

    def get_summary(self) -> str:
        """Get a one-line summary of risks."""
        if not self.risks:
            return "Code appears safe"

        counts = {}
        for risk in self.risks:
            counts[risk.level] = counts.get(risk.level, 0) + 1

        parts = []
        for level in [RiskLevel.CRITICAL, RiskLevel.HIGH, RiskLevel.MEDIUM, RiskLevel.LOW]:
            if level in counts:
                parts.append(f"{counts[level]} {level.value}")

        return f"Detected: {', '.join(parts)}"


# Dangerous module imports
DANGEROUS_MODULES = {
    # Subprocess/shell execution
    "subprocess": RiskLevel.HIGH,
    "os": RiskLevel.MEDIUM,  # os has many safe uses, but also dangerous ones
    "pty": RiskLevel.HIGH,
    "popen2": RiskLevel.HIGH,
    "commands": RiskLevel.HIGH,

    # Network
    "socket": RiskLevel.HIGH,
    "urllib": RiskLevel.MEDIUM,
    "urllib.request": RiskLevel.MEDIUM,
    "http.client": RiskLevel.MEDIUM,
    "ftplib": RiskLevel.MEDIUM,
    "smtplib": RiskLevel.MEDIUM,
    "telnetlib": RiskLevel.HIGH,

    # Code execution
    "code": RiskLevel.HIGH,
    "codeop": RiskLevel.HIGH,
    "imp": RiskLevel.MEDIUM,
    "importlib": RiskLevel.MEDIUM,

    # System
    "ctypes": RiskLevel.HIGH,
    "multiprocessing": RiskLevel.MEDIUM,

    # File operations
    "shutil": RiskLevel.MEDIUM,
    "tempfile": RiskLevel.LOW,
    "pathlib": RiskLevel.LOW,  # Can be dangerous if used to access external paths
}

# Dangerous function calls
DANGEROUS_CALLS = {
    # File operations
    "open": (RiskLevel.MEDIUM, "file_access", "Opens files - check path"),
    "file": (RiskLevel.MEDIUM, "file_access", "Opens files - check path"),

    # Code execution
    "exec": (RiskLevel.CRITICAL, "code_execution", "Arbitrary code execution"),
    "eval": (RiskLevel.CRITICAL, "code_execution", "Arbitrary code evaluation"),
    "compile": (RiskLevel.HIGH, "code_execution", "Code compilation"),
    "__import__": (RiskLevel.HIGH, "code_execution", "Dynamic module import"),

    # OS operations
    "input": (RiskLevel.LOW, "user_input", "User input - could hang"),
}

# Dangerous attribute access patterns
DANGEROUS_ATTRS = {
    # os module
    ("os", "system"): (RiskLevel.CRITICAL, "subprocess", "Shell command execution"),
    ("os", "popen"): (RiskLevel.CRITICAL, "subprocess", "Shell command execution"),
    ("os", "spawn"): (RiskLevel.HIGH, "subprocess", "Process spawning"),
    ("os", "spawnl"): (RiskLevel.HIGH, "subprocess", "Process spawning"),
    ("os", "spawnle"): (RiskLevel.HIGH, "subprocess", "Process spawning"),
    ("os", "spawnlp"): (RiskLevel.HIGH, "subprocess", "Process spawning"),
    ("os", "spawnlpe"): (RiskLevel.HIGH, "subprocess", "Process spawning"),
    ("os", "spawnv"): (RiskLevel.HIGH, "subprocess", "Process spawning"),
    ("os", "spawnve"): (RiskLevel.HIGH, "subprocess", "Process spawning"),
    ("os", "spawnvp"): (RiskLevel.HIGH, "subprocess", "Process spawning"),
    ("os", "spawnvpe"): (RiskLevel.HIGH, "subprocess", "Process spawning"),
    ("os", "execl"): (RiskLevel.CRITICAL, "subprocess", "Process replacement"),
    ("os", "execle"): (RiskLevel.CRITICAL, "subprocess", "Process replacement"),
    ("os", "execlp"): (RiskLevel.CRITICAL, "subprocess", "Process replacement"),
    ("os", "execlpe"): (RiskLevel.CRITICAL, "subprocess", "Process replacement"),
    ("os", "execv"): (RiskLevel.CRITICAL, "subprocess", "Process replacement"),
    ("os", "execve"): (RiskLevel.CRITICAL, "subprocess", "Process replacement"),
    ("os", "execvp"): (RiskLevel.CRITICAL, "subprocess", "Process replacement"),
    ("os", "execvpe"): (RiskLevel.CRITICAL, "subprocess", "Process replacement"),
    ("os", "fork"): (RiskLevel.HIGH, "subprocess", "Process forking"),
    ("os", "forkpty"): (RiskLevel.HIGH, "subprocess", "Process forking with pty"),
    ("os", "kill"): (RiskLevel.HIGH, "process", "Process killing"),
    ("os", "killpg"): (RiskLevel.HIGH, "process", "Process group killing"),
    ("os", "environ"): (RiskLevel.MEDIUM, "env_access", "Environment variable access"),
    ("os", "getenv"): (RiskLevel.LOW, "env_access", "Environment variable read"),
    ("os", "putenv"): (RiskLevel.MEDIUM, "env_access", "Environment variable modification"),
    ("os", "chdir"): (RiskLevel.MEDIUM, "file_access", "Directory change"),
    ("os", "chroot"): (RiskLevel.CRITICAL, "file_access", "Chroot escape"),
    ("os", "chmod"): (RiskLevel.MEDIUM, "file_access", "Permission modification"),
    ("os", "chown"): (RiskLevel.MEDIUM, "file_access", "Ownership modification"),
    ("os", "link"): (RiskLevel.MEDIUM, "file_access", "Hard link creation"),
    ("os", "symlink"): (RiskLevel.MEDIUM, "file_access", "Symlink creation"),
    ("os", "remove"): (RiskLevel.MEDIUM, "file_access", "File deletion"),
    ("os", "unlink"): (RiskLevel.MEDIUM, "file_access", "File deletion"),
    ("os", "rmdir"): (RiskLevel.MEDIUM, "file_access", "Directory deletion"),
    ("os", "rename"): (RiskLevel.MEDIUM, "file_access", "File renaming"),
    ("os", "replace"): (RiskLevel.MEDIUM, "file_access", "File replacement"),
    ("os", "makedirs"): (RiskLevel.MEDIUM, "file_access", "Directory creation"),
    ("os", "mkdir"): (RiskLevel.MEDIUM, "file_access", "Directory creation"),

    # subprocess module
    ("subprocess", "run"): (RiskLevel.CRITICAL, "subprocess", "Subprocess execution"),
    ("subprocess", "call"): (RiskLevel.CRITICAL, "subprocess", "Subprocess execution"),
    ("subprocess", "check_call"): (RiskLevel.CRITICAL, "subprocess", "Subprocess execution"),
    ("subprocess", "check_output"): (RiskLevel.CRITICAL, "subprocess", "Subprocess execution"),
    ("subprocess", "Popen"): (RiskLevel.CRITICAL, "subprocess", "Subprocess execution"),
    ("subprocess", "getoutput"): (RiskLevel.CRITICAL, "subprocess", "Shell execution"),
    ("subprocess", "getstatusoutput"): (RiskLevel.CRITICAL, "subprocess", "Shell execution"),

    # shutil module
    ("shutil", "rmtree"): (RiskLevel.HIGH, "file_access", "Recursive directory deletion"),
    ("shutil", "move"): (RiskLevel.MEDIUM, "file_access", "File/directory moving"),
    ("shutil", "copy"): (RiskLevel.LOW, "file_access", "File copying"),
    ("shutil", "copy2"): (RiskLevel.LOW, "file_access", "File copying with metadata"),
    ("shutil", "copytree"): (RiskLevel.MEDIUM, "file_access", "Directory tree copying"),

    # Reflection/introspection for bypass
    ("builtins", "exec"): (RiskLevel.CRITICAL, "code_execution", "Builtin exec access"),
    ("builtins", "eval"): (RiskLevel.CRITICAL, "code_execution", "Builtin eval access"),
    ("builtins", "open"): (RiskLevel.MEDIUM, "file_access", "Builtin open access"),
    ("builtins", "__import__"): (RiskLevel.HIGH, "code_execution", "Builtin import access"),
}

# Patterns that indicate external path access
EXTERNAL_PATH_PATTERNS = [
    r'^/',                    # Absolute Unix paths
    r'^[A-Za-z]:',            # Windows drive letters
    r'\.\.',                  # Parent directory traversal
    r'^~',                    # Home directory
    r'/home/',                # Unix home directories
    r'/etc/',                 # System config
    r'/usr/',                 # System programs
    r'/var/',                 # Variable data
    r'/root/',                # Root home
    r'/proc/',                # Process info
    r'/sys/',                 # System info
    r'/dev/',                 # Devices
]


class CodeAnalyzer:
    """Analyzes Python code for security risks."""

    def __init__(
        self,
        workspace_root: Optional[str] = None,
        allowed_paths: Optional[List[str]] = None,
        strict_mode: bool = False,
        plugin_registry: Any = None,
        allow_tmp: bool = True,
    ):
        """Initialize the analyzer.

        Args:
            workspace_root: The allowed workspace directory (for path validation).
            allowed_paths: Additional paths that are allowed.
            strict_mode: If True, flag more patterns as risky.
            plugin_registry: Optional PluginRegistry for external path authorization.
            allow_tmp: Whether to allow /tmp/** access (default: True).
        """
        self.workspace_root = workspace_root
        self.allowed_paths = allowed_paths or []
        self.strict_mode = strict_mode
        self.plugin_registry = plugin_registry
        self.allow_tmp = allow_tmp

        # Track imports seen during analysis
        self._imported_names: Set[str] = set()
        self._import_aliases: dict = {}  # alias -> module name

    def analyze(self, code: str) -> AnalysisResult:
        """Analyze code and return detected risks.

        Args:
            code: Python source code to analyze.

        Returns:
            AnalysisResult with detected risks and metadata.
        """
        result = AnalysisResult()
        self._imported_names = set()
        self._import_aliases = {}

        # Check for shell command prefix (!)
        if code.strip().startswith('!'):
            cmd = code.strip()[1:]
            result.risks.append(DetectedRisk(
                category="subprocess",
                description="Shell command execution",
                level=RiskLevel.CRITICAL,
                line_number=1,
                code_snippet=code.strip()[:80],
                details=f"Command: {cmd[:60]}..." if len(cmd) > 60 else f"Command: {cmd}",
            ))
            # Also check for paths in shell command
            self._check_shell_command_paths(cmd, result)
            return result

        # Parse AST
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            # Can't analyze invalid syntax
            result.risks.append(DetectedRisk(
                category="parse_error",
                description=f"Syntax error: {e.msg}",
                level=RiskLevel.LOW,
                line_number=e.lineno or 1,
                code_snippet=code[:80] if code else "",
            ))
            return result

        # First pass: collect imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self._imported_names.add(alias.name.split('.')[0])
                    if alias.asname:
                        self._import_aliases[alias.asname] = alias.name
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    self._imported_names.add(node.module.split('.')[0])
                    result.imports.add(node.module)
                    for alias in node.names:
                        if alias.asname:
                            self._import_aliases[alias.asname] = f"{node.module}.{alias.name}"

        # Analyze the tree
        self._analyze_node(tree, code, result)

        return result

    def _analyze_node(self, node: ast.AST, code: str, result: AnalysisResult) -> None:
        """Recursively analyze AST nodes."""
        code_lines = code.split('\n')

        for child in ast.walk(node):
            line_no = getattr(child, 'lineno', 1)
            snippet = code_lines[line_no - 1] if line_no <= len(code_lines) else ""

            # Check imports
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                self._check_import(child, snippet, result)

            # Check function calls
            elif isinstance(child, ast.Call):
                self._check_call(child, snippet, result)

            # Check name references (for __builtins__, etc.)
            elif isinstance(child, ast.Name):
                self._check_name(child, snippet, result)

            # Check attribute access
            elif isinstance(child, ast.Attribute):
                self._check_attribute(child, snippet, result)

            # Check string literals for paths
            elif isinstance(child, ast.Constant) and isinstance(child.value, str):
                self._check_string_literal(child, child.value, result)

            # For Python < 3.8 compatibility
            elif isinstance(child, ast.Str):
                self._check_string_literal(child, child.s, result)

    def _check_import(
        self, node: ast.AST, snippet: str, result: AnalysisResult
    ) -> None:
        """Check import statements for dangerous modules."""
        line_no = getattr(node, 'lineno', 1)

        if isinstance(node, ast.Import):
            for alias in node.names:
                module = alias.name
                if module in DANGEROUS_MODULES:
                    level = DANGEROUS_MODULES[module]
                    result.risks.append(DetectedRisk(
                        category="dangerous_import",
                        description=f"Import of '{module}' module",
                        level=level,
                        line_number=line_no,
                        code_snippet=snippet.strip(),
                    ))
                result.imports.add(module)

        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            base_module = module.split('.')[0]
            if base_module in DANGEROUS_MODULES:
                level = DANGEROUS_MODULES[base_module]
                result.risks.append(DetectedRisk(
                    category="dangerous_import",
                    description=f"Import from '{module}' module",
                    level=level,
                    line_number=line_no,
                    code_snippet=snippet.strip(),
                ))
            result.imports.add(module)

    def _check_call(
        self, node: ast.Call, snippet: str, result: AnalysisResult
    ) -> None:
        """Check function calls for dangerous patterns."""
        line_no = getattr(node, 'lineno', 1)
        func = node.func

        # Direct function call: open(), exec(), etc.
        if isinstance(func, ast.Name):
            func_name = func.id
            if func_name in DANGEROUS_CALLS:
                level, category, desc = DANGEROUS_CALLS[func_name]
                risk = DetectedRisk(
                    category=category,
                    description=f"Call to '{func_name}()': {desc}",
                    level=level,
                    line_number=line_no,
                    code_snippet=snippet.strip(),
                )

                # For open(), try to extract the path argument
                if func_name == "open" and node.args:
                    path = self._extract_string_value(node.args[0])
                    if path:
                        is_external = self._is_external_path(path)
                        if is_external:
                            risk.level = RiskLevel.HIGH
                            risk.details = f"Path: {path}"
                            result.external_paths.append(path)

                result.risks.append(risk)

        # Method call: os.system(), subprocess.run(), etc.
        elif isinstance(func, ast.Attribute):
            self._check_method_call(node, func, snippet, result)

    def _check_method_call(
        self,
        node: ast.Call,
        func: ast.Attribute,
        snippet: str,
        result: AnalysisResult,
    ) -> None:
        """Check method calls for dangerous patterns."""
        line_no = getattr(node, 'lineno', 1)
        method_name = func.attr

        # Get the object being called on
        obj = func.value
        obj_name = None

        if isinstance(obj, ast.Name):
            obj_name = obj.id
            # Check for alias
            if obj_name in self._import_aliases:
                obj_name = self._import_aliases[obj_name].split('.')[0]
        elif isinstance(obj, ast.Attribute):
            # Nested attribute like os.path.join
            obj_name = self._get_attribute_chain(obj)

        if obj_name:
            key = (obj_name, method_name)
            if key in DANGEROUS_ATTRS:
                level, category, desc = DANGEROUS_ATTRS[key]
                risk = DetectedRisk(
                    category=category,
                    description=f"Call to '{obj_name}.{method_name}()': {desc}",
                    level=level,
                    line_number=line_no,
                    code_snippet=snippet.strip(),
                )

                # Extract path argument if applicable
                if node.args and category == "file_access":
                    path = self._extract_string_value(node.args[0])
                    if path:
                        if self._is_external_path(path):
                            risk.level = RiskLevel.HIGH
                            risk.details = f"Path: {path}"
                            result.external_paths.append(path)

                result.risks.append(risk)

    def _check_name(
        self, node: ast.Name, snippet: str, result: AnalysisResult
    ) -> None:
        """Check name references for dangerous builtins."""
        line_no = getattr(node, 'lineno', 1)
        name = node.id

        # Check for direct __builtins__ reference
        if name == "__builtins__":
            result.risks.append(DetectedRisk(
                category="reflection",
                description="Direct access to __builtins__",
                level=RiskLevel.HIGH,
                line_number=line_no,
                code_snippet=snippet.strip(),
                details="Can be used to access restricted builtins",
            ))

        # Check for dunder names that indicate introspection attacks
        if name.startswith("__") and name.endswith("__") and name not in ("__name__", "__doc__", "__file__"):
            # Skip common safe dunders
            if name in ("__class__", "__bases__", "__subclasses__", "__mro__",
                        "__globals__", "__code__", "__func__", "__builtins__"):
                result.risks.append(DetectedRisk(
                    category="reflection",
                    description=f"Access to {name}",
                    level=RiskLevel.HIGH,
                    line_number=line_no,
                    code_snippet=snippet.strip(),
                    details="Can be used for introspection attacks",
                ))

    def _check_attribute(
        self, node: ast.Attribute, snippet: str, result: AnalysisResult
    ) -> None:
        """Check attribute access (even without call)."""
        line_no = getattr(node, 'lineno', 1)
        attr_name = node.attr

        # Check for __builtins__ access
        if attr_name == "__builtins__":
            result.risks.append(DetectedRisk(
                category="reflection",
                description="Access to __builtins__",
                level=RiskLevel.HIGH,
                line_number=line_no,
                code_snippet=snippet.strip(),
                details="Can be used to access restricted builtins",
            ))

        # Check for __class__ / __bases__ / __subclasses__ (sandbox escape patterns)
        if attr_name in ("__class__", "__bases__", "__subclasses__", "__mro__"):
            result.risks.append(DetectedRisk(
                category="reflection",
                description=f"Access to {attr_name}",
                level=RiskLevel.HIGH,
                line_number=line_no,
                code_snippet=snippet.strip(),
                details="Can be used for class hierarchy traversal attacks",
            ))

        # Check for __globals__ / __code__ access
        if attr_name in ("__globals__", "__code__", "__func__"):
            result.risks.append(DetectedRisk(
                category="reflection",
                description=f"Access to {attr_name}",
                level=RiskLevel.HIGH,
                line_number=line_no,
                code_snippet=snippet.strip(),
                details="Can access function internals and globals",
            ))

    def _check_string_literal(
        self, node: ast.AST, value: str, result: AnalysisResult
    ) -> None:
        """Check string literals for external path references."""
        if self._is_external_path(value):
            line_no = getattr(node, 'lineno', 1)
            result.risks.append(DetectedRisk(
                category="external_path",
                description="External path reference",
                level=RiskLevel.MEDIUM,
                line_number=line_no,
                code_snippet=value[:60] + "..." if len(value) > 60 else value,
                details=f"Path may be outside workspace: {value}",
            ))
            result.external_paths.append(value)

    def _check_shell_command_paths(self, cmd: str, result: AnalysisResult) -> None:
        """Check shell command for external path references."""
        # Simple heuristic: look for path-like patterns
        for pattern in EXTERNAL_PATH_PATTERNS:
            matches = re.findall(pattern + r'[^\s]*', cmd)
            for match in matches:
                result.external_paths.append(match)
                result.risks.append(DetectedRisk(
                    category="external_path",
                    description="External path in shell command",
                    level=RiskLevel.HIGH,
                    line_number=1,
                    code_snippet=cmd[:60],
                    details=f"Path: {match}",
                ))

    def _is_external_path(self, path: str) -> bool:
        """Check if a path is external using the shared sandbox utility.

        Uses check_path_with_jaato_containment() which handles:
        - Workspace boundary checking
        - .jaato symlink escape (contained within .jaato boundary)
        - /tmp access allowance (configurable)
        - Plugin registry authorization for external paths
        """
        if not path:
            return False

        # If no workspace root configured, we can't determine if path is external
        # Use pattern-based fallback for obvious external paths
        if not self.workspace_root:
            for pattern in EXTERNAL_PATH_PATTERNS:
                if re.match(pattern, path):
                    return True
            return False

        # Use the shared sandbox utility for consistent path validation
        # This respects .jaato symlinks, /tmp access, and plugin registry auth
        is_allowed = check_path_with_jaato_containment(
            path=path,
            workspace_root=self.workspace_root,
            plugin_registry=self.plugin_registry,
            allow_tmp=self.allow_tmp,
        )

        # If path is NOT allowed by sandbox, it's external
        return not is_allowed

    def _extract_string_value(self, node: ast.AST) -> Optional[str]:
        """Try to extract a string value from an AST node."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Str):
            return node.s
        if isinstance(node, ast.JoinedStr):
            # f-string - can't fully evaluate but try to get static parts
            parts = []
            for value in node.values:
                if isinstance(value, ast.Constant):
                    parts.append(str(value.value))
                elif isinstance(value, ast.Str):
                    parts.append(value.s)
            return "".join(parts) if parts else None
        return None

    def _get_attribute_chain(self, node: ast.Attribute) -> str:
        """Get the full attribute chain (e.g., 'os.path')."""
        parts = [node.attr]
        current = node.value
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        return ".".join(reversed(parts))


def analyze_code(
    code: str,
    workspace_root: Optional[str] = None,
    plugin_registry: Any = None,
    allow_tmp: bool = True,
) -> AnalysisResult:
    """Convenience function to analyze code.

    Args:
        code: Python source code to analyze.
        workspace_root: Optional workspace root for path validation.
        plugin_registry: Optional PluginRegistry for external path authorization.
        allow_tmp: Whether to allow /tmp/** access (default: True).

    Returns:
        AnalysisResult with detected risks.
    """
    analyzer = CodeAnalyzer(
        workspace_root=workspace_root,
        plugin_registry=plugin_registry,
        allow_tmp=allow_tmp,
    )
    return analyzer.analyze(code)
