"""Path and command sanitization for security.

This module provides validation to prevent:
- Path traversal attacks (accessing files outside allowed directories)
- Shell injection attacks (command chaining, variable expansion)
"""

import os
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# Shell metacharacters that could enable injection attacks
SHELL_METACHARACTERS = {
    ';',   # Command separator
    '|',   # Pipe
    '&',   # Background / AND
    '`',   # Command substitution (backticks)
    '$',   # Variable expansion / command substitution
    '(',   # Subshell
    ')',
    '{',   # Brace expansion
    '}',
    '<',   # Redirection
    '>',
    '\n',  # Newline (command separator)
    '\r',
}

# Patterns that indicate shell injection attempts
SHELL_INJECTION_PATTERNS = [
    r'\$\(',        # $(command)
    r'\$\{',        # ${variable}
    r'`[^`]+`',     # `command`
    r';\s*\w',      # ; command
    r'\|\s*\w',     # | command
    r'&&\s*\w',     # && command
    r'\|\|\s*\w',   # || command
    r'>\s*/',       # > /path (redirect to absolute)
    r'>>\s*/',      # >> /path
    r'<\s*/',       # < /path
]

# Dangerous commands that should typically be blocked
DANGEROUS_COMMANDS = {
    # Privilege escalation
    'sudo', 'su', 'doas', 'pkexec',
    # System control
    'shutdown', 'reboot', 'halt', 'poweroff', 'init',
    # Destructive
    'rm', 'rmdir', 'mkfs', 'dd', 'shred',
    # Network exfiltration
    'curl', 'wget', 'nc', 'netcat', 'ncat', 'ssh', 'scp', 'rsync', 'ftp', 'sftp',
    # Process manipulation
    'kill', 'killall', 'pkill',
    # Permission changes
    'chmod', 'chown', 'chgrp',
    # Mount operations
    'mount', 'umount',
}


@dataclass
class SanitizationResult:
    """Result of command sanitization check."""
    is_safe: bool
    reason: str
    violations: List[str] = field(default_factory=list)


@dataclass
class PathScopeConfig:
    """Configuration for path scope validation."""
    allowed_roots: List[str] = field(default_factory=lambda: ["."])
    block_absolute: bool = True
    block_parent_traversal: bool = True
    resolve_symlinks: bool = True
    allow_home: bool = False


@dataclass
class SanitizationConfig:
    """Configuration for command sanitization."""
    block_shell_metacharacters: bool = True
    block_dangerous_commands: bool = True
    allowed_dangerous_commands: Set[str] = field(default_factory=set)
    custom_blocked_commands: Set[str] = field(default_factory=set)
    path_scope: Optional[PathScopeConfig] = None


def check_shell_injection(command: str) -> SanitizationResult:
    """Check for shell injection attempts in a command string.

    Args:
        command: The command string to check

    Returns:
        SanitizationResult indicating if command is safe
    """
    violations = []

    # Check for metacharacters
    for char in SHELL_METACHARACTERS:
        if char in command:
            violations.append(f"Contains shell metacharacter: {repr(char)}")

    # Check for injection patterns
    for pattern in SHELL_INJECTION_PATTERNS:
        if re.search(pattern, command):
            violations.append(f"Matches injection pattern: {pattern}")

    if violations:
        return SanitizationResult(
            is_safe=False,
            reason="Potential shell injection detected",
            violations=violations
        )

    return SanitizationResult(is_safe=True, reason="No injection detected")


def check_dangerous_command(command: str, config: SanitizationConfig) -> SanitizationResult:
    """Check if command uses dangerous executables.

    Args:
        command: The command string to check
        config: Sanitization configuration

    Returns:
        SanitizationResult indicating if command is safe
    """
    violations = []

    try:
        # Parse command to get executable
        parts = shlex.split(command)
        if not parts:
            return SanitizationResult(is_safe=True, reason="Empty command")

        executable = os.path.basename(parts[0])

        # Check against dangerous commands
        blocked = DANGEROUS_COMMANDS | config.custom_blocked_commands
        allowed = config.allowed_dangerous_commands

        if executable in blocked and executable not in allowed:
            violations.append(f"Dangerous command: {executable}")

    except ValueError as e:
        # shlex.split failed - malformed command
        violations.append(f"Malformed command: {e}")

    if violations:
        return SanitizationResult(
            is_safe=False,
            reason="Dangerous command detected",
            violations=violations
        )

    return SanitizationResult(is_safe=True, reason="Command is allowed")


def extract_paths_from_command(command: str) -> List[str]:
    """Extract potential file paths from a command string.

    Args:
        command: The command string to parse

    Returns:
        List of potential paths found in the command
    """
    paths = []

    try:
        parts = shlex.split(command)
    except ValueError:
        # Malformed command, return empty
        return paths

    for part in parts[1:]:  # Skip the command itself
        # Skip flags
        if part.startswith('-'):
            continue
        # Check if it looks like a path
        if '/' in part or part.startswith('.') or part.startswith('~'):
            paths.append(part)
        # Also check for paths without prefix that could be relative
        elif not part.startswith('-') and '.' in part:
            # Could be a filename like "file.txt"
            paths.append(part)

    return paths


def resolve_path(path: str, cwd: str) -> Tuple[bool, str]:
    """Resolve a path to its absolute canonical form.

    Args:
        path: The path to resolve
        cwd: Current working directory

    Returns:
        Tuple of (success, resolved_path_or_error)
    """
    try:
        # Expand ~ to home directory
        expanded = os.path.expanduser(path)

        # Make absolute relative to cwd
        if not os.path.isabs(expanded):
            expanded = os.path.join(cwd, expanded)

        # Resolve to canonical path (handles .., symlinks)
        resolved = os.path.realpath(expanded)

        return True, resolved
    except Exception as e:
        return False, str(e)


def check_path_scope(
    path: str,
    config: PathScopeConfig,
    cwd: Optional[str] = None
) -> SanitizationResult:
    """Check if a path is within allowed scope.

    Args:
        path: The path to check
        config: Path scope configuration
        cwd: Current working directory (defaults to os.getcwd())

    Returns:
        SanitizationResult indicating if path is allowed
    """
    violations = []
    cwd = cwd or os.getcwd()

    # Check for obvious violations before resolving
    if config.block_absolute and path.startswith('/'):
        violations.append(f"Absolute path not allowed: {path}")

    if config.block_parent_traversal and '..' in path:
        violations.append(f"Parent traversal not allowed: {path}")

    if not config.allow_home and path.startswith('~'):
        violations.append(f"Home directory access not allowed: {path}")

    # Early return if obvious violations
    if violations:
        return SanitizationResult(
            is_safe=False,
            reason="Path scope violation",
            violations=violations
        )

    # Resolve path to check actual location
    success, resolved = resolve_path(path, cwd)
    if not success:
        return SanitizationResult(
            is_safe=False,
            reason=f"Cannot resolve path: {resolved}",
            violations=[f"Path resolution failed: {path}"]
        )

    # Check if resolved path is within allowed roots
    allowed = False
    for root in config.allowed_roots:
        # Resolve the allowed root too
        _, resolved_root = resolve_path(root, cwd)
        if resolved.startswith(resolved_root + os.sep) or resolved == resolved_root:
            allowed = True
            break

    if not allowed:
        violations.append(f"Path outside allowed scope: {path} -> {resolved}")
        return SanitizationResult(
            is_safe=False,
            reason="Path outside allowed directories",
            violations=violations
        )

    return SanitizationResult(is_safe=True, reason="Path within allowed scope")


def sanitize_command(
    command: str,
    config: SanitizationConfig,
    cwd: Optional[str] = None
) -> SanitizationResult:
    """Full sanitization check for a command.

    Performs all configured checks:
    - Shell injection detection
    - Dangerous command blocking
    - Path scope validation

    Args:
        command: The command string to check
        config: Sanitization configuration
        cwd: Current working directory for path checks

    Returns:
        SanitizationResult with combined results
    """
    all_violations = []

    # Check shell injection
    if config.block_shell_metacharacters:
        result = check_shell_injection(command)
        if not result.is_safe:
            all_violations.extend(result.violations)

    # Check dangerous commands
    if config.block_dangerous_commands:
        result = check_dangerous_command(command, config)
        if not result.is_safe:
            all_violations.extend(result.violations)

    # Check path scope
    if config.path_scope:
        paths = extract_paths_from_command(command)
        for path in paths:
            result = check_path_scope(path, config.path_scope, cwd)
            if not result.is_safe:
                all_violations.extend(result.violations)

    if all_violations:
        return SanitizationResult(
            is_safe=False,
            reason="Command failed sanitization",
            violations=all_violations
        )

    return SanitizationResult(is_safe=True, reason="Command passed all checks")


def create_strict_config(cwd: Optional[str] = None) -> SanitizationConfig:
    """Create a strict sanitization config for sandboxed execution.

    This configuration:
    - Blocks all shell metacharacters
    - Blocks dangerous commands
    - Restricts paths to current working directory only
    - Blocks absolute paths, parent traversal, and home access

    Args:
        cwd: Current working directory (defaults to ".")

    Returns:
        Strict SanitizationConfig
    """
    return SanitizationConfig(
        block_shell_metacharacters=True,
        block_dangerous_commands=True,
        allowed_dangerous_commands=set(),
        custom_blocked_commands=set(),
        path_scope=PathScopeConfig(
            allowed_roots=[cwd or "."],
            block_absolute=True,
            block_parent_traversal=True,
            resolve_symlinks=True,
            allow_home=False,
        )
    )


def create_permissive_config() -> SanitizationConfig:
    """Create a permissive config that only blocks obvious attacks.

    This configuration:
    - Blocks shell injection attempts
    - Blocks the most dangerous commands (sudo, rm, etc.)
    - Does NOT restrict paths

    Returns:
        Permissive SanitizationConfig
    """
    return SanitizationConfig(
        block_shell_metacharacters=True,
        block_dangerous_commands=True,
        allowed_dangerous_commands={'rm'},  # Allow rm for basic file operations
        custom_blocked_commands=set(),
        path_scope=None,  # No path restrictions
    )
