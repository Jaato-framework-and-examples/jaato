"""Permission policy evaluation engine.

This module provides the core logic for evaluating tool execution permissions
based on blacklist/whitelist rules. The blacklist always takes priority.

Optionally includes sanitization checks for:
- Shell injection prevention
- Path scope validation
- Dangerous command blocking
"""

import fnmatch
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from .sanitization import (
    SanitizationConfig,
    SanitizationResult,
    sanitize_command,
    create_strict_config,
)


class PermissionDecision(Enum):
    """Possible decisions from permission evaluation."""
    ALLOW = "allow"
    DENY = "deny"
    ASK_ACTOR = "ask_actor"  # Policy undecided, needs actor approval


@dataclass
class PolicyMatch:
    """Details about why a permission decision was made."""
    decision: PermissionDecision
    reason: str
    matched_rule: Optional[str] = None
    rule_type: Optional[str] = None  # "blacklist", "whitelist", "default", "sanitization"
    violations: Optional[List[str]] = None  # For sanitization failures


@dataclass
class PermissionPolicy:
    """Policy engine for evaluating tool execution permissions.

    Evaluation order:
    1. Sanitization checks (if enabled) -> DENY if violations found
    2. Check blacklist (tools, patterns, arguments) -> DENY if matched
    3. Check whitelist (tools, patterns, arguments) -> ALLOW if matched
    4. Apply default_policy

    Blacklist ALWAYS takes priority over whitelist.
    """

    default_policy: str = "deny"  # "allow" or "deny"

    # Blacklist rules
    blacklist_tools: Set[str] = field(default_factory=set)
    blacklist_patterns: List[str] = field(default_factory=list)
    blacklist_arguments: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)

    # Whitelist rules
    whitelist_tools: Set[str] = field(default_factory=set)
    whitelist_patterns: List[str] = field(default_factory=list)
    whitelist_arguments: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)

    # Session-level dynamic rules (added via actor responses)
    session_blacklist: Set[str] = field(default_factory=set)
    session_whitelist: Set[str] = field(default_factory=set)

    # Sanitization configuration (None = disabled)
    sanitization_config: Optional[SanitizationConfig] = None
    cwd: Optional[str] = None  # Working directory for path checks

    def check(self, tool_name: str, args: Dict[str, Any]) -> PolicyMatch:
        """Evaluate permission for a tool call.

        Args:
            tool_name: Name of the tool being called
            args: Arguments being passed to the tool

        Returns:
            PolicyMatch with decision and reasoning
        """
        # Build a signature for pattern matching
        signature = self._build_signature(tool_name, args)

        # 0. Run sanitization checks FIRST (highest priority for security)
        if self.sanitization_config is not None:
            sanitization_match = self._check_sanitization(tool_name, args, signature)
            if sanitization_match:
                return sanitization_match

        # 1. Check session blacklist first (highest priority)
        if self._matches_session_blacklist(tool_name, signature):
            return PolicyMatch(
                decision=PermissionDecision.DENY,
                reason=f"Tool '{tool_name}' is blacklisted for this session",
                rule_type="session_blacklist"
            )

        # 2. Check static blacklist
        blacklist_match = self._check_blacklist(tool_name, args, signature)
        if blacklist_match:
            return blacklist_match

        # 3. Check session whitelist
        if self._matches_session_whitelist(tool_name, signature):
            return PolicyMatch(
                decision=PermissionDecision.ALLOW,
                reason=f"Tool '{tool_name}' is whitelisted for this session",
                rule_type="session_whitelist"
            )

        # 4. Check static whitelist
        whitelist_match = self._check_whitelist(tool_name, args, signature)
        if whitelist_match:
            return whitelist_match

        # 5. Apply default policy
        if self.default_policy == "allow":
            return PolicyMatch(
                decision=PermissionDecision.ALLOW,
                reason="Allowed by default policy",
                rule_type="default"
            )
        elif self.default_policy == "deny":
            return PolicyMatch(
                decision=PermissionDecision.DENY,
                reason="Denied by default policy",
                rule_type="default"
            )
        else:
            # default_policy == "ask" or unknown -> ask actor
            return PolicyMatch(
                decision=PermissionDecision.ASK_ACTOR,
                reason="No matching rule, requires actor approval",
                rule_type="default"
            )

    def _check_sanitization(
        self, tool_name: str, args: Dict[str, Any], signature: str
    ) -> Optional[PolicyMatch]:
        """Run sanitization checks. Returns PolicyMatch if blocked, None otherwise.

        Sanitization checks run BEFORE blacklist/whitelist evaluation and include:
        - Shell injection detection (metacharacters, command substitution)
        - Dangerous command blocking (sudo, rm, etc.)
        - Path scope validation (restrict to allowed directories)
        """
        if self.sanitization_config is None:
            return None

        # Only sanitize CLI commands
        if tool_name != "cli_based_tool":
            return None

        command = args.get("command", "")
        if not command:
            return None

        result = sanitize_command(command, self.sanitization_config, self.cwd)

        if not result.is_safe:
            return PolicyMatch(
                decision=PermissionDecision.DENY,
                reason=f"Sanitization failed: {result.reason}",
                matched_rule="sanitization",
                rule_type="sanitization",
                violations=result.violations
            )

        return None

    def _build_signature(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Build a command signature for pattern matching.

        For CLI tools, this extracts the command string.
        For other tools, it creates a representation like "tool_name(arg1=val1, ...)".
        """
        if tool_name == "cli_based_tool":
            command = args.get("command", "")
            arg_list = args.get("args", [])
            if arg_list:
                return f"{command} {' '.join(str(a) for a in arg_list)}"
            return command
        else:
            # For non-CLI tools, create a simple signature
            return f"{tool_name}({', '.join(f'{k}={v}' for k, v in sorted(args.items()))})"

    def _matches_session_blacklist(self, tool_name: str, signature: str) -> bool:
        """Check if tool matches any session blacklist entry."""
        for pattern in self.session_blacklist:
            if fnmatch.fnmatch(tool_name, pattern) or fnmatch.fnmatch(signature, pattern):
                return True
        return False

    def _matches_session_whitelist(self, tool_name: str, signature: str) -> bool:
        """Check if tool matches any session whitelist entry."""
        for pattern in self.session_whitelist:
            if fnmatch.fnmatch(tool_name, pattern) or fnmatch.fnmatch(signature, pattern):
                return True
        return False

    def _check_blacklist(
        self, tool_name: str, args: Dict[str, Any], signature: str
    ) -> Optional[PolicyMatch]:
        """Check blacklist rules. Returns match if blocked, None otherwise."""

        # Check tool name blacklist
        if tool_name in self.blacklist_tools:
            return PolicyMatch(
                decision=PermissionDecision.DENY,
                reason=f"Tool '{tool_name}' is blacklisted",
                matched_rule=tool_name,
                rule_type="blacklist"
            )

        # Check pattern blacklist (glob-style matching)
        for pattern in self.blacklist_patterns:
            if fnmatch.fnmatch(signature, pattern):
                return PolicyMatch(
                    decision=PermissionDecision.DENY,
                    reason=f"Command matches blacklist pattern: {pattern}",
                    matched_rule=pattern,
                    rule_type="blacklist"
                )

        # Check argument blacklist
        if tool_name in self.blacklist_arguments:
            arg_rules = self.blacklist_arguments[tool_name]
            for arg_name, blocked_values in arg_rules.items():
                arg_value = args.get(arg_name, "")
                # For string arguments, check if it starts with any blocked value
                if isinstance(arg_value, str):
                    for blocked in blocked_values:
                        if arg_value.startswith(blocked) or blocked in arg_value.split():
                            return PolicyMatch(
                                decision=PermissionDecision.DENY,
                                reason=f"Argument '{arg_name}' contains blocked value: {blocked}",
                                matched_rule=f"{arg_name}={blocked}",
                                rule_type="blacklist"
                            )

        return None

    def _check_whitelist(
        self, tool_name: str, args: Dict[str, Any], signature: str
    ) -> Optional[PolicyMatch]:
        """Check whitelist rules. Returns match if allowed, None otherwise."""

        # Check tool name whitelist
        if tool_name in self.whitelist_tools:
            return PolicyMatch(
                decision=PermissionDecision.ALLOW,
                reason=f"Tool '{tool_name}' is whitelisted",
                matched_rule=tool_name,
                rule_type="whitelist"
            )

        # Check pattern whitelist (glob-style matching)
        for pattern in self.whitelist_patterns:
            if fnmatch.fnmatch(signature, pattern):
                return PolicyMatch(
                    decision=PermissionDecision.ALLOW,
                    reason=f"Command matches whitelist pattern: {pattern}",
                    matched_rule=pattern,
                    rule_type="whitelist"
                )

        # Check argument whitelist
        if tool_name in self.whitelist_arguments:
            arg_rules = self.whitelist_arguments[tool_name]
            for arg_name, allowed_values in arg_rules.items():
                arg_value = args.get(arg_name, "")
                if isinstance(arg_value, str):
                    for allowed in allowed_values:
                        if arg_value.startswith(allowed):
                            return PolicyMatch(
                                decision=PermissionDecision.ALLOW,
                                reason=f"Argument '{arg_name}' matches allowed value: {allowed}",
                                matched_rule=f"{arg_name}={allowed}",
                                rule_type="whitelist"
                            )

        return None

    def add_session_blacklist(self, pattern: str) -> None:
        """Add a pattern to the session blacklist."""
        self.session_blacklist.add(pattern)

    def add_session_whitelist(self, pattern: str) -> None:
        """Add a pattern to the session whitelist.

        Note: Session blacklist still takes priority over session whitelist.
        """
        self.session_whitelist.add(pattern)

    def clear_session_rules(self) -> None:
        """Clear all session-level rules."""
        self.session_blacklist.clear()
        self.session_whitelist.clear()

    def set_sanitization(
        self,
        config: Optional[SanitizationConfig] = None,
        cwd: Optional[str] = None
    ) -> None:
        """Enable or update sanitization configuration.

        Args:
            config: SanitizationConfig instance, or None to disable
            cwd: Working directory for path scope checks
        """
        self.sanitization_config = config
        self.cwd = cwd

    def enable_strict_sandbox(self, cwd: Optional[str] = None) -> None:
        """Enable strict sandboxing with all protections.

        This enables:
        - Shell injection blocking
        - Dangerous command blocking
        - Path scope restricted to cwd only
        """
        self.sanitization_config = create_strict_config(cwd)
        self.cwd = cwd

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'PermissionPolicy':
        """Create a PermissionPolicy from a configuration dict.

        Expected config structure:
        {
            "defaultPolicy": "deny",
            "blacklist": {
                "tools": ["tool1", "tool2"],
                "patterns": ["rm *", "sudo *"],
                "arguments": {
                    "cli_based_tool": {"command": ["rm", "sudo"]}
                }
            },
            "whitelist": {
                "tools": ["safe_tool"],
                "patterns": ["git *"],
                "arguments": {
                    "cli_based_tool": {"command": ["git", "npm"]}
                }
            },
            "sanitization": {
                "enabled": true,
                "block_shell_metacharacters": true,
                "block_dangerous_commands": true,
                "allowed_dangerous_commands": ["rm"],
                "path_scope": {
                    "enabled": true,
                    "allowed_roots": ["."],
                    "block_absolute": true,
                    "block_parent_traversal": true,
                    "allow_home": false
                }
            }
        }
        """
        from .sanitization import SanitizationConfig, PathScopeConfig

        blacklist = config.get("blacklist", {})
        whitelist = config.get("whitelist", {})

        # Parse sanitization config
        sanitization_config = None
        san_cfg = config.get("sanitization", {})
        if san_cfg.get("enabled", False):
            path_scope = None
            ps_cfg = san_cfg.get("path_scope", {})
            if ps_cfg.get("enabled", False):
                path_scope = PathScopeConfig(
                    allowed_roots=ps_cfg.get("allowed_roots", ["."]),
                    block_absolute=ps_cfg.get("block_absolute", True),
                    block_parent_traversal=ps_cfg.get("block_parent_traversal", True),
                    resolve_symlinks=ps_cfg.get("resolve_symlinks", True),
                    allow_home=ps_cfg.get("allow_home", False),
                )

            sanitization_config = SanitizationConfig(
                block_shell_metacharacters=san_cfg.get("block_shell_metacharacters", True),
                block_dangerous_commands=san_cfg.get("block_dangerous_commands", True),
                allowed_dangerous_commands=set(san_cfg.get("allowed_dangerous_commands", [])),
                custom_blocked_commands=set(san_cfg.get("custom_blocked_commands", [])),
                path_scope=path_scope,
            )

        return cls(
            default_policy=config.get("defaultPolicy", "deny"),
            blacklist_tools=set(blacklist.get("tools", [])),
            blacklist_patterns=blacklist.get("patterns", []),
            blacklist_arguments=blacklist.get("arguments", {}),
            whitelist_tools=set(whitelist.get("tools", [])),
            whitelist_patterns=whitelist.get("patterns", []),
            whitelist_arguments=whitelist.get("arguments", {}),
            sanitization_config=sanitization_config,
            cwd=config.get("cwd"),
        )
