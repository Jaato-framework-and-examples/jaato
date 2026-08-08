"""Runtime-injectable permission evaluators.

Evaluators are Python scripts referenced from permission configs that
run at permission-check time. They can make authoritative decisions
(ALLOW/DENY) or defer to the standard policy pipeline (FALLBACK).

Config example::

    {
      "evaluators": {
        "default": "policies/global_evaluator.py",
        "cli_based_tool": "policies/cli_evaluator.py"
      }
    }

Evaluator script contract::

    from shared.plugins.permission.evaluator import PolicyDecision, EvalContext

    def evaluate(tool_name: str, args: dict, context: EvalContext) -> PolicyDecision:
        if some_condition:
            return PolicyDecision.DENY
        return PolicyDecision.FALLBACK

Path resolution: workspace ``.jaato/`` → user ``~/.jaato/`` → absolute.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable

from shared.script_loader import load_script_symbol, resolve_script_path

logger = logging.getLogger(__name__)


class PolicyDecision(Enum):
    """Decision returned by a permission evaluator.

    Mirrors the script-relevant subset of ``ChannelDecision`` from the
    permission channels, plus ``FALLBACK`` for deferring to the standard
    policy pipeline.

    Attributes:
        ALLOW: Permit this tool execution (equivalent to user ``y``).
        ALLOW_ONCE: Permit but don't remember (equivalent to ``once``).
        ALLOW_TURN: Permit all remaining tools this turn (``t``).
        ALLOW_UNTIL_IDLE: Permit until session goes idle (``i``).
        ALLOW_SESSION: Add to session whitelist (``a``/``always``).
        ALLOW_ALL: Pre-approve all future requests in session (``all``).
        ALLOW_WITH_COMMENT: Permit and inject a comment into the tool
            result so the model sees advisory feedback (e.g. drift
            warnings).  Use ``EvalResult`` with the comment text.
        DENY: Block this tool execution (``n``).
        DENY_SESSION: Add to session blacklist (``never``).
        DENY_WITH_COMMENT: Deny and pass a comment to the model (``c:``).
            When returning this, use ``EvalResult`` with the comment text.
        FALLBACK: Defer to the standard policy pipeline
            (blacklist/whitelist/default).
    """
    ALLOW = "allow"
    ALLOW_ONCE = "allow_once"
    ALLOW_TURN = "allow_turn"
    ALLOW_UNTIL_IDLE = "allow_until_idle"
    ALLOW_SESSION = "allow_session"
    ALLOW_ALL = "allow_all"
    ALLOW_WITH_COMMENT = "allow_with_comment"
    DENY = "deny"
    DENY_SESSION = "deny_session"
    DENY_WITH_COMMENT = "deny_with_comment"
    FALLBACK = "fallback"


@dataclass
class EvalResult:
    """Result from a permission evaluator, pairing a decision with an
    optional comment.

    Evaluators can return either a bare ``PolicyDecision`` or an
    ``EvalResult`` for richer responses (e.g. deny-with-comment).

    Attributes:
        decision: The permission decision.
        comment: Optional comment text passed to the model when
            ``decision`` is ``DENY_WITH_COMMENT``.
    """
    decision: PolicyDecision
    comment: Optional[str] = None


@dataclass
class EvalContext:
    """Context passed to permission evaluators.

    Only carries information the evaluator can't obtain on its own
    (agent identity, session/workspace context).  Evaluators can
    import stdlib modules (``datetime``, ``os``, etc.) for anything else.

    Attributes:
        tool_name: Name of the tool being checked.
        args: Arguments passed to the tool.
        agent_type: ``"main"`` or ``"subagent"``.
        agent_name: Optional agent/profile name.
        session_id: Daemon session manager ID (if available).
        workspace_path: Workspace directory path (if available).
        turn_index: Current turn number within the session (1-based).
        model_preamble: Text the model emitted before this tool call
            in the current response.  May be None if the model called
            the tool without any preamble text.
        execution_log: Read-only snapshot of the session's PRIOR permission
            decisions, oldest-first — a list of ``{"tool_name", "arguments",
            "decision", "reason"}`` dicts.  Lets an evaluator decide based on
            what happened earlier this session (e.g. deny a tool after N prior
            uses, or escalate once a risky tool has run).  Does NOT include the
            current call being evaluated (that decision hasn't been made yet).
            A snapshot, not the live log — mutating it does not affect the
            plugin's audit trail.  Empty when no evaluator context supplies it.
        extra: Extensible dict for future context fields.
    """
    tool_name: str
    args: Dict[str, Any]
    agent_type: str = "main"
    agent_name: Optional[str] = None
    session_id: Optional[str] = None
    workspace_path: Optional[str] = None
    turn_index: Optional[int] = None
    model_preamble: Optional[str] = None
    execution_log: List[Dict[str, Any]] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class PermissionEvaluator(Protocol):
    """Protocol for permission evaluator functions.

    Evaluators are typically plain functions (not classes), but this
    protocol documents the expected signature for type checking.
    """

    def evaluate(
        self,
        tool_name: str,
        args: Dict[str, Any],
        context: EvalContext,
    ) -> PolicyDecision:
        """Evaluate whether a tool execution should be permitted.

        Args:
            tool_name: Name of the tool being checked.
            args: Arguments passed to the tool.
            context: Evaluation context with agent/session info.

        Returns:
            PolicyDecision: ALLOW, DENY, or FALLBACK.
        """
        ...


def load_evaluators(
    config: Dict[str, str],
    workspace_path: Optional[str] = None,
) -> Dict[str, Callable]:
    """Load permission evaluators from a config dict.

    Args:
        config: Dict mapping tool names (or ``"default"``) to script paths.
        workspace_path: Workspace directory for relative path resolution.

    Returns:
        Dict mapping tool names to loaded ``evaluate`` callables.
        Entries that fail to load are omitted (with warnings logged).
    """
    evaluators: Dict[str, Callable] = {}

    for key, path in config.items():
        resolved = resolve_script_path(path, workspace_path)
        if resolved is None:
            logger.warning(
                "Evaluator script not found for '%s': %s "
                "(searched workspace .jaato/ and ~/.jaato/)",
                key, path,
            )
            continue

        fn = load_script_symbol(
            resolved,
            symbol="evaluate",
            module_prefix="_jaato_evaluator",
        )
        if fn is not None:
            evaluators[key] = fn
            logger.info("Loaded evaluator for '%s' from %s", key, resolved)

    return evaluators


def run_evaluator(
    evaluators: Dict[str, Callable],
    tool_name: str,
    args: Dict[str, Any],
    context: EvalContext,
) -> EvalResult:
    """Run the appropriate evaluator for a tool.

    Checks for a tool-specific evaluator first, then falls back to
    the ``"default"`` evaluator.  If no evaluator matches or the
    evaluator raises an exception, returns FALLBACK.

    Evaluators can return:
    - A ``PolicyDecision`` enum value
    - An ``EvalResult`` (for richer responses like deny-with-comment)
    - A string (``"allow"``, ``"deny"``, ``"fallback"``)
    - A tuple ``(decision, comment)`` for deny-with-comment

    Args:
        evaluators: Loaded evaluator dict from ``load_evaluators()``.
        tool_name: Name of the tool being checked.
        args: Arguments passed to the tool.
        context: Evaluation context.

    Returns:
        EvalResult from the evaluator, or FALLBACK on error/miss.
    """
    _FALLBACK = EvalResult(PolicyDecision.FALLBACK)

    # Tool-specific evaluator takes precedence
    fn = evaluators.get(tool_name) or evaluators.get("default")
    if fn is None:
        return _FALLBACK

    try:
        result = fn(tool_name, args, context)

        # Already an EvalResult
        if isinstance(result, EvalResult):
            return result

        # Bare PolicyDecision
        if isinstance(result, PolicyDecision):
            return EvalResult(result)

        # Tuple (decision, comment) for deny-with-comment
        if isinstance(result, tuple) and len(result) == 2:
            decision, comment = result
            if isinstance(decision, PolicyDecision):
                return EvalResult(decision, comment=str(comment) if comment else None)
            if isinstance(decision, str):
                try:
                    return EvalResult(
                        PolicyDecision(decision.lower()),
                        comment=str(comment) if comment else None,
                    )
                except ValueError:
                    pass

        # Accept string values ("allow", "deny", "fallback")
        if isinstance(result, str):
            try:
                return EvalResult(PolicyDecision(result.lower()))
            except ValueError:
                pass

        logger.warning(
            "Evaluator for '%s' returned invalid result: %r "
            "(expected PolicyDecision, EvalResult, or (decision, comment) tuple)",
            tool_name, result,
        )
        return _FALLBACK
    except Exception as exc:
        logger.warning(
            "Evaluator for '%s' raised %s: %s — falling back to policy",
            tool_name, type(exc).__name__, exc,
        )
        return _FALLBACK
