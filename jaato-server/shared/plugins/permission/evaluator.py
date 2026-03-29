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

import importlib.util
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


class PolicyDecision(Enum):
    """Decision returned by a permission evaluator.

    Attributes:
        ALLOW: Permit the tool execution unconditionally.
        DENY: Block the tool execution unconditionally.
        FALLBACK: Defer to the standard policy pipeline
            (blacklist/whitelist/default).
    """
    ALLOW = "allow"
    DENY = "deny"
    FALLBACK = "fallback"


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
        extra: Extensible dict for future context fields.
    """
    tool_name: str
    args: Dict[str, Any]
    agent_type: str = "main"
    agent_name: Optional[str] = None
    session_id: Optional[str] = None
    workspace_path: Optional[str] = None
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


def _resolve_evaluator_path(
    path: str,
    workspace_path: Optional[str] = None,
) -> Optional[Path]:
    """Resolve an evaluator script path.

    Resolution order:
    1. Absolute path → use directly
    2. Relative → ``{workspace}/.jaato/{path}``
    3. Relative → ``~/.jaato/{path}``

    Args:
        path: Evaluator file path (absolute or relative).
        workspace_path: Workspace directory for relative resolution.

    Returns:
        Resolved Path, or None if not found.
    """
    p = Path(path)
    if p.is_absolute():
        return p if p.is_file() else None

    # Workspace-relative
    if workspace_path:
        ws_path = Path(workspace_path) / ".jaato" / path
        if ws_path.is_file():
            return ws_path

    # User-level
    home_path = Path.home() / ".jaato" / path
    if home_path.is_file():
        return home_path

    return None


def _load_evaluate_function(file_path: Path) -> Optional[Callable]:
    """Load the ``evaluate`` function from a Python script.

    Args:
        file_path: Absolute path to the evaluator script.

    Returns:
        The ``evaluate`` callable, or None on failure.
    """
    module_name = f"_jaato_evaluator_{file_path.stem}"
    try:
        spec = importlib.util.spec_from_file_location(module_name, str(file_path))
        if spec is None or spec.loader is None:
            logger.warning("Cannot load evaluator module from %s", file_path)
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        fn = getattr(module, "evaluate", None)
        if fn is None:
            logger.warning(
                "Evaluator script %s has no 'evaluate' function", file_path
            )
            return None
        if not callable(fn):
            logger.warning(
                "Evaluator script %s: 'evaluate' is not callable", file_path
            )
            return None
        return fn
    except Exception as exc:
        logger.warning(
            "Failed to load evaluator from %s: %s", file_path, exc
        )
        return None


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
        resolved = _resolve_evaluator_path(path, workspace_path)
        if resolved is None:
            logger.warning(
                "Evaluator script not found for '%s': %s "
                "(searched workspace .jaato/ and ~/.jaato/)",
                key, path,
            )
            continue

        fn = _load_evaluate_function(resolved)
        if fn is not None:
            evaluators[key] = fn
            logger.info("Loaded evaluator for '%s' from %s", key, resolved)

    return evaluators


def run_evaluator(
    evaluators: Dict[str, Callable],
    tool_name: str,
    args: Dict[str, Any],
    context: EvalContext,
) -> PolicyDecision:
    """Run the appropriate evaluator for a tool.

    Checks for a tool-specific evaluator first, then falls back to
    the ``"default"`` evaluator.  If no evaluator matches or the
    evaluator raises an exception, returns FALLBACK.

    Args:
        evaluators: Loaded evaluator dict from ``load_evaluators()``.
        tool_name: Name of the tool being checked.
        args: Arguments passed to the tool.
        context: Evaluation context.

    Returns:
        PolicyDecision from the evaluator, or FALLBACK on error/miss.
    """
    # Tool-specific evaluator takes precedence
    fn = evaluators.get(tool_name) or evaluators.get("default")
    if fn is None:
        return PolicyDecision.FALLBACK

    try:
        result = fn(tool_name, args, context)
        if isinstance(result, PolicyDecision):
            return result
        # Accept string values too ("allow", "deny", "fallback")
        if isinstance(result, str):
            try:
                return PolicyDecision(result.lower())
            except ValueError:
                pass
        logger.warning(
            "Evaluator for '%s' returned invalid result: %r (expected PolicyDecision)",
            tool_name, result,
        )
        return PolicyDecision.FALLBACK
    except Exception as exc:
        logger.warning(
            "Evaluator for '%s' raised %s: %s — falling back to policy",
            tool_name, type(exc).__name__, exc,
        )
        return PolicyDecision.FALLBACK
