"""Dynamic-instructions expansion for agent system prompts.

Recognises ``{{!py:script.py [args]}}`` placeholders and replaces them
with the output of user scripts loaded from ``.jaato/`` via the standard
``script_loader`` resolution chain.

This is the **input-side** symmetric counterpart to reactor actions and
permission evaluators: scripts run on the framework's authority before
the agent's first turn, with full access to the session's runtime
state.  The agent never sees them as choices to make — only as content
already present in its system prompt.

Use cases (per ``project_backlog_dynamic_instructions``):

- **Mandatory prefetch.** Service calls the agent must have made
  anyway — push them out of the agent's discretion.  Script calls
  ``call_service`` and embeds the structured response in the prompt.
- **Live state.** Memory snapshots, ledger usage, recent references —
  values that should be visible at session start.
- **Forwarded context.** Snippets pulled from ``agent_params`` (e.g.
  forwarded ``case_data``) without manual re-formatting.

Scripts must define a top-level ``def render(context, args) -> str``.
Errors embed inline as ``[script error: ...]`` so the agent sees
failure as observable evidence rather than as a silent gap.

Execution-context contract (per the backlog memory's 2026-04-30
addendum):

> If a tool can call it, a script can render it; if a tool can't, a
> script can't either.

Scripts run synchronously during ``JaatoSession.configure()``, after
the system_instruction is assembled but before it is finalised.  The
session's env stack (``JAATO_WORKSPACE_ROOT`` / ``JAATO_CONFIG_ROOT`` /
profile env overlay) is already pushed by the time configure() runs.
The :class:`RenderContext` exposes runtime / registry / workspace_path
/ config_root / agent_params for explicit access.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional

from .script_loader import resolve_script_path, load_script_symbol

logger = logging.getLogger(__name__)


# ``{{!py[?]:path/to/script.py optional space-separated args}}``
# Path is everything up to the first whitespace; args are the rest of
# the placeholder body up to the closing braces.  Newlines inside the
# placeholder are not supported (kept on one line).
#
# The optional ``?`` modifier (server 0.6.48+) marks the placeholder
# as best-effort: any failure (script-not-found, load-error, render
# raise, sentinel-string return, non-string return) is swallowed and
# substituted with the error sentinel string in the prompt — same as
# pre-0.6.48 behaviour.  Without ``?`` the framework now ABORTS session
# creation on any of those error paths, propagating a structured
# ErrorEvent to the client.  Default-strict matches the load-bearing
# nature of most prefetches; opt-in best-effort for genuinely optional
# scripts (memory snapshot, ledger usage, ambient reference data).
_PY_PLACEHOLDER = re.compile(r"\{\{!py(\?)?:([^\s}]+)(?:\s+([^}]*))?\}\}")


class DynamicInstructionsError(Exception):
    """Raised when a non-optional ``{{!py:...}}`` placeholder fails to
    render (server 0.6.48+).

    Carries the failing placeholder's script reference and the
    underlying error message so callers can surface a structured
    ``ErrorEvent`` to the client.  Without this, the prior swallow-
    and-substitute behaviour produced agents running with a hollow
    prompt that fabricated outputs at T=0 (false byte-identicality
    diagnosed by 7:3 in the kb-enablement-2.0 cascade probe v6).

    Optional placeholders (``{{!py?:script.py}}``) DO NOT raise; their
    error sentinel is substituted into the prompt as before, since
    those scripts opted in to best-effort semantics.
    """

    def __init__(self, script_ref: str, reason: str) -> None:
        super().__init__(f"dynamic-instructions abort: {script_ref}: {reason}")
        self.script_ref = script_ref
        self.reason = reason


# Sentinel-prefix list (server 0.6.48+).  When a script's render returns
# a string starting with any of these prefixes, the framework treats it
# as a deliberate failure signal — the convention 7:3 surfaced for
# returning ``[prefetch error: ...]`` from inside render().  Plus the
# framework's own emitted-on-error sentinels (kept for compatibility
# with scripts that catch their own errors and rebuild the same shape).
_FAILURE_SENTINEL_PREFIXES = (
    "[prefetch error:",
    "[script error:",
    "[script not found:",
    "[script load error:",
)


@dataclass
class RenderContext:
    """Per-render context handed to user scripts.

    Mirrors jaato-premium's reactor ``ActionContext`` shape but for the
    dynamic-instructions render path on the input side.  Scripts read
    state through these handles rather than hunting for it via globals.

    Attributes:
        session: The owning ``JaatoSession`` instance.  Scripts can
            read ``session.workspace_path``, ``session.history``, etc.
        runtime: The session's ``JaatoRuntime``.  Provides cross-session
            shared state (registered providers, ledger, ...).
        registry: The session's ``PluginRegistry``.  Use
            ``registry.get_plugin("service_connector")`` to call services
            with the same URL resolution / auth as the session's tools.
        workspace_path: Session's workspace dir, or ``None``.  Already
            propagated as ``JAATO_WORKSPACE_ROOT`` in the process env.
        config_root: Session's read-only-config root override, or ``None``.
            Already propagated as ``JAATO_CONFIG_ROOT`` in the process env.
        agent_params: The dict the supervisor passed via
            ``spawn_subagent(agent_params={...})`` — typically carries
            the forwarded ``case_data`` and any other per-spawn fields.
            Empty for top-level sessions (orchestrator-driven prompts
            embed case data in the message text instead).
        env: Snapshot of ``os.environ`` taken at expansion time.  Use
            this for *deterministic* env reads; scripts may also call
            ``os.environ.get`` directly since the session env is
            already in effect, but the snapshot is preferred for
            reproducibility.
        logger: Per-script logger; defaults to a shared one if the
            caller doesn't pass a script-specific logger.
        tool_calls: Pre-computed ledger of every function_call +
            function_response in the session, paired by call_id.
            Populated only at completion-processor invocation time
            (``LifecycleTools._execute_signal_completion``); empty
            list when the context is built for input-side prefetch.
            Each entry is a dict with keys ``name`` / ``args`` /
            ``result`` / ``success`` / ``call_id`` / ``turn_index``.
            See :func:`shared.completion_processors.build_tool_call_ledger`
            for the contract.
        session_id: Daemon-side session identifier resolved at
            construction time with the parent-walk fallback (server
            0.6.172+).  Mirrors the resolution at
            ``JaatoSession._daemon_session_id`` lookup in
            ``jaato_session.py:621-628``: prefers the immediate
            session's value but walks up the ``_parent_session``
            chain so subagent sessions inherit the root daemon
            session_id.  ``None`` when the session has not been
            registered with a daemon (e.g. unit tests with bare
            ``JaatoSession`` instances).  Kb-side processors and
            prefetch scripts read this for per-session disk
            artifact paths (e.g.
            ``cascade_state/audit/<session_id>.jsonl``) WITHOUT a
            defensive fallback — None here means the kb caller
            should treat the session as non-daemon-attached and
            either skip the artifact or fail loudly per the
            no-fallback rule.
    """

    session: Any
    runtime: Any
    registry: Any
    workspace_path: Optional[str]
    config_root: Optional[str]
    agent_params: Dict[str, Any] = field(default_factory=dict)
    env: Mapping[str, str] = field(default_factory=dict)
    logger: logging.Logger = field(
        default_factory=lambda: logging.getLogger(
            "shared.dynamic_instructions"
        )
    )
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    session_id: Optional[str] = None


def expand_py_placeholders(content: str, context: RenderContext) -> str:
    """Walk ``{{!py:script.py args}}`` placeholders and substitute them.

    Each placeholder resolves to a script via the standard
    ``script_loader.resolve_script_path`` tier (workspace → user →
    premium-fallback).  The script's top-level ``render(context, args)``
    callable is loaded and invoked; its return value (coerced to ``str``
    if needed) replaces the placeholder.

    Failure modes are non-fatal: the placeholder is replaced with a
    bracketed error marker that the agent can see and reason about.
    This matches the behaviour of the existing ``{{!command}}`` shell
    expansion in ``prompt_library`` — agents are better off with
    *visible* failure evidence than silent gaps.

    Failure markers used:

    - ``[script not found: <ref>]`` — resolution miss.
    - ``[script load error: <ref>]`` — file present but import or
      symbol-lookup failed (logged with traceback).
    - ``[script error: <ref>: <exception>]`` — script raised at runtime.

    Args:
        content: Agent template content (already param-substituted).
        context: :class:`RenderContext` with the session's handles.

    Returns:
        The same content with every ``{{!py:...}}`` placeholder replaced.
        Returns the original content unchanged when no placeholders
        match (early-out via substring check).
    """
    # Early-out fast path.  Match the strict ``{{!py:`` form OR the
    # optional ``{{!py?:`` form (server 0.6.48+).  Either prefix
    # warrants the regex pass below.
    if "{{!py:" not in content and "{{!py?:" not in content:
        return content

    def _replace(match: re.Match) -> str:
        # Group 1: optional `?` (best-effort marker, server 0.6.48+).
        # Group 2: script_ref.  Group 3: optional args.
        is_optional = match.group(1) == "?"
        script_ref = match.group(2).strip()
        args_str = match.group(3) or ""
        args: List[str] = args_str.split() if args_str else []

        def _fail(reason: str, sentinel: str) -> str:
            """Either raise (default strict) or return sentinel (optional).

            When ``is_optional`` is True the placeholder is best-effort:
            log + substitute the sentinel into the prompt (pre-0.6.48
            behaviour).  Otherwise raise ``DynamicInstructionsError`` so
            the session-creation path can convert it to a structured
            ErrorEvent and abort cleanly — preventing silent fabrication.
            """
            if is_optional:
                return sentinel
            raise DynamicInstructionsError(script_ref, reason)

        path = resolve_script_path(
            script_ref,
            workspace_path=context.workspace_path,
            config_root=context.config_root,
        )
        if path is None:
            logger.warning(
                "dynamic-instructions: script not found: %s "
                "(workspace=%s, config_root=%s)",
                script_ref, context.workspace_path, context.config_root,
            )
            return _fail(
                f"script not found (workspace={context.workspace_path}, "
                f"config_root={context.config_root})",
                f"[script not found: {script_ref}]",
            )

        fn = load_script_symbol(
            path, symbol="render", module_prefix="_jaato_dynprompt",
        )
        if fn is None:
            return _fail(
                f"script load failed (path={path}, render symbol missing "
                f"or import error)",
                f"[script load error: {script_ref}]",
            )

        try:
            result = fn(context, args)
        except Exception as exc:
            logger.exception(
                "dynamic-instructions: render failed for %s", script_ref,
            )
            return _fail(
                f"render raised {type(exc).__name__}: {exc}",
                f"[script error: {script_ref}: {exc}]",
            )

        # Coerce non-string returns.  Pre-0.6.48 silently called str(result);
        # server 0.6.48+ treats it as a contract violation by default and
        # raises (optional placeholders preserve the legacy coerce).
        if not isinstance(result, str):
            return _fail(
                f"render returned non-string ({type(result).__name__})",
                str(result),
            )

        # Sentinel-string detection (server 0.6.48+).  Convention from
        # 7:3's q-message: scripts deliberately signal failure by
        # returning a string starting with ``[prefetch error: ...]``.
        # Also catches the framework's own emitted-on-error sentinels
        # (kept for compatibility with scripts that catch their own
        # errors and rebuild the same shape).  Optional placeholders
        # let the sentinel through; default-strict raises.
        if result.startswith(_FAILURE_SENTINEL_PREFIXES):
            # Strip the leading bracket-tag so the abort reason is just
            # the script's own error message, not the framework's
            # wrapping — keeps the ErrorEvent message readable.
            return _fail(
                f"render returned failure sentinel: {result.strip()}",
                result,
            )

        return result

    return _PY_PLACEHOLDER.sub(_replace, content)


def build_render_context(
    session: Any,
    agent_params: Optional[Dict[str, Any]] = None,
    tool_calls: Optional[List[Dict[str, Any]]] = None,
) -> RenderContext:
    """Build a :class:`RenderContext` from a configured ``JaatoSession``.

    Convenience helper used by ``JaatoSession.configure()`` to package
    its handles for the expansion call site.  Pulls runtime / registry
    / paths off the session and snapshots ``os.environ`` at the moment
    the context is built (so even if the env stack pops later, the
    snapshot still reflects the session's effective env).

    Args:
        session: The configured ``JaatoSession``.
        agent_params: Optional dict to expose to scripts as
            ``context.agent_params``.  Pass through whatever was passed
            to ``configure(agent_params=...)``.
        tool_calls: Optional pre-computed tool-call ledger (server
            0.6.125+) to expose as ``context.tool_calls``.  Populated
            by ``LifecycleTools._execute_signal_completion`` so
            ``completion_processors`` can cross-check payload claims
            against the session's actual tool history; left empty
            for input-side prefetch where session history isn't yet
            interesting.

    Returns:
        Fully populated :class:`RenderContext`.
    """
    runtime = getattr(session, "_runtime", None) or getattr(session, "runtime", None)
    registry = getattr(runtime, "registry", None) if runtime else None
    workspace_path = getattr(session, "_workspace_path", None) or getattr(
        session, "workspace_path", None,
    )
    config_root = (
        getattr(session, "_config_root", None)
        or (getattr(runtime, "_config_root", None) if runtime else None)
    )
    session_id = _resolve_session_id(session)
    return RenderContext(
        session=session,
        runtime=runtime,
        registry=registry,
        workspace_path=workspace_path,
        config_root=config_root,
        agent_params=dict(agent_params or {}),
        env=dict(os.environ),
        tool_calls=list(tool_calls) if tool_calls else [],
        session_id=session_id,
    )


def _resolve_session_id(session: Any) -> Optional[str]:
    """Resolve the daemon session_id with parent-walk fallback
    (server 0.6.172+).

    Mirrors the canonical resolution at
    ``JaatoSession._daemon_session_id`` lookup in
    ``shared.jaato_session:621-628``: prefer the immediate session's
    ``_daemon_session_id`` value, but walk up the ``_parent_session``
    chain when None so subagent sessions inherit the root daemon's
    session_id.

    Subagents are spawned with a fresh JaatoSession that doesn't
    receive its own ``_daemon_session_id`` (the daemon-side session
    manager only registers the root agent's session).  Without the
    walk, every subagent's ``RenderContext.session_id`` would be
    None, breaking per-session artifact paths and audit ledger
    naming for any kb-side code that relies on a non-None
    session_id.  The walk closes that gap deterministically — same
    contract as the telemetry-side resolver, single source of
    behavioral truth.

    Defensive ``getattr`` reads tolerate non-JaatoSession callers
    (e.g. unit tests passing a bare object as ``session``) so this
    helper returns None cleanly rather than raising.

    Args:
        session: A ``JaatoSession`` instance (or any object;
            non-session inputs return None).

    Returns:
        The resolved daemon session_id, or None when no ancestor in
        the parent chain has one set.
    """
    sid = getattr(session, "_daemon_session_id", None)
    if sid:
        return sid
    parent = getattr(session, "_parent_session", None)
    while parent is not None:
        sid = getattr(parent, "_daemon_session_id", None)
        if sid:
            return sid
        parent = getattr(parent, "_parent_session", None)
    return None
