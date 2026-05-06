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


@dataclass
class ArtifactRenderResult:
    """Outcome of running a profile's ``completion_artifacts`` list.

    Each artifact lands in exactly one of three buckets, mirroring the
    profile-declared ``on_error`` policy:

    Attributes:
        written: Absolute paths of artefacts the framework successfully
            wrote.  Surfaced into the signal_completion result so the
            agent (and downstream consumers) know what landed.
        warned: List of ``(artifact, error_message)`` tuples for
            artefacts whose ``on_error="warn"`` policy let them fail
            non-fatally.  Logged but the completion still succeeds.
        failed: List of ``(artifact, error_message)`` tuples for
            artefacts whose ``on_error="fail_completion"`` policy
            forces a hard failure.  Caller MUST treat the agent's
            ``signal_completion`` as failed and return a
            self-correction prompt.
    """
    written: List[str] = field(default_factory=list)
    warned: List[tuple] = field(default_factory=list)
    failed: List[tuple] = field(default_factory=list)


def _resolve_output_path(
    template: str,
    payload: Dict[str, Any],
    context: "RenderContext",
) -> str:
    """Substitute ``{field}`` placeholders in an output path template.

    Lookup precedence: payload fields → agent_params → a small set of
    session-derived values (``case_id``, ``workspace_path``).  Unknown
    placeholders raise ``KeyError`` deliberately — silent miss would
    write files to weirdly-named paths.

    Relative paths resolve under ``context.workspace_path`` so a
    template like ``output/{case_id}/policy.md`` lands inside the
    session's sandbox, not the daemon's cwd.
    """
    fields: Dict[str, Any] = {}
    if context.workspace_path:
        fields["workspace_path"] = context.workspace_path
    fields.update(context.agent_params or {})
    if isinstance(payload, dict):
        fields.update(payload)
    case_id_val = (
        payload.get("case_id") if isinstance(payload, dict) else None
    ) or context.agent_params.get("case_id")
    if case_id_val:
        fields["case_id"] = case_id_val

    rendered = template.format_map(fields)

    import os as _os
    if not _os.path.isabs(rendered) and context.workspace_path:
        rendered = _os.path.join(context.workspace_path, rendered)
    return rendered


def render_completion_artifacts(
    payload: Dict[str, Any],
    context: "RenderContext",
    artifacts: List[Any],
) -> ArtifactRenderResult:
    """Run profile-declared renderers and write artefact files.

    Output-side counterpart to :func:`expand_py_placeholders`: where
    that function injects script output INTO the agent's prompt, this
    one writes script output OUT to disk after the agent has finished
    its turn.  Called from
    :meth:`LifecycleTools._execute_signal_completion` after the
    payload validates against the profile's
    ``completion_payload_schema``.

    Each artifact is a ``CompletionArtifact`` (typed as ``Any`` here
    to avoid a top-level subagent-config import).  For each entry:

    1. Resolve ``renderer`` via the standard ``script_loader`` tier.
    2. Load its ``render(payload, context) -> str | bytes`` callable.
    3. Resolve the ``output`` template against payload + agent_params
       + session-derived fields (``case_id``, ``workspace_path``).
    4. Write the rendered content atomically (``.tmp`` + ``rename``).
    5. Bucket success / warned-error / failed-error per the entry's
       ``on_error`` policy.

    Args:
        payload: The agent's validated ``signal_completion`` payload.
        context: :class:`RenderContext` (same shape used by
            ``expand_py_placeholders`` on the input side).
        artifacts: List of ``CompletionArtifact`` entries from the
            session's profile.

    Returns:
        :class:`ArtifactRenderResult` summarising the outcome.
    """
    result = ArtifactRenderResult()
    if not artifacts:
        return result

    import os as _os

    for artifact in artifacts:
        renderer_ref = getattr(artifact, "renderer", None)
        output_template = getattr(artifact, "output", None)
        on_error = getattr(artifact, "on_error", "fail_completion")
        if not renderer_ref:
            err = f"missing renderer ({artifact!r})"
            (result.failed if on_error == "fail_completion" else result.warned).append(
                (artifact, err)
            )
            continue
        # output is optional — when None, this entry is a
        # validator-as-renderer (runs for side-effect, no file write).

        path = resolve_script_path(
            renderer_ref,
            workspace_path=context.workspace_path,
            config_root=context.config_root,
        )
        if path is None:
            err = f"renderer not found: {renderer_ref}"
            logger.warning("completion-artifact: %s", err)
            (result.failed if on_error == "fail_completion" else result.warned).append(
                (artifact, err)
            )
            continue

        fn = load_script_symbol(
            path, symbol="render", module_prefix="_jaato_artifact",
        )
        if fn is None:
            err = f"renderer load error: {renderer_ref}"
            (result.failed if on_error == "fail_completion" else result.warned).append(
                (artifact, err)
            )
            continue

        try:
            content = fn(payload, context)
        except Exception as exc:
            logger.exception(
                "completion-artifact: render failed for %s", renderer_ref,
            )
            err = f"renderer exception: {renderer_ref}: {exc}"
            (result.failed if on_error == "fail_completion" else result.warned).append(
                (artifact, err)
            )
            continue

        # Validator-as-renderer: ``output`` is None → run for side-effect,
        # log the (typically empty) return, do not write a file.  The
        # renderer's value here is whatever it raised (caught above) or
        # whatever it returned (logged).  ``on_error`` still governs how
        # raises propagate.
        if output_template is None:
            if isinstance(content, (str, bytes, bytearray)) and content:
                preview = (
                    content[:200] if isinstance(content, str)
                    else content[:200].decode("utf-8", errors="replace")
                )
                logger.info(
                    "completion-artifact: validator-only renderer %r emitted: %s",
                    renderer_ref, preview,
                )
            # Track as written-with-no-path so callers can count
            # validator-passed entries; the empty-string sentinel marks it.
            result.written.append("")
            continue

        try:
            output_path = _resolve_output_path(output_template, payload, context)
        except KeyError as exc:
            err = (
                f"output path template references unknown field {exc} "
                f"(template={output_template!r})"
            )
            (result.failed if on_error == "fail_completion" else result.warned).append(
                (artifact, err)
            )
            continue

        try:
            _os.makedirs(_os.path.dirname(output_path), exist_ok=True)
            tmp_path = output_path + ".tmp"
            mode = "wb" if isinstance(content, (bytes, bytearray)) else "w"
            kwargs: Dict[str, Any] = {"mode": mode}
            if mode == "w":
                kwargs["encoding"] = "utf-8"
            with open(tmp_path, **kwargs) as fh:
                fh.write(content)
            _os.replace(tmp_path, output_path)
            result.written.append(output_path)
        except OSError as exc:
            logger.exception(
                "completion-artifact: write failed for %s", output_path,
            )
            err = f"write failed: {output_path}: {exc}"
            (result.failed if on_error == "fail_completion" else result.warned).append(
                (artifact, err)
            )

    return result


def build_render_context(session: Any, agent_params: Optional[Dict[str, Any]] = None) -> RenderContext:
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
    return RenderContext(
        session=session,
        runtime=runtime,
        registry=registry,
        workspace_path=workspace_path,
        config_root=config_root,
        agent_params=dict(agent_params or {}),
        env=dict(os.environ),
    )
