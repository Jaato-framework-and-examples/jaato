"""Shared trace logging utility.

Provides a single place for trace file writing with automatic parent
directory creation. Replaces the duplicated _trace() pattern across
36+ files.

Two trace channels:
- Application trace: JAATO_TRACE_LOG (plugins, client, session)
- Provider trace: JAATO_PROVIDER_TRACE (model provider SDKs)

Per-agent provider trace:
    When an agent context is set via ``set_trace_agent_context()``,
    ``provider_trace()`` writes to a per-agent file derived from the
    base path.  Two forms, and the path itself selects which:

    - **Explicit** — the path names a placeholder from
      :data:`TRACE_PATH_PLACEHOLDERS` (``{agent}`` / ``{agent_suffix}``),
      which is substituted wherever the author put it::

          logs/{agent}/provider.log   ->  logs/subagent_1/provider.log
          provider{agent_suffix}.log  ->  provider_subagent_1.log

    - **Implicit** — the path names no placeholder, and the agent id is
      appended before the extension, as it always has been::

          provider_trace.log          ->  provider_trace_subagent_1.log

    The main agent (agent_id ``"main"`` or ``None``) writes to the base
    path under the implicit form; under the explicit form it renders
    ``{agent}`` as ``main`` and ``{agent_suffix}`` as the empty string,
    so an author who asks for the id gets it for every agent.

Usage:
    from jaato_sdk.trace import trace, provider_trace, trace_write, resolve_trace_path

    # Most plugins (writes to JAATO_TRACE_LOG):
    trace("MyPlugin", "some message")
    trace("MyPlugin", "error occurred", include_traceback=True)

    # Provider plugins (writes to JAATO_PROVIDER_TRACE):
    provider_trace("google_genai", "streaming chunk received")

    # Custom path resolution (e.g. jaato_client checks both env vars):
    path = resolve_trace_path("JAATO_TRACE_LOG", "JAATO_PROVIDER_TRACE",
                              default_filename="provider_trace.log")
    trace_write("jaato_client", msg, path)
"""

import os
import re
import tempfile
import traceback as _traceback_module
from contextvars import ContextVar
from datetime import datetime
from typing import Dict, Optional, Set


# Cache of directories we've already ensured exist, to avoid
# repeated os.makedirs calls on every trace write.
_ensured_dirs: Set[str] = set()

# ContextVar holding the current agent ID for per-agent provider trace routing.
# ``None`` or ``"main"`` means the main agent; any other value (e.g.
# ``"subagent_1"``) routes to a per-agent file — see _agent_trace_path.
_trace_agent_id: ContextVar[Optional[str]] = ContextVar(
    'trace_agent_id', default=None
)

#: The agent id used when no agent context is set.  Named rather than
#: repeated so the profile-side validator and ``jaato-scaffold explain``
#: can quote the real value instead of restating it.
MAIN_AGENT_ID = "main"

#: Placeholders a trace path may carry, resolved by the READER (here) at
#: write time -- NOT by ``expand_variables`` at profile-resolution.
#:
#: That split is the whole reason these exist as their own vocabulary with
#: their own ``{...}`` syntax.  ``${VAR}`` is substituted daemon-side, once,
#: while a profile is being resolved; the agent writing a given trace line is
#: not known until the line is written, and differs between concurrent threads
#: of ONE session.  A ``${agent}`` would therefore have to be a context var
#: expand_variables cannot supply, and would sit in a syntax that promises
#: resolution at a time this value does not exist.  Two syntaxes, two
#: resolution times, no overlap.
#:
#: THE REGISTRY IS THE CONTRACT.  ``shared.plugins.subagent.config`` refuses a
#: profile naming a token that is not in here (a typo'd ``{agent_id}`` would
#: otherwise survive validation and be created as a literal directory --
#: exactly the #775 shape), and ``jaato-scaffold explain`` renders the table
#: from it.  Adding a placeholder therefore means adding it HERE and nowhere
#: else.
TRACE_PATH_PLACEHOLDERS: Dict[str, str] = {
    "{agent}": ("the agent writing the line -- `main` for the main agent, "
                "`subagent_1`, `subagent_2`, ... for spawned subagents"),
    "{agent_suffix}": ("`_subagent_1` for a subagent and EMPTY for the main "
                       "agent -- what the implicit form appends, placeable "
                       "anywhere in the path"),
}

#: Matches any ``{token}`` so an unknown one can be reported rather than
#: silently surviving into a directory name.  Deliberately not anchored to the
#: known set: the point is to SEE ``{agent_id}``, not to skip it.
#:
#: The ``(?<!\$)`` is load-bearing and was found by the first test written
#: against it: ``${HOME}`` CONTAINS ``{HOME}``, so without the lookbehind
#: every legitimate variable reference reads as an unknown placeholder and a
#: correct profile is refused at load.  The two vocabularies overlap
#: textually and must be told apart by the ``$``.
_PLACEHOLDER_TOKEN_RE = re.compile(r"(?<!\$)\{[A-Za-z_][A-Za-z0-9_]*\}")

#: Characters an agent id may contribute to a filename.  An id reaches this
#: module from ``set_trace_agent_context`` and is framework-generated
#: (``subagent_<n>``), but it now lands wherever an AUTHOR put ``{agent}`` --
#: mid-path, not only in a suffix -- so a separator in it would silently
#: redirect the write.  Substituted, never rejected: a trace path is a
#: diagnostic, and losing the diagnosis to a validation error is the worse
#: outcome.
_UNSAFE_AGENT_CHARS = re.compile(r"[^A-Za-z0-9_.-]")


def unknown_trace_placeholders(path: str) -> list:
    """Return the ``{token}`` placeholders in *path* that nothing resolves.

    The one shared answer to "is this trace path's vocabulary real?", called
    by the profile loader (which refuses) and by ``jaato-scaffold validate``
    (which reports).  Both ask this module rather than restating the set.

    Args:
        path: A trace path, expanded or not.

    Returns:
        Sorted, de-duplicated unknown tokens (``["{agent_id}"]``), empty when
        every token present is in :data:`TRACE_PATH_PLACEHOLDERS`.
    """
    if not path or "{" not in path:
        return []
    found = set(_PLACEHOLDER_TOKEN_RE.findall(path))
    return sorted(found - set(TRACE_PATH_PLACEHOLDERS))


def set_trace_agent_context(agent_id: Optional[str] = None) -> None:
    """Set the agent ID for per-agent provider trace routing.

    Call this at the start of a thread or task to direct subsequent
    ``provider_trace()`` calls to an agent-specific file.

    Args:
        agent_id: Agent identifier (e.g. ``"main"``, ``"subagent_1"``).
            ``None`` or ``"main"`` routes to the default provider_trace.log.
    """
    _trace_agent_id.set(agent_id)


def clear_trace_agent_context() -> None:
    """Clear the agent trace context, reverting to the default log file."""
    _trace_agent_id.set(None)


def _agent_trace_path(base_path: Optional[str]) -> Optional[str]:
    """Derive a per-agent trace path from a base path.

    Two forms, selected by the path itself — an author who names a
    placeholder is asking for it to be honoured *where they put it*, and
    appending the id as well would write to a file neither form describes::

        EXPLICIT (path names a placeholder from TRACE_PATH_PLACEHOLDERS)
            logs/{agent}/provider.log   →  logs/subagent_1/provider.log
            provider{agent_suffix}.log  →  provider_subagent_1.log
            provider{agent_suffix}.log  →  provider.log          (main agent)

        IMPLICIT (no placeholder — the historical behaviour, unchanged)
            /tmp/provider_trace.log     →  /tmp/provider_trace_subagent_1.log
            /tmp/provider_trace.log     →  /tmp/provider_trace.log  (main agent)

    The forms differ deliberately for the main agent: implicit leaves the
    path alone (so every pre-existing deployment keeps writing where it
    always did), while explicit renders ``{agent}`` as ``main``, because an
    author who asked for the id in the name wants it on every file rather
    than one anonymous file among named siblings.

    Unknown tokens are left alone rather than substituted or rejected — the
    profile loader refuses them at load and ``jaato-scaffold validate``
    reports them, both via :func:`unknown_trace_placeholders`, which are the
    two places that can say so usefully.  Here, tracing must not raise.

    Args:
        base_path: The resolved base trace path (may be ``None``).

    Returns:
        Agent-specific path, or ``None`` if *base_path* is ``None``.
    """
    if not base_path:
        return base_path

    if _has_agent_placeholder(base_path):
        return _substitute_agent_placeholders(base_path)

    agent_id = _current_agent_id()
    if agent_id == MAIN_AGENT_ID:
        return base_path

    root, ext = os.path.splitext(base_path)
    return f"{root}_{agent_id}{ext}"


def _current_agent_id() -> str:
    """The active agent id, filename-safe.

    Defaults to :data:`MAIN_AGENT_ID` when no context is set, and replaces
    any character that is not safe in a path segment (see
    :data:`_UNSAFE_AGENT_CHARS`).
    """
    raw_id = _trace_agent_id.get() or MAIN_AGENT_ID
    safe = _UNSAFE_AGENT_CHARS.sub("_", raw_id)
    # `.` and `..` survive the character filter (both are legal in a filename)
    # and are the two segments that still MEAN something to a path resolver —
    # `logs/{agent}/p.log` with an id of `..` writes one directory up.  The
    # character class closes separators; this closes the rest.
    if safe in ("", ".", ".."):
        return MAIN_AGENT_ID
    return safe


#: The known placeholders, as one alternation with the same ``$`` lookbehind
#: :data:`_PLACEHOLDER_TOKEN_RE` uses -- so ``${agent}`` (a variable named
#: ``agent``) is left for ``expand_variables`` instead of being half-rewritten
#: into ``$subagent_1``.
_KNOWN_PLACEHOLDER_RE = re.compile(
    r"(?<!\$)\{(" + "|".join(
        re.escape(t[1:-1]) for t in sorted(TRACE_PATH_PLACEHOLDERS)) + r")\}")


def _has_agent_placeholder(path: str) -> bool:
    """Whether *path* names any placeholder this module resolves."""
    return bool(_KNOWN_PLACEHOLDER_RE.search(path))


def _substitute_agent_placeholders(path: Optional[str]) -> Optional[str]:
    """Replace every known ``{token}`` in *path* with the active agent's value.

    Substitution only — never the implicit suffix.  This is the whole of what
    the SESSION trace channel does (:func:`trace`), so an author who writes
    ``{agent}`` into ``JAATO_TRACE_LOG`` gets it honoured while a path that
    names no placeholder keeps landing exactly where it always did.  The
    PROVIDER channel wraps this with the implicit-suffix fallback it has
    always had (:func:`_agent_trace_path`).

    Args:
        path: A resolved trace path (may be ``None``).

    Returns:
        The substituted path, or *path* unchanged when it names no
        placeholder (or is ``None``).
    """
    if not path or not _has_agent_placeholder(path):
        return path
    agent_id = _current_agent_id()
    values = {
        "agent": agent_id,
        "agent_suffix": "" if agent_id == MAIN_AGENT_ID else f"_{agent_id}",
    }
    return _KNOWN_PLACEHOLDER_RE.sub(lambda m: values[m.group(1)], path)


def _resolve_trace_file(file_path: str) -> str:
    """Resolve a trace file path, using JAATO_WORKSPACE_ROOT for relative paths.

    Relative paths (like ``.jaato/logs/provider_trace.log`` from a ``.env``
    file) should resolve against the session workspace, not the server
    process's ``cwd()``.
    """
    if os.path.isabs(file_path):
        return file_path
    workspace = os.environ.get("JAATO_WORKSPACE_ROOT")
    if workspace:
        return os.path.join(workspace, file_path)
    return os.path.abspath(file_path)


def _ensure_parent_dirs(file_path: str) -> None:
    """Create parent directories for a file path if they don't exist."""
    parent = os.path.dirname(file_path)
    if parent not in _ensured_dirs:
        os.makedirs(parent, exist_ok=True)
        _ensured_dirs.add(parent)


def resolve_trace_path(
    *env_vars: str,
    default_filename: str = "rich_client_trace.log",
) -> Optional[str]:
    """Resolve trace file path from environment variables.

    Checks env vars in order. An empty string value means tracing is
    explicitly disabled. If no env var is set, falls back to a file
    in the system temp directory.

    Args:
        *env_vars: Environment variable names to check, in priority order.
        default_filename: Fallback filename in temp directory.

    Returns:
        Resolved file path, or None if tracing is disabled.
    """
    for var in env_vars:
        value = os.environ.get(var)
        if value == "":
            return None  # Explicitly disabled
        if value:
            return value

    # No env var set - use default in temp directory
    return os.path.join(tempfile.gettempdir(), default_filename)


def trace_write(
    component: str,
    msg: str,
    trace_path: Optional[str],
    *,
    include_traceback: bool = False,
) -> None:
    """Write a trace message to the given path.

    Creates parent directories automatically. Never raises - tracing
    errors are silently ignored to avoid breaking the application.

    Args:
        component: Component name for the log prefix (e.g., "MCP", "jaato_client").
        msg: Message to write.
        trace_path: File path to write to. If None, does nothing.
        include_traceback: If True, append the current exception traceback.
    """
    if not trace_path:
        return
    try:
        resolved = _resolve_trace_file(trace_path)
        _ensure_parent_dirs(resolved)
        with open(resolved, "a") as f:
            ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            f.write(f"[{ts}] [{component}] {msg}\n")
            if include_traceback:
                tb = _traceback_module.format_exc()
                if tb and tb.strip() != "NoneType: None":
                    f.write(f"[{ts}] [{component}] Traceback:\n{tb}\n")
            f.flush()
    except Exception:
        pass  # Never let tracing errors break the application


def trace(
    component: str,
    msg: str,
    *,
    include_traceback: bool = False,
) -> None:
    """Write a trace message to the application trace log.

    Resolves path from JAATO_TRACE_LOG env var.
    Fallback: rich_client_trace.log in temp directory.

    Placeholders from :data:`TRACE_PATH_PLACEHOLDERS` are substituted, so a
    session trace can be split per agent the same way a provider trace can.
    Unlike :func:`provider_trace` this channel applies NO implicit suffix: a
    session trace is one conversation's story and joining a subagent's lines
    into it is usually what an operator wants, so splitting it is opt-in by
    writing the placeholder.

    Args:
        component: Component name for the log prefix.
        msg: Message to write.
        include_traceback: If True, append the current exception traceback.
    """
    path = _substitute_agent_placeholders(
        resolve_trace_path("JAATO_TRACE_LOG",
                           default_filename="rich_client_trace.log"))
    trace_write(component, msg, path, include_traceback=include_traceback)


def provider_trace(
    component: str,
    msg: str,
    *,
    include_traceback: bool = False,
) -> None:
    """Write a trace message to the provider trace log.

    Resolves the base path from JAATO_PROVIDER_TRACE env var (fallback:
    ``provider_trace.log`` in temp directory), then derives a per-agent
    path when an agent context is active (see
    :func:`set_trace_agent_context`).

    Args:
        component: Component name for the log prefix.
        msg: Message to write.
        include_traceback: If True, append the current exception traceback.
    """
    base_path = resolve_trace_path("JAATO_PROVIDER_TRACE",
                                   default_filename="provider_trace.log")
    path = _agent_trace_path(base_path)
    trace_write(component, msg, path, include_traceback=include_traceback)
