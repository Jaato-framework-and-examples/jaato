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
    base path (e.g. ``provider_trace_subagent_1.log`` instead of
    ``provider_trace.log``).  The main agent (agent_id ``"main"`` or
    ``None``) always writes to the base path.

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
import tempfile
import traceback as _traceback_module
from contextvars import ContextVar
from datetime import datetime
from typing import Optional, Set


# Cache of directories we've already ensured exist, to avoid
# repeated os.makedirs calls on every trace write.
_ensured_dirs: Set[str] = set()

# ContextVar holding the current agent ID for per-agent provider trace routing.
# ``None`` or ``"main"`` means use the base provider_trace.log; any other
# value (e.g. ``"subagent_1"``) produces a sibling file like
# ``provider_trace_subagent_1.log``.
_trace_agent_id: ContextVar[Optional[str]] = ContextVar(
    'trace_agent_id', default=None
)


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

    For the main agent (or no context), returns *base_path* unchanged.
    For subagents, inserts the agent ID before the file extension::

        /tmp/provider_trace.log  →  /tmp/provider_trace_subagent_1.log

    Args:
        base_path: The resolved base trace path (may be ``None``).

    Returns:
        Agent-specific path, or ``None`` if *base_path* is ``None``.
    """
    if not base_path:
        return base_path

    agent_id = _trace_agent_id.get()
    if not agent_id or agent_id == "main":
        return base_path

    root, ext = os.path.splitext(base_path)
    return f"{root}_{agent_id}{ext}"


def _ensure_parent_dirs(file_path: str) -> None:
    """Create parent directories for a file path if they don't exist."""
    parent = os.path.dirname(os.path.abspath(file_path))
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
        _ensure_parent_dirs(trace_path)
        with open(trace_path, "a") as f:
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

    Args:
        component: Component name for the log prefix.
        msg: Message to write.
        include_traceback: If True, append the current exception traceback.
    """
    path = resolve_trace_path("JAATO_TRACE_LOG",
                              default_filename="rich_client_trace.log")
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
