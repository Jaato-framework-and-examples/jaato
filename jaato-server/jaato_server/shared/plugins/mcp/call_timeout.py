"""How long the MCP plugin waits for one tool call (#1646).

Stdlib only.  One rule, read by the plugin when a call starts and rendered
by ``jaato-scaffold explain plugin mcp`` from :data:`CALL_TIMEOUT_RULES`,
so the page cannot describe a precedence the plugin does not apply.

Until #1646 the wait was a literal 30 s with no knob, and a remote tool
whose normal answer took 60-90 s was discarded every time it ran long.

Precedence, highest first:

1. ``"timeout"`` on the server's ``.mcp.json`` entry;
2. ``plugin_configs.mcp.call_timeout``;
3. :data:`DEFAULT_CALL_TIMEOUT`.

``0`` means "no ceiling" (the provider-deadline convention).  A value that
is not a non-negative number is IGNORED with a WARNING and the next layer
decides: "unbounded" is never what a typo should buy.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)

#: Seconds, when neither the server entry nor the plugin config says.
DEFAULT_CALL_TIMEOUT = 30.0

#: The ``plugin_configs.mcp`` key.
PLUGIN_KEY = "call_timeout"

#: The key on one ``.mcp.json`` server entry.
SERVER_KEY = "timeout"

CALL_TIMEOUT_RULES: Tuple[str, ...] = (
    f"a tool call waits \"{SERVER_KEY}\" seconds from its .mcp.json server "
    f"entry, else plugin_configs.mcp.{PLUGIN_KEY}, else "
    f"{DEFAULT_CALL_TIMEOUT:g}",
    "0 means no ceiling; a value that is not a non-negative number is "
    "ignored with a WARNING and the next layer decides",
    "a call abandoned on that ceiling, or cancelled by the session, is "
    "cancelled on the server too (MCP notifications/cancelled), so the "
    "server stops working on an answer nobody will read",
    "each call has its own reply channel, so a late answer to an abandoned "
    "call is dropped rather than read by the next call",
)


def _seconds(value: Any) -> Optional[float]:
    """The value as non-negative seconds, or ``None`` when it is not one."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value != value or value < 0:  # NaN or negative
        return None
    return float(value)


def resolve_call_timeout(
    server_spec: Optional[Mapping[str, Any]],
    plugin_value: Any,
    *,
    server: str = "",
) -> Tuple[Optional[float], str]:
    """Return ``(seconds, source)`` for one call; ``seconds`` None = no ceiling.

    ``source`` names the layer that decided (``server``, ``plugin`` or
    ``default``), so the timeout message can say which knob to change.
    """
    layers = (
        ("server", (server_spec or {}).get(SERVER_KEY)
         if isinstance(server_spec, Mapping) else None,
         f".mcp.json server '{server}' \"{SERVER_KEY}\""),
        ("plugin", plugin_value, f"plugin_configs.mcp.{PLUGIN_KEY}"),
    )
    for source, raw, where in layers:
        if raw is None:
            continue
        seconds = _seconds(raw)
        if seconds is None:
            logger.warning(
                "MCP: %s=%r is not a non-negative number of seconds; "
                "ignored", where, raw)
            continue
        return (None if seconds == 0 else seconds), source
    return DEFAULT_CALL_TIMEOUT, "default"


def timeout_message(seconds: float, source: str, server: str) -> str:
    """The error a timed-out call returns: what happened and which knob."""
    if source == "server":
        knob = f"\"{SERVER_KEY}\" on the .mcp.json entry for '{server}'"
    elif source == "plugin":
        knob = f"plugin_configs.mcp.{PLUGIN_KEY}"
    else:
        knob = (f"plugin_configs.mcp.{PLUGIN_KEY}, or \"{SERVER_KEY}\" on the "
                f".mcp.json entry for '{server}'")
    return (f"MCP tool call timed out after {seconds:g}s and was cancelled on "
            f"the server; raise {knob} if this tool needs longer")
