"""Client-side rendering of a ``PermissionRequestedEvent``'s content.

WHY THIS EXISTS.  A permission ASK reaches the TUI on two shapes of wire
traffic, and until this module the TUI rendered only one of them:

* **Daemon-local sessions**: the daemon's own permission hook renders the
  prompt (summary, the diff for a file edit, analyzer warnings) into an
  ``AgentOutputEvent`` with ``source="permission"``, then emits a
  ``PermissionInputModeEvent``.  ``OutputBuffer.append("permission", ...)``
  buffers the text and ``set_tool_awaiting_approval`` attaches it to the
  tool awaiting approval.
* **Runner-tier sessions** (the pre-warm pool / confined runner, on by
  default): the runner relays the ASK over the ``client.prompt_operator``
  RPC and the daemon emits a ``PermissionRequestedEvent`` carrying
  ``prompt_lines`` / ``format_hint`` / ``warnings`` / ``warning_level``,
  then the input-mode event.  No ``AgentOutputEvent`` is ever emitted on
  this path, and the TUI had no branch for the requested event, so the
  person saw "Permission required" and the options bar with nothing
  between them -- no diff, no warning.

So the requested event is rendered into the SAME text the daemon-local
hook would have produced, and appended under the same ``permission``
source, which is what lets ``set_tool_awaiting_approval`` attach it to
the tool exactly as before.  The security-warning block uses the marker
``OutputBuffer._extract_and_render_security_warnings`` already parses.
"""

from typing import Any, Callable, Iterable, Optional

from jaato_sdk.events import PermissionRequestedEvent


def security_warning_block(warnings: Optional[str], warning_level: Optional[str]) -> str:
    """The ``<security-warning level="...">`` block the buffer styles.

    Byte-compatible with the block the daemon-local hook in
    ``server/core.py`` emits, so one parser serves both wires.  Empty when
    there is nothing to warn about.
    """
    if not warnings:
        return ""
    level = warning_level or "warning"
    return f'<security-warning level="{level}">\n{warnings}\n</security-warning>\n'


def permission_prompt_content(
    prompt_lines: Optional[Iterable[str]],
    warnings: Optional[str] = None,
    warning_level: Optional[str] = None,
) -> str:
    """Join a requested event's content into one ``permission`` block.

    Returns the empty string when the event carries neither lines nor a
    warning, so the caller appends nothing and the tool tree renders the
    bare header it always did rather than an empty content block.
    """
    parts = []
    block = security_warning_block(warnings, warning_level)
    if block:
        parts.append(block)
    lines = [str(line) for line in (prompt_lines or [])]
    if lines:
        parts.append("\n".join(lines))
    return "\n".join(parts)


def render_permission_requested(event, agent_registry) -> bool:
    """Append a ``PermissionRequestedEvent``'s content to the agent's buffer.

    Resolves the buffer the way the batch-clarification path does -- by
    ``agent_id`` when it names one, else the selected buffer -- because a
    relayed ASK carries the plugin's agent name, often empty for the main
    agent, and a miss must not swallow the prompt.  Returns whether
    anything was appended.
    """
    content = permission_prompt_content(
        getattr(event, "prompt_lines", None),
        getattr(event, "warnings", None),
        getattr(event, "warning_level", None),
    )
    if not content:
        return False
    agent_id = getattr(event, "agent_id", "") or ""
    buffer = agent_registry.get_buffer(agent_id) if agent_id else None
    if buffer is None:
        buffer = agent_registry.get_selected_buffer()
    if buffer is None:
        return False
    buffer.append("permission", content, "write")
    return True


def maybe_render_permission_requested(
    event: Any,
    agent_registry: Any,
    display: Any,
    trace: Callable[[str], None] = lambda _msg: None,
) -> bool:
    """The event-loop entry point: a no-op for every other event.

    Called unconditionally for each event the loop receives rather than
    as a branch of its ``isinstance`` chain -- ``handle_events`` is frozen
    at the top of the cyclomatic-complexity ratchet, so the branch lives
    here.  Refreshes the display when something was appended.
    """
    if not isinstance(event, PermissionRequestedEvent):
        return False
    trace(f"  PermissionRequestedEvent: tool={event.tool_name}, id={event.request_id}, lines={len(event.prompt_lines or [])}")
    if not render_permission_requested(event, agent_registry):
        return False
    display.refresh()
    return True
