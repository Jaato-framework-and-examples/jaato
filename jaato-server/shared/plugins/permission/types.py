"""Typed payload shapes for the runner ↔ daemon permission RPC.

Phase 3 task §3.2.1 — generic ``client.prompt_operator`` primitive
(parent design §4.5).  These dataclasses cross the RPC wire as JSON
dicts (via :func:`to_dict` / :func:`from_dict`); the SDK's
:class:`PermissionRequestedEvent` / :class:`PermissionResponseRequest`
event classes carry the same field shapes when the daemon-side
handler emits to the connected client.

Why distinct types vs. the SDK events directly?  The daemon-side
handler accepts a generic ``PromptPayload`` (string + options +
session-id + tool-name) so future operator-interaction plugins
(interactive consent, free-text confirmation dialogs, etc.) can
re-use the same RPC primitive without adding a sibling RPC.  Per
parent §4.5: "the primitive intentionally takes a PromptPayload
... rather than permission-specific args."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PromptPayload:
    """Operator-prompt request crossing the runner→daemon RPC boundary.

    Mirrors the field shape of
    :class:`jaato_sdk.events.PermissionRequestedEvent` but stays
    decoupled from the event class so the RPC contract is stable
    against SDK-side schema additions.

    The daemon-side handler builds a
    :class:`PermissionRequestedEvent` from these fields, emits it to
    the connected client, and awaits the matching
    :class:`PermissionResponseRequest`.

    Attributes:
        request_id: Caller-allocated id correlating prompt → response.
            Phase 3 callers (the runner-side permission plugin)
            generate UUIDs; tests use simple strings.
        session_id: Session identifier — informational; the daemon's
            event-emit routing already knows which session this
            handler runs for.  Carried in the payload so audit/log
            lines surface the right session id even when the handler
            runs out-of-band.
        tool_name: Tool whose execution is being approved.
        tool_args: Args dict the tool was called with.  May be
            redacted by the runner-side stub before crossing the wire
            (when the permission policy declares an arg-redaction
            transformer; out of scope for §3.2.1).
        prompt_lines: Optional pre-formatted prompt text (line list)
            the runner already rendered.  When ``None`` the daemon
            renders a default prompt from tool_name + tool_args.
        response_options: List of response-key options the operator
            may choose (e.g. ``[{"key": "y", "label": "yes"}, ...]``).
            Same shape as ``PermissionRequestedEvent.response_options``.
        format_hint: Optional hint for the client renderer (e.g.
            ``"diff"`` for colored diff display).
        warnings: Optional security/analysis warnings to display
            separately from the prompt body.
        warning_level: ``"info"`` / ``"warning"`` / ``"error"``.
        agent_id: Agent identifier (for cascade workloads where
            multiple agents hit the same client).
    """

    request_id: str
    session_id: str
    tool_name: str
    tool_args: Dict[str, Any] = field(default_factory=dict)
    prompt_lines: Optional[List[str]] = None
    response_options: List[Dict[str, str]] = field(default_factory=list)
    format_hint: Optional[str] = None
    warnings: Optional[str] = None
    warning_level: Optional[str] = None
    agent_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "session_id": self.session_id,
            "tool_name": self.tool_name,
            "tool_args": dict(self.tool_args),
            "prompt_lines": list(self.prompt_lines) if self.prompt_lines else None,
            "response_options": [dict(o) for o in self.response_options],
            "format_hint": self.format_hint,
            "warnings": self.warnings,
            "warning_level": self.warning_level,
            "agent_id": self.agent_id,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PromptPayload":
        return cls(
            request_id=str(d.get("request_id", "")),
            session_id=str(d.get("session_id", "")),
            tool_name=str(d.get("tool_name", "")),
            tool_args=dict(d.get("tool_args") or {}),
            prompt_lines=list(d["prompt_lines"]) if d.get("prompt_lines") else None,
            response_options=list(d.get("response_options") or []),
            format_hint=d.get("format_hint"),
            warnings=d.get("warnings"),
            warning_level=d.get("warning_level"),
            agent_id=str(d.get("agent_id", "")),
        )


@dataclass
class PromptResponse:
    """Operator's response to a :class:`PromptPayload`.

    Mirrors :class:`jaato_sdk.events.PermissionResponseRequest`.

    Attributes:
        request_id: Echoes the payload's id so the caller can
            correlate against multi-prompt fan-outs.
        response: Response key (``"y"`` / ``"n"`` / ``"a"`` /
            ``"never"`` / etc.).
        edited_arguments: When the operator chose to edit the tool
            args before approving (response key ``"e"``), the new
            args dict.  ``None`` for non-edit responses.
        comment: Optional free-text comment the operator attached to
            the response (used by some channels for audit).
    """

    request_id: str
    response: str
    edited_arguments: Optional[Dict[str, Any]] = None
    comment: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "response": self.response,
            "edited_arguments": (
                dict(self.edited_arguments)
                if self.edited_arguments is not None
                else None
            ),
            "comment": self.comment,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PromptResponse":
        edited = d.get("edited_arguments")
        return cls(
            request_id=str(d.get("request_id", "")),
            response=str(d.get("response", "")),
            edited_arguments=dict(edited) if edited else None,
            comment=d.get("comment"),
        )
