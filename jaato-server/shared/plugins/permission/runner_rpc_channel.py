"""Runner-side ASK relay channel (Phase 3 §3.7 deeper).

When the permission plugin runs runner-side, the daemon-side
``Channel`` subclasses (ConsoleChannel / WebhookChannel / etc.)
can't reach the connected client directly — the runner is in a
kernel-confined process with no client connection.  ASK decisions
must relay through the daemon's ``client.prompt_operator`` RPC
primitive (§3.2.1).

This module provides :class:`RunnerRPCChannel`, a :class:`Channel`
implementation whose ``request_permission`` builds a
:class:`PromptPayload`, calls the runner-side
``RunnerRPCClient.prompt_operator(payload)`` (which transports via
the existing bidirectional RPC channel from §3.2), translates the
returned :class:`PromptResponse` to a :class:`ChannelResponse`,
and returns it to the caller.

The runner-side permission plugin selects this channel by default
when it detects a runner-side execution context (`registry.runner_rpc_client`
attribute set by the runner's ``__main__``).  Falls back to the
daemon-side channels when no RPC client is attached — preserving
in-process behaviour for non-runner / pre-Phase-3 sessions.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from .channels import (
    Channel,
    ChannelDecision,
    ChannelResponse,
    PermissionRequest,
    PermissionResponseOption,
)


logger = logging.getLogger(__name__)


# Type alias for the prompt-operator callable.  The runner's
# :class:`server.runner.rpc_client.RunnerRPCClient.prompt_operator`
# matches this signature; tests inject a stub.
PromptOperatorFn = Callable[
    [Any],  # PromptPayload (forward type to avoid circular import)
    Any,    # PromptResponse
]


# Map of response-key strings (the value the operator's client sends
# back via ``PermissionResponseRequest.response``) to ChannelDecision.
# Mirrors the in-process channels' parsing of ``ConsoleChannel`` /
# ``QueueChannel``.  When the wire response is not in this table the
# channel defaults to DENY — same posture as the in-process channels
# treat ambiguous input.
_RESPONSE_KEY_TO_DECISION: Dict[str, ChannelDecision] = {
    "y": ChannelDecision.ALLOW,
    "yes": ChannelDecision.ALLOW,
    "allow": ChannelDecision.ALLOW,
    "n": ChannelDecision.DENY,
    "no": ChannelDecision.DENY,
    "deny": ChannelDecision.DENY,
    "a": ChannelDecision.ALLOW_SESSION,
    "always": ChannelDecision.ALLOW_SESSION,
    "allow_session": ChannelDecision.ALLOW_SESSION,
    "never": ChannelDecision.DENY_SESSION,
    "deny_session": ChannelDecision.DENY_SESSION,
    "all": ChannelDecision.ALLOW_ALL,
    "allow_all": ChannelDecision.ALLOW_ALL,
    "t": ChannelDecision.ALLOW_TURN,
    "turn": ChannelDecision.ALLOW_TURN,
    "allow_turn": ChannelDecision.ALLOW_TURN,
    "i": ChannelDecision.ALLOW_UNTIL_IDLE,
    "idle": ChannelDecision.ALLOW_UNTIL_IDLE,
    "allow_until_idle": ChannelDecision.ALLOW_UNTIL_IDLE,
    "once": ChannelDecision.ALLOW_ONCE,
    "allow_once": ChannelDecision.ALLOW_ONCE,
    "e": ChannelDecision.EDIT,
    "edit": ChannelDecision.EDIT,
    "c": ChannelDecision.COMMENT,
    "comment": ChannelDecision.COMMENT,
    # The option is labelled "deny-comment" so it reads as the pair of
    # "allow-comment"; "comment" stays accepted for older clients.
    "deny-comment": ChannelDecision.COMMENT,
    "deny_comment": ChannelDecision.COMMENT,
    "yc": ChannelDecision.ALLOW_COMMENT,
    "allow-comment": ChannelDecision.ALLOW_COMMENT,
    "allow_comment": ChannelDecision.ALLOW_COMMENT,
}


class RunnerRPCChannel(Channel):
    """Permission channel that relays ASK decisions via runner-RPC.

    Phase 3 §3.7 deeper.  When the permission plugin runs runner-
    side and the model invokes a tool whose policy returns "ASK"
    (no static rule, no evaluator decision), the plugin's
    ``_get_channel()`` returns this channel.  ``request_permission``
    builds a :class:`PromptPayload`, sends it to the daemon via the
    runner-side ``RunnerRPCClient.prompt_operator(payload)``
    method, and translates the daemon's response back into a
    :class:`ChannelResponse`.

    Lifecycle:
    - Constructed by the runner-side permission plugin's
      ``_get_channel`` lookup.  The plugin checks
      ``registry.runner_rpc_client`` for an attached runner-RPC
      wrapper; when present, an instance of this channel is used.
    - Channel is stateless beyond the captured ``prompt_operator``
      callable + an optional default response-options list.

    Errors:
    - When the daemon-side handler reports an error envelope, the
      RPC wrapper raises ``RunnerRPCError``.  This channel catches
      and returns a DENY ``ChannelResponse`` with the error in the
      reason field — same posture as the in-process channels treat
      transport / handler failures.
    - When the daemon-side ``prompt_operator`` future is set with
      an exception (e.g., the handler shutdown mid-prompt), the
      same DENY-with-reason posture applies.
    """

    def __init__(
        self,
        prompt_operator: PromptOperatorFn,
        *,
        default_options: Optional[List[PermissionResponseOption]] = None,
    ) -> None:
        """Construct the channel.

        Args:
            prompt_operator: Callable that takes a
                :class:`PromptPayload` and returns a
                :class:`PromptResponse`.  In production this is
                bound to the runner-side
                :meth:`RunnerRPCClient.prompt_operator`; tests pass
                a stub.
            default_options: Optional override for the response
                options the daemon-side handler advertises to the
                client.  When ``None``, falls back to the request's
                own ``response_options`` list (which itself
                defaults to the framework's standard options).
        """
        self._prompt_operator = prompt_operator
        self._default_options = default_options

    @property
    def name(self) -> str:
        return "runner_rpc"

    def request_permission(
        self, request: PermissionRequest,
    ) -> ChannelResponse:
        """Relay the ASK to the daemon via ``client.prompt_operator``.

        Translates ``PermissionRequest`` → ``PromptPayload``, calls
        the RPC, translates ``PromptResponse`` → ``ChannelResponse``.

        Args:
            request: The permission request to escalate to the
                operator.

        Returns:
            ``ChannelResponse`` with the operator's decision, OR
            a DENY response with an error reason if the relay
            failed.
        """
        # Lazy-import to avoid a circular dependency at module load
        # time (PromptPayload lives alongside the daemon-side
        # handler in shared/plugins/permission/types.py, which may
        # import channels itself in the future).
        from shared.plugins.permission.types import (
            PromptPayload,
            PromptResponse,
        )

        # Build the payload.  Session-id + agent-id are populated
        # by the runner-side permission plugin when it constructs
        # the request; carried through ``request.context`` to keep
        # the Channel ABC unchanged.
        session_id = ""
        agent_id = ""
        call_id: Optional[str] = None
        if isinstance(request.context, dict):
            session_id = str(request.context.get("session_id", "") or "")
            agent_id = str(request.context.get("agent_id", "") or "")
            # Phase 4 §4.1 (J.A): tool-call correlator threaded via
            # context dict (same channel-ABC-stable pattern as
            # session_id/agent_id).  The runner-side permission
            # plugin populates this when a permission check fires
            # from inside an active tool call.  None for ASKs not
            # bound to a specific call.
            ctx_call_id = request.context.get("call_id")
            if isinstance(ctx_call_id, str) and ctx_call_id:
                call_id = ctx_call_id

        # Choose the response-options list the daemon-side handler
        # will advertise to the connected client.
        options = self._default_options or request.response_options
        response_options_dicts = [
            {
                "key": opt.short,
                "label": opt.full,
                "description": opt.description,
            }
            for opt in options
        ]

        # Phase 4 §4.2 (J.B): convert request.editable
        # (EditableContent | None, already resolved by the permission
        # plugin via _get_tool_schema at check_permission time —
        # plugin.py:1411-1412) into the canonical wire-shape dict.
        # Mirrors the pre-§7c daemon-side hook at core.py:3062-3072.
        # Template field NOT included — it's an editor-rendering
        # concern that the runner consumes locally; daemon/TUI need
        # only parameters + format for input-mode signaling.
        editable_metadata: Optional[Dict[str, Any]] = None
        editable = getattr(request, "editable", None)
        if editable is not None:
            try:
                editable_metadata = {
                    "parameters": list(
                        getattr(editable, "parameters", None) or []
                    ),
                    "format": str(getattr(editable, "format", "yaml") or "yaml"),
                }
            except Exception:  # noqa: BLE001 — boundary; never fail
                # the ASK relay because of a malformed editable.
                editable_metadata = None

        payload = PromptPayload(
            request_id=request.request_id,
            session_id=session_id,
            tool_name=request.tool_name,
            tool_args=dict(request.arguments),
            response_options=response_options_dicts,
            agent_id=agent_id,
            call_id=call_id,
            editable_metadata=editable_metadata,
        )

        try:
            response: PromptResponse = self._prompt_operator(payload)
        except Exception as exc:  # noqa: BLE001 — RPC transport boundary
            logger.warning(
                "RunnerRPCChannel: prompt_operator raised %s — "
                "denying permission request %s",
                type(exc).__name__, request.request_id,
            )
            return ChannelResponse(
                request_id=request.request_id,
                decision=ChannelDecision.DENY,
                reason=f"runner-RPC ASK relay failed: {exc}",
            )

        decision = _RESPONSE_KEY_TO_DECISION.get(
            (response.response or "").lower().strip(),
            ChannelDecision.DENY,
        )

        was_edited = (
            response.edited_arguments is not None
            and decision in (
                ChannelDecision.EDIT,
                ChannelDecision.ALLOW,
                ChannelDecision.ALLOW_ONCE,
                ChannelDecision.ALLOW_SESSION,
                ChannelDecision.ALLOW_TURN,
                ChannelDecision.ALLOW_UNTIL_IDLE,
                ChannelDecision.ALLOW_ALL,
                ChannelDecision.ALLOW_COMMENT,
            )
        )

        return ChannelResponse(
            request_id=request.request_id,
            decision=decision,
            reason=response.comment or "",
            edited_arguments=(
                dict(response.edited_arguments)
                if response.edited_arguments
                else None
            ),
            was_edited=was_edited,
        )
