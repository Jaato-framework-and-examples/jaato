"""Daemon-side ``client.prompt_operator`` handler (Phase 3 §3.2.1).

Receives a :class:`PromptPayload` from the runner-side permission
plugin, relays it to the connected client as a
:class:`PermissionRequestedEvent`, and awaits the matching
:class:`PermissionResponseRequest` from the client.  Returns a
:class:`PromptResponse` over the RPC.

Wire flow (parent design §4.5):

1. Runner-side permission plugin's ASK path constructs a
   ``PromptPayload``.
2. Runner-side ``rpc_client.prompt_operator(payload)`` sends the RPC.
3. **THIS handler** emits a ``PermissionRequestedEvent`` on the
   daemon's event channel and awaits the matching response on a
   request-id-keyed futures dict.
4. Connected client (TUI / WS / IPC) receives the event, surfaces
   the prompt, sends back ``PermissionResponseRequest``.
5. Daemon's transport layer routes the response into
   :meth:`PromptOperatorHandler.resolve_response`, which sets the
   future's result.
6. Step 3's await returns; the handler builds a ``PromptResponse``
   and returns it.

Channel-future correlation pattern: analogous to the in-process
``shared/plugins/permission/plugin.py:_get_channel()`` →
``wait_for_response()`` flow, but with the futures dict on the
handler instead of the channel — when the permission plugin moves
runner-side in §3.7, the channel goes with it; the daemon-side
handler is just the relay surface.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, Dict, Optional

from jaato_sdk.events import PermissionInputModeEvent, PermissionRequestedEvent

from shared.plugins.permission.types import PromptPayload, PromptResponse


logger = logging.getLogger(__name__)


# Type aliases for the daemon-side hooks the handler depends on.
# These are passed in at construction time so the handler is testable
# without standing up a full daemon — tests inject stubs.

# Emit an event to the connected client(s).  The real implementation
# calls JaatoServer.emit() (or SessionManager._emit_to_session) —
# both end up on the client's event stream.  Path J: widened from
# ``PermissionRequestedEvent`` to ``Any`` because the handler now
# emits TWO event types per ASK (the requested-event +
# the input-mode-control event).
EmitEventFn = Callable[[Any], None]


class PromptOperatorHandler:
    """Stateful handler for the ``client.prompt_operator`` RPC.

    Owns a futures dict keyed by ``request_id`` so concurrent ASKs
    (different sessions, or parallel tool calls in a cascade) can
    each resolve independently.

    Lifecycle:

    1. Constructed by ``_bootstrap_session(envelope)`` (§3.12.0) for
       each session that opts into AppArmor.
    2. Registered on the session's ``RunnerRPCServer`` as the
       ``client.prompt_operator`` handler.
    3. The session's response-routing layer (today: the IPC + WS
       transports' ``PermissionResponseRequest`` dispatch) calls
       :meth:`resolve_response` when a client response arrives.
    4. On session shutdown, :meth:`shutdown` cancels in-flight
       futures with a clean error.
    """

    def __init__(
        self,
        emit_event: EmitEventFn,
        *,
        prompt_timeout: Optional[float] = None,
    ) -> None:
        """Construct the handler.

        Args:
            emit_event: Callable that publishes a
                :class:`PermissionRequestedEvent` to the connected
                client.  In production this is bound to
                ``JaatoServer.emit`` or
                ``SessionManager._emit_to_session``.  Tests pass a
                stub.
            prompt_timeout: Optional wall-clock cap on how long a
                single prompt waits for a client response.  ``None``
                means no timeout — the prompt holds until the client
                attaches and responds, OR the handler shuts down.
                Phase 3 callers should generally NOT set this in
                production (the operator may walk away from a
                cascade for legitimate reasons); tests use a small
                value to avoid hanging.
        """
        self._emit_event = emit_event
        self._prompt_timeout = prompt_timeout
        self._pending: Dict[str, "asyncio.Future[PromptResponse]"] = {}
        self._closed = False

    async def handle(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """RPC handler entry point.

        Registers the request, emits the event, awaits the response,
        returns the dict-encoded :class:`PromptResponse`.  Raises on
        timeout or shutdown.
        """
        if self._closed:
            raise RuntimeError("PromptOperatorHandler is closed")

        try:
            payload = PromptPayload.from_dict(args)
        except (KeyError, ValueError) as exc:
            raise ValueError(f"prompt_operator: malformed payload: {exc}")

        if not payload.request_id:
            raise ValueError("prompt_operator: payload.request_id is required")
        if payload.request_id in self._pending:
            raise ValueError(
                f"prompt_operator: duplicate request_id={payload.request_id!r}"
            )

        loop = asyncio.get_running_loop()
        fut: "asyncio.Future[PromptResponse]" = loop.create_future()
        self._pending[payload.request_id] = fut

        # Build + emit the event.  The event-emit may throw — we
        # catch and clean the futures dict so a transport-level
        # failure doesn't leak pending entries.
        event = PermissionRequestedEvent(
            agent_id=payload.agent_id,
            request_id=payload.request_id,
            tool_name=payload.tool_name,
            tool_args=dict(payload.tool_args),
            response_options=list(payload.response_options),
            prompt_lines=list(payload.prompt_lines) if payload.prompt_lines else None,
            format_hint=payload.format_hint,
            warnings=payload.warnings,
            warning_level=payload.warning_level,
        )
        # Path J (cycle 12) Layer 10: companion control event that
        # switches the TUI's input mode to "answer the prompt".
        # Pre-§7c BOTH events were emitted from the daemon-side
        # permission plugin's hook (core.py:3105) — that path is
        # dead post-§7c since the runner-side permission plugin is
        # the one in the loop, and its ASK arrives HERE via the
        # PromptOperator RPC instead of via the daemon-side plugin.
        # The TUI listens for PermissionInputModeEvent (not
        # PermissionRequestedEvent) to enter permission mode; without
        # this emit the user's 'y' keypress is queued as a chat
        # message instead of an ASK response.
        #
        # Phase 4 §4.1 (J.A): ``call_id`` propagates from the runner-
        # side ASK origin via ``PromptPayload.call_id``.  TUI uses it
        # for per-tool-block correlation when multiple tools are in
        # flight simultaneously.
        #
        # Phase 4 §4.2 (J.B): ``editable_metadata`` propagates from
        # the runner-side ASK origin via
        # ``PromptPayload.editable_metadata``.  Runner-side already
        # resolved the editable schema via
        # ``PermissionRequest.editable`` (an ``EditableContent``
        # instance from the permission plugin's
        # ``check_permission`` flow at plugin.py:1411-1412); the
        # runner-RPC channel converted it to the canonical wire
        # shape (parameters + format).  Enables the TUI's edit-and-
        # approve flow without daemon-side schema coupling.
        input_mode_event = PermissionInputModeEvent(
            agent_id=payload.agent_id,
            request_id=payload.request_id,
            tool_name=payload.tool_name,
            call_id=payload.call_id,
            response_options=list(payload.response_options),
            tool_args=dict(payload.tool_args) if payload.tool_args else None,
            editable_metadata=(
                dict(payload.editable_metadata)
                if payload.editable_metadata is not None else None
            ),
        )
        try:
            self._emit_event(event)
            # Order matters: content event first, then the control
            # event that switches input mode.  Matches the pre-§7c
            # daemon-side emit order at core.py.
            self._emit_event(input_mode_event)
        except Exception:
            self._pending.pop(payload.request_id, None)
            raise

        try:
            if self._prompt_timeout is not None:
                response = await asyncio.wait_for(fut, self._prompt_timeout)
            else:
                response = await fut
            return response.to_dict()
        finally:
            self._pending.pop(payload.request_id, None)

    def resolve_response(
        self,
        request_id: str,
        response: str,
        *,
        edited_arguments: Optional[Dict[str, Any]] = None,
        comment: Optional[str] = None,
    ) -> bool:
        """Resolve the pending future for *request_id*.

        Called by the daemon's transport layer when a
        ``PermissionResponseRequest`` arrives from the connected
        client.  Returns ``True`` if a future was pending and got
        resolved; ``False`` if no future was waiting (e.g. the
        client responded after the prompt timed out, or the runner
        cancelled the call before the response arrived).
        """
        fut = self._pending.get(request_id)
        if fut is None or fut.done():
            return False
        prompt_response = PromptResponse(
            request_id=request_id,
            response=response,
            edited_arguments=edited_arguments,
            comment=comment,
        )
        fut.set_result(prompt_response)
        return True

    def shutdown(self) -> None:
        """Cancel all in-flight prompts with a clean error.

        Called when the session ends or the daemon shuts down.
        Pending futures get a ``RuntimeError("handler shutting down")``;
        callers blocked in
        :meth:`server.runner.rpc.RunnerRPC.outgoing_call` see this as
        the future's exception and translate it into a clean error
        envelope on the runner side.
        """
        if self._closed:
            return
        self._closed = True
        for request_id, fut in list(self._pending.items()):
            if not fut.done():
                fut.set_exception(
                    RuntimeError(
                        f"prompt_operator: shutting down "
                        f"(in-flight request_id={request_id!r})"
                    )
                )
        self._pending.clear()


def register(
    rpc_server: Any,  # server.runner_rpc_server.RunnerRPCServer
    handler: PromptOperatorHandler,
) -> None:
    """Register *handler*'s ``handle`` method as the
    ``client.prompt_operator`` RPC method on *rpc_server*.

    Convenience indirection so the bootstrap site (today's
    ``_spawn_session_runner``, Phase 3 ``_bootstrap_session``)
    doesn't repeat the method-name string literal.
    """
    rpc_server.register("client.prompt_operator", handler.handle)
