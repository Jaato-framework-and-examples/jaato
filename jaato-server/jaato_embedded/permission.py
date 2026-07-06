"""Permission ``Channel`` for the embedded (in-process) facade.

The in-process analog of ``RunnerRPCChannel``: when a gated tool requests
permission, :meth:`InProcessChannel.request_permission` — running on the
embedded runtime's worker thread (the blocking ``send_message`` dispatched via
``asyncio.to_thread``) — emits a ``PERMISSION_REQUESTED`` event to the facade
(marshalled onto the event loop) and **blocks** until the facade answers via
``InProcessClient.respond_to_permission``, then maps the response string to a
:class:`ChannelResponse`.

No RPC hop: the daemon's ``RunnerRPCChannel`` does the same shape over the
runner→daemon RPC; this does it in-process via a ``threading.Event``
rendezvous. The response-string → decision mapping is reused verbatim from
``runner_rpc_channel`` so "y"/"n"/"a"/"t"/... mean exactly what they mean on
the daemon path.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Dict, Optional

from jaato_sdk.events import PermissionRequestedEvent
from shared.plugins.permission.channels import (
    Channel,
    ChannelDecision,
    ChannelResponse,
    PermissionRequest,
)
from shared.plugins.permission.runner_rpc_channel import _RESPONSE_KEY_TO_DECISION


class PendingPermissions:
    """Cross-thread rendezvous between ``request_permission`` (worker thread)
    and ``respond_to_permission`` (event loop).

    ``register`` (worker) creates a per-request ``threading.Event``; ``resolve``
    (loop) stores the response and sets the event; ``take`` (worker, after the
    wait) pops the stored response. Thread-safe.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: Dict[str, Dict[str, Any]] = {}

    def register(self, request_id: str) -> threading.Event:
        event = threading.Event()
        with self._lock:
            self._pending[request_id] = {"event": event, "response": None}
        return event

    def resolve(self, request_id: str, response: str) -> bool:
        """Store ``response`` for ``request_id`` and wake the waiter.

        Returns False if no request is pending under that id (already resolved
        or unknown) — so a stray response can't blow up the facade.
        """
        with self._lock:
            entry = self._pending.get(request_id)
            if entry is None:
                return False
            entry["response"] = response
            entry["event"].set()
        return True

    def take(self, request_id: str) -> Optional[str]:
        with self._lock:
            entry = self._pending.pop(request_id, None)
        return entry["response"] if entry is not None else None


def decision_for(response: Optional[str]) -> ChannelDecision:
    """Map a facade response string ("y"/"n"/"a"/"t"/...) to a
    ``ChannelDecision`` — the same table the daemon's runner channel uses.
    Unknown / missing responses fail closed to ``DENY``."""
    return _RESPONSE_KEY_TO_DECISION.get(
        str(response if response is not None else "n").strip().lower(),
        ChannelDecision.DENY,
    )


class InProcessChannel(Channel):
    """A permission ``Channel`` that bridges a gated tool's permission request
    to the in-process facade and back.

    Constructed with ``emit_threadsafe`` (marshals an event onto the loop) and a
    shared :class:`PendingPermissions` registry (also held by the
    ``InProcessClient`` so ``respond_to_permission`` resolves the same request).
    """

    def __init__(
        self,
        emit_threadsafe: Callable[[Any], None],
        pending: PendingPermissions,
    ) -> None:
        self._emit = emit_threadsafe
        self._pending = pending

    @property
    def name(self) -> str:
        return "in_process"

    def request_permission(self, request: PermissionRequest) -> ChannelResponse:
        event = self._pending.register(request.request_id)
        self._emit(
            PermissionRequestedEvent(
                agent_id=str((request.context or {}).get("agent_id", "") or "main"),
                request_id=request.request_id,
                tool_name=request.tool_name,
                tool_args=dict(request.arguments),
            )
        )
        # Block the worker thread until the facade answers. No timeout — the
        # facade's Session always responds (it denies when no on_permission is
        # set, rather than hang — convenience.py:_on_perm).
        event.wait()
        decision = decision_for(self._pending.take(request.request_id))
        return ChannelResponse(request_id=request.request_id, decision=decision)
