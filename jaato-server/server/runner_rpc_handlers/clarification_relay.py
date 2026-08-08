"""Daemon-side ``client.request_clarification`` handler.

Symmetric counterpart of
:mod:`server.runner_rpc_handlers.prompt_operator` (the permission ASK
relay), for clarification.  Post-seat-flip the clarification plugin
runs runner-side; its ``QueueChannel`` has no daemon connection (the
daemon-side ``_setup_queue_channels`` wiring targets the daemon plugin
instance, not the runner one), so a runner-tier (confined / pool)
session could never deliver a clarification to the connected client —
``request_clarification`` returned ``cancelled`` immediately.  This
handler closes that gap exactly the way ``prompt_operator`` closed it
for permissions.

Wire flow:

1. Runner-side clarification plugin resolves a
   :class:`shared.plugins.clarification.runner_rpc_channel.RunnerRPCClarificationChannel`
   (preferred when ``registry.runner_rpc_client`` is present) and calls
   ``request_clarification`` on it.
2. The channel builds a batch payload and calls
   ``rpc_client.request_clarification(payload)``.
3. **THIS handler** emits a :class:`ClarificationBatchEvent` (the same
   event the legacy daemon-side hook emitted, so the connected client's
   existing ClarificationHandler renders it) and awaits the matching
   answers on a request-id-keyed futures dict.
4. The connected client surfaces the questions, collects answers, and
   sends back a ``ClarificationBatchResponseRequest`` which the daemon
   transport routes into :meth:`JaatoServer.respond_to_clarification_batch`.
5. That method calls :meth:`resolve_response`, which sets the future's
   result; step 3's await returns and the handler replies over the RPC.

Batch-only: a clarification is delivered as one batch (all questions at
once) and answered as one batch (ordered answer strings), matching the
WS ``ClarificationBatchEvent`` / ``respond_to_clarification_batch``
contract.  Per-question streaming (the TUI's legacy in-process flow) is
unchanged for daemon-local sessions; runner-tier sessions use the batch
relay.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional

from jaato_sdk.events import ClarificationBatchEvent


logger = logging.getLogger(__name__)


# Emit an event to the connected client(s).  Bound to ``JaatoServer.emit``
# in production; tests pass a stub.
EmitEventFn = Callable[[Any], None]


class ClarificationRelayHandler:
    """Stateful handler for the ``client.request_clarification`` RPC.

    Owns a futures dict keyed by ``request_id`` so concurrent
    clarifications resolve independently.

    Lifecycle (mirror of :class:`PromptOperatorHandler`):

    1. Constructed + registered in
       :meth:`JaatoServer.set_runner_rpc` (next to the prompt-operator
       handler) for every session with a runner RPC handle.
    2. :meth:`resolve_response` is called by
       :meth:`JaatoServer.respond_to_clarification_batch` when the
       client's answers arrive.
    3. :meth:`shutdown` (from ``JaatoServer.shutdown``) cancels
       in-flight clarifications with a clean error.
    """

    def __init__(
        self,
        emit_event: EmitEventFn,
        *,
        prompt_timeout: Optional[float] = None,
    ) -> None:
        """Construct the handler.

        Args:
            emit_event: Publishes a :class:`ClarificationBatchEvent`
                to the connected client.  Bound to ``JaatoServer.emit``
                in production.
            prompt_timeout: Optional wall-clock cap on how long a
                clarification waits for a client response.  ``None``
                (production default) means hold until the client
                answers or the handler shuts down.  Tests pass a small
                value.
        """
        self._emit_event = emit_event
        self._prompt_timeout = prompt_timeout
        # request_id -> Future[{"cancelled": bool, "answers": List[str]}]
        self._pending: Dict[str, "asyncio.Future[Dict[str, Any]]"] = {}
        self._closed = False

    async def handle(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """RPC handler entry point.

        Registers the request, emits the batch event, awaits the
        answers, returns ``{"cancelled": bool, "answers": List[str]}``.
        Raises on timeout or shutdown (the runner-side channel treats
        any raise as a fail-closed cancel).
        """
        if self._closed:
            raise RuntimeError("ClarificationRelayHandler is closed")

        request_id = str(args.get("request_id", ""))
        if not request_id:
            raise ValueError(
                "request_clarification: payload.request_id is required"
            )
        if request_id in self._pending:
            raise ValueError(
                f"request_clarification: duplicate request_id={request_id!r}"
            )

        loop = asyncio.get_running_loop()
        fut: "asyncio.Future[Dict[str, Any]]" = loop.create_future()
        self._pending[request_id] = fut

        event = ClarificationBatchEvent(
            agent_id=str(args.get("agent_id", "") or ""),
            request_id=request_id,
            tool_name=str(args.get("tool_name", "request_clarification")),
            context=str(args.get("context", "") or ""),
            questions=list(args.get("questions") or []),
        )
        try:
            self._emit_event(event)
        except Exception:
            self._pending.pop(request_id, None)
            raise

        try:
            if self._prompt_timeout is not None:
                result = await asyncio.wait_for(fut, self._prompt_timeout)
            else:
                result = await fut
            return result
        finally:
            self._pending.pop(request_id, None)

    def resolve_response(
        self,
        request_id: str,
        answers: List[str],
        *,
        cancelled: bool = False,
    ) -> bool:
        """Resolve the pending future for *request_id*.

        Called by ``JaatoServer.respond_to_clarification_batch`` when
        the client's ordered answer strings arrive.  Returns ``True`` if
        a future was pending and got resolved; ``False`` otherwise (no
        future waiting — e.g. a daemon-local session whose answers go
        through the legacy input queue instead).
        """
        fut = self._pending.get(request_id)
        if fut is None or fut.done():
            return False
        fut.set_result({"cancelled": bool(cancelled), "answers": list(answers)})
        return True

    def shutdown(self) -> None:
        """Cancel all in-flight clarifications with a clean error.

        Pending futures get a ``RuntimeError("shutting down")``; the
        runner-side channel translates the raise into a clean cancelled
        response.
        """
        if self._closed:
            return
        self._closed = True
        for request_id, fut in list(self._pending.items()):
            if not fut.done():
                fut.set_exception(
                    RuntimeError(
                        f"request_clarification: shutting down "
                        f"(in-flight request_id={request_id!r})"
                    )
                )
        self._pending.clear()


def register(
    rpc_server: Any,  # server.runner_rpc_server.RunnerRPCServer
    handler: ClarificationRelayHandler,
) -> None:
    """Register *handler*'s ``handle`` as the ``client.request_clarification``
    RPC method on *rpc_server* (mirror of ``prompt_operator.register``)."""
    rpc_server.register("client.request_clarification", handler.handle)
