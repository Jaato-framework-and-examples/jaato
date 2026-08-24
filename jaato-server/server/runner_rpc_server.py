"""Daemon-side dispatch table for runner → daemon RPCs.

Pairs with :class:`server.runner_rpc_client.RunnerRPCClient`.  Phase 2
shipped the daemon→runner direction (the client sends outgoing
requests + receives stream/response frames); Phase 3 task 3.2 adds
the **incoming** direction so the runner can call back into the
daemon for capabilities only the daemon owns:

- ``client.prompt_operator`` — relay an ASK prompt to the connected
  client (used by the runner-side permission plugin).  See §3.2.1.
- ``apparmor.add_reference_fragment`` — load a path-grant fragment
  into the kernel for the running session's AppArmor profile.  Used
  by the runner-side references plugin when ``selectReferences``
  admits a new external path.  See §3.2.2.
- ``telemetry.publish`` — Phase 3 §3.15 cleanup target; runner
  publishes OTel sub-spans to the daemon's forwarder.

API shape:

- :meth:`register(method, handler)` — bind ``method`` (str) to an
  async ``handler(args: dict) -> Any``.  Handler may raise; the
  dispatcher catches and translates to the typed envelope's error
  payload.  Re-registering an existing method overwrites — the
  daemon's session-init flow is the single registrar so collisions
  shouldn't happen, but we want loud failures if they do.
- :meth:`dispatch(env: RequestEnvelope) -> ResponseEnvelope` —
  invoke the handler for ``env.method``; return the typed envelope
  ready for serialization.

Threading: handlers run on whatever asyncio loop calls
:meth:`dispatch`.  The :class:`RunnerRPCClient` calls dispatch
inside its read-loop task but spawns each call in a child task
(``asyncio.create_task``) so a handler awaiting a slow client
prompt doesn't block subsequent frames.

This module hosts NO transport code — the read loop owns the
socket; the server just owns the handler table + the dispatch
contract.  Same pattern as today's ``shared.framing`` (transport
is one level up).
"""

from __future__ import annotations

import logging
import traceback
from typing import Any, Awaitable, Callable, Dict, Optional

from server.runner.envelope import (
    TOOL_ERROR_TYPE,
    ErrorPayload,
    ExecutorOutcome,
    RequestEnvelope,
    ResponseEnvelope,
)
from server.runner.sanitize import sanitize_traceback


logger = logging.getLogger(__name__)


# Type alias for an async runner→daemon RPC handler.  Handler
# receives the request's ``args`` dict and returns a result-shaped
# value (typically a dict, sometimes a primitive).  Exceptions get
# caught at the dispatch boundary and serialized into the error
# payload — same contract as the runner-side dispatcher in
# ``server/runner/rpc.py``.
HandlerFn = Callable[[Dict[str, Any]], Awaitable[Any]]


def _extract_outcome_error(payload: Any) -> str:
    """Best-effort human-readable message for a domain-failure payload.

    Mirrors ``server/runner/rpc.py``'s ``_extract_error_message`` for the
    opposite direction: the message is for LOGS and for clients that show
    ``envelope.error.message``.  The structured payload still travels in
    ``result`` — this never replaces it, so nothing is lost when the
    payload isn't dict-shaped.
    """
    if isinstance(payload, dict):
        for key in ("error", "message"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                return value
    return "executor reported failure"


class RunnerRPCServer:
    """Handler registry for runner → daemon RPCs.

    One instance per :class:`RunnerRPCClient`.  Constructed by the
    client at session-init; handlers register against it as part of
    ``_spawn_session_runner`` (or its Phase 3 successor inside
    ``_bootstrap_session(envelope)``).
    """

    def __init__(self, *, workspace_root: Optional[str] = None) -> None:
        """Construct the server.

        Args:
            workspace_root: Per-session workspace root used for
                handler-traceback sanitization (§3.1).  Mirrors the
                runner-side ``RunnerRPC.__init__`` parameter so both
                directions of the channel get the same redaction.
        """
        self._handlers: Dict[str, HandlerFn] = {}
        self._workspace_root = workspace_root

    def register(self, method: str, handler: HandlerFn) -> None:
        """Bind *method* to an async *handler*.

        Re-registering an existing method overwrites with a warning.
        The single-registrar contract means this only fires under bug
        conditions; we surface it loudly rather than silently
        winning-by-last-write.
        """
        if method in self._handlers:
            logger.warning(
                "RunnerRPCServer: re-registering handler for %r — "
                "overwriting existing", method,
            )
        self._handlers[method] = handler

    def has_handler(self, method: str) -> bool:
        return method in self._handlers

    async def dispatch(self, env: RequestEnvelope) -> ResponseEnvelope:
        """Run the handler for *env.method*; return the typed envelope.

        Mirrors the runner-side dispatcher
        (`server/runner/rpc.py:RunnerRPC._handle_request`):
        - Unknown method → ``ok=False`` with a useful error message.
        - Handler raises → ``ok=False`` with traceback (sanitized
          per §3.1) in the error payload.
        - Handler returns :class:`ExecutorOutcome` → its ``ok`` and
          ``payload`` become the envelope's ``ok`` / ``result``, with a
          :data:`TOOL_ERROR_TYPE` error stamped on failure.
        - Handler returns anything else → ``ok=True`` with the result.

        The ``ExecutorOutcome`` branch is what makes this a real mirror.
        The runner-side dispatcher has always split an executor's
        ``(ok, result)`` into the envelope's own halves; this side
        previously mirrored only the EXCEPTION paths, so a handler
        dispatching a plugin executor had nowhere to put the flag and
        returned the raw tuple as ``result``.  JSON has no tuple type,
        so it arrived as ``[False, {...}]`` and the flag was read as
        data — see :class:`ExecutorOutcome` for the full failure mode.
        """
        handler = self._handlers.get(env.method)
        if handler is None:
            return ResponseEnvelope(
                id=env.id,
                ok=False,
                result=None,
                error=ErrorPayload(
                    type="UnknownMethod",
                    message=f"no handler registered for method={env.method!r}",
                ),
            )

        try:
            result = await handler(env.args)
        except Exception as exc:  # noqa: BLE001 — boundary surface
            tb = sanitize_traceback(
                traceback.format_exc(), self._workspace_root,
            )
            logger.exception(
                "RunnerRPCServer: handler for %r raised", env.method,
            )
            return ResponseEnvelope(
                id=env.id,
                ok=False,
                result=None,
                error=ErrorPayload(
                    type=type(exc).__name__,
                    message=sanitize_traceback(str(exc), self._workspace_root),
                    traceback=tb,
                ),
            )

        # A BARE TUPLE can no longer be shipped as ``result``.  JSON has
        # no tuple type, so it would arrive as a list and any
        # ``isinstance(x, tuple)`` check downstream would silently read
        # the flag as data — the exact defect ``ExecutorOutcome`` exists
        # to remove.  No handler returns a tuple today, so this cannot
        # fire for existing callers; it fires for the NEXT handler that
        # tries to express the ``(ok, payload)`` contract by shape.
        #
        # Loud, not silent: a wrong-but-plausible envelope is what made
        # the original bug survive four consumer-side error checks.
        if isinstance(result, tuple):
            return ResponseEnvelope(
                id=env.id,
                ok=False,
                result=None,
                error=ErrorPayload(
                    type="HandlerContractError",
                    message=(
                        f"handler for {env.method!r} returned a bare tuple; "
                        f"a tuple cannot cross the JSON wire intact. Return "
                        f"ExecutorOutcome(ok=..., payload=...) to express "
                        f"the (ok, payload) executor contract."
                    ),
                ),
            )

        # Domain failure declared by the handler — unpack into the
        # envelope's own halves rather than shipping the pair as data.
        # ``error`` is stamped with the SAME discriminator the
        # runner-side dispatcher uses, so a client can tell "the
        # executor returned (False, payload)" from "the call crashed"
        # without inspecting the payload.
        if isinstance(result, ExecutorOutcome):
            if result.ok:
                return ResponseEnvelope(
                    id=env.id, ok=True, result=result.payload,
                )
            return ResponseEnvelope(
                id=env.id,
                ok=False,
                result=result.payload,
                error=ErrorPayload(
                    type=TOOL_ERROR_TYPE,
                    message=_extract_outcome_error(result.payload),
                ),
            )

        return ResponseEnvelope(
            id=env.id,
            ok=True,
            result=result,
        )
