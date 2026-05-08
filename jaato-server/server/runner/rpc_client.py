"""Runner-side high-level RPC client (Phase 3 §3.2).

Thin wrapper over :class:`server.runner.rpc.RunnerRPC.outgoing_call`
exposing named methods for the daemon-tier capabilities runner-side
plugins consume:

- :meth:`prompt_operator` — relay an ASK prompt to the connected
  client (used by the runner-side permission plugin in §3.7).
- :meth:`add_reference_fragment` — kernel-side fragment load for
  the running session's profile (used by references in §3.8;
  lands in the §3.2.2 commit).
- :meth:`publish_telemetry` — runner publishes OTel sub-spans to
  the daemon's forwarder (Phase 3 §3.15).

The wrapper exists so plugins don't repeat the string-literal
method name + dict-shape construction at every call site, and so
the runner→daemon API surface is a single read-once import.

Synchronous because plugin code runs in worker threads, not asyncio
— mirrors the runner-side ``RunnerRPC.outgoing_call`` contract.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from shared.plugins.permission.types import PromptPayload, PromptResponse

from .envelope import ResponseEnvelope
from .rpc import RunnerRPC


logger = logging.getLogger(__name__)


class RunnerRPCError(RuntimeError):
    """Raised when a runner→daemon RPC fails at the protocol level
    (handler raised, transport closed, timeout) or returns an
    ``ok=False`` envelope.

    Distinguishes daemon-side failures from in-runner errors so
    plugin code can decide whether to retry, surface to the model,
    or fail-fast.
    """


class RunnerRPCClient:
    """Named-method wrapper over :class:`RunnerRPC.outgoing_call`.

    Constructed once per runner process and shared across all
    runner-side plugins (the API is stateless beyond the underlying
    RPC).
    """

    def __init__(self, rpc: RunnerRPC) -> None:
        """Construct the wrapper.

        Args:
            rpc: The runner's bidirectional dispatcher.  The wrapper
                holds a reference and routes outgoing calls through
                ``rpc.outgoing_call(method, args, timeout)``.
        """
        self._rpc = rpc

    # ------------------------- prompt_operator -------------------------

    def prompt_operator(
        self,
        payload: PromptPayload,
        *,
        timeout: Optional[float] = None,
    ) -> PromptResponse:
        """Relay an ASK prompt to the connected client; await response.

        Phase 3 §3.2.1.  The runner-side permission plugin's ASK path
        calls this when its policy doesn't have a static rule for a
        tool invocation.

        Args:
            payload: The prompt to relay (tool-name + args + prompt
                text + response options).
            timeout: Optional wall-clock cap.  ``None`` means wait
                indefinitely — the operator may legitimately walk
                away from a cascade for hours.  Phase 5+ may add a
                daemon-level configurable default.

        Returns:
            The :class:`PromptResponse` carrying the operator's
            decision (response key + optional edited args).

        Raises:
            RunnerRPCError: when the daemon-side handler reported
                an error (envelope ``ok=False``) or the channel is
                closed.
            concurrent.futures.TimeoutError: when *timeout* fires.
        """
        env: ResponseEnvelope = self._rpc.outgoing_call(
            "client.prompt_operator",
            payload.to_dict(),
            timeout=timeout,
        )
        if not env.ok or env.error is not None:
            err_type = env.error.type if env.error else "UnknownError"
            err_msg = env.error.message if env.error else "no error message"
            raise RunnerRPCError(
                f"client.prompt_operator failed: {err_type}: {err_msg}"
            )
        if not isinstance(env.result, dict):
            raise RunnerRPCError(
                f"client.prompt_operator: unexpected result type "
                f"{type(env.result).__name__}; expected dict"
            )
        return PromptResponse.from_dict(env.result)
