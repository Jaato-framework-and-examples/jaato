"""Mixin for daemon-side stubs of runner-tier plugins (Phase 3 §3.4).

When a plugin's ``PLUGIN_TIER == "runner"`` and a session is hosted
runner-side, the daemon-side instance becomes a thin RPC stub that
forwards every tool call via ``tool.execute`` over the runner-RPC
channel.  The actual subprocess.Popen / FS work happens runner-side
inside the AppArmor-confined runner process.

Phase 2 demonstrated this pattern for ``cli``: its ``_execute`` method
hand-rolled the runner-RPC forward.  Phase 3's wave-1 plugin
migrations would multiply that pattern across ~20 plugins, each with
multiple ``_execute_*`` methods (todo has ~12, file_edit has ~8).
This mixin replaces the per-method check with a single
``get_executors`` wrap-point so each plugin migration is a 2-line
change.

Usage:

    from shared.plugins.runner_forwarding import RunnerForwardingMixin

    class MultimodalPlugin(RunnerForwardingMixin):
        def get_executors(self) -> Dict[str, Callable]:
            raw = {
                "showImageWithCaption": self._execute_show_image,
            }
            return self.wrap_executors_for_runner_forwarding(raw)

The mixin inspects ``self._plugin_registry`` (set via the standard
``set_plugin_registry`` plugin-protocol method) for a
``runner_rpc`` attribute.  ``JaatoServer.set_runner_rpc`` (Phase 2
§2.3) attaches this attribute when the IPC apparmor pre-init hook
spawns a runner.  When the attribute is absent or ``None``, the
wrapped executor calls the in-process implementation — preserving
the legacy path for sessions without a runner.

Streaming + cancellation: each forwarded call reads
``get_current_tool_output_callback()`` /
``get_current_cancel_token()`` from
``shared.ai_tool_runner._thread_local`` and threads them through
``RunnerRPCClient.call_threadsafe(...)``.  The runner-side
dispatcher routes stream frames back to the same thread-local
callback (via :func:`server.runner.rpc.RunnerRPC._dispatch_via_session_executor`),
so the model + history pipeline see the same stream contract whether
the call was in-process or runner-forwarded.

Result translation: the runner returns a typed
:class:`server.runner.envelope.ResponseEnvelope`; the wrapper
converts it to the legacy result-dict shape every plugin's
``_execute_*`` returns today:

- ``envelope.ok=True`` + dict result → returned unchanged.
- ``envelope.ok=False`` + structured result (e.g. exec-not-found) →
  result dict returned unchanged so the model sees the same shape
  as the in-process path.
- ``envelope.error.type == "CancelledException"`` → re-raised so
  the in-process pipeline classifies the call as cancelled.
- Transport / handler crash → translated into a clean error-dict
  ``{"error": "..."}``.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional


logger = logging.getLogger(__name__)


# Type alias for the in-process plugin executor signature.  ``args``
# dict in, plugin-defined value out (typically dict, sometimes str).
ExecutorFn = Callable[[Dict[str, Any]], Any]


class RunnerForwardingMixin:
    """Wrap a plugin's executors so they forward via runner-RPC when
    a runner is attached, else call the in-process body."""

    # ``_plugin_registry`` is set by the standard plugin lifecycle
    # protocol (``set_plugin_registry``).  Re-declared here as a
    # type annotation so mypy + readers see the contract.
    _plugin_registry: Any = None

    def wrap_executors_for_runner_forwarding(
        self,
        executors: Dict[str, ExecutorFn],
    ) -> Dict[str, ExecutorFn]:
        """Replace each executor with a forwarding wrapper.

        Args:
            executors: The plugin's ``{tool_name: executor_fn}`` dict
                (the same shape ``get_executors`` returns today).

        Returns:
            A new dict with each value replaced by a forwarder.  Iter-
            order preserved so plugins that rely on it
            (introspection, deferred-tool ordering) keep their shape.
        """
        return {
            name: self._make_runner_forwarding_executor(name, fn)
            for name, fn in executors.items()
        }

    def _runner_rpc_handle(self) -> Optional[Any]:
        """Return the attached :class:`RunnerRPCClient`, or ``None``.

        Looks up ``runner_rpc`` on the plugin registry the
        ``set_plugin_registry`` hook stashed.  ``JaatoServer.set_runner_rpc``
        (Phase 2 task 2.3) sets this attribute when the runner is
        spawned; ``None`` means the session runs in-process.
        """
        registry = getattr(self, "_plugin_registry", None)
        if registry is None:
            return None
        return getattr(registry, "runner_rpc", None)

    def _make_runner_forwarding_executor(
        self,
        tool_name: str,
        in_process_fn: ExecutorFn,
    ) -> ExecutorFn:
        """Build the per-tool forwarder.

        Re-resolves ``runner_rpc`` on every call (the registry
        attribute is set after plugin instantiation but before any
        execute call, so a one-shot capture at decorate time misses
        it).  The check is cheap (one attribute lookup) and matches
        Phase 2 cli's per-call resolution.
        """

        def _forwarder(args: Dict[str, Any]) -> Any:
            rpc = self._runner_rpc_handle()
            if rpc is None:
                # No runner attached — invoke the in-process body.
                # This is the legacy / no-apparmor / cli-only-runner
                # path.
                return in_process_fn(args)

            # Runner attached — forward via tool.execute RPC.  Same
            # shape Phase 2 cli's ``_execute_via_runner`` implements
            # by hand; this wrapper encapsulates it.
            return _forward_via_runner(rpc, tool_name, args)

        # Preserve the underlying function's __name__ + __module__
        # for traces / debuggers.  Plugins sometimes name their
        # executors with leading underscores; the wrapper keeps that.
        try:
            _forwarder.__name__ = getattr(in_process_fn, "__name__", tool_name)
            _forwarder.__module__ = getattr(
                in_process_fn, "__module__",
                _forwarder.__module__,
            )
        except (AttributeError, TypeError):
            pass

        return _forwarder


def _forward_via_runner(
    rpc: Any,  # server.runner_rpc_client.RunnerRPCClient
    tool_name: str,
    args: Dict[str, Any],
) -> Any:
    """Dispatch *tool_name* via the runner's ``tool.execute`` RPC.

    Reads the streaming callback + cancel token from
    ``shared.ai_tool_runner._thread_local`` and threads them through
    so the runner-side dispatcher can route stream frames back +
    honour cancel.  Translates the typed envelope back to the
    legacy result-dict shape every plugin's ``_execute_*`` returns
    today.

    Module-level (not instance method) so it can be unit-tested
    without constructing a plugin instance.
    """
    from shared.ai_tool_runner import (
        get_current_cancel_token,
        get_current_tool_output_callback,
    )
    from jaato_sdk.plugins.model_provider.types import CancelledException

    on_output = get_current_tool_output_callback()
    cancel_token = get_current_cancel_token()

    try:
        envelope = rpc.call_threadsafe(
            "tool.execute",
            {"name": tool_name, "args": dict(args)},
            on_output=on_output,
            cancel_token=cancel_token,
            timeout=None,
        )
    except Exception as exc:  # noqa: BLE001 — RPC transport boundary
        # Transport-level failure (peer disconnect, malformed frame,
        # etc.).  Translate to legacy error-dict shape so the model
        # sees a clean message rather than an opaque traceback.
        return {
            "error": (
                f"{tool_name}: runner RPC failed: "
                f"{type(exc).__name__}: {exc}"
            ),
        }

    # Cancellation flows through as a typed error — re-raise so the
    # in-process ToolExecutor classifies the call as cancelled (per
    # the Phase 2 cli pattern + plan §4.1.1 contract).
    if (
        envelope.error is not None
        and envelope.error.type == "CancelledException"
    ):
        raise CancelledException(envelope.error.message or "Cancelled")

    # ok=True with structured result → unchanged.
    if envelope.ok and isinstance(envelope.result, dict):
        return envelope.result

    # Domain failure with structured result → unchanged so the model
    # sees the same shape (e.g. ``{"error": "executable not found",
    # "hint": "..."}``).
    if isinstance(envelope.result, dict):
        return envelope.result

    # Domain failure without a result dict → synthesise one from the
    # error payload.
    if envelope.error is not None:
        return {"error": envelope.error.message}

    return {"error": f"{tool_name}: empty response from runner"}
