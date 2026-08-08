"""Mixin for runner-side stubs of daemon-executed tool plugins.

The symmetric mirror of :mod:`shared.plugins.runner_forwarding`.

``RunnerForwardingMixin`` solves the case where a plugin's tool is
discoverable + invokable daemon-side but the **body** must execute
runner-side (filesystem, subprocess, per-session in-memory state).
The daemon-side instance is a stub; calls forward via ``tool.execute``
RPC over the existing daemon→runner channel.

``DaemonForwardingMixin`` solves the inverse: a plugin's tool schema
must surface to the model (which runs runner-side post-seat-flip), but
the **body** must execute daemon-side because it needs daemon-only
state — the canonical case being ``SessionManager`` access for
cross-session introspection (premium ``session_ops`` plugin).  The
runner-side instance is a stub; calls forward via
``daemon.plugin_execute`` RPC over the runner→daemon channel (the same
``RunnerRPC.outgoing_call`` channel today's ``client.prompt_operator``
and ``apparmor.add_reference_fragment`` ride).

Usage:

    from shared.plugins.daemon_forwarding import DaemonForwardingMixin

    class SessionOpsPlugin(DaemonForwardingMixin):
        def get_executors(self) -> Dict[str, Callable]:
            raw = {
                "interrogate_session": self._execute_interrogate_session,
            }
            return self.wrap_executors_for_daemon_forwarding(raw)

The mixin inspects ``self._plugin_registry`` (set via the standard
``set_plugin_registry`` plugin-protocol method) for a
``runner_rpc_client`` attribute.  ``RunnerRPC._handle_session_bootstrap``
attaches this attribute on the runner-side registry once session
bootstrap completes (``server/runner/rpc.py:1114``).  When the
attribute is present, the wrapped executor forwards to the daemon;
when absent — i.e. the plugin is being invoked daemon-side — the
in-process body runs.

The "side check" is by **registry attribute**, not process detection:

- Daemon-side plugin instance:
  ``self._plugin_registry.runner_rpc_client`` does NOT exist (only the
  opposite-direction ``runner_rpc`` exists daemon-side).  Mixin falls
  through to in-process execution.  This is also the path taken when
  the daemon-side handler for ``daemon.plugin_execute`` re-enters the
  plugin's executor — the wrapped function self-routes correctly.

- Runner-side plugin instance:
  ``self._plugin_registry.runner_rpc_client`` is a
  :class:`server.runner.rpc_client.RunnerRPCClient`.  Mixin forwards
  via ``runner_rpc_client.daemon_plugin_execute(...)``.

Cross-tier discovery: the plugin must declare ``PLUGIN_TIER =
"daemon_callable"`` so the registry's ``tier_filter="runner"`` sweep
loads it on the runner side (for schema surfacing) AND the daemon's
unfiltered ``discover()`` loads it daemon-side (for real execution).
See ``shared/plugins/CLAUDE.md`` for the full pattern + the
build-fail static-analysis gate.

Result translation: the daemon returns a typed
:class:`server.runner.envelope.ResponseEnvelope`; the wrapper
converts it to the legacy result-dict shape every plugin's
``_execute_*`` returns today:

- ``envelope.ok=True`` + dict result → returned unchanged.
- ``envelope.ok=False`` + structured result → result dict returned
  unchanged so the model sees the same shape as the in-process path.
- ``envelope.error.type == "CancelledException"`` → re-raised so the
  in-process pipeline classifies the call as cancelled.
- Transport / handler crash → translated into a clean error-dict
  ``{"error": "..."}``.

Streaming + cancellation: deliberately NOT threaded through in v1.
The canonical use case (``interrogate_session``) is a single-shot
call; ``replay_in_workspace`` is also single-shot.  When a future
``daemon_callable`` plugin needs streaming, extend
``_forward_via_daemon`` to read
``shared.ai_tool_runner._thread_local`` the way
``runner_forwarding._forward_via_runner`` does today.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional


logger = logging.getLogger(__name__)


# Same alias shape as runner_forwarding.ExecutorFn so the two mixins
# can be composed (in principle) on a single plugin without import-
# alias collisions.  No use case for both today.
ExecutorFn = Callable[[Dict[str, Any]], Any]


class DaemonForwardingMixin:
    """Wrap a plugin's executors so they forward via runner→daemon RPC
    when invoked on a runner-side plugin instance, else call the
    in-process body (the daemon-side path)."""

    # ``_plugin_registry`` is set by the standard plugin lifecycle
    # protocol (``set_plugin_registry``).  Re-declared here as a
    # type annotation so mypy + readers see the contract.
    _plugin_registry: Any = None

    def wrap_executors_for_daemon_forwarding(
        self,
        executors: Dict[str, ExecutorFn],
    ) -> Dict[str, ExecutorFn]:
        """Replace each executor with a forwarding wrapper.

        Args:
            executors: The plugin's ``{tool_name: executor_fn}`` dict
                (the same shape ``get_executors`` returns today).

        Returns:
            A new dict with each value replaced by a forwarder.  Iter-
            order preserved so plugins that rely on it (introspection,
            deferred-tool ordering) keep their shape.
        """
        return {
            name: self._make_daemon_forwarding_executor(name, fn)
            for name, fn in executors.items()
        }

    def _runner_rpc_client_handle(self) -> Optional[Any]:
        """Return the attached runner-side :class:`RunnerRPCClient`, or
        ``None``.

        Looks up ``runner_rpc_client`` on the plugin registry the
        ``set_plugin_registry`` hook stashed.  Attribute is set
        runner-side by ``RunnerRPC._handle_session_bootstrap`` after
        bootstrap completes; daemon-side registries never gain this
        attribute (their counterpart is ``runner_rpc`` for the
        opposite direction).
        """
        registry = getattr(self, "_plugin_registry", None)
        if registry is None:
            return None
        return getattr(registry, "runner_rpc_client", None)

    def _make_daemon_forwarding_executor(
        self,
        tool_name: str,
        in_process_fn: ExecutorFn,
    ) -> ExecutorFn:
        """Build the per-tool forwarder.

        Re-resolves ``runner_rpc_client`` on every call — the registry
        attribute is set after plugin instantiation but before any
        execute call, so a one-shot capture at decorate time misses it.
        Matches the pattern in ``runner_forwarding`` (per-call
        resolution).
        """

        plugin_name = getattr(self, "name", None) or type(self).__name__

        def _forwarder(args: Dict[str, Any]) -> Any:
            rpc_client = self._runner_rpc_client_handle()
            if rpc_client is None:
                # No runner→daemon channel attached — invoke the
                # in-process body.  This is the daemon-side path: the
                # daemon-side plugin instance executes the real body
                # (either directly, or re-entered from the daemon-side
                # ``daemon.plugin_execute`` handler).
                return in_process_fn(args)

            # Runner-side path — forward via daemon.plugin_execute RPC.
            return _forward_via_daemon(
                rpc_client, plugin_name, tool_name, args,
            )

        # Preserve the underlying function's __name__ + __module__ for
        # traces / debuggers.  Mirrors runner_forwarding's preservation
        # so stack traces under both directions read symmetrically.
        try:
            _forwarder.__name__ = getattr(in_process_fn, "__name__", tool_name)
            _forwarder.__module__ = getattr(
                in_process_fn, "__module__",
                _forwarder.__module__,
            )
        except (AttributeError, TypeError):
            pass

        return _forwarder


def _forward_via_daemon(
    rpc_client: Any,  # server.runner.rpc_client.RunnerRPCClient
    plugin_name: str,
    tool_name: str,
    args: Dict[str, Any],
) -> Any:
    """Dispatch *tool_name* via the daemon's ``daemon.plugin_execute``
    RPC.

    Translates the typed envelope back to the legacy result-dict shape
    every plugin's ``_execute_*`` returns today.  Symmetric with
    ``runner_forwarding._forward_via_runner`` minus the streaming hooks
    (see module docstring rationale).

    Module-level (not instance method) so it can be unit-tested
    without constructing a plugin instance.
    """
    from jaato_sdk.plugins.model_provider.types import CancelledException

    try:
        result = rpc_client.daemon_plugin_execute(
            plugin_name=plugin_name,
            tool_name=tool_name,
            args=dict(args),
        )
    except _RunnerRPCError as exc:
        # Domain failure surfaced by the wrapper as a typed exception.
        # Cancellation needs to flow through as CancelledException so
        # the in-process pipeline classifies it correctly.
        message = str(exc)
        if "CancelledException" in message:
            raise CancelledException(message)
        return {
            "error": (
                f"{tool_name}: daemon RPC failed: {message}"
            ),
        }
    except Exception as exc:  # noqa: BLE001 — RPC transport boundary
        # Transport-level failure (channel closed, malformed frame,
        # etc.).  Translate to legacy error-dict shape so the model
        # sees a clean message rather than an opaque traceback.
        return {
            "error": (
                f"{tool_name}: daemon RPC failed: "
                f"{type(exc).__name__}: {exc}"
            ),
        }

    # Daemon handler returns whatever the underlying executor returns
    # (typically a dict, sometimes a primitive).  No envelope-shaped
    # translation needed because the wrapper method on
    # RunnerRPCClient already unwrapped the ResponseEnvelope.
    return result


# Late import sentinel — avoids a hard dependency on server.runner.*
# at shared/plugins/ import time (the shared/ tree must remain
# importable from contexts that don't load the server-side package).
try:
    from server.runner.rpc_client import RunnerRPCError as _RunnerRPCError
except ImportError:  # pragma: no cover — server package always present in jaato-server
    class _RunnerRPCError(RuntimeError):
        """Fallback when server.runner.rpc_client is unavailable."""
