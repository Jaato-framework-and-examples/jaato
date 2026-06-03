"""Daemon-side ``daemon.plugin_execute`` handler.

Pairs with :class:`shared.plugins.daemon_forwarding.DaemonForwardingMixin`
and ``RunnerRPCClient.daemon_plugin_execute`` — the runner→daemon
direction for cross-tier plugins (``PLUGIN_TIER = "daemon_callable"``).

When the model running runner-side invokes a daemon-callable tool
(e.g. ``session_ops.interrogate_session``), the runner-side stub
wraps the call in a ``daemon.plugin_execute`` envelope.  This handler:

1. Validates the request shape (plugin_name + tool_name + args).
2. Looks up the plugin in the daemon-side ``JaatoServer.registry``
   (which holds the *real* instance with its daemon-only state — e.g.
   ``session_ops``'s ``SessionManager`` reference wired at
   ``SessionManager._bootstrap_session``).
3. Looks up the executor in the plugin's ``get_executors()`` dict.
4. Runs the executor in a worker thread (the daemon-side body can
   block on long-running operations like interrogate_session forking
   another session; running on the asyncio loop would stall every
   other RPC).
5. Returns the result via the typed envelope.

Re-entry note: the executor returned by ``get_executors()`` is the
**wrapped** one (DaemonForwardingMixin's forwarder).  Re-entering it
daemon-side IS the intended path — the wrapper re-checks
``registry.runner_rpc_client`` (absent daemon-side) and falls through
to the in-process body.  No bypass mechanism needed because the
mixin self-routes by registry attribute.

Pattern reference: ``prompt_operator.py``, ``apparmor_fragment.py``,
``spawn_isolated_runner.py`` — same four-layer pattern (handler +
runner-side wrapper + register helper + tests).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)


# Required keys in the request args dict.  Validated up-front so
# the handler never reaches plugin lookup with garbage.
_REQUIRED_KEYS = ("plugin_name", "tool_name", "args")


class DaemonPluginExecuteHandler:
    """Stateful handler for the ``daemon.plugin_execute`` RPC.

    Confused-deputy protection: the handler holds a reference to the
    parent session's ``JaatoServer`` so the registry lookup is bound
    to that session's plugin instances — a runner sending a request
    via a different session's channel reaches a different handler
    with a different server reference.

    Lifecycle:

    1. Constructed by ``JaatoServer.set_runner_rpc()`` alongside
       ``PromptOperatorHandler`` + ``SpawnIsolatedRunnerHandler``.
    2. Registered on the session's ``RunnerRPCServer`` as the
       ``daemon.plugin_execute`` handler.
    3. Held on ``JaatoServer._daemon_plugin_execute_handler`` so
       ``shutdown()`` can cascade.
    4. On session shutdown, :meth:`shutdown` marks the handler closed.
       In-flight executor calls finish naturally; the daemon-side
       plugin instance is owned by the server's registry, not this
       handler, so cleanup is upstream.
    """

    def __init__(self, server: Any) -> None:
        """Construct the handler.

        Args:
            server: The parent session's ``JaatoServer`` instance.
                Holds the registry the handler dispatches against.
                Type-erased to ``Any`` to avoid a forward-import cycle
                with ``server.core``.

        Raises:
            ValueError: if ``server`` is ``None`` — fail-fast at
                construction so a misconfigured registration site
                surfaces immediately rather than at first RPC.
        """
        if server is None:
            raise ValueError(
                "DaemonPluginExecuteHandler: server must not be None"
            )
        self._server = server
        self._closed = False

    async def handle(self, args: Dict[str, Any]) -> Any:
        """RPC handler entry point.

        Expected args:

            plugin_name (str, required): target plugin's ``name``
                attribute (e.g. ``"session_ops"``).
            tool_name (str, required): key within the plugin's
                ``get_executors()`` dict.
            args (dict, required): tool args — passed verbatim to the
                executor.  Empty dict ``{}`` is valid for no-arg
                tools.

        Returns:
            The executor's return value, returned via the typed
            envelope's ``result`` field (the dispatcher wraps).

        Raises:
            ValueError: malformed args (missing keys, wrong types,
                plugin/tool not found).  Translates to a typed error
                envelope on the wire.
            RuntimeError: handler closed (shutdown was called).
            Exception: re-raises whatever the underlying executor
                raises.  The dispatcher catches at the boundary and
                serialises into the error envelope (with a sanitised
                traceback per §3.1).
        """
        if self._closed:
            raise RuntimeError("DaemonPluginExecuteHandler is closed")

        # ── Validate required keys + types ─────────────────────
        for key in _REQUIRED_KEYS:
            if key not in args:
                raise ValueError(
                    f"daemon.plugin_execute: missing required arg {key!r}"
                )

        plugin_name = args["plugin_name"]
        tool_name = args["tool_name"]
        tool_args = args["args"]

        if not isinstance(plugin_name, str) or not plugin_name:
            raise ValueError(
                "daemon.plugin_execute: plugin_name must be a non-empty str"
            )
        if not isinstance(tool_name, str) or not tool_name:
            raise ValueError(
                "daemon.plugin_execute: tool_name must be a non-empty str"
            )
        if not isinstance(tool_args, dict):
            raise ValueError(
                "daemon.plugin_execute: args must be a dict"
            )

        # ── Look up plugin + executor ──────────────────────────
        registry = getattr(self._server, "registry", None)
        if registry is None:
            raise ValueError(
                f"daemon.plugin_execute({plugin_name}.{tool_name}): "
                f"server has no registry (session not fully bootstrapped)"
            )
        plugin = registry.get_plugin(plugin_name)
        if plugin is None:
            raise ValueError(
                f"daemon.plugin_execute: plugin {plugin_name!r} not found "
                f"in daemon-side registry — likely cause: PLUGIN_TIER is "
                f"not 'daemon_callable' or the plugin failed to discover."
            )
        executors = plugin.get_executors()
        executor = executors.get(tool_name)
        if executor is None:
            raise ValueError(
                f"daemon.plugin_execute: tool {tool_name!r} not found in "
                f"plugin {plugin_name!r}'s executors. Available: "
                f"{sorted(executors.keys())}"
            )

        # ── Run the executor in a worker thread ────────────────
        # The body may block (interrogate_session forks another session
        # and waits for the model turn).  Running on the asyncio loop
        # would stall every other RPC on this channel.  ``run_in_executor``
        # with ``None`` uses asyncio's default ThreadPoolExecutor.
        #
        # Re-entry note: ``executor`` is the DaemonForwardingMixin's
        # forwarder.  On daemon-side ``registry.runner_rpc_client``
        # doesn't exist, so the forwarder falls through to the
        # in-process body — same shape ``runner_forwarding`` uses
        # when daemon-side ``registry.runner_rpc`` is None.
        logger.debug(
            "daemon.plugin_execute: dispatching %s.%s (args keys=%s)",
            plugin_name, tool_name, sorted(tool_args.keys()),
        )
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, executor, tool_args)
        return result

    def shutdown(self) -> None:
        """Mark the handler closed.

        In-flight ``run_in_executor`` calls finish naturally — the
        underlying plugin instance is owned by the server's registry,
        not this handler, so we don't need to interrupt them.
        Idempotent.
        """
        self._closed = True


def register(
    rpc_server: Any,  # server.runner_rpc_server.RunnerRPCServer
    handler: DaemonPluginExecuteHandler,
) -> None:
    """Register *handler*'s ``handle`` method as the
    ``daemon.plugin_execute`` RPC method on *rpc_server*.

    Convenience indirection so the bootstrap site
    (``JaatoServer.set_runner_rpc()``) doesn't repeat the method-name
    string literal.  Mirrors ``register`` in ``prompt_operator.py``
    and ``spawn_isolated_runner.py``.
    """
    rpc_server.register("daemon.plugin_execute", handler.handle)
