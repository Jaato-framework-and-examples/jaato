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

    # --------------------- add_reference_fragment ----------------------

    def add_reference_fragment(
        self,
        ref_id: str,
        path: str,
        *,
        session_id: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Ask the daemon to load a reference-fragment into the
        running session's AppArmor profile.

        Phase 3 §3.2.2.  The runner-side references plugin's
        ``selectReferences`` admit path calls this when an external
        path is selected.

        Args:
            ref_id: Stable identifier for the reference (used as the
                fragment filename suffix).
            path: Absolute path to grant readonly access to.  Must
                pass the daemon's
                ``_validate_path_for_fragment`` (no relative paths,
                no AppArmor glob metacharacters, no newlines).
            session_id: Optional echo of the session id for sanity-
                check on the daemon side.  Authoritative session id
                is the handler-bound one; mismatches here surface
                as a ``ValueError`` from the daemon.
            timeout: Optional wall-clock cap.  ``apparmor_parser -r``
                takes 10-30s on slow hosts; 60s is a reasonable
                default.

        Returns:
            ``{"ok": True}`` on success, or
            ``{"ok": False, "error": "<reason>"}`` on validation or
            kernel failure.  Domain failures (validation reject,
            ``add_reference_fragment`` returning False) flow as
            ``ok=False`` dicts; transport / handler crashes raise
            :class:`RunnerRPCError`.

        Raises:
            RunnerRPCError: transport-level failure or handler crash.
        """
        args: Dict[str, Any] = {"ref_id": ref_id, "path": path}
        if session_id is not None:
            args["session_id"] = session_id
        env = self._rpc.outgoing_call(
            "apparmor.add_reference_fragment",
            args,
            timeout=timeout,
        )
        if not env.ok or env.error is not None:
            err_type = env.error.type if env.error else "UnknownError"
            err_msg = env.error.message if env.error else "no error message"
            raise RunnerRPCError(
                f"apparmor.add_reference_fragment failed: "
                f"{err_type}: {err_msg}"
            )
        if not isinstance(env.result, dict):
            raise RunnerRPCError(
                f"apparmor.add_reference_fragment: unexpected result "
                f"type {type(env.result).__name__}; expected dict"
            )
        return dict(env.result)

    # --------------------- spawn_isolated_runner -----------------------

    def spawn_isolated_runner(
        self,
        *,
        parent_session_id: str,
        subagent_id: str,
        profile_payload: Dict[str, Any],
        task: str,
        workspace_path: str,
        agent_params: Optional[Dict[str, Any]] = None,
        display_name: Optional[str] = None,
        parent_agent_id: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Ask the daemon to spawn an isolated runner subprocess for
        a subagent (Phase 4 §4.3.2).

        The eventual consumer is the runner-side subagent plugin's
        ``agent_params.isolated=true`` opt-in branch (§4.3.7).  Until
        that branch is wired, this wrapper is registered but not
        called in production — exists so §4.3.3-§4.3.6 can land
        their respective machinery against a stable surface.

        Args:
            parent_session_id: The current (parent) session's id.
                Echoed in the request for confused-deputy sanity
                check; the daemon-side handler is bound to its own
                authoritative copy and rejects mismatches.
            subagent_id: Pre-generated subagent id from the runner-
                side subagent plugin (today's ``_next_agent_id``).
            profile_payload: Serialized ``SubagentProfile`` as a
                dict — see Audit 5 in
                ``docs/design/phase4_implementation_audits.md`` for
                the field list (model, provider, plugins,
                plugin_configs, system_instructions, etc.).
            task: First-turn prompt for the isolated runner.
            workspace_path: Inherited from parent (§4.3 invariant).
            agent_params: Forwarded case data for ``{{name}}``
                substitution / ``RenderContext.agent_params``.  The
                ``isolated`` key is stripped daemon-side (control
                flag, not template data).  Optional.
            display_name: Custom display name; defaults to
                ``profile_payload.name``.  Optional.
            parent_agent_id: For multi-hop subagent trees (a
                subagent spawning another isolated subagent).
                Optional.
            timeout: Wall-clock cap on the spawn RPC.  Spawning a
                runner subprocess can take ~50-200ms (fork +
                AppArmor change_profile + plugin discovery).  60s
                is a reasonable default; ``None`` waits
                indefinitely.

        Returns:
            On success (post-§4.3.7 readiness):
                ``{"ok": True, "session_id": "...", "subagent_id":
                "...", "runner_pid": <int>, "apparmor_profile":
                "...", "cgroup_path": "..."}``.

            On domain failure (validation reject, spawn failure,
            sub-profile / sub-cgroup / forwarding failure):
                ``{"ok": False, "error": "...", "stage": "..."}``
                where ``stage`` is one of ``validation`` /
                ``sub_profile`` / ``spawn`` / ``sub_cgroup`` /
                ``forwarding``.  Caller branches on ``ok`` and
                surfaces ``stage`` for precise diagnostics.

            Phase 4 §4.3.2 stub status: any request with valid args
            returns ``{"ok": False, "error": "...not yet
            implemented...", "stage": "spawn"}`` until the §4.3.3-
            §4.3.7 sub-commits land.

        Raises:
            RunnerRPCError: transport-level failure (channel closed,
                handler crashed, malformed envelope).  Domain
                failures (handler returned ``ok=False``) are NOT
                raised — they return as the success-path dict so
                callers can inspect ``stage``.
        """
        args: Dict[str, Any] = {
            "parent_session_id": parent_session_id,
            "subagent_id": subagent_id,
            "profile_payload": profile_payload,
            "task": task,
            "workspace_path": workspace_path,
        }
        if agent_params is not None:
            args["agent_params"] = agent_params
        if display_name is not None:
            args["display_name"] = display_name
        if parent_agent_id is not None:
            args["parent_agent_id"] = parent_agent_id

        env = self._rpc.outgoing_call(
            "subagent.spawn_isolated_runner",
            args,
            timeout=timeout,
        )
        if not env.ok or env.error is not None:
            err_type = env.error.type if env.error else "UnknownError"
            err_msg = env.error.message if env.error else "no error message"
            raise RunnerRPCError(
                f"subagent.spawn_isolated_runner failed: "
                f"{err_type}: {err_msg}"
            )
        if not isinstance(env.result, dict):
            raise RunnerRPCError(
                f"subagent.spawn_isolated_runner: unexpected result "
                f"type {type(env.result).__name__}; expected dict"
            )
        return dict(env.result)

    # --------------------- daemon_plugin_execute -----------------------

    def daemon_plugin_execute(
        self,
        *,
        plugin_name: str,
        tool_name: str,
        args: Dict[str, Any],
        timeout: Optional[float] = None,
    ) -> Any:
        """Ask the daemon to execute a tool on a daemon-tier plugin
        instance.

        The runner-side counterpart for cross-tier plugins declared
        with ``PLUGIN_TIER = "daemon_callable"``.  Such plugins are
        discovered both sides — the runner-side instance is a
        :class:`shared.plugins.daemon_forwarding.DaemonForwardingMixin`
        stub whose executors call this method; the daemon-side
        instance holds the real state (e.g. ``SessionManager``
        reference for ``session_ops``) and executes the body.

        Args:
            plugin_name: Name of the target plugin as returned by
                ``plugin.name`` (e.g. ``"session_ops"``).  Must
                match an entry in the daemon-side
                ``server.registry`` for the parent session.
            tool_name: Name of the tool to dispatch within the
                plugin (e.g. ``"interrogate_session"``).  Must
                match a key in that plugin's
                ``get_executors()`` dict.
            args: Tool-arg dict — the same shape the in-process
                executor would receive.  Serialised verbatim onto
                the wire; values must be JSON-encodable.
            timeout: Optional wall-clock cap.  ``None`` waits
                indefinitely — the canonical use case
                (``session_ops.interrogate_session``) forks another
                session and may wait many seconds for the model
                turn to complete.

        Returns:
            The executor's return value, unwrapped from the typed
            envelope.  Plugins return dicts (``{"answer": ..., ...}``
            for interrogate_session) but primitives + lists are also
            supported.

        Raises:
            RunnerRPCError: when the daemon-side handler reports an
                error (envelope ``ok=False``) — typically:
                  - plugin not found in the daemon-side registry
                    (cross-tier discovery skipped it)
                  - tool not found in the plugin's executors
                  - executor raised (traceback in ``err_msg``)
                  - the channel was closed mid-call.
            concurrent.futures.TimeoutError: when *timeout* fires.
        """
        env: ResponseEnvelope = self._rpc.outgoing_call(
            "daemon.plugin_execute",
            {
                "plugin_name": plugin_name,
                "tool_name": tool_name,
                "args": dict(args),
            },
            timeout=timeout,
        )
        if not env.ok or env.error is not None:
            err_type = env.error.type if env.error else "UnknownError"
            err_msg = env.error.message if env.error else "no error message"
            raise RunnerRPCError(
                f"daemon.plugin_execute({plugin_name}.{tool_name}) "
                f"failed: {err_type}: {err_msg}"
            )
        return env.result
