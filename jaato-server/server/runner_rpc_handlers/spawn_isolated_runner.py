"""Daemon-side ``subagent.spawn_isolated_runner`` handler (Phase 4 §4.3.2).

Wire-shape stub for the isolated-subagent opt-in primitive.  The
runner-side subagent plugin's ``agent_params.isolated=true`` branch
(§4.3.7) will eventually call this handler to ask the daemon to
spawn a new runner subprocess with its own AppArmor sub-profile
(``jaato-ws-{parent_session}//{subagent_id}``) and its own
sub-cgroup.

This commit (§4.3.2) wires the four layers needed to land that
opt-in branch incrementally:

1. **This module** — daemon-side handler class + register helper.
2. ``server/runner/rpc_client.py`` — runner-side
   ``spawn_isolated_runner`` wrapper method on ``RunnerRPCClient``.
3. ``server/core.py:set_runner_rpc()`` — per-``JaatoServer``
   handler instantiation alongside ``PromptOperatorHandler``.
4. ``server/runner_rpc_handlers/tests/test_spawn_isolated_runner.py``
   + wrapper tests — pin the args validation + stub body + register/
   shutdown lifecycle.

The handler body is intentionally a stub.  Args validation runs
(missing keys, type checks, parent_session_id echo check) and
returns a typed error envelope on failure.  When validation
passes, the handler returns:

    {"ok": False, "error": "isolated-runner spawn machinery not
    yet implemented (Phase 4 §4.3.3-§4.3.7)", "stage": "spawn"}

The ``stage`` enum lets the runner-side caller surface precise
diagnostics — ``validation`` for malformed requests vs ``spawn`` /
``sub_profile`` / ``sub_cgroup`` / ``forwarding`` once the
respective sub-commits land.

Pattern reference: Phase 3 §3.2 handlers
(``prompt_operator.py``, ``apparmor_fragment.py``) — same four-
layer pattern, same confused-deputy protection via handler-bound
session id (parent_session_id here, session_id there).

Wire flow (post-§4.3.7 readiness):

1. Runner-side subagent plugin's opt-in branch detects
   ``agent_params.isolated=true`` (§4.3.1 stub becomes a routing
   decision in §4.3.7).
2. Runner-side ``RunnerRPCClient.spawn_isolated_runner(...)``
   sends the RPC.
3. **This handler** validates args, asks the daemon's
   ``SessionManager`` to spawn a new runner with a sub-AppArmor
   profile + sub-cgroup (§4.3.3-§4.3.5), wires cross-runner
   forwarding (§4.3.6), and returns the new runner's session id.
4. Runner-side caller receives the new ``session_id`` and forwards
   the first-turn prompt INTO the new runner; output forwards BACK
   to the parent's injection queue via the daemon as relay.

Until §4.3.7 wires the call site, this handler is registered but
not exercised in production.  The §4.3.1 stub still fires for
``isolated=true`` callers.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)


# Stage enum values — used in the failure response's ``stage`` key
# so callers can surface precise diagnostics.  Each downstream
# sub-commit (§4.3.3-§4.3.6) fills in the body for its own stage.
STAGE_VALIDATION = "validation"
STAGE_SUB_PROFILE = "sub_profile"
STAGE_SPAWN = "spawn"
STAGE_SUB_CGROUP = "sub_cgroup"
STAGE_FORWARDING = "forwarding"


# Required keys in the request args dict.  Validated up-front so
# the stub never gets garbage requests through to spawn machinery.
_REQUIRED_KEYS = (
    "parent_session_id",
    "subagent_id",
    "profile_payload",
    "task",
    "workspace_path",
)

# Optional keys — pass through but don't require.
_OPTIONAL_KEYS = (
    "agent_params",
    "display_name",
    "parent_agent_id",
)


class SpawnIsolatedRunnerHandler:
    """Stateful handler for the ``subagent.spawn_isolated_runner`` RPC.

    Phase 4 §4.3.2 stub.  Validates request args and returns a
    typed "not yet implemented" envelope.  Full spawn body lands
    incrementally in §4.3.3 (helper plumbing), §4.3.4 (sub-AppArmor
    profile), §4.3.5 (sub-cgroup), §4.3.6 (forwarding).

    Confused-deputy protection: the handler is constructed with the
    **parent session's id** at session-bootstrap time.  Request
    args must echo the same id (or omit it); a mismatch raises
    ``ValueError``, which the dispatcher translates to a typed
    error envelope on the wire.  Prevents a runner from spawning a
    sub-runner "under" a different parent — would otherwise
    produce a sub-AppArmor profile name with the wrong parent
    encoded in it.

    Lifecycle:

    1. Constructed by ``JaatoServer.set_runner_rpc()`` for each
       session that has a runner subprocess (§3.2 / Phase 3 §7c
       infrastructure).
    2. Registered on the session's ``RunnerRPCServer`` as the
       ``subagent.spawn_isolated_runner`` handler.
    3. Held on ``JaatoServer._spawn_isolated_runner_handler`` so
       ``shutdown()`` can cascade.
    4. On session shutdown, :meth:`shutdown` is a no-op for now
       (the stub holds no in-flight state).  §4.3.6 will add
       in-flight spawn tracking + cleanup.
    """

    def __init__(
        self,
        parent_session_id: str,
    ) -> None:
        """Construct the handler.

        Args:
            parent_session_id: The session this handler is bound
                to.  Used for the confused-deputy check on every
                incoming request: payload args' ``parent_session_id``
                must echo (or omit) this value.

        Raises:
            ValueError: if ``parent_session_id`` is empty.  Empty
                parent IDs would break the sub-profile name template
                (``jaato-ws-{parent}//{sub}``) and cause silent
                mis-attribution.  Fail-fast at construction.

        Note (§4.3.2 stub): the handler holds NO spawn-machinery
        references (no ``SessionManager``, no ``AppArmorManager``,
        no cgroup manager).  §4.3.3 will add a setter
        (``set_spawn_dependencies``) for the spawn helper +
        related managers when the real spawn body is wired.  The
        stub deliberately keeps the constructor minimal so the
        registration site in ``core.py:set_runner_rpc()`` doesn't
        need a ``SessionManager`` reference (which ``JaatoServer``
        doesn't have today — passing it would require a wider
        signature change to ``spawn_session_runner`` and is left
        for §4.3.3).
        """
        if not isinstance(parent_session_id, str) or not parent_session_id:
            raise ValueError(
                "SpawnIsolatedRunnerHandler: parent_session_id must be "
                "a non-empty str"
            )
        self._parent_session_id = parent_session_id
        self._closed = False
        # Phase 4 §4.3.3: spawn dependencies are wired post-construction
        # via :meth:`set_spawn_dependencies`.  None until the daemon's
        # ``SessionManager._spawn_session_runner_unconditional`` calls
        # the setter after the runner subprocess + RPC channel are up.
        # When None, ``handle()`` returns the §4.3.2 "RPC primitive not
        # yet bridged" stub envelope.  When set, ``handle()`` routes
        # through ``session_manager._spawn_isolated_runner(...)``.
        self._session_manager: Optional[Any] = None

    def set_spawn_dependencies(
        self,
        session_manager: Any,  # server.session_manager.SessionManager
    ) -> None:
        """Wire the SessionManager reference post-construction
        (Phase 4 §4.3.3).

        Called by ``SessionManager._spawn_session_runner_unconditional``
        after the parent session's runner subprocess + RPC channel are
        up (the handler is registered earlier in
        ``JaatoServer.set_runner_rpc()``, but the SessionManager
        reference isn't available at THAT seam — passing it would
        require widening ``spawn_session_runner``'s signature, which
        was kept narrow in §4.3.2).

        Idempotent: re-calling with the same session_manager is a
        no-op; re-calling with a different session_manager replaces
        the reference (no defensive guard — the caller is trusted).

        Once dependencies are set, :meth:`handle` routes valid
        requests through ``session_manager._spawn_isolated_runner(...)``
        instead of returning the §4.3.2 "RPC primitive not yet bridged"
        stub.  Going from "stub" to "routed" is the §4.3.3 milestone;
        the routed helper itself still returns ``stage=sub_profile``
        until §4.3.4 fills in the next stage.

        Args:
            session_manager: The daemon's ``SessionManager`` instance.
                Holds the ``_spawn_isolated_runner`` helper added in
                §4.3.3.  Type-erased to ``Any`` to avoid a forward-
                import cycle with ``server.session_manager``.
        """
        self._session_manager = session_manager

    async def handle(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """RPC handler entry point.

        Expected args (see Audit 5 in
        ``docs/design/phase4_implementation_audits.md``):

            parent_session_id (str, required): echo of the parent
                session's id; must match the handler-bound id.
            subagent_id (str, required): pre-generated by the
                runner-side subagent plugin.
            profile_payload (dict, required): serialized
                SubagentProfile (model, provider, plugins,
                plugin_configs, system_instructions, etc.).
            task (str, required): first-turn prompt for the
                isolated runner.
            workspace_path (str, required): inherited from parent.
            agent_params (dict, optional): forwarded case data for
                {{name}} substitution.  The ``isolated`` key is
                stripped before pass-through (it's a control flag,
                not template data).
            display_name (str, optional): defaults to
                ``profile_payload.name``.
            parent_agent_id (str, optional): for multi-hop subagent
                trees.

        Returns:
            On success (post-§4.3.7):
                {"ok": True, "session_id": "...", "subagent_id":
                "...", "runner_pid": <int>, "apparmor_profile":
                "...", "cgroup_path": "..."}

            On domain failure (this is what the stub always returns
            once args validate, until §4.3.3-§4.3.7 fill in real
            spawn logic):
                {"ok": False, "error": "...", "stage": "..."}

        Raises:
            ValueError: for malformed args (missing required keys,
                wrong types, parent_session_id mismatch).  The
                dispatcher translates these to typed envelope
                errors on the wire.
            RuntimeError: if the handler is closed (shutdown was
                called).
        """
        if self._closed:
            raise RuntimeError(
                "SpawnIsolatedRunnerHandler is closed"
            )

        # ── Validate required keys + types ─────────────────────
        for key in _REQUIRED_KEYS:
            if key not in args:
                raise ValueError(
                    f"subagent.spawn_isolated_runner: missing "
                    f"required arg {key!r}"
                )

        if not isinstance(args["parent_session_id"], str) or not args["parent_session_id"]:
            raise ValueError(
                "subagent.spawn_isolated_runner: parent_session_id "
                "must be a non-empty str"
            )
        if not isinstance(args["subagent_id"], str) or not args["subagent_id"]:
            raise ValueError(
                "subagent.spawn_isolated_runner: subagent_id must "
                "be a non-empty str"
            )
        if not isinstance(args["profile_payload"], dict):
            raise ValueError(
                "subagent.spawn_isolated_runner: profile_payload "
                "must be a dict"
            )
        if not isinstance(args["task"], str):
            raise ValueError(
                "subagent.spawn_isolated_runner: task must be a str"
            )
        if not isinstance(args["workspace_path"], str) or not args["workspace_path"]:
            raise ValueError(
                "subagent.spawn_isolated_runner: workspace_path "
                "must be a non-empty str"
            )

        # Optional type checks — only run when key is present.
        if "agent_params" in args and not isinstance(args["agent_params"], (dict, type(None))):
            raise ValueError(
                "subagent.spawn_isolated_runner: agent_params, if "
                "present, must be a dict or None"
            )
        if "display_name" in args and not isinstance(args["display_name"], (str, type(None))):
            raise ValueError(
                "subagent.spawn_isolated_runner: display_name, if "
                "present, must be a str or None"
            )
        if "parent_agent_id" in args and not isinstance(args["parent_agent_id"], (str, type(None))):
            raise ValueError(
                "subagent.spawn_isolated_runner: parent_agent_id, "
                "if present, must be a str or None"
            )

        # ── Confused-deputy: parent_session_id echo check ──────
        if args["parent_session_id"] != self._parent_session_id:
            raise ValueError(
                f"subagent.spawn_isolated_runner: payload "
                f"parent_session_id "
                f"{args['parent_session_id']!r} != handler-bound "
                f"parent_session_id {self._parent_session_id!r}"
            )

        logger.info(
            "subagent.spawn_isolated_runner: validation passed for "
            "parent_session=%s subagent_id=%s (routed=%s)",
            self._parent_session_id, args["subagent_id"],
            self._session_manager is not None,
        )

        # ── Route to SessionManager helper if dependencies wired ──
        # Phase 4 §4.3.3: when ``set_spawn_dependencies`` has been
        # called, hand off to the daemon's
        # ``SessionManager._spawn_isolated_runner(...)`` helper.  The
        # helper itself still returns ``stage=sub_profile`` today —
        # progressing past that stage is §4.3.4's job.  Routing here
        # in §4.3.3 means the handler→helper bridge exists and is
        # exercised; subsequent sub-commits only need to extend the
        # helper body, not touch the handler.
        if self._session_manager is not None:
            # Strip the ``isolated`` control flag from agent_params
            # before forwarding — it's a routing flag, not template
            # data the subagent should see.  Per Audit 5 in
            # ``phase4_implementation_audits.md``.
            forwarded_agent_params: Optional[Dict[str, Any]] = None
            raw_agent_params = args.get("agent_params")
            if raw_agent_params is not None:
                forwarded_agent_params = {
                    k: v for k, v in raw_agent_params.items()
                    if k != "isolated"
                }
            return self._session_manager._spawn_isolated_runner(
                parent_session_id=args["parent_session_id"],
                subagent_id=args["subagent_id"],
                profile_payload=args["profile_payload"],
                task=args["task"],
                workspace_path=args["workspace_path"],
                agent_params=forwarded_agent_params,
                display_name=args.get("display_name"),
                parent_agent_id=args.get("parent_agent_id"),
            )

        # ── Unwired stub (no SessionManager reference yet) ─────
        # Returned when ``set_spawn_dependencies`` hasn't been called
        # — typically a transient state during session bootstrap, or
        # a misconfigured test harness.  Distinct ``stage=spawn``
        # message from the §4.3.4+ helper stages so operators can
        # tell "RPC primitive not bridged" apart from "helper
        # reached but next stage not implemented".
        return {
            "ok": False,
            "error": (
                "subagent.spawn_isolated_runner handler not yet "
                "wired to SessionManager (Phase 4 §4.3.3 "
                "set_spawn_dependencies bridge missing).  This is "
                "typically a transient state during session "
                "bootstrap or a misconfigured test harness.  If "
                "you see this in production, the parent session's "
                "runner spawn may have failed to call "
                "handler.set_spawn_dependencies(session_manager) — "
                "see SessionManager._spawn_session_runner_unconditional.  "
                "Workaround: omit agent_params.isolated to use the "
                "default-share path."
            ),
            "stage": STAGE_SPAWN,
        }

    def shutdown(self) -> None:
        """Mark the handler closed.

        Phase 4 §4.3.2 stub: the handler holds no in-flight state,
        so shutdown is a no-op beyond the closed flag.  §4.3.6's
        cross-runner forwarding will add in-flight spawn tracking +
        cleanup; the corresponding shutdown ladder lands with that
        commit.

        Idempotent — safe to call multiple times.
        """
        self._closed = True


def register(
    rpc_server: Any,  # server.runner_rpc_server.RunnerRPCServer
    handler: SpawnIsolatedRunnerHandler,
) -> None:
    """Register *handler*'s ``handle`` method as the
    ``subagent.spawn_isolated_runner`` RPC method on *rpc_server*.

    Convenience indirection so the bootstrap site
    (``JaatoServer.set_runner_rpc()``) doesn't repeat the method-
    name string literal.
    """
    rpc_server.register("subagent.spawn_isolated_runner", handler.handle)
