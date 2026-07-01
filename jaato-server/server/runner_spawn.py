"""Per-session runner subprocess spawn helper (Phase 3 §3.12).

Extracted from ``server/__main__.py:_spawn_session_runner`` (Phase 2
task 2.3) so both the IPC apparmor pre-init hook and the WS-server
apparmor pre-init hook can spawn runners through the same code
path.

Lifecycle:

1. Caller (an apparmor pre-init hook) finishes
   ``apparmor.provision_profile`` for the session.
2. Caller invokes :func:`spawn_session_runner`.
3. Function spawns the runner via :class:`RunnerSpawner`, opens a
   :class:`RunnerRPCClient` against the parent end of the
   socketpair, starts the read-loop on the daemon's asyncio loop,
   and attaches the RPC handle onto the JaatoServer via
   ``server.set_runner_rpc(rpc, spawned)``.
4. After this returns, plugins discovered during
   ``server.initialize()`` see ``registry.runner_rpc`` set at
   configure time.
5. Caller invokes :func:`dispatch_bootstrap_envelope` to send the
   ``session.bootstrap`` RPC so the runner-side
   :class:`shared.jaato_session.JaatoSession` host is populated.

Failures raise; the caller catches and downgrades the session to
``sandbox_mode = "soft"`` per the §4.6 fallback contract.

Phase 3 §7c step 2: the bootstrap-envelope dispatch + the
envelope builder live in this module so both IPC + WS callers
share the implementation.  Pre-§7c-step-2 the helpers lived only
in ``server/__main__.py`` (the IPC entry point) and the WS path
had no bootstrap dispatch at all — every WS session left the
runner-side ``JaatoSession`` host unpopulated, blocking the
seat-flip on WS sessions.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING


if TYPE_CHECKING:  # pragma: no cover — types only
    from shared.session_envelope import SessionInitEnvelope


logger = logging.getLogger(__name__)


def _pool_enabled() -> bool:
    """Read the ``JAATO_RUNNER_POOL_ENABLED`` env var.

    Pool PR 5e: default flipped to **enabled**.  Sessions consume
    pre-warm pool slots unless the operator explicitly disables the
    pool by setting ``JAATO_RUNNER_POOL_ENABLED=false`` (or
    ``0`` / ``no`` / ``off``).

    Rationale: PRs 4-5d shipped the pool's structural correctness,
    operational robustness (subreaper + watchdog), startup determinism
    (READY handshake), and observability (telemetry).  Default-on
    delivers the cascade speedup to every operator without per-deployment
    configuration.

    Disable cases (set env var to a falsy value):
      - Suspected pool regression — bisect against cold-spawn.
      - Operator host with extremely tight memory budget where the
        idle pool's ~150-300 MiB footprint is undesirable.
      - Custom plugin set that doesn't work with fork-from-template
        (would need to be a runner-tier plugin with module-global
        non-fork-safe state — none exist today per the 2026-05-13
        audit).

    Pre-PR-5e the default was opt-in (off); explicit ``true`` was
    required.  Operators who set ``true`` explicitly are unaffected
    by the flip.  Operators who never set the var get the pool
    automatically post-PR-5e.
    """
    raw = os.environ.get("JAATO_RUNNER_POOL_ENABLED", "").strip().lower()
    # Empty (unset) → enabled.  Explicit-falsy → disabled.  Anything
    # else (truthy or unrecognised) → enabled.
    return raw not in ("0", "false", "no", "off")


def spawn_session_runner(
    *,
    server: Any,  # JaatoServer (forward-typed; importing the real
                  # type creates a cycle through server/core.py).
    session_id: str,
    workspace_path: str,
    profile_name: str,
    daemon_loop: asyncio.AbstractEventLoop,
    disable_confine: bool = False,
    cgroup_attach: Optional[Callable[[], None]] = None,
    pool_manager: Any = None,
    cascade_driver_id: Optional[str] = None,
) -> None:
    """Spawn the per-session runner subprocess and wire its RPC handle
    onto the JaatoServer.

    Pool PR 4: when *pool_manager* is supplied AND
    ``JAATO_RUNNER_POOL_ENABLED`` is set AND the pool has an idle
    slot, the session reuses that slot's pre-warm runner subprocess
    instead of paying the per-session fork+exec+plugin-imports cost
    (~10-15s on v62 step 6).  The slot's daemon-side socket is
    wrapped in a :class:`SpawnedRunner` and the rest of the bootstrap
    flow (RunnerRPCClient + session.bootstrap RPC) is identical to
    the cold-spawn path — by design, so PR 4 doesn't fork a parallel
    code path that has to be maintained alongside the existing one.

    Args:
        server: The session's :class:`JaatoServer` instance.
        session_id: Session identifier (passed via env to the runner).
        workspace_path: Session workspace; used both as the runner's
            cwd and as the prefix for the per-session log file path
            (plan §5.1).
        profile_name: AppArmor profile name (already loaded in the
            kernel).  Required unless *disable_confine* is set —
            then it can be empty (the runner runs unconfined).
        daemon_loop: The daemon's main asyncio loop — needed to run
            ``RunnerRPCClient.start()`` since it's async.
        disable_confine: Phase 3 §7a — skip kernel-level
            confinement.  Used by the always-spawn path when the
            client did not opt into apparmor.  The runner spawns
            with ``JAATO_RUNNER_DISABLE_CONFINE=1``; tool execution
            runs in the runner subprocess but without an AppArmor
            profile applied.  The runner-RPC dispatch surface is
            still available; the trade-off is process isolation
            without kernel-enforced FS confinement.
        cgroup_attach: Phase 3 §7d — optional zero-arg callable
            forwarded to ``RunnerSpawner.spawn`` to migrate the
            forked child into the per-session cgroup before exec.
            Caller (the WS pre-init hook) obtains via
            ``CgroupsManager.make_attach_callback(session_id)``
            after provisioning the cgroup.  ``None`` means no
            cgroup attach — the IPC session path (no cgroup
            provisioned) passes ``None``.
        cascade_driver_id: Phase 2 cascade-sharing tenant ID.  When
            non-None, the pool acquire walks for a slot already
            affined to this cascade (warm plugin state, warm LSP
            connections from prior sessions).  ``None`` (default)
            means standalone session — pool acquires any PURE IDLE
            slot, no cascade affinity stamped.
        pool_manager: Pool PR 4 — the daemon's
            :class:`server.runner_pool.PoolManager`.  When non-None
            AND the pool routing flag is enabled AND the pool has an
            idle slot, the session is served by a pre-warm slot
            instead of a cold-spawned runner.  ``None`` (or empty
            pool, or flag disabled) falls back to cold-spawn.

            Pool routing gates (post-PR 5a):
              - ``pool_manager`` is wired by the daemon.
              - ``JAATO_RUNNER_POOL_ENABLED`` is set.
              - No ``cgroup_attach`` supplied (PR 5b territory —
                slot is template-child, mid-life cgroup migration
                requires subreaper coordination).

            Per-slot AppArmor self-confinement landed in PR 5a:
            slots accept ``envelope.profile_name`` and call
            ``aa_change_profile`` in ``bootstrap_session`` step 1c
            BEFORE plugin initialize / prefetch.  The pre-PR-5a
            ``disable_confine`` gate is removed — sessions with
            AppArmor opt-in are now eligible for pool routing.

    Raises:
        RuntimeError: when *daemon_loop* is None or the runner-RPC
            start times out.  Caller catches and downgrades to
            ``sandbox_mode = "soft"`` (or omits the field entirely
            for the always-spawn-no-apparmor path).
        Exception: any spawn / RPC failure.  Caller catches and
            downgrades.
    """
    from server.runner_spawner import SpawnedRunner, RunnerSpawner
    from server.runner_rpc_client import RunnerRPCClient

    if daemon_loop is None:
        raise RuntimeError(
            "spawn_session_runner: daemon loop unavailable; cannot "
            "start RunnerRPCClient"
        )

    log_path: Optional[str] = None
    if workspace_path:
        log_dir = os.path.join(workspace_path, ".jaato", "logs")
        log_path = os.path.join(log_dir, f"runner-{session_id}.log")

    # ----- Pool routing (pool PR 4 + 5a) -----
    # Pool-served path is gated to sessions that:
    #   (a) Have a pool_manager wired by the daemon.
    #   (b) Have JAATO_RUNNER_POOL_ENABLED set.
    #   (c) Don't need a cgroup_attach.  Pool slots are forked from
    #       the template (already in daemon cgroup); migrating
    #       mid-life is PR 5b + subreaper-fix territory.
    # (PR 5a removed the disable_confine gate — apparmor sessions are
    # now eligible for pool routing because the slot self-confines
    # to envelope.profile_name in bootstrap_session step 1c.)
    spawned: Optional[SpawnedRunner] = None
    pool_served = False
    if (
        pool_manager is not None
        and _pool_enabled()
        and cgroup_attach is None
    ):
        slot = pool_manager.acquire_slot(
            cascade_driver_id=cascade_driver_id,
        )
        if slot is not None:
            spawned = SpawnedRunner(
                pid=slot.pid,
                parent_socket=slot.sock,
                # Slot will transition to ``profile_name`` itself in
                # bootstrap_session step 1c.  Record the target on
                # the SpawnedRunner for audit / diagnostic surfaces.
                profile_name=profile_name,
                session_id=session_id,
            )
            spawned.pool_slot = slot  # Phase 2: keep ref for return path.
            pool_served = True
            # Phase 2 cascade-sharing teardown path: server.shutdown()
            # needs the pool_manager to return the slot after a
            # successful session_end RPC.  Stash it alongside the
            # SpawnedRunner so JaatoServer.shutdown() can find it.
            # Only set when the slot actually came from the pool; cold-
            # spawn paths leave it None.
            server._pool_manager_ref = pool_manager
            logger.info(
                "spawn_session_runner: session %s served by pool slot "
                "pid=%d cascade=%s (warm imports inherited; slot will "
                "self-confine to profile=%s)",
                session_id, slot.pid, slot.cascade_id or "(standalone)",
                profile_name or "(unconfined)",
            )
        else:
            logger.info(
                "spawn_session_runner: session %s — pool empty, falling "
                "back to cold-spawn", session_id,
            )

    # ----- Cold-spawn fallback (pre-PR-4 behavior) -----
    if spawned is None:
        spawner = RunnerSpawner()

        # Phase 5 §5.1b: forward the app-layer ``RuntimeLimits`` fields
        # via ``RunnerSpawner.spawn``'s ``max_output_chars`` /
        # ``tool_timeout_seconds`` kwargs.  The spawner translates them
        # into the ``JAATO_RUNNER_MAX_OUTPUT_CHARS`` /
        # ``JAATO_RUNNER_TOOL_TIMEOUT_SECONDS`` env vars the runner-side
        # cli plugin reads at startup.  Source of truth is
        # ``server._profile.runtime_limits`` — same field the WS path
        # consults for cgroup provision (see
        # ``server/websocket.py:620-624``).  No defaulting on the
        # mainline path: a profile that omits ``runtime_limits`` keeps
        # the runner's compile-time defaults (§5.1's
        # ``apply_isolated_defaults`` is specific to
        # ``agent_params.isolated=true``).  See
        # docs/design/phase5_5_1b_mainline_runtime_limits_passthrough_audit.md.
        profile = getattr(server, "_profile", None)
        runtime_limits = getattr(profile, "runtime_limits", None) if profile else None
        max_output_chars = (
            runtime_limits.max_output_bytes
            if runtime_limits is not None else None
        )
        tool_timeout_seconds = (
            runtime_limits.tool_timeout_seconds
            if runtime_limits is not None else None
        )

        spawned = spawner.spawn(
            profile_name=profile_name,
            session_id=session_id,
            workspace_path=workspace_path,
            log_path=log_path,
            max_output_chars=max_output_chars,
            tool_timeout_seconds=tool_timeout_seconds,
            disable_confine=disable_confine,
            cgroup_attach=cgroup_attach,
        )

    # Phase 3 cascade-sharing hotfix (server 0.6.150+): reuse the
    # slot's existing RunnerRPCClient when it's a returned pool slot.
    # Creating a second RunnerRPCClient on the same socket fails —
    # the asyncio transport adopted by ``rpc.start`` binds the
    # socket exclusively.  Subsequent ``start`` calls on the same
    # socket fail; ``server._runner_rpc`` ends up None; sessions
    # crash with ``NoneType.session_send_message_threadsafe``.
    # See PR #173.
    pool_slot = getattr(spawned, "pool_slot", None) if pool_served else None
    existing_rpc = getattr(pool_slot, "rpc", None) if pool_slot else None
    if existing_rpc is not None:
        # Returned slot — reuse the rpc client.  Per-session state
        # was cleared by the slot-return path's
        # reset_for_slot_reuse call (see core.py shutdown cascade
        # branch).  Transport (reader/writer/read_task) stays
        # bound to the slot's socket.
        rpc = existing_rpc
        logger.info(
            "spawn_session_runner: session %s reused slot's rpc client "
            "(no new transport adopt needed)", session_id,
        )
    else:
        rpc = RunnerRPCClient(
            spawned.parent_socket,
            runner_pid=spawned.pid,
            loop=daemon_loop,
        )
        fut = asyncio.run_coroutine_threadsafe(rpc.start(), daemon_loop)
        fut.result(timeout=10.0)
        # First session on this slot — stash the rpc on the slot
        # so subsequent same-cascade sessions can reuse it.
        if pool_slot is not None:
            pool_slot.rpc = rpc
            logger.debug(
                "spawn_session_runner: session %s — new rpc client "
                "created + stashed on pool slot pid=%d",
                session_id, pool_slot.pid,
            )

    server.set_runner_rpc(rpc, spawned)
    logger.info(
        "runner spawned for session %s: pid=%d profile=%s log=%s "
        "confined=%s pool_served=%s",
        session_id, spawned.pid,
        profile_name or "(none)",
        log_path or "(inherited)",
        not disable_confine,
        pool_served,
    )


def build_session_envelope(
    *,
    server: Any,  # JaatoServer (forward-typed; importing the real type
                  # creates a cycle through server/core.py).
    session_id: str,
    workspace_path: Optional[str],
    profile_name: str,
) -> "SessionInitEnvelope":
    """Build a :class:`SessionInitEnvelope` from a pre-init JaatoServer.

    Phase 3 §7c step 2: relocated from ``server/__main__.py`` so
    both IPC and WS callers share the implementation.  Reads the
    resolved profile from the server (set in
    ``SessionManager._create_session_impl`` before the pre-init
    hooks fire) and constructs the envelope the runner-side host
    needs.

    Fallback rules (no hardcoded defaults):
    - ``model_name``: profile.model → ``session_env["MODEL_NAME"]``.
      Empty if neither declares; runner-side ``_validate_envelope``
      raises ``BootstrapError(stage="validate")`` audibly.
    - ``provider_name``: profile.provider → ``session_env["JAATO_PROVIDER"]``.
      Same empty-stays-empty rule.

    Args:
        server: The :class:`JaatoServer` instance — has ``_profile``
            set to a :class:`SubagentProfile` if a profile was
            resolved.  ``None`` for inline-spec / no-profile sessions.
        session_id: Stable session identifier.
        workspace_path: Session's workspace; ``None`` for headless.
        profile_name: AppArmor profile name (informational; the
            envelope's ``profile_name`` field carries it for
            audit attribution).

    Returns:
        A :class:`SessionInitEnvelope` ready for
        :meth:`server.runner_rpc_client.RunnerRPCClient.bootstrap_session_threadsafe`.
    """
    from shared.session_envelope import SessionInitEnvelope

    profile = getattr(server, "_profile", None)
    provider_name = ""
    model_name = ""
    plugin_specs: list = []
    plugin_configs_dict: dict = {}
    preloaded: set = set()
    system_instructions: Optional[str] = None
    gc_dict: Optional[dict] = None
    env_overrides: dict = {}
    model_tiers_dict: Optional[Dict[str, Any]] = None

    if profile is not None:
        provider_name = getattr(profile, "provider", None) or ""
        model_name = getattr(profile, "model", None) or ""
        names = list(getattr(profile, "plugins", []) or [])
        preloaded = set(getattr(profile, "preloaded_plugins", set()) or set())
        # v3 (2026-05-14): forward profile.model_tiers to the runner so
        # it can resolve ``ModelTierConfig`` and register the
        # ``enter_tier`` lifecycle tool.  Pre-v3 this never reached the
        # runner — sessions silently ran in single-model mode regardless
        # of profile config.  Empty dict means "no tiers declared";
        # serialise as ``None`` to keep the envelope minimal.
        raw_tiers = getattr(profile, "model_tiers", None) or None
        if raw_tiers:
            model_tiers_dict = {str(k): v for k, v in raw_tiers.items()}
        # Phase 4 §C: carry the full profile.plugin_configs at the
        # envelope's top level (schema v2) so auto-loaded plugins not
        # named in profile.plugins (e.g. permission) receive their
        # profile overrides.  Closes backlog §3.3c.X.  Per-entry
        # ``config`` is no longer set — entries are {name, preload} only.
        #
        # Server 0.6.123+: values flow through ``expand_plugin_configs``
        # so ``${VAR}`` references AND secret URIs (``pass://`` /
        # ``vault://``) resolve daemon-side before the runner sees
        # them.  Pre-0.6.123 the envelope copied profile.plugin_configs
        # LITERALLY — any ``pass://`` in
        # ``plugin_configs.<provider>.api_key`` reached the runner
        # unresolved, and the AppArmor-confined runner can't exec
        # ``pass`` to resolve it itself (per
        # ``feedback_secret_resolution_stays_daemon_side`` memory).
        # Symmetric to the ``envelope.session_env`` resolution channel
        # (PR #91 → #92).  Same trust posture: resolved plaintext on
        # the daemon↔runner socketpair, never logged or forwarded.
        from shared.plugins.subagent.config import expand_plugin_configs
        raw_plugin_configs = {
            k: dict(v)
            for k, v in (getattr(profile, "plugin_configs", {}) or {}).items()
        }
        plugin_configs_dict = expand_plugin_configs(
            raw_plugin_configs,
            workspace_root_override=getattr(server, "_workspace_path", None),
        )
        # Quirks injection (server 0.6.194+).  Top-level
        # ``profile.quirks`` is threaded into the provider's
        # plugin_configs namespace under the ``"quirks"`` key so the
        # runner-side provider plugin reads it from
        # ``ProviderConfig.extra["quirks"]`` at session init.  Lives
        # here at the envelope-build site (NOT at
        # ``core.py:_build_profile_session_kwargs`` — that's dead code
        # per PR #240's diagnosis comment in runner/session.py:~1015)
        # so the value actually reaches the runner.  Mirror in
        # ``shared/plugins/subagent/plugin.py`` covers the
        # daemon-spawned subagent path.  See
        # ``SubagentProfile.quirks`` docstring +
        # ``feedback_llama31_vllm_auto_mode_stringifies_args``.
        profile_quirks = getattr(profile, "quirks", None) or {}
        effective_provider_for_quirks = (
            provider_name or getattr(profile, "provider", None)
        )
        if profile_quirks and effective_provider_for_quirks:
            provider_cfg = dict(
                plugin_configs_dict.get(effective_provider_for_quirks) or {}
            )
            provider_cfg["quirks"] = dict(profile_quirks)
            plugin_configs_dict[effective_provider_for_quirks] = provider_cfg
        profile_tool_scopes = getattr(profile, "tool_scopes", {}) or {}
        for name in names:
            spec = {"name": name, "preload": name in preloaded}
            # Per-plugin tool allow-list (profile ``tools:[...]`` modifier)
            # rides alongside name/preload on the envelope entry so the
            # runner-side ``_build_session`` can scope this session's
            # wire surface.  Absent → all of the plugin's tools exposed.
            scope = profile_tool_scopes.get(name)
            if scope:
                spec["tools"] = list(scope)
            plugin_specs.append(spec)
        system_instructions = getattr(profile, "system_instructions", None)
        gc_obj = getattr(profile, "gc", None)
        if gc_obj is not None:
            # GCProfileConfig has ``type`` + ``config`` (dict).  Flatten
            # to a single dict for the envelope.
            gc_type = getattr(gc_obj, "type", None)
            gc_config = getattr(gc_obj, "config", None) or {}
            if gc_type:
                gc_dict = {"type": gc_type, **dict(gc_config)}
        env_overrides = dict(getattr(profile, "env", {}) or {})

    # Profile-less fallback: read ``MODEL_NAME`` / ``JAATO_PROVIDER``
    # from the daemon's resolved session env (workspace ``.env`` +
    # profile.env + env_overrides, populated by
    # ``JaatoServer._resolve_session_env``).  Closes the profile-less
    # ``jaato --new-session`` regression introduced by §7c step 1
    # (commit 6406fe35, 2026-05-09) which made the runner-side
    # bootstrap always run + always validate ``envelope.model_name``.
    # Pre-§7c profile-less worked because the runner never validated
    # the envelope's model_name field (it was read from the daemon's
    # session env post-fork).
    #
    # No hardcoded default — if neither the profile NOR the session
    # env declares ``MODEL_NAME`` / ``JAATO_PROVIDER``, the field
    # stays empty and the runner-side ``_validate_envelope`` raises
    # ``BootstrapError(stage="validate")`` with the missing field
    # surfaced.  Loud failure beats a silent guess; the user's
    # standing rule against hardcoded fallbacks (global CLAUDE.md)
    # is honored here.  Pre-PR-100 (this PR) the provider had a
    # silent ``"anthropic"`` default which papered over the
    # configuration gap and could mask a Vertex / OpenRouter
    # / etc. session creation slipping through with wrong provider.
    session_env = getattr(server, "_session_env", {}) or {}
    if not model_name:
        model_name = session_env.get("MODEL_NAME", "") or ""
    if not provider_name:
        provider_name = session_env.get("JAATO_PROVIDER", "") or ""

    # Phase 3 post-Step-7 Path C: read PROJECT_ID + LOCATION daemon-
    # side so the envelope carries the provider-connect args the
    # runner-side ``bootstrap_session`` needs to call
    # ``runtime.connect(project, location)`` before
    # ``runtime.create_session`` (which guards on ``_connected``).
    # Non-Vertex providers tolerate empty strings; Vertex AI sessions
    # pick up real values from env.  Mirrors the daemon-side reads
    # at ``core.py:1550-1551``.
    try:
        import os as _os
        project_val = _os.environ.get("PROJECT_ID", "") or ""
        location_val = _os.environ.get("LOCATION", "") or ""
    except Exception:
        project_val = ""
        location_val = ""

    # Phase 4 §D: read agent_params from the per-session JaatoServer.
    # SessionManager._construct_and_initialize_server stashes the
    # originating create_session.agent_params there so this builder
    # can forward them on the wire envelope.  Pre-§D this was
    # hard-coded to ``{}`` and any runner-side prefetch script
    # reading ``context.agent_params`` saw an empty dict.
    agent_params_dict = dict(getattr(server, "_agent_params", {}) or {})

    # Read the daemon-side resolved agent identity + profile's
    # completion_payload_schema.  Both were previously hardcoded /
    # omitted:
    #
    # - ``agent_id`` was hardcoded ``"main"`` regardless of
    #   ``--agent <name>`` resolution.  Result: the runner-side
    #   ``JaatoSession._agent_id`` stayed at its ``__init__`` default
    #   of ``"main"`` for EVERY session, because the only attribute
    #   that updates ``_agent_id`` post-construction is
    #   ``set_ui_hooks`` — and the runner-side bootstrap installs the
    #   UI hooks shim via direct attribute write (rpc.py:3178-3185),
    #   bypassing ``set_ui_hooks``.  Downstream consequence:
    #   ``AgentCompletedEvent.agent_id`` always carried ``"main"``;
    #   reactor where-clauses keying on the logical agent identity
    #   (e.g. ``agent_id == "discovery"``) silently missed.
    #
    # - ``completion_payload_schema`` was missing from the
    #   constructor entirely.  Even though
    #   :class:`SessionInitEnvelope` declared the field,
    #   ``build_session_envelope`` never populated it — the runner-
    #   side ``JaatoSession._completion_payload_schema`` stayed
    #   ``None`` for profile-declared payload schemas, and
    #   ``LifecycleTools._execute_signal_completion`` fell back to
    #   the legacy ``summary`` string path instead of validating
    #   the typed payload.
    #
    # Both regressions surfaced 2026-05-12 by the kb-enablement-2.0
    # cascade smoke test.  See:
    # - ``docs/design/per_session_confined_runner_phase5_plan.md``
    # - The §7c step-2 relocation commit that copied the body
    #   verbatim from server/__main__.py without catching the gaps.
    profile_completion_schema = None
    # Profile-declared completion processors (server 0.6.125+).
    # Replaces the prior split between completion_artifacts +
    # completion_validators.  Both were broken in the runner path
    # pre-0.6.122 (envelope didn't ship them); collapsed into one
    # unified surface as of 0.6.125.  Serialise to wire-dict shape
    # the runner side reconstructs without importing
    # CompletionProcessor on the wire boundary.
    profile_completion_processors: List[Dict[str, Any]] = []
    if profile is not None:
        profile_completion_schema = getattr(
            profile, "completion_payload_schema", None,
        )
        raw_processors = getattr(profile, "completion_processors", []) or []
        for entry in raw_processors:
            if hasattr(entry, "script"):
                profile_completion_processors.append({
                    "script": getattr(entry, "script", None),
                    "output": getattr(entry, "output", None),
                    "on_error": getattr(entry, "on_error", "fail_completion"),
                    "description": getattr(entry, "description", None),
                    "phase": getattr(entry, "phase", "finalization"),
                })
            elif isinstance(entry, dict):
                profile_completion_processors.append(dict(entry))

    # PR #91 Y fix: ship the FULLY-RESOLVED per-session env to the
    # runner.  ``server._session_env`` is populated by the daemon's
    # :meth:`JaatoServer._resolve_session_env` from workspace ``.env``
    # + profile.env + env_overrides, with ``${VAR}`` cross-references
    # expanded AND secret URIs (``pass://`` / ``vault://`` / etc.)
    # resolved via the daemon's SecretResolver entry points.  The
    # runner applies this dict to ``os.environ`` verbatim during
    # bootstrap — no resolver discovery, no ``pass`` exec (which
    # AppArmor correctly blocks).
    #
    # Trust posture: the runner-rpc socketpair (daemon ↔ runner) is
    # FD-pass only — not in the filesystem, not on the network.
    # Resolved secrets transit that channel in plaintext, same as
    # pre-PR-91 fork-inherit ``os.environ`` semantics.  The audit
    # behind PR #92 verified envelope.session_env is never logged,
    # persisted, or forwarded to clients.
    resolved_session_env = dict(getattr(server, "_session_env", {}) or {})

    return SessionInitEnvelope(
        session_id=session_id,
        workspace_path=workspace_path,
        profile_name=profile_name,
        provider_name=provider_name,
        model_name=model_name,
        plugins=plugin_specs,
        plugin_configs=plugin_configs_dict,
        system_instructions=system_instructions,
        agent_id=getattr(server, "_main_agent_id", "main"),
        gc=gc_dict,
        agent_params=agent_params_dict,
        config_root=getattr(server, "config_root", None),
        env_overrides=env_overrides,
        session_env=resolved_session_env,
        project=project_val,
        location=location_val,
        completion_payload_schema=profile_completion_schema,
        completion_processors=profile_completion_processors,
        model_tiers=model_tiers_dict,
        # Phase 2 cascade-sharing (envelope v4): forward the cascade
        # tenant ID stashed on the server by
        # ``SessionManager._construct_and_initialize_server``.  Runner
        # stashes onto JaatoSession so subagent create_session calls
        # auto-inherit via runtime.create_session().
        cascade_driver_id=getattr(server, "_cascade_driver_id", None),
        # 2026-06-06: ferry the two daemon-resolved system-instruction
        # knobs to the runner.  See SessionInitEnvelope field docstrings
        # for the bug history — both knobs were set correctly on
        # ``JaatoServer`` daemon-side but never reached the runner's
        # ``JaatoSession.configure`` over the wire, making them silent
        # no-ops.  ``getattr`` with the default keeps backward compat
        # with daemons that predate these attributes (the attributes
        # are set in JaatoServer.__init__ from BootstrapEnvelope so
        # they should always be present, but the defaults are also the
        # documented "no-op" values for both knobs).
        suppress_base_instructions=getattr(
            server, "_suppress_base_instructions", False,
        ),
        system_instruction_override=getattr(
            server, "_system_instruction_override", None,
        ),
        # 2026-06-21: client-provided ("host") tools registered via the WS/IPC
        # protocol BEFORE session.new (e.g. a telegram client's send_to_telegram),
        # ferried so the RUNNER-tier model SEES them in list_tools.  Pre-fix they
        # registered only on the daemon registry and the runner model was blind
        # (#344-sibling daemon-vs-runner split).  Execution forwards back to the
        # daemon's proxy executor via daemon.plugin_execute (sentinel name).
        client_tools=list(
            getattr(server, "client_tool_schemas", {}).values()
        ),
    )


def dispatch_bootstrap_envelope(
    *,
    server: Any,  # JaatoServer (forward-typed; see above).
    session_id: str,
    workspace_path: Optional[str],
    profile_name: str,
    timeout: float = 30.0,
) -> None:
    """Send the ``session.bootstrap`` RPC so the runner-side
    :class:`shared.jaato_session.JaatoSession` host is populated.

    Phase 3 §7c step 2: shared between the IPC + WS spawn paths.
    Pre-§7c-step-2 the dispatch lived only in
    ``server/__main__.py`` (IPC); WS sessions left the runner-side
    host unpopulated.

    Bootstrap failure does NOT propagate — the daemon-side
    :class:`JaatoSession` is still authoritative during the §7c
    rollout window (steps 3-7 progressively migrate authority away).
    Failures log at WARNING so operators notice the runner host
    isn't actually populated.

    Args:
        server: The session's :class:`JaatoServer` instance.  Must
            have ``runner_rpc`` set (i.e. :func:`spawn_session_runner`
            already ran).
        session_id: Session identifier.
        workspace_path: Session workspace; threaded into the
            envelope.
        profile_name: AppArmor profile name; threaded into the
            envelope for audit attribution.
        timeout: Wall-clock cap on the bootstrap RPC, seconds.
            Default 30s — generous to absorb runner-side plugin
            discovery + provider connect latency.
    """
    rpc = server.runner_rpc
    if rpc is None:
        # Defensive: a caller invoking this without spawn_session_runner
        # having succeeded means the spawn helper raised.  Log + return;
        # no point dispatching to a None handle.
        logger.debug(
            "dispatch_bootstrap_envelope: server.runner_rpc is None for "
            "session %s — skipping bootstrap (spawn likely failed)",
            session_id,
        )
        # Server 0.6.169+ (bootstrap-time visibility): surface this
        # as a terminal event so cascade observers + reactor rules on
        # ``session.terminated where reason='error'`` can react.
        # Without this, spawn-helper failures dead-end the cascade
        # silently (driver hits IPC timeout instead of getting the
        # actionable error_type/error_summary).  See
        # ``_emit_bootstrap_terminated`` helper for the exception-safe
        # emit + the rationale memory
        # ``project_backlog_bootstrap_time_visibility_gap``.
        _emit_bootstrap_terminated(
            server=server,
            session_id=session_id,
            exc=RuntimeError(
                "spawn_session_runner did not populate "
                "server.runner_rpc — session never reached bootstrap "
                "phase.  Check earlier ERROR logs in this turn for "
                "the spawn-helper failure (apparmor compose, slot "
                "acquisition, runner-spawn fork, etc.)."
            ),
        )
        return

    try:
        envelope = build_session_envelope(
            server=server,
            session_id=session_id,
            workspace_path=workspace_path,
            profile_name=profile_name,
        )
        result = rpc.bootstrap_session_threadsafe(envelope, timeout=timeout)
        logger.info(
            "runner session.bootstrap acknowledged for %s: %s",
            session_id, result,
        )
    except Exception as exc:  # noqa: BLE001 — boundary surface
        # %s on `exc` collapses arg-less exceptions (e.g. ``TimeoutError()``)
        # to "", leaving the colon trailing into the suffix and the operator
        # blind to root cause.  Surface ``error_type=`` + ``error=`` (same
        # shape as MODEL_THREAD_TERMINAL_ERROR in core.py) so the class name
        # is always visible, plus ``exc_info=True`` for the traceback when
        # the logger config preserves it.
        logger.warning(
            "runner session.bootstrap failed for %s: error_type=%s error=%s — "
            "daemon-side JaatoSession remains authoritative",
            session_id, type(exc).__name__, exc, exc_info=True,
        )
        # Server 0.6.169+ (bootstrap-time visibility): emit
        # SessionTerminatedEvent so cascade observers + reactor rules
        # see the failure.  Covers ANY bootstrap-time failure class
        # (SecretResolutionError, apparmor compose, runner RPC timeout,
        # plugin discovery errors) — generic ``except Exception``
        # mechanically catches everything that escapes the bootstrap
        # RPC.  Empirical motivation (peer 7:1, 2026-05-31): gpg-agent
        # passphrase expiry → SecretResolutionError → bootstrap WARNING
        # logged but cascade.py hung 3min on IPC timeout because no
        # terminal event surfaced.
        _emit_bootstrap_terminated(
            server=server, session_id=session_id, exc=exc,
        )
    finally:
        # Mark the runner ready REGARDLESS of bootstrap outcome: on success it
        # can service mid-session client-tool pushes + sends; on the
        # daemon-authoritative failure path the daemon-side JaatoSession still
        # handles the turn — either way, don't strand the push / send-gate on a
        # 30s readiness timeout.  This bootstrap-settled point is what the gates
        # now wait for, instead of racing the reused warm pool slot's
        # live-but-not-ready rpc handle (the re-attach client-tool-push stall).
        server.mark_runner_ready()
        # Runner is bootstrap-settled — re-emit the tool-id registry OFF the
        # event loop so the runner-tier tool names (prompt.* etc.) reach the
        # client.  Every ON-loop emit caller (emit_current_state / initialize /
        # _register_client_tools) now skips runner-tier to avoid the
        # daemon-side prompt_library filesystem walk on the loop (the re-attach
        # self-block); those names come ONLY from the runner
        # (session_get_tool_schemas), which can only run off-loop — here.  On a
        # bootstrap failure the runner RPC yields [] and this maps daemon-tier
        # only (harmless).
        try:
            server._emit_tool_id_registry_from_schemas()
        except Exception:  # noqa: BLE001 — re-emit must not strand bootstrap
            logger.debug(
                "post-bootstrap tool-id re-emit failed for %s",
                session_id, exc_info=True,
            )


def _emit_bootstrap_terminated(
    *, server: Any, session_id: str, exc: BaseException,
) -> None:
    """Emit ``SessionTerminatedEvent(reason="error")`` for a
    bootstrap-time failure.

    Server 0.6.169+ helper used by :func:`dispatch_bootstrap_envelope`
    at both failure paths (spawn-didn't-populate-rpc + bootstrap-rpc-
    raised).  By the time ``dispatch_bootstrap_envelope`` is invoked,
    ``server.set_event_callback`` has already been wired by
    ``session_manager.create_session`` (line ~4217), so emitted events
    flow through ``_emit_to_session`` → ``_dispatch_to_cascade_clients``
    and reach cascade observers + reactor rules.

    Routes through the single error-termination chokepoint
    ``JaatoServer._emit_error_termination_from_exc`` — which emits
    ``AgentErrorEvent`` (recovery first refusal; bootstrap has no auto-retry to
    wait on, the framework is out of moves immediately) THEN
    ``SessionTerminatedEvent(reason="error")`` (carrying ``error_summary`` /
    ``error_type``) — so the "AgentErrorEvent precedes every reason=error"
    invariant is structural here too.  ``agent_id`` falls back to ``"main"`` when
    ``server._main_agent_id`` isn't set (early bootstrap fail).

    The call is wrapped in a defensive try/except: a failure of the visibility
    path must not mask the underlying bootstrap failure or disrupt the
    session-creation caller's error handling.  Logs the failure to keep the
    audit trail intact.
    """
    try:
        agent_id = getattr(server, "_main_agent_id", None) or "main"
        server._emit_error_termination_from_exc(
            exc, session_id=session_id, agent_id=agent_id,
        )
    except Exception as emit_exc:  # noqa: BLE001 — defensive
        logger.warning(
            "_emit_bootstrap_terminated: SessionTerminatedEvent emit "
            "failed for session %s (root cause was %s: %s); cascade "
            "observers will not see the bootstrap failure for this "
            "session — investigate emit chain.  Emit error: %s: %s",
            session_id, type(exc).__name__, exc,
            type(emit_exc).__name__, emit_exc,
            exc_info=True,
        )
