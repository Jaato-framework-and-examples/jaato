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
from typing import Any, Callable, Optional, TYPE_CHECKING


if TYPE_CHECKING:  # pragma: no cover — types only
    from shared.session_envelope import SessionInitEnvelope


logger = logging.getLogger(__name__)


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
) -> None:
    """Spawn the per-session runner subprocess and wire its RPC handle
    onto the JaatoServer.

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

    Raises:
        RuntimeError: when *daemon_loop* is None or the runner-RPC
            start times out.  Caller catches and downgrades to
            ``sandbox_mode = "soft"`` (or omits the field entirely
            for the always-spawn-no-apparmor path).
        Exception: any spawn / RPC failure.  Caller catches and
            downgrades.
    """
    from server.runner_spawner import RunnerSpawner
    from server.runner_rpc_client import RunnerRPCClient

    if daemon_loop is None:
        raise RuntimeError(
            "spawn_session_runner: daemon loop unavailable; cannot "
            "start RunnerRPCClient"
        )

    spawner = RunnerSpawner()

    log_path: Optional[str] = None
    if workspace_path:
        log_dir = os.path.join(workspace_path, ".jaato", "logs")
        log_path = os.path.join(log_dir, f"runner-{session_id}.log")

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

    rpc = RunnerRPCClient(
        spawned.parent_socket,
        runner_pid=spawned.pid,
        loop=daemon_loop,
    )

    fut = asyncio.run_coroutine_threadsafe(rpc.start(), daemon_loop)
    fut.result(timeout=10.0)

    server.set_runner_rpc(rpc, spawned)
    logger.info(
        "runner spawned for session %s: pid=%d profile=%s log=%s confined=%s",
        session_id, spawned.pid,
        profile_name or "(none)",
        log_path or "(inherited)",
        not disable_confine,
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

    Defaults applied for fields that would otherwise be empty:
    - ``provider_name`` → ``"anthropic"`` (the framework default).
    - ``model_name`` → ``""`` (the runner-side validate stage will
      reject; surfaced loudly via the daemon's bootstrap-failure
      WARNING).

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

    if profile is not None:
        provider_name = getattr(profile, "provider", None) or ""
        model_name = getattr(profile, "model", None) or ""
        names = list(getattr(profile, "plugins", []) or [])
        preloaded = set(getattr(profile, "preloaded_plugins", set()) or set())
        # Phase 4 §C: carry the full profile.plugin_configs at the
        # envelope's top level (schema v2) so auto-loaded plugins not
        # named in profile.plugins (e.g. permission) receive their
        # profile overrides.  Closes backlog §3.3c.X.  Per-entry
        # ``config`` is no longer set — entries are {name, preload} only.
        plugin_configs_dict = {
            k: dict(v)
            for k, v in (getattr(profile, "plugin_configs", {}) or {}).items()
        }
        for name in names:
            plugin_specs.append({"name": name, "preload": name in preloaded})
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

    # Provider fallback — the JaatoRuntime default is "google_genai"
    # but Phase 3's runner-tier plugins are most-tested against
    # anthropic.  When neither the profile nor the env explicitly
    # specifies, fall back to anthropic which has the broadest
    # plugin compat coverage.
    if not provider_name:
        provider_name = "anthropic"

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

    return SessionInitEnvelope(
        session_id=session_id,
        workspace_path=workspace_path,
        profile_name=profile_name,
        provider_name=provider_name,
        model_name=model_name,
        plugins=plugin_specs,
        plugin_configs=plugin_configs_dict,
        system_instructions=system_instructions,
        agent_id="main",
        gc=gc_dict,
        agent_params=agent_params_dict,
        config_root=getattr(server, "config_root", None),
        env_overrides=env_overrides,
        project=project_val,
        location=location_val,
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
        logger.warning(
            "runner session.bootstrap failed for %s: %s — "
            "daemon-side JaatoSession remains authoritative",
            session_id, exc, exc_info=True,
        )
