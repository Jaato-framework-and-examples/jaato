"""Session Manager for multi-session support.

This module manages multiple named sessions, each with its own
JaatoServer instance and conversation state.

Sessions are:
- Persisted to disk via the Session Plugin
- Loaded on-demand when clients attach
- Saved periodically and on shutdown
- Identified by consistent IDs across memory and disk

Integration with Session Plugin:
- SessionManager uses SessionPlugin for persistence
- SessionState from the plugin is used for save/load
- Session IDs are consistent between runtime and storage
"""

import json
import logging
import os
import re
import sys
import pathlib
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

# Add project root to path
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.plugins.session import (
    create_plugin as create_session_plugin,
    load_session_config,
    SessionPlugin,
    SessionState,
    SessionConfig,
    SessionInfo as PluginSessionInfo,
)

from shared.instruction_token_cache import InstructionTokenCache
from shared.runtime_limits import RuntimeLimits, apply_isolated_defaults
from shared.session_envelope import BootstrapEnvelope
from .core import JaatoServer
from .session_logging import set_logging_context, clear_logging_context, get_session_handler
from jaato_sdk.events import (
    Event,
    EventType,
    SystemMessageEvent,
    ErrorEvent,
    SessionInfoEvent,
    SessionDescriptionUpdatedEvent,
    SessionProfilesEvent,
    SessionRestoredEvent,
    ContextUpdatedEvent,
    UsageBreakdown,
    AgentCreatedEvent,
    InstructionBudgetEvent,
    InterruptedTurnRecoveredEvent,
    ToolCallStartEvent,
    ToolCallEndEvent,
    TurnCompletedEvent,
    AgentStatusChangedEvent,
    WorkspaceFilesChangedEvent,
    WorkspaceFilesSnapshotEvent,
)
from .workspace_monitor import WorkspaceMonitor


logger = logging.getLogger(__name__)


@dataclass
class RuntimeSessionInfo:
    """Metadata about a session (runtime + persisted)."""
    session_id: str
    name: str
    description: Optional[str]
    created_at: str
    last_activity: str
    model_provider: str
    model_name: str
    is_processing: bool
    is_loaded: bool  # True if currently in memory
    client_count: int
    turn_count: int
    workspace_path: Optional[str] = None
    created_by: Optional[str] = None  # Authenticated user who created the session


@dataclass
class Session:
    """A managed session with its JaatoServer."""
    session_id: str
    name: str
    server: JaatoServer
    created_at: str
    last_activity: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    attached_clients: Set[str] = field(default_factory=set)
    description: Optional[str] = None
    is_dirty: bool = False  # True if has unsaved changes
    workspace_path: Optional[str] = None  # Client's working directory
    config_root: Optional[str] = None  # Read-only framework-config root override
    user_inputs: List[str] = field(default_factory=list)  # Command history for prompt restoration
    interrupted_turn: Optional[Dict[str, Any]] = None  # Turn interruption state for recovery
    provisioned: bool = False  # True if workspace was auto-provisioned by server
    created_by: Optional[str] = None  # Authenticated user who created the session
    sandbox_mode: Optional[str] = None  # "apparmor" or "soft" when workspace sandboxing is active
    # Phase 3 §3.12 disk-restore + peer-review M5/N1: True when this
    # Session was loaded from disk and is awaiting its first
    # client-attach.  While True, ``check_permission`` ASK paths
    # **hold** tool calls (queue the prompt rather than deny) so a
    # session restored after a daemon restart doesn't silently fail
    # in-flight work.  Cleared on first client attach, at which
    # point the daemon emits ``SessionRestoredEvent`` so the client
    # surfaces a "review pending tool calls" prompt and drains the
    # queue.  False for fresh / never-restored sessions.
    restored_pending_attach: bool = False


@dataclass
class SubRunnerHandle:
    """Daemon-side bookkeeping for an isolated-subagent sub-runner
    (Phase 4 §4.3.6a).

    Holds the RPC client + spawned subprocess handle for a sub-runner
    spawned by the §4.3 isolated-subagent opt-in.  Distinct from
    ``Session`` (which tracks top-level user-visible sessions) — these
    handles live in their own dict keyed by isolated_session_id, off
    the parent session's lifecycle.

    Lifecycle (each §4.3.6 sub-commit advances):
    - §4.3.6a: created when the helper's spawn invocation succeeds.
    - §4.3.6b: cross-runner output forwarding subscribes by handle.
    - §4.3.6c: cross-runner prompt forwarding dispatches via ``rpc``.
    - §4.3.6d: parent-cascade teardown reads + cleans up.

    Fields:
        parent_session_id: The session that owns this isolated
            subagent.  Used for parent-cascade teardown lookups.
        subagent_id: Pre-generated id from the runner-side subagent
            plugin.  Combined with ``parent_session_id`` to derive
            the isolated_session_id, sub-AppArmor profile name, and
            sub-cgroup path.
        isolated_session_id: ``{parent}__sub_{subagent}`` — used as
            the session_id for daemon-side RPC + kernel-resource
            names.
        rpc: The :class:`RunnerRPCClient` started against the
            sub-runner's parent socket.  Daemon talks to the
            sub-runner via this handle.
        spawned: The :class:`SpawnedRunner` (pid + socket pair).
            Held for teardown.
        sub_apparmor_profile: The kernel-visible sub-profile name
            (``jaato-ws-{parent}//{subagent}``).  Empty when
            unconfined fallback was taken (not the normal path).
        cgroup_path: Absolute path to the sub-cgroup directory.
            Empty when no cgroup was created (no runtime_limits or
            cgroups unavailable).
        created_at: Timestamp for diagnostics + leak detection.
    """
    parent_session_id: str
    subagent_id: str
    isolated_session_id: str
    rpc: Any  # server.runner_rpc_client.RunnerRPCClient
    spawned: Any  # server.runner_spawner.SpawnedRunner
    sub_apparmor_profile: str
    cgroup_path: str
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc),
    )


class SessionManager:
    """Manages multiple named sessions with persistence.

    Integrates with the Session Plugin to provide:
    - Persistent storage of session history
    - Load sessions on-demand from disk
    - Save sessions periodically and on shutdown
    - Unified view of in-memory and on-disk sessions

    Each session has its own JaatoServer with isolated:
    - Conversation history
    - Agent state
    - Plugin state
    - Token accounting

    Clients can:
    - Create new sessions
    - Attach to existing sessions (loads from disk if needed)
    - List all sessions (memory + disk)
    - Save/checkpoint sessions
    """

    def __init__(
        self,
        storage_path: Optional[str] = None,
    ):
        """Initialize the session manager.

        Args:
            storage_path: Override for session storage path.
        """

        # Initialize session plugin for persistence.
        # storage_path stays relative (e.g. ".jaato/sessions") — it is
        # resolved per-workspace via _session_storage_dir() at each call site.
        self._session_plugin: SessionPlugin = create_session_plugin()
        self._session_config: SessionConfig = load_session_config()

        if storage_path:
            self._session_config.storage_path = storage_path

        # Initialize with relative storage_path. The plugin's self._storage_path
        # acts as a fallback for standalone JaatoClient usage; SessionManager
        # always passes an explicit storage_dir resolved per-workspace.
        self._session_plugin.initialize({
            'storage_path': self._session_config.storage_path
        })

        # In-memory session storage
        self._sessions: Dict[str, Session] = {}
        # Use RLock (reentrant) because initialize() may emit events during session load
        self._lock = threading.RLock()

        # Phase 4 §4.3.6a: bookkeeping for isolated-subagent sub-runners.
        # Keyed by isolated_session_id (``{parent}__sub_{subagent}``).
        # Distinct from ``_sessions`` — these are off the top-level
        # session lifecycle and are owned by the parent session for
        # parent-cascade teardown.  Guarded by ``_lock`` (same as
        # ``_sessions``) — both dicts are mutated together at
        # parent-shutdown time.
        self._isolated_sub_runners: Dict[str, SubRunnerHandle] = {}

        # Path H (cycle 10): serialize concurrent async saves so
        # parallel ToolCallStartEvents (parallel tool execution)
        # don't trample each other.  atomic_write_json already
        # prevents file tear; this lock gives consistent last-
        # writer-wins ordering across concurrent _save_session_async
        # invocations for the same session.
        self._async_save_lock = threading.Lock()

        # Client to session mapping
        self._client_to_session: Dict[str, str] = {}

        # Per-client configuration (presentation context, working_dir, etc.)
        self._client_config: Dict[str, Dict[str, Any]] = {}

        # Event routing callback
        self._event_callback: Optional[Callable[[str, Event], None]] = None
        # Broadcast callback — wired to CompositeEventSink.broadcast_event
        # by the daemon (see __main__.py).  Daemon-wide events (currently
        # HandoffGate transitions from jaato-premium) flow through this.
        self._broadcast_callback: Optional[Callable[[Event], None]] = None

        # Workspace file monitors keyed by session_id
        self._workspace_monitors: Dict[str, WorkspaceMonitor] = {}

        # Shared instruction token cache — survives across session
        # creates/restores within the same daemon process.
        self._instruction_token_cache = InstructionTokenCache()

        # Session hooks — callbacks invoked after each session is initialized.
        # Registered by daemon extensions via ``add_session_hook()``.
        self._session_hooks: List[Callable] = []
        # Registered by transports via ``add_pre_initialize_hook()``.
        # Fire BEFORE ``server.initialize()`` so kernel-level provisioning
        # (AppArmor profile, cgroup) is in place before prefetch scripts
        # run (server 0.6.49+).
        self._pre_initialize_hooks: List[Callable] = []

        logger.info(f"SessionManager initialized with storage template: {self._session_config.storage_path}")

    def _session_storage_dir(self, workspace_path: str) -> pathlib.Path:
        """Resolve session storage directory for a workspace.

        Combines the workspace path with the configured (relative) storage_path
        to produce an absolute directory, e.g.
        ``/home/user/project`` + ``.jaato/sessions`` → ``/home/user/project/.jaato/sessions``.

        Args:
            workspace_path: Absolute path to the client's workspace.

        Returns:
            Absolute Path to the session storage directory.

        Raises:
            ValueError: If workspace_path is empty/None.
        """
        if not workspace_path:
            raise ValueError("workspace_path required for session storage")
        return pathlib.Path(workspace_path) / self._session_config.storage_path

    @staticmethod
    def _resolve_agent(
        agent_name: str,
        params: Optional[Dict[str, str]],
        workspace_path: Optional[str],
        config_root: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Resolve an agent by name from .jaato/agents/ and .jaato/prompts/.

        Scans agent directories (workspace then user-level), reads the
        markdown file, parses frontmatter, substitutes params, and returns
        the rendered system instructions.

        Args:
            agent_name: Agent name (filename stem).
            params: Parameter values for ``{{param}}`` placeholders.
            workspace_path: Workspace directory for agent resolution.
            config_root: Optional override for the workspace tier.  When
                set, scans ``<config_root>/agents/`` and
                ``<config_root>/prompts/`` instead of the
                workspace-anchored paths.  See
                :func:`shared.config_resolver.resolve_config_search_path`.

        Returns:
            Dict with ``system_instructions``, ``description``,
            ``default_profile``, ``missing_params``, or ``None`` if not found.
        """
        import re

        search_dirs = []
        if config_root:
            cr = pathlib.Path(config_root).expanduser().resolve()
            search_dirs.append(cr / "agents")
            search_dirs.append(cr / "prompts")
        elif workspace_path:
            search_dirs.append(pathlib.Path(workspace_path) / ".jaato" / "agents")
            search_dirs.append(pathlib.Path(workspace_path) / ".jaato" / "prompts")
        search_dirs.append(pathlib.Path.home() / ".jaato" / "agents")
        search_dirs.append(pathlib.Path.home() / ".jaato" / "prompts")

        # Find the agent file
        agent_path = None
        for search_dir in search_dirs:
            if not search_dir.is_dir():
                continue
            # Single file: agents/gen-references.md
            candidate = search_dir / f"{agent_name}.md"
            if candidate.is_file():
                agent_path = candidate
                break
            # Directory: agents/gen-references/PROMPT.md
            candidate_dir = search_dir / agent_name
            if candidate_dir.is_dir():
                for entry_name in ("PROMPT.md", "SKILL.md"):
                    entry = candidate_dir / entry_name
                    if entry.is_file():
                        agent_path = entry
                        break
                if agent_path:
                    break

        if not agent_path:
            return None

        raw = agent_path.read_text(encoding="utf-8")

        # Parse YAML frontmatter
        frontmatter: Dict[str, Any] = {}
        body = raw
        if raw.startswith("---"):
            match = re.match(r"^---\s*\n(.*?)\n---\s*\n", raw, re.DOTALL)
            if match:
                try:
                    import yaml
                    frontmatter = yaml.safe_load(match.group(1)) or {}
                except Exception:
                    pass
                body = raw[match.end():]

        # Substitute params
        effective_params = dict(params or {})
        param_defs = frontmatter.get("params", {})

        # Apply frontmatter defaults for params not provided
        if isinstance(param_defs, dict):
            for pname, pdef in param_defs.items():
                if pname not in effective_params:
                    if isinstance(pdef, dict) and "default" in pdef:
                        default = pdef["default"]
                        if default is not None:
                            effective_params[pname] = str(default)

        # Pre-scan: collect inline ``{{name:default}}`` defaults declared
        # anywhere in the body so a later bare ``{{name}}`` can fall back
        # to the same default.  Without this, an agent that uses a
        # parameter both with and without an inline default would mark it
        # missing on the bare occurrences and leave literal ``{{name}}``
        # placeholders in the rendered system instructions — which then
        # bloat every turn's prompt.
        inline_defaults: Dict[str, str] = {}
        inline_pattern = re.compile(r"\{\{(\w+)(?::([^}]*))?\}\}")
        for m in inline_pattern.finditer(body):
            name = m.group(1)
            default = m.group(2)
            if default is not None and name not in inline_defaults:
                inline_defaults[name] = default

        # Use a set for O(1) dedup; the public missing list is built
        # once at the end so the same name never appears twice.
        missing_set: set = set()

        def replace_param(m: re.Match) -> str:
            name = m.group(1)
            inline_default = m.group(2)

            if name in effective_params:
                return effective_params[name]
            if inline_default is not None:
                return inline_default
            if name in inline_defaults:
                return inline_defaults[name]
            missing_set.add(name)
            return m.group(0)  # Keep unresolved (debugging signal)

        rendered = inline_pattern.sub(replace_param, body)

        return {
            "system_instructions": rendered,
            "description": frontmatter.get("description", ""),
            "default_profile": frontmatter.get("default_profile"),
            "missing_params": sorted(missing_set),
            "source_path": str(agent_path),
        }

    def _resolve_profile(
        self,
        profile_name: str,
        workspace_path: str,
        config_root: Optional[str] = None,
        env_file: Optional[str] = None,
    ) -> Tuple[Optional[Any], Optional[str]]:
        """Resolve an agent profile by name.

        Discovers profiles from the workspace tier (``<config_root>/profiles/``
        when set, else ``<workspace_path>/.jaato/profiles/``), the user tier
        (``~/.jaato/profiles/``), and premium entry points, then returns the
        matching ``SubagentProfile``.  When the profile is not found, the
        second element carries a user-friendly error message (e.g. a JSON
        parse error for a broken profile file).

        Args:
            profile_name: Name of the profile to look up.
            workspace_path: Workspace directory containing ``.jaato/profiles/``.
            config_root: Optional override for the workspace tier — see
                :func:`shared.config_resolver.resolve_config_search_path`.
            env_file: Optional path to the session's ``.env`` file.  When
                provided, the file is parsed and overlaid onto the
                session-scoped ``ContextVar`` BEFORE
                :func:`discover_profiles` runs — closes the wire-gap
                where ``JAATO_PROFILE_SET`` (and any other env-var
                read by profile discovery) is empty because the
                normal ``_with_session_env`` block hasn't fired yet.
                Discovered 2026-05-18 with the kb-enablement-2.0
                handoff_test profile-set refactor: ``profiles/<set>/``
                subdir was silently skipped because the contextvar
                wasn't populated at this point in
                ``_create_session_impl``.  Symmetric to PR #139
                (completion_validators wire-gap) and PR #140
                (plugin_configs pass:// wire-gap).  Server 0.6.124+.

        Returns:
            Tuple of ``(profile, None)`` on success, or
            ``(None, error_message)`` when the profile cannot be resolved.
        """
        from shared.plugins.subagent.config import discover_profiles

        # Overlay env_file onto the session-scoped ContextVar so
        # ``discover_profiles`` sees the workspace-declared
        # ``JAATO_PROFILE_SET`` (and any other env-affected reads it
        # may grow).  Same parsing pipeline as
        # ``JaatoServer._resolve_session_env`` — ``dotenv_values`` +
        # ``expand_variables`` so ``pass://`` / ``vault://`` / ``${VAR}``
        # cross-references resolve daemon-side, never reaching the
        # AppArmor-confined runner literal.  Saved/restored via
        # try/finally so concurrent callers don't clobber each other's
        # contextvar state.
        from shared.session_context import (
            _session_env as _session_env_var,
            set_session_env,
        )
        previous_env = _session_env_var.get()
        overlay_applied = False
        if env_file:
            try:
                from dotenv import dotenv_values
                from shared.plugins.subagent.config import expand_variables
                raw = dotenv_values(env_file)
                raw_filtered = {k: v for k, v in raw.items() if v is not None}
                resolved = expand_variables(raw_filtered, context=raw_filtered)
                set_session_env(resolved)
                overlay_applied = True
            except Exception as exc:  # noqa: BLE001
                # Failure to parse env_file is non-fatal for profile
                # resolution — discover_profiles still falls back to
                # the contextvar's existing value / ``os.environ``.
                # Log at INFO so operators see the source of any
                # profile-set-not-found surprises that follow.
                logger.info(
                    "_resolve_profile: env_file overlay skipped for "
                    "%r (parse failed: %s); discover_profiles will "
                    "fall back to contextvar / os.environ",
                    env_file, exc,
                )

        try:
            result = discover_profiles(
                ".jaato/profiles",
                base_path=workspace_path,
                config_root=config_root,
            )
        finally:
            if overlay_applied:
                _session_env_var.set(previous_env)
        profile = result.profiles.get(profile_name)
        if profile is not None:
            return profile, None

        # Profile not in the successfully parsed set — check if there was
        # a parse error for a file matching the requested name.
        if profile_name in result.errors:
            return None, (
                f"Profile '{profile_name}' exists but failed to parse: "
                f"{result.errors[profile_name]}"
            )
        return None, f"Agent profile '{profile_name}' not found in .jaato/profiles/"

    def add_session_hook(self, hook: Callable) -> None:
        """Register a callback invoked after each session is initialized.

        Session hooks are the primary mechanism for daemon extensions to
        inject per-session functionality (e.g., registering custom
        environment aspects, wiring remote spawn handlers).

        The hook is called with two arguments:

        1. ``server`` — the ``JaatoServer`` instance for the newly created
           or loaded session.  Fully initialized: plugin registry populated,
           provider connected, tools configured.
        2. ``session_id`` — the unique session identifier string
           (e.g., ``"20260321_132926"``).

        Hooks are called in registration order.  If a hook raises an
        exception, it is logged and subsequent hooks still run.

        Args:
            hook: A callable with signature
                ``(server: JaatoServer, session_id: str) -> None``.

        Example (from a daemon extension)::

            def _on_session_ready(self, server, session_id):
                env = server.registry.get_plugin("environment")
                if env and hasattr(env, 'register_aspect'):
                    env.register_aspect("my_aspect", self._handler)

            # In the extension's start():
            ctx.session_manager.add_session_hook(self._on_session_ready)
        """
        self._session_hooks.append(hook)

    # ------------------------------------------------------------------
    # IPC AppArmor + runner spawn (Phase 3 §3.13)
    # ------------------------------------------------------------------
    #
    # Phase 2 §2.3 wired this as a pre-initialize hook in
    # server/__main__.py.  Phase 3 §3.13 relocates the logic into the
    # SessionManager itself so it lives next to the bootstrap helper
    # rather than reaching across module boundaries via a hook
    # registration: the hook indirection was a transitional step,
    # not the design endpoint.  The helper is now invoked inline from
    # ``_bootstrap_session`` AFTER ``JaatoServer`` construction and
    # BEFORE ``_run_pre_initialize_hooks`` (which still fires for the
    # WS-side pre-init hook + any third-party hooks).
    #
    # The AppArmorManager + daemon_loop dependencies the relocated
    # method needs are wired at daemon startup via
    # :meth:`set_apparmor_dependencies` (called from
    # ``server/__main__.py``).  Tests construct a SessionManager
    # without these dependencies; the method is a clean no-op when
    # they're unset.

    def set_apparmor_dependencies(
        self,
        ws_server: Any = None,
        daemon_loop: Any = None,
        pool_manager: Any = None,
    ) -> None:
        """Wire the IPC AppArmor + runner-spawn dependencies (§3.13).

        Phase 3 §3.13.  Called once at daemon startup by
        ``server/__main__.py:JaatoDaemon.start``.  The relocated IPC
        apparmor logic in :meth:`_provision_ipc_apparmor_and_spawn_runner`
        reads these to:

        - Skip provisioning for sessions whose workspace lives under a
          running WS server's root (those are handled by the WS hook,
          not by us — avoids double-provisioning).
        - Pass the daemon's main asyncio loop to the
          :class:`AppArmorManager` so confined-worker mutations
          dispatch back to the unconfined main loop, AND to the
          :func:`spawn_session_runner` helper which runs
          ``RunnerRPCClient.start()`` async.

        Args:
            ws_server: Optional reference to the running WS server.
                ``None`` for IPC-only daemons.
            daemon_loop: The daemon's main asyncio loop.  ``None`` is
                accepted but disables apparmor provisioning (no
                AppArmor manager can be constructed without it).
            pool_manager: Pool PR 4 — the daemon's
                :class:`server.runner_pool.PoolManager`, threaded
                into :func:`spawn_session_runner` so IPC sessions can
                claim a pre-warm pool slot instead of cold-spawning
                a runner.  ``None`` disables the pool path for IPC
                sessions (cold-spawn fallback applies).
        """
        self._ws_server_ref = ws_server
        self._daemon_loop = daemon_loop
        self._pool_manager_ref = pool_manager

    def _provision_ipc_apparmor_and_spawn_runner(
        self,
        server: 'JaatoServer',
        session_id: str,
        workspace_path: Optional[str],
        client_id: Optional[str],
        *,
        apparmor_override: Optional[bool] = None,
        config_root_override: Optional[str] = None,
        env_file_override: Optional[str] = None,
        apparmor_fragments_override: Optional[List[str]] = None,
    ) -> Optional[str]:
        """Provision IPC AppArmor (opt-in) + spawn the per-session runner.

        Phase 3 §3.13 + §7a.  §3.13 relocated this from the legacy
        pre-init hook in ``server/__main__.py``.  §7a refactored the
        body into two composed helpers — apparmor provisioning is
        opt-in, but the runner spawn is unconditional (every IPC
        session with a workspace gets a runner; the runner-RPC
        dispatch surface is always available for the seat-flip's
        ``self._jaato.X`` migrations).

        Lifecycle:
        1. Skip when ``client_id`` is ``None`` (non-client paths
           supply their own bootstrap; this method is IPC-only).
        2. Skip when the session has no ``workspace_path`` — the
           runner needs a cwd; sessions without one don't get one
           (matches pre-§7a behavior).
        3. Skip when the session's workspace lives under a running
           WS server's workspace_root — the WS hook owns this
           lane; don't double-spawn.
        4. **Apparmor (opt-in)**: if ``client_config["apparmor"]``
           is set, call :meth:`_provision_apparmor_for_session` to
           load the profile.  Returns the resolved profile_name +
           sandbox_mode (``"apparmor"`` on success, ``"soft"`` on
           provisioning failure).
        5. **Spawn (unconditional)**: call
           :meth:`_spawn_session_runner_unconditional` with the
           profile_name from step 4 (or empty + disable_confine=True
           when no opt-in).  Spawn failure downgrades the return
           value to ``"soft"`` (or leaves it ``None`` for the
           no-opt-in case).

        Returns:
            The planned ``sandbox_mode`` for the Session record:
            ``"apparmor"`` (kernel-confined, runner spawned),
            ``"soft"`` (apparmor downgrade due to provisioning /
            spawn failure), or ``None`` (no apparmor opt-in or
            spawn was unconditional but unconfined — sandbox_mode
            stays None for the runner-without-confinement case
            since the field semantically tracks confinement, not
            runner presence).
        """
        # Phase 7a: only IPC client-driven sessions.
        # Non-client-driven bootstrap paths (loaded-from-disk,
        # ephemeral, standalone WS) supply their own bootstrap.
        if client_id is None:
            return None

        # PR-A (2026-05-14): resolve AppArmor opt-in across three
        # sources, in precedence order:
        #   1. ``apparmor_override`` kwarg — explicit caller intent
        #      from ``SessionManager.create_headless_session`` /
        #      reactor surface.  Wins outright when set.
        #   2. ``client_config["apparmor"]`` — IPC ClientConfigRequest
        #      opt-in (the existing pre-PR-A signal).  Used by the
        #      TUI / direct-IPC clients.
        #   3. ``server._profile.apparmor`` — profile-declared baseline.
        #      Default ``False`` today; PR-B will flip the default to
        #      ``True`` after kb-enablement-2.0 has validated the field.
        #
        # The two earlier sources are "explicit yes/no"; the profile
        # field is the "what does this profile prefer" baseline.  A
        # TUI user setting ``--apparmor`` on the wire still wins; a
        # reactor passing ``apparmor=True`` via kwarg still wins.
        client_config = self._client_config.get(client_id, {})
        client_config_signal = client_config.get("apparmor")
        profile = getattr(server, "_profile", None)
        profile_apparmor = bool(getattr(profile, "apparmor", False)) if profile else False
        if apparmor_override is not None:
            opt_in_apparmor = bool(apparmor_override)
        elif client_config_signal is not None:
            opt_in_apparmor = bool(client_config_signal)
        else:
            opt_in_apparmor = profile_apparmor

        # 2026-05-14 unification: resolve ``config_root`` + ``env_file``
        # for the AppArmor policy generator across all entry points.
        # Caller-supplied envelope override wins (used by headless +
        # disk-restore + ephemeral subagent paths that don't populate
        # ``client_config``); otherwise fall back to the IPC
        # ``client_config`` (populated by ``ClientConfigRequest``);
        # otherwise ``None`` (policy template's ``{config_root_rules}``
        # renders empty).  Pre-unification, this helper read ONLY from
        # ``client_config[client_id]`` — closes the v76 reactor-spawned
        # cascade crash class where headless callers couldn't supply
        # config_root to the policy generator.  See
        # ``project_backlog_apparmor_config_root_threading_for_headless``.
        if config_root_override is not None:
            config_root = config_root_override
        else:
            config_root = client_config.get("config_root")
        if env_file_override is not None:
            env_file = env_file_override
        else:
            env_file = client_config.get("env_file")

        # Piece 1 (2026-05-14): per-profile apparmor fragment scoping.
        # Resolution: explicit kwarg override wins; otherwise read
        # from the resolved profile's ``apparmor_fragments`` field
        # (already inheritance-merged with child-replaces semantics
        # at profile-load time).  ``None`` (default) preserves the
        # pre-Piece-1 "compose all fragments" behaviour;
        # non-``None`` (including ``[]``) restricts the compose set.
        if apparmor_fragments_override is not None:
            requested_fragments: Optional[List[str]] = list(apparmor_fragments_override)
        else:
            profile_fragments = getattr(profile, "apparmor_fragments", None) if profile else None
            requested_fragments = list(profile_fragments) if profile_fragments is not None else None

        # Phase 0 (template v20, 2026-05-16): plugin-contribution hook.
        # Walk ``profile.plugins`` to resolve each plugin's class via the
        # server registry, then call its optional
        # ``get_apparmor_rules`` classmethod with session context.
        # ``None`` when no profile is set OR no plugin contributes —
        # ``_render_profile`` treats both as "no contributions".  See
        # docs/design/plugin-apparmor-contribution.md.
        from server.apparmor import resolve_plugin_apparmor_rules
        plugin_rules = resolve_plugin_apparmor_rules(
            server=server,
            profile=profile,
            session_id=session_id,
            workspace_path=workspace_path,
            config_root=config_root,
        )

        # Spawn requires a workspace (cwd target).  Sessions without
        # one don't get a runner; pre-§7a behavior preserved.
        if not workspace_path:
            if opt_in_apparmor:
                # Surface the apparmor downgrade if confinement was
                # requested.  Silent skip otherwise — most IPC sessions
                # without workspace are short-lived / headless and don't
                # expect a runner.
                self._notify_apparmor(
                    client_id, session_id,
                    "requested but session has no workspace_path — "
                    "running unconfined",
                    style="warning",
                )
            return None

        # WS-overlap precedence: WS hook handles confinement +
        # runner spawn for sessions whose workspace is under the
        # WS server's root.  Skip both apparmor and spawn here so
        # we don't duplicate.
        if self._workspace_under_ws_root(workspace_path):
            return None

        # ----- Step 4: apparmor (opt-in) -----
        profile_name = ""
        sandbox_mode: Optional[str] = None
        if opt_in_apparmor:
            profile_name, sandbox_mode = self._provision_apparmor_for_session(
                session_id=session_id,
                workspace_path=workspace_path,
                client_id=client_id,
                config_root=config_root,
                env_file=env_file,
                requested_fragments=requested_fragments,
                plugin_rules=plugin_rules,
            )
            if profile_name == "" and sandbox_mode == "soft":
                # Apparmor unavailable / provisioning failed.
                # Continue to the unconditional spawn but the
                # runner is unconfined — that's the §7a intent
                # (always have a runner; confinement is layered).
                pass

        # ----- Step 5: spawn (unconditional) -----
        spawn_ok = self._spawn_session_runner_unconditional(
            server=server,
            session_id=session_id,
            workspace_path=workspace_path,
            client_id=client_id,
            profile_name=profile_name,
        )
        if not spawn_ok:
            # Spawn failed.  If apparmor was opted-in, downgrade
            # to "soft"; otherwise the session continues with
            # in-process tool execution (no sandbox_mode change).
            if opt_in_apparmor:
                return "soft"
            return None

        # ----- Step 5b: success notification + return -----
        if opt_in_apparmor and sandbox_mode == "apparmor":
            # ``config_root`` is the resolved value from above (envelope
            # override first, then client_config); the notification
            # reflects what the policy was actually generated with.
            self._notify_apparmor(
                client_id, session_id,
                f"profile provisioned (workspace={workspace_path}, "
                f"config_root={config_root or '(none)'}); runner spawned",
                style="info",
            )
            return "apparmor"
        if opt_in_apparmor and sandbox_mode == "soft":
            # Apparmor opt-in but provisioning failed; runner
            # spawned anyway (always-spawn).
            return "soft"
        # No apparmor opt-in: runner spawned unconfined.
        # sandbox_mode stays None (semantically tracks confinement
        # — there's none here, even though there IS a runner).
        return None

    def _notify_apparmor(
        self,
        client_id: str,
        session_id: str,
        message: str,
        style: str,
    ) -> None:
        """Surface an apparmor-status line to the client.

        Helper extracted for reuse between the apparmor-provision
        path and the no-workspace warning path.  Routes via
        ``_emit_to_client`` directly because at the call point the
        Session record doesn't exist in ``_sessions`` yet (we're
        pre-init).  Failures are swallowed — emit must not break
        session creation.
        """
        from jaato_sdk.events import SystemMessageEvent
        logger.info("[apparmor] %s", message)
        try:
            self._emit_to_client(client_id, SystemMessageEvent(
                message=f"[apparmor] {message}",
                style=style,
            ))
        except Exception:
            logger.warning(
                "Failed to emit apparmor status event for %s",
                session_id, exc_info=True,
            )

    def _workspace_under_ws_root(self, workspace_path: str) -> bool:
        """Return True iff *workspace_path* is under a running WS
        server's ``_workspace_root``.

        Used by ``_provision_ipc_apparmor_and_spawn_runner`` to
        skip its work for sessions the WS hook owns — preventing
        double-provision + double-spawn.
        """
        ws_server = getattr(self, "_ws_server_ref", None)
        if ws_server is None:
            return False
        ws_root = getattr(ws_server, "_workspace_root", None)
        if not ws_root:
            return False
        try:
            ws_root_real = os.path.realpath(ws_root)
            sess_real = os.path.realpath(workspace_path)
            return (
                sess_real == ws_root_real
                or sess_real.startswith(ws_root_real + os.sep)
            )
        except OSError:
            return False

    def _provision_apparmor_for_session(
        self,
        session_id: str,
        workspace_path: str,
        client_id: str,
        *,
        config_root: Optional[str],
        env_file: Optional[str],
        requested_fragments: Optional[List[str]] = None,
        plugin_rules: Optional[List[str]] = None,
    ) -> "Tuple[str, Optional[str]]":
        """Provision the AppArmor profile for a session
        (Phase 3 §7a — opt-in only).

        Caller has already checked the apparmor opt-in flag and
        the workspace gates.  This method does only the apparmor
        lifecycle: lazy-init manager + provision_profile.

        2026-05-14: signature unified to take ``config_root`` +
        ``env_file`` as explicit kwargs.  Pre-fix the helper read
        these from ``client_config[client_id]``, which made the
        reactor-spawned headless path (no client_config) silently
        omit the ``{config_root_rules}`` placeholder and produce a
        profile that didn't grant reads on the repo-root
        ``.jaato/`` for the kb-enablement-2.0 "framework config at
        parent, sandbox inside" layout.  Now every entry point's
        envelope carries these fields explicitly; the helper reads
        the envelope's values uniformly.  See
        ``project_backlog_apparmor_config_root_threading_for_headless``.

        Returns:
            ``(profile_name, sandbox_mode)``:
            - ``("<name>", "apparmor")`` on success.
            - ``("", "soft")`` when AppArmor is unavailable on the
              host or provisioning failed.  Caller should still
              spawn the runner (with disable_confine=True) — that's
              the §7a always-spawn intent.
        """
        # Lazy-init the AppArmor manager.
        if getattr(self, "_apparmor_manager", None) is None:
            from server.apparmor import AppArmorManager
            daemon_loop = getattr(self, "_daemon_loop", None)
            self._apparmor_manager = AppArmorManager(
                workspace_root=workspace_path,
                loop=daemon_loop,
            )

        apparmor = self._apparmor_manager
        if not apparmor.is_available():
            self._notify_apparmor(
                client_id, session_id,
                "requested but AppArmor is unavailable on this "
                "host (non-Linux, kernel module not loaded, or "
                "apparmor_parser missing) — running unconfined",
                style="warning",
            )
            return "", "soft"

        if not apparmor.provision_profile(
            session_id,
            workspace_path,
            config_root=config_root,
            env_file=env_file,
            requested_fragments=requested_fragments,
            plugin_rules=plugin_rules,
        ):
            self._notify_apparmor(
                client_id, session_id,
                "profile provisioning failed (see daemon log) — "
                "running unconfined",
                style="warning",
            )
            return "", "soft"

        return apparmor.get_profile_name(session_id), "apparmor"

    def _spawn_session_runner_unconditional(
        self,
        server: 'JaatoServer',
        session_id: str,
        workspace_path: str,
        client_id: str,
        profile_name: str,
    ) -> bool:
        """Spawn the per-session runner subprocess (Phase 3 §7a —
        always-called for IPC sessions with a workspace).

        Args:
            server: The session's JaatoServer instance.
            session_id: Session identifier.
            workspace_path: Session workspace (the runner's cwd).
            client_id: For warning notifications on failure.
            profile_name: AppArmor profile to self-confine to.
                Empty string ``""`` means run unconfined (no
                apparmor opt-in or provisioning failed) — the
                spawn helper passes ``disable_confine=True`` to
                ``RunnerSpawner``.

        Returns:
            True on successful spawn; False on failure.  Caller
            decides whether to downgrade ``sandbox_mode`` based on
            the apparmor opt-in flag.
        """
        try:
            from server.runner_spawn import (
                spawn_session_runner,
                dispatch_bootstrap_envelope,
            )
            spawn_session_runner(
                server=server,
                session_id=session_id,
                workspace_path=workspace_path,
                profile_name=profile_name,
                daemon_loop=getattr(self, "_daemon_loop", None),
                disable_confine=(profile_name == ""),
                pool_manager=getattr(self, "_pool_manager_ref", None),
            )
            # Phase 3 post-Step-7 regression fix (Path B):
            # synchronously dispatch ``session.bootstrap`` so the
            # runner-side ``JaatoSession`` host is populated BEFORE
            # ``server.initialize()`` runs.  Pre-fix the IPC path
            # spawned the runner but never sent the bootstrap RPC,
            # leaving ``RunnerRPC._session_host = None`` for the
            # session lifetime.  Every daemon-side
            # ``self._runner_rpc.session_X_threadsafe()`` call
            # then raced against an unbootstrapped runner-side
            # session — handler correctly returned
            # ``stage="no_session"`` per the §3.3c surface
            # defensive contract, but the wrapper raised
            # ``RunnerCallError`` and crashed daemon-side init.
            #
            # The WS path has shipped this synchronous dispatch
            # since §7c step 2 (commit 6e31d375 era) at
            # ``websocket.py:690``.  This commit mirrors that
            # pattern for IPC sessions, closing the structural
            # asymmetry.
            #
            # Inline try/except so a bootstrap-dispatch hiccup
            # doesn't roll back the spawn-success return — spawn
            # already succeeded by this point.  Bootstrap-RPC
            # failures log WARNING via ``dispatch_bootstrap_envelope``
            # itself; the inline guard here additionally tolerates
            # ``server.runner_rpc`` being absent (test-stub
            # JaatoServer fakes that don't fully replicate the
            # post-spawn attribute set).
            try:
                dispatch_bootstrap_envelope(
                    server=server,
                    session_id=session_id,
                    workspace_path=workspace_path,
                    profile_name=profile_name,
                )
            except Exception as exc:  # noqa: BLE001 — best-effort
                logger.warning(
                    "IPC session.bootstrap dispatch failed for %s "
                    "(%s: %s) — session will start with an "
                    "unbootstrapped runner-side host; downstream "
                    "session.* RPCs may race-fail until the "
                    "runner-side bootstrap completes some other way",
                    session_id, type(exc).__name__, exc,
                )
            # Phase 4 §4.3.3: wire the spawn_isolated_runner handler
            # with this SessionManager so the §4.3.7 opt-in branch
            # can spawn additional runners for isolated subagents.
            # The handler itself was registered earlier inside
            # ``JaatoServer.set_runner_rpc()`` (§4.3.2); here we
            # bridge it to our own ``_spawn_isolated_runner`` helper
            # by calling ``set_spawn_dependencies(self)``.  The
            # SessionManager reference isn't available at the
            # ``set_runner_rpc`` seam (would require widening
            # ``spawn_session_runner``'s signature) so the wire
            # lands here instead, where SessionManager is ``self``.
            # Best-effort: failure to wire the bridge logs WARNING
            # but doesn't roll back the spawn-success return — the
            # parent session is up and the runner is healthy; the
            # only impact is that ``agent_params.isolated=true``
            # subagents (§4.3.7) would get the "handler not yet
            # wired" stub envelope instead of the routed helper.
            handler = getattr(
                server, "_spawn_isolated_runner_handler", None,
            )
            if handler is not None:
                try:
                    handler.set_spawn_dependencies(session_manager=self)
                except Exception as exc:  # noqa: BLE001 — best-effort
                    logger.warning(
                        "set_spawn_dependencies failed for session "
                        "%s (%s: %s) — agent_params.isolated=true "
                        "subagents will receive the handler-not-"
                        "wired stub until next session restart",
                        session_id, type(exc).__name__, exc,
                    )
            return True
        except Exception as exc:  # noqa: BLE001 — boundary
            self._notify_apparmor(
                client_id, session_id,
                f"runner spawn failed ({type(exc).__name__}: {exc}) "
                "— falling back to in-process tool execution; "
                "session is NOT kernel-confined",
                style="warning",
            )
            logger.exception(
                "runner spawn failed for session %s", session_id,
            )
            return False

    def _spawn_isolated_runner(
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
        sub_profile_tightenings: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Daemon-side helper for the isolated-subagent opt-in
        (Phase 4 §4.3.3).

        Invoked by ``SpawnIsolatedRunnerHandler.handle()`` (once
        ``set_spawn_dependencies(self)`` has been called) when a
        runner-side subagent plugin sends a
        ``subagent.spawn_isolated_runner`` RPC.  Per parent design
        §4.3, the supervisor's ``agent_params.isolated=true`` opt-in
        is meant to spawn the subagent in its own runner subprocess
        with a fresh AppArmor sub-profile
        (``jaato-ws-{parent}//{subagent}``) and its own sub-cgroup.

        §4.3.3 scope (THIS commit):

        1. Reconstruct ``SubagentProfile`` from the wire-shape
           ``profile_payload`` dict via ``build_inline_profile`` —
           same path SDK inline-spec callers use, so the JSON shape
           is identical to ``session.new`` inline specs.
        2. Generate the isolated session id
           (``{parent_session_id}__sub_{subagent_id}``).
        3. Return a ``stage="sub_profile"`` envelope with would-be
           values for diagnostic visibility — the next stage
           (sub-AppArmor profile generation + provisioning) lands
           in §4.3.4.

        Deferred to subsequent sub-commits:

        - §4.3.4: sub-AppArmor profile name generation + load.
        - §4.3.5: sub-cgroup creation + attach.
        - §4.3.6: cross-runner forwarding (prompt-forward INTO new
          runner; output-forward BACK to parent).
        - The actual ``_spawn_session_runner_unconditional(...)``
          call lands once both §4.3.4 and §4.3.5 are wired — passing
          ``profile_name=""`` today would spawn an unconfined
          sub-runner, which is a worse security posture than the
          §4.3 default-share path the §4.3.1 stub points callers
          to.  Refusing to spawn until the sub-profile is ready
          keeps the security gradient monotonic.

        Args:
            parent_session_id: Parent session's id.  Already echo-
                checked daemon-side by the handler; we re-use here
                only for the isolated session-id template.
            subagent_id: Pre-generated subagent id from the runner-
                side subagent plugin.
            profile_payload: Serialized ``SubagentProfile`` as a
                dict.  Field set mirrors ``build_inline_profile``'s
                contract (model, provider, plugins, plugin_configs,
                system_instructions, gc, env, runtime_limits,
                completion_payload_schema, spawn_payload_schema,
                completion_artifacts, model_tiers,
                suppress_base_instructions, max_turns).
            task: First-turn prompt for the isolated runner.  §4.3.3
                does NOT consume this; §4.3.6's forwarding will.
            workspace_path: Inherited from parent (§4.3 invariant).
                Forwarded for diagnostics only in §4.3.3.
            agent_params: Forwarded ``case_data``.  ``isolated`` key
                already stripped daemon-side by the handler.
            display_name: Custom display name; defaults to
                ``profile_payload.name``.
            parent_agent_id: For multi-hop subagent trees.

        Returns:
            Domain-failure envelope shape (parallels
            ``SpawnIsolatedRunnerHandler.handle``'s return):

                {"ok": False,
                 "error": "<reason>",
                 "stage": "validation" | "sub_profile" | "spawn" |
                          "sub_cgroup" | "forwarding",
                 # Diagnostic fields (would-be values for the next
                 # stage to debug against — only present when the
                 # stage advanced past the corresponding failure):
                 "isolated_session_id": "...",
                 "profile_name": "..."}

            On full success (post-§4.3.6 readiness):

                {"ok": True,
                 "session_id": "...",
                 "subagent_id": "...",
                 "runner_pid": <int>,
                 "apparmor_profile": "...",
                 "cgroup_path": "..."}
        """
        # ── Stage: validation — reconstruct SubagentProfile ────
        # ``build_inline_profile`` is the canonical "dict →
        # SubagentProfile" path used by SDK inline-spec session
        # creation; reuse here so the wire shape stays consistent
        # with what session.new accepts.  This also gives the
        # caller a precise validation-failure message if the
        # payload is malformed (e.g., bad gc / runtime_limits dict).
        try:
            from shared.plugins.subagent.config import build_inline_profile
            profile = build_inline_profile(
                profile_payload,
                name=profile_payload.get("name") or "<isolated>",
                description=profile_payload.get("description") or (
                    "Isolated subagent profile (Phase 4 §4.3 opt-in)"
                ),
            )
        except (ValueError, KeyError, TypeError) as exc:
            return {
                "ok": False,
                "error": (
                    f"profile_payload reconstruction failed "
                    f"({type(exc).__name__}: {exc}).  Expected the "
                    f"same dict shape as session.new inline specs — "
                    f"see shared/plugins/subagent/config.py:"
                    f"build_inline_profile."
                ),
                "stage": "validation",
            }

        # ── Phase 5 §5.1: apply isolated-subagent runtime defaults ────
        # ``agent_params.isolated=true`` establishes the "isolation
        # implies bounds" invariant.  Profiles that omit
        # ``runtime_limits`` (or supply only a subset of fields) would
        # otherwise skip cgroup provision entirely and inherit the
        # daemon's default cgroup — the §4.3.9 item 1 hardening gap.
        # Merge supplied limits with the conservative default
        # (per-field; supplied wins on non-None) so the effective
        # value below carries the safety floor regardless of profile
        # content.  See
        # docs/design/phase5_5_1_isolated_default_runtime_limits_audit.md.
        effective_runtime_limits = apply_isolated_defaults(
            profile.runtime_limits,
        )
        if profile.runtime_limits != effective_runtime_limits:
            logger.info(
                "_spawn_isolated_runner: applied isolated-subagent "
                "runtime-limits defaults for parent=%s subagent=%s "
                "(supplied=%s, effective=%s)",
                parent_session_id, subagent_id,
                profile.runtime_limits, effective_runtime_limits,
            )

        # ── Stage: id generation ───────────────────────────────
        # Template ``{parent}__sub_{subagent}`` keeps the isolated
        # session id parseable + correlatable back to its parent
        # for log / trace inspection.  Same template used in
        # parent design §4.3's sub-profile name
        # (``jaato-ws-{session_id}//{subagent_id}``) so the two
        # stay in sync — when §4.3.4 generates the sub-profile
        # name, it can derive it from this session id.
        isolated_session_id = f"{parent_session_id}__sub_{subagent_id}"

        # ── Stage: sub_profile (next stage — §4.3.4) ───────────
        # Stop here.  Refusing to spawn until §4.3.4 provisions a
        # sub-AppArmor profile is intentional: the alternative
        # would be ``profile_name=""`` (unconfined sub-runner),
        # which weakens security relative to the §4.3 default-
        # share path callers can use today.  Monotonic security
        # gradient through the sub-track.
        # ── Stage: sub_profile — provision sub-AppArmor profile ──
        # Phase 4 §4.3.4: ask the daemon's AppArmorManager to write
        # + load a sub-profile named ``jaato-ws-{parent}//{subagent}``.
        # Standalone-with-prefix-name (not a true hat) per Audit 6.
        # When AppArmor isn't available (host doesn't support it, or
        # AppArmorManager isn't wired into this SessionManager
        # instance), we treat the sub-profile as absent and return
        # ``stage=sub_profile`` — the isolated-runner spawn requires
        # kernel confinement.
        apparmor_manager = self._resolve_apparmor_manager()
        if apparmor_manager is None or not apparmor_manager.is_available():
            return {
                "ok": False,
                "error": (
                    f"sub-AppArmor profile cannot be provisioned: "
                    f"AppArmorManager unavailable on this host.  "
                    f"Isolated-runner spawn requires kernel-level "
                    f"confinement.  Profile reconstruction succeeded "
                    f"(name={profile.name!r}, model={profile.model!r}).  "
                    f"Would-be isolated session: {isolated_session_id!r}.  "
                    f"Workaround: set agent_params.isolated=false (or "
                    f"omit) to use the default-share path (subagent "
                    f"runs in the parent's runner) — works end-to-end "
                    f"today.  See docs/design/phase4_implementation_audits.md."
                ),
                "stage": "sub_profile",
                "isolated_session_id": isolated_session_id,
                "profile_name": profile.name,
            }

        ok, sub_profile_or_err = apparmor_manager.provision_sub_profile(
            parent_session_id=parent_session_id,
            subagent_id=subagent_id,
            workspace_path=workspace_path,
            tightenings=sub_profile_tightenings,
        )
        if not ok:
            logger.warning(
                "_spawn_isolated_runner: sub-profile provision failed "
                "for parent=%s subagent=%s: %s",
                parent_session_id, subagent_id, sub_profile_or_err,
            )
            return {
                "ok": False,
                "error": (
                    f"sub-AppArmor profile provision failed: "
                    f"{sub_profile_or_err}.  Profile reconstruction "
                    f"succeeded (name={profile.name!r}).  Would-be "
                    f"isolated session: {isolated_session_id!r}.  "
                    f"Workaround: omit agent_params.isolated."
                ),
                "stage": "sub_profile",
                "isolated_session_id": isolated_session_id,
                "profile_name": profile.name,
            }

        sub_profile_name = sub_profile_or_err
        logger.info(
            "_spawn_isolated_runner: sub-profile provisioned for "
            "parent=%s subagent=%s (sub_profile=%s)",
            parent_session_id, subagent_id, sub_profile_name,
        )

        # ── Stage: sub_cgroup — provision sub-cgroup ───────────
        # Phase 4 §4.3.5: when cgroups available + profile declares
        # kernel-enforceable runtime_limits, create the sub-cgroup
        # via the existing CgroupsManager.provision_cgroup API
        # (passing isolated_session_id as session_id — the cgroup
        # path naturally lands at jaato-ws-{parent}__sub_{subagent}/
        # via get_cgroup_name's template).  Sibling-not-nested
        # structure per Audit 7 (cgroup v2 "no internal processes"
        # rule + design intent).
        #
        # Graceful degradation: cgroups unavailable on this host →
        # skip cgroup creation, sub-runner inherits daemon's default
        # cgroup.  AppArmor isolation still applies.
        #
        # Phase 5 §5.1: ``effective_runtime_limits`` now always carries
        # kernel-enforceable fields (memory_max_mb / pids_max /
        # cpu_weight) via :func:`apply_isolated_defaults`, so the
        # ``has_kernel_limits()`` predicate is True for the default.
        # The cgroup-skip branch can only fire when ``CgroupsManager``
        # is unavailable (kernel without cgroup v2, or daemon wired
        # without one), which is a host-capability gap rather than a
        # profile-content gap.
        cgroup_path = ""  # Empty = no sub-cgroup provisioned.
        cgroups_manager = self._resolve_cgroups_manager()
        runtime_limits = effective_runtime_limits
        if (cgroups_manager is not None
                and cgroups_manager.is_available()
                and runtime_limits.has_kernel_limits()):
            cgroup_ok = cgroups_manager.provision_cgroup(
                isolated_session_id, runtime_limits,
            )
            if not cgroup_ok:
                # Roll back the §4.3.4 sub-AppArmor profile before
                # returning — otherwise it remains kernel-loaded
                # but unused, and the next provision attempt with
                # the same subagent_id sees the loaded profile.
                # Best-effort: teardown failure logs but doesn't
                # change the §4.3.5 return shape.  Idempotent
                # re-load via provision_sub_profile would succeed
                # anyway, so a stuck-loaded profile is not blocking.
                try:
                    apparmor_manager.teardown_sub_profile(
                        parent_session_id=parent_session_id,
                        subagent_id=subagent_id,
                    )
                except Exception:  # noqa: BLE001 — best-effort
                    logger.exception(
                        "_spawn_isolated_runner: sub-AppArmor "
                        "rollback failed after sub-cgroup provision "
                        "failure for parent=%s subagent=%s",
                        parent_session_id, subagent_id,
                    )
                logger.warning(
                    "_spawn_isolated_runner: sub-cgroup provision "
                    "failed for parent=%s subagent=%s "
                    "(isolated_session_id=%s)",
                    parent_session_id, subagent_id, isolated_session_id,
                )
                return {
                    "ok": False,
                    "error": (
                        f"sub-cgroup provision failed for isolated "
                        f"session {isolated_session_id!r}.  Sub-AppArmor "
                        f"profile {sub_profile_name!r} rolled back.  "
                        f"Workaround: omit agent_params.isolated to "
                        f"use default-share path."
                    ),
                    "stage": "sub_cgroup",
                    "isolated_session_id": isolated_session_id,
                    "profile_name": profile.name,
                    "apparmor_profile": "",  # Rolled back.
                }
            cgroup_path = str(
                cgroups_manager.get_cgroup_path(isolated_session_id),
            )
            logger.info(
                "_spawn_isolated_runner: sub-cgroup provisioned for "
                "parent=%s subagent=%s (cgroup_path=%s)",
                parent_session_id, subagent_id, cgroup_path,
            )
            # Phase 5 §5.2: nesting-visibility instrumentation.
            # Sibling cgroup structure (today, per §4.3.5) means
            # sub bounds don't compose under the parent's.  When
            # sub's kernel-enforced cap exceeds parent's, true
            # nesting WOULD have capped at parent's bound — log
            # so operators get visibility into where a Phase 6
            # nested-layout migration would change behavior.
            # Observability-only; no behavior change.  See
            # docs/design/phase5_5_2_nested_cgroup_deferral_audit.md.
            self._log_nesting_visibility(
                parent_session_id=parent_session_id,
                subagent_id=subagent_id,
                sub_limits=effective_runtime_limits,
            )
        else:
            logger.info(
                "_spawn_isolated_runner: sub-cgroup skipped for "
                "parent=%s subagent=%s "
                "(cgroups_available=%s, has_kernel_limits=%s) — "
                "sub-runner will inherit default cgroup",
                parent_session_id, subagent_id,
                cgroups_manager is not None
                and cgroups_manager.is_available(),
                runtime_limits.has_kernel_limits(),
            )

        # ── Stage: forwarding — spawn sub-runner subprocess ────
        # Phase 4 §4.3.6a: actually spawn the sub-runner with the
        # provisioned sub-AppArmor profile + (optional) sub-cgroup
        # attach.  Sub-runner self-confines via ``change_profile``
        # on spawn; pre-exec preexec_fn migrates into the sub-cgroup.
        #
        # Stays at ``stage=forwarding`` because cross-runner event
        # forwarding (sub→parent_runner) is §4.3.6b's job and the
        # first-turn prompt dispatch is §4.3.6c's job.  §4.3.6a
        # only proves the spawn happens + handle is bookkept.
        #
        # On spawn failure: roll back sub-cgroup + sub-AppArmor in
        # order to keep kernel state consistent (no orphaned
        # profile/cgroup if the subprocess never starts).
        try:
            sub_handle = self._do_spawn_isolated_runner(
                parent_session_id=parent_session_id,
                subagent_id=subagent_id,
                isolated_session_id=isolated_session_id,
                workspace_path=workspace_path,
                sub_apparmor_profile=sub_profile_name,
                cgroup_path=cgroup_path,
                profile=profile,
                effective_runtime_limits=effective_runtime_limits,
                agent_params=agent_params,
            )
        except Exception as spawn_exc:  # noqa: BLE001 — boundary
            logger.warning(
                "_spawn_isolated_runner: subprocess spawn failed "
                "for parent=%s subagent=%s: %s",
                parent_session_id, subagent_id, spawn_exc,
                exc_info=True,
            )
            # Rollback chain — cgroup then AppArmor.  Best-effort;
            # rollback failures log but don't change return shape.
            self._rollback_isolated_resources(
                parent_session_id=parent_session_id,
                subagent_id=subagent_id,
                isolated_session_id=isolated_session_id,
                cgroup_path=cgroup_path,
            )
            return {
                "ok": False,
                "error": (
                    f"sub-runner subprocess spawn failed: "
                    f"{type(spawn_exc).__name__}: {spawn_exc}.  "
                    f"Sub-cgroup + sub-AppArmor profile rolled back.  "
                    f"Workaround: omit agent_params.isolated to use "
                    f"default-share path."
                ),
                "stage": "forwarding",
                "isolated_session_id": isolated_session_id,
                "profile_name": profile.name,
                "apparmor_profile": "",  # Rolled back.
                "cgroup_path": "",       # Rolled back.
            }

        # Register the handle so §4.3.6b/c/d can find it by id.
        with self._lock:
            self._isolated_sub_runners[isolated_session_id] = sub_handle
        logger.info(
            "_spawn_isolated_runner: sub-runner spawned for "
            "parent=%s subagent=%s (isolated_session_id=%s "
            "pid=%d sub_profile=%s cgroup=%s)",
            parent_session_id, subagent_id, isolated_session_id,
            sub_handle.spawned.pid, sub_profile_name,
            cgroup_path or "(none)",
        )

        # ── Phase 4 §4.3.6c — session.bootstrap + first-turn dispatch ──
        # The sub-runner subprocess is up but its runner-side
        # JaatoSession isn't yet — dispatch session.bootstrap so
        # the sub-runner constructs its session host before we
        # send the first-turn prompt.  Bootstrap failure is
        # surfaced as a stage=forwarding error envelope; the
        # sub-runner is left registered (caller can attempt
        # explicit teardown).
        envelope = self._build_isolated_envelope(
            profile=profile,
            isolated_session_id=isolated_session_id,
            workspace_path=workspace_path,
            sub_apparmor_profile=sub_profile_name,
            agent_params=agent_params,
        )
        bootstrap_ok = self._dispatch_isolated_session_bootstrap(
            sub_handle, envelope,
        )
        if not bootstrap_ok:
            return {
                "ok": False,
                "error": (
                    f"sub-runner subprocess spawned but "
                    f"session.bootstrap dispatch failed.  Sub-runner "
                    f"will not run the task.  Sub-resources remain "
                    f"provisioned until parent teardown (§4.3.6d "
                    f"will add cascade)."
                ),
                "stage": "forwarding",
                "isolated_session_id": isolated_session_id,
                "profile_name": profile.name,
                "apparmor_profile": sub_profile_name,
                "cgroup_path": cgroup_path,
                "sub_runner_pid": sub_handle.spawned.pid,
                "sub_session_id": isolated_session_id,
            }

        # Schedule the first-turn prompt as a background task — the
        # sub-runner runs the task; streaming output forwards back
        # via the §4.3.6b chain.  Returns immediately so the caller
        # (subagent plugin) can continue.
        self._schedule_isolated_first_turn(sub_handle, task)

        # §4.3.6c milestone — isolated subagent is running end-to-end.
        # ok=True signals the supervisor that the spawn succeeded;
        # streaming output will arrive via inject_prompt over the
        # cross-runner chain.
        return {
            "ok": True,
            "session_id": isolated_session_id,
            "subagent_id": subagent_id,
            "runner_pid": sub_handle.spawned.pid,
            "apparmor_profile": sub_profile_name,
            "cgroup_path": cgroup_path,
            # Diagnostic fields kept for backward-compat with §4.3.6a/b
            # test assertions + audit-trail logging.
            "isolated_session_id": isolated_session_id,
            "profile_name": profile.name,
            "sub_runner_pid": sub_handle.spawned.pid,
            "sub_session_id": isolated_session_id,
        }

    def _do_spawn_isolated_runner(
        self,
        *,
        parent_session_id: str,
        subagent_id: str,
        isolated_session_id: str,
        workspace_path: str,
        sub_apparmor_profile: str,
        cgroup_path: str,
        profile: Any,  # SubagentProfile
        effective_runtime_limits: RuntimeLimits,
        agent_params: Optional[Dict[str, Any]],
    ) -> SubRunnerHandle:
        """Spawn the sub-runner subprocess + initialize its RPC
        channel (Phase 4 §4.3.6a).

        Mirrors ``runner_spawn.spawn_session_runner`` but for the
        isolated-subagent path: the daemon doesn't have a full
        JaatoServer for the sub-session (and doesn't need one — the
        runner-side JaatoSession holds all the state).  We hold the
        spawn handles on a :class:`SubRunnerHandle` instead.

        Raises any exception from the spawn / RPC start path; caller
        catches + rolls back kernel resources.
        """
        import asyncio
        from server.runner_rpc_client import RunnerRPCClient
        from server.runner_spawner import RunnerSpawner

        daemon_loop = getattr(self, "_daemon_loop", None)
        if daemon_loop is None:
            raise RuntimeError(
                "_do_spawn_isolated_runner: daemon loop unavailable; "
                "cannot start RunnerRPCClient"
            )

        spawner = RunnerSpawner()
        log_path: Optional[str] = None
        if workspace_path:
            log_dir = os.path.join(workspace_path, ".jaato", "logs")
            log_path = os.path.join(
                log_dir, f"runner-{isolated_session_id}.log",
            )

        # Cgroup attach: when §4.3.5 provisioned a sub-cgroup, build
        # the preexec_fn that migrates the forked child in.  When no
        # cgroup, pass ``None`` — sub-runner inherits daemon's
        # default cgroup (the documented Phase 5+ hardening gap).
        cgroup_attach = None
        if cgroup_path:
            cgroups_manager = self._resolve_cgroups_manager()
            if cgroups_manager is not None:
                cgroup_attach = cgroups_manager.make_attach_callback(
                    isolated_session_id,
                )

        # Phase 5 §5.1: forward the app-layer caps via the
        # ``JAATO_RUNNER_MAX_OUTPUT_CHARS`` / ``JAATO_RUNNER_TOOL_TIMEOUT_SECONDS``
        # env-passthrough already wired in :class:`RunnerSpawner`.  The
        # runner-side cli plugin reads those env vars at startup.  Without
        # this hookup the kernel-layer defaults would apply (via cgroup)
        # but the application-layer defaults — tool wall-clock + output
        # truncation — would silently no-op on the sub-runner subprocess.
        spawned = spawner.spawn(
            profile_name=sub_apparmor_profile,
            session_id=isolated_session_id,
            workspace_path=workspace_path,
            log_path=log_path,
            max_output_chars=effective_runtime_limits.max_output_bytes,
            tool_timeout_seconds=effective_runtime_limits.tool_timeout_seconds,
            disable_confine=False,  # Always confined for §4.3.6.
            cgroup_attach=cgroup_attach,
        )

        rpc = RunnerRPCClient(
            spawned.parent_socket,
            runner_pid=spawned.pid,
            loop=daemon_loop,
        )
        fut = asyncio.run_coroutine_threadsafe(rpc.start(), daemon_loop)
        fut.result(timeout=10.0)

        return SubRunnerHandle(
            parent_session_id=parent_session_id,
            subagent_id=subagent_id,
            isolated_session_id=isolated_session_id,
            rpc=rpc,
            spawned=spawned,
            sub_apparmor_profile=sub_apparmor_profile,
            cgroup_path=cgroup_path,
        )

    def _build_isolated_envelope(
        self,
        *,
        profile: Any,  # SubagentProfile
        isolated_session_id: str,
        workspace_path: str,
        sub_apparmor_profile: str,
        agent_params: Optional[Dict[str, Any]],
    ) -> Any:
        """Build a :class:`SessionInitEnvelope` for an isolated
        subagent's runner-side bootstrap (Phase 4 §4.3.6c).

        Mirrors ``server.runner_spawn.build_session_envelope`` but
        sources fields from the reconstructed SubagentProfile
        directly (no daemon-side JaatoServer with ``_profile``
        attribute — isolated subagents don't get one).
        """
        from shared.session_envelope import SessionInitEnvelope

        provider_name = getattr(profile, "provider", None) or ""
        model_name = getattr(profile, "model", None) or ""
        plugins_list = list(getattr(profile, "plugins", []) or [])
        preloaded = set(
            getattr(profile, "preloaded_plugins", set()) or set(),
        )
        # Server 0.6.123+: values flow through ``expand_plugin_configs``
        # so ``${VAR}`` references AND secret URIs (``pass://`` /
        # ``vault://``) resolve daemon-side before the isolated
        # subagent's runner sees them.  Same wire-gap class as PR #139
        # (completion_validators) and the v118 zhipuai diagnosis.
        # AppArmor-confined runners can't exec ``pass``, so resolution
        # must happen here.
        from shared.plugins.subagent.config import expand_plugin_configs
        raw_plugin_configs = dict(
            getattr(profile, "plugin_configs", {}) or {},
        )
        plugin_configs = expand_plugin_configs(
            raw_plugin_configs,
            workspace_root_override=workspace_path,
        )
        plugin_specs = []
        for name in plugins_list:
            entry = {"name": name, "preload": name in preloaded}
            cfg = plugin_configs.get(name)
            if cfg:
                entry["config"] = dict(cfg)
            plugin_specs.append(entry)

        system_instructions = getattr(profile, "system_instructions", None)
        gc_dict = None
        gc_obj = getattr(profile, "gc", None)
        if gc_obj is not None:
            gc_type = getattr(gc_obj, "type", None)
            gc_config = getattr(gc_obj, "config", None) or {}
            if gc_type:
                gc_dict = {"type": gc_type, **dict(gc_config)}
        env_overrides = dict(getattr(profile, "env", {}) or {})

        if not provider_name:
            provider_name = "anthropic"

        try:
            project_val = os.environ.get("PROJECT_ID", "") or ""
            location_val = os.environ.get("LOCATION", "") or ""
        except Exception:  # noqa: BLE001
            project_val = ""
            location_val = ""

        return SessionInitEnvelope(
            session_id=isolated_session_id,
            workspace_path=workspace_path,
            profile_name=sub_apparmor_profile,
            provider_name=provider_name,
            model_name=model_name,
            plugins=plugin_specs,
            # Phase 4 §C (envelope schema v2): carry the full
            # profile.plugin_configs map at the top level so
            # auto-loaded plugins (permission, gc_*, etc.) that
            # aren't named in ``plugins`` still receive their
            # profile overrides on the runner side.
            plugin_configs=plugin_configs,
            system_instructions=system_instructions,
            agent_id="main",
            gc=gc_dict,
            agent_params=dict(agent_params or {}),
            config_root=None,  # Isolated subagent doesn't inherit.
            env_overrides=env_overrides,
            project=project_val,
            location=location_val,
            completion_payload_schema=getattr(
                profile, "completion_payload_schema", None,
            ),
            completion_validators=list(
                getattr(profile, "completion_validators", []) or []
            ),
            completion_artifacts=[
                {
                    "renderer": getattr(a, "renderer", None),
                    "output": getattr(a, "output", None),
                    "on_error": getattr(a, "on_error", "warn"),
                    "description": getattr(a, "description", None),
                }
                for a in (getattr(profile, "completion_artifacts", []) or [])
                if hasattr(a, "renderer")
            ],
        )

    def _dispatch_isolated_session_bootstrap(
        self,
        sub_handle: SubRunnerHandle,
        envelope: Any,  # SessionInitEnvelope
        *,
        timeout: float = 30.0,
    ) -> bool:
        """Send the ``session.bootstrap`` RPC to the sub-runner so
        its runner-side JaatoSession host is populated before the
        first-turn prompt arrives (Phase 4 §4.3.6c).

        Synchronous: blocks until the sub-runner acknowledges.
        Returns True on success, False on failure (logged).
        """
        try:
            result = sub_handle.rpc.bootstrap_session_threadsafe(
                envelope, timeout=timeout,
            )
            logger.info(
                "isolated session.bootstrap acknowledged for %s: %s",
                sub_handle.isolated_session_id, result,
            )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "isolated session.bootstrap failed for %s: %s",
                sub_handle.isolated_session_id, exc,
                exc_info=True,
            )
            return False

    def _schedule_isolated_first_turn(
        self,
        sub_handle: SubRunnerHandle,
        task: str,
    ) -> None:
        """Schedule the first-turn prompt dispatch as an async task
        on the daemon loop (Phase 4 §4.3.6c).

        Returns immediately — the actual ``session.send_message``
        runs in the background.  Output streams back to the parent
        runner via the §4.3.6b ``subagent.forward_event`` RPC chain
        triggered by the ``on_output`` / ``on_notification``
        callbacks registered here.

        On completion: forwards final response (as event_kind=output)
        + status=done.
        On exception: forwards status=error with the exception message.
        """
        import asyncio

        daemon_loop = getattr(self, "_daemon_loop", None)
        if daemon_loop is None:
            logger.warning(
                "_schedule_isolated_first_turn: no daemon loop; "
                "sub-runner %s will not receive task",
                sub_handle.isolated_session_id,
            )
            self._forward_subagent_event_to_parent(
                sub_handle, "error",
                {"message": "no daemon loop to dispatch first-turn prompt"},
            )
            return

        # on_output: forward streaming text chunks to parent.
        def on_output(source: str, text: str, mode: str) -> None:
            try:
                self._forward_subagent_event_to_parent(
                    sub_handle, "output",
                    {"text": text, "source": source},
                )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "on_output forwarding failed for %s",
                    sub_handle.isolated_session_id,
                )

        # on_notification: lifecycle notifications (instruction
        # budget, retry, etc.).  Phase 4 §4.3.6c forwards them
        # as status events; finer-grained kinds are Phase 5+.
        def on_notification(event_type: str, payload: Any) -> None:
            try:
                self._forward_subagent_event_to_parent(
                    sub_handle, "status",
                    {
                        "status": str(event_type),
                        "payload": payload if isinstance(payload, dict) else {},
                    },
                )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "on_notification forwarding failed for %s",
                    sub_handle.isolated_session_id,
                )

        async def _run_first_turn():
            try:
                response = await sub_handle.rpc.session_send_message(
                    prompt=task,
                    on_output=on_output,
                    on_notification=on_notification,
                )
                # Final response — forward as terminal output.
                self._forward_subagent_event_to_parent(
                    sub_handle, "output",
                    {"text": response, "source": "final"},
                )
                self._forward_subagent_event_to_parent(
                    sub_handle, "status",
                    {"status": "done"},
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "_run_first_turn: session.send_message failed for %s",
                    sub_handle.isolated_session_id,
                )
                self._forward_subagent_event_to_parent(
                    sub_handle, "error",
                    {"message": f"{type(exc).__name__}: {exc}"},
                )

        asyncio.run_coroutine_threadsafe(_run_first_turn(), daemon_loop)

    def _forward_subagent_event_to_parent(
        self,
        sub_handle: SubRunnerHandle,
        event_kind: str,
        event_payload: Dict[str, Any],
        *,
        timeout: float = 5.0,
    ) -> bool:
        """Forward a sub-runner event to the parent runner via the
        ``subagent.forward_event`` RPC (Phase 4 §4.3.6b).

        Looks up the parent session's runner-RPC handle (from the
        parent's JaatoServer in ``self._sessions``) and dispatches
        the event.  Runner-side handler routes to the SubagentPlugin's
        ``receive_forwarded_event``, which calls ``inject_prompt``
        on the parent session.

        Args:
            sub_handle: The sub-runner's :class:`SubRunnerHandle`.
                Provides ``parent_session_id`` for parent lookup.
            event_kind: ``"output"`` | ``"status"`` | ``"error"``.
            event_payload: Event-kind-specific payload dict.
            timeout: Wall-clock cap for the RPC.  Default 5s — the
                runner-side handler is fast (lookup + inject_prompt
                + return), no plugin discovery / provider connect.

        Returns:
            True when the forward succeeded; False when the parent
            runner is unavailable / the RPC failed / the plugin
            rejected the event.  Caller logs but doesn't retry —
            cross-runner forwarding is best-effort (CLAUDE.md §4.4).

        Phase 4 §4.3.6b: this is the daemon-side helper.  §4.3.6c
        wires the subscription that triggers it via the first-turn
        ``session.send_message`` call's ``on_notification`` callback.
        """
        with self._lock:
            parent_session = self._sessions.get(sub_handle.parent_session_id)
        if parent_session is None:
            logger.warning(
                "_forward_subagent_event_to_parent: parent session %s "
                "not found; dropping event_kind=%s for subagent_id=%s",
                sub_handle.parent_session_id, event_kind,
                sub_handle.subagent_id,
            )
            return False

        parent_runner_rpc = getattr(
            parent_session.server, "runner_rpc", None,
        )
        if parent_runner_rpc is None:
            logger.warning(
                "_forward_subagent_event_to_parent: parent session %s "
                "has no runner_rpc; dropping event_kind=%s for "
                "subagent_id=%s",
                sub_handle.parent_session_id, event_kind,
                sub_handle.subagent_id,
            )
            return False

        try:
            env = parent_runner_rpc.call_threadsafe(
                "subagent.forward_event",
                {
                    "subagent_id": sub_handle.subagent_id,
                    "event_kind": event_kind,
                    "event_payload": event_payload,
                },
                timeout=timeout,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "_forward_subagent_event_to_parent: RPC failed for "
                "subagent_id=%s event_kind=%s: %s",
                sub_handle.subagent_id, event_kind, exc,
            )
            return False

        if not env.ok:
            err_msg = (
                env.error.message if env.error else "no error message"
            )
            logger.warning(
                "_forward_subagent_event_to_parent: handler rejected "
                "subagent_id=%s event_kind=%s: %s",
                sub_handle.subagent_id, event_kind, err_msg,
            )
            return False
        return True

    def _cascade_teardown_isolated_subagents(
        self,
        parent_session_id: str,
    ) -> int:
        """Tear down every isolated sub-runner owned by a parent
        session (Phase 4 §4.3.6d).

        Called when a parent session is unloaded or shut down.
        Iterates ``_isolated_sub_runners`` filtering by
        ``parent_session_id``; for each handle:
          1. Close the sub-runner's RPC client (sends EOF, waits
             for the sub-runner subprocess to exit).
          2. Tear down the sub-cgroup (``cgroup.kill`` atomic
             termination, then rmdir).
          3. Tear down the sub-AppArmor profile (``apparmor_parser -R``).
          4. Remove the handle from ``_isolated_sub_runners``.

        Best-effort throughout — each step wrapped in try/except so
        a single teardown failure doesn't strand the others.  Returns
        the count of handles torn down for logging / metrics.

        Args:
            parent_session_id: Parent session whose isolated
                sub-runners should be torn down.

        Returns:
            Number of handles processed (whether or not each
            individual teardown step succeeded).
        """
        import asyncio

        with self._lock:
            owned_handles = [
                handle for handle in self._isolated_sub_runners.values()
                if handle.parent_session_id == parent_session_id
            ]

        if owned_handles:
            logger.info(
                "_cascade_teardown_isolated_subagents: tearing down "
                "%d sub-runner(s) for parent_session=%s",
                len(owned_handles), parent_session_id,
            )
        # Phase 5 §5.3: don't short-circuit on empty owned_handles —
        # the orphan-scan tail still runs to catch kernel state left
        # behind by rollback failures or crashes that never registered
        # a handle.  Per the ledger, the leak audit runs at every
        # parent teardown, not only when known handles existed.

        for handle in owned_handles:
            # 1. Close RPC client (signals EOF; sub-runner exits).
            try:
                daemon_loop = getattr(self, "_daemon_loop", None)
                if daemon_loop is not None and hasattr(handle.rpc, "close"):
                    fut = asyncio.run_coroutine_threadsafe(
                        handle.rpc.close(), daemon_loop,
                    )
                    try:
                        fut.result(timeout=5.0)
                    except Exception:  # noqa: BLE001
                        logger.warning(
                            "cascade teardown: RPC close timed out "
                            "for %s — continuing",
                            handle.isolated_session_id,
                        )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "cascade teardown: RPC close failed for %s",
                    handle.isolated_session_id,
                )

            # 2. Tear down sub-cgroup (if one was provisioned).
            if handle.cgroup_path:
                try:
                    cgroups_manager = self._resolve_cgroups_manager()
                    if cgroups_manager is not None:
                        cgroups_manager.teardown_cgroup(
                            handle.isolated_session_id,
                        )
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "cascade teardown: sub-cgroup teardown "
                        "failed for %s",
                        handle.isolated_session_id,
                    )

            # 3. Tear down sub-AppArmor profile (if one was loaded).
            if handle.sub_apparmor_profile:
                try:
                    apparmor_manager = self._resolve_apparmor_manager()
                    if apparmor_manager is not None:
                        apparmor_manager.teardown_sub_profile(
                            parent_session_id=handle.parent_session_id,
                            subagent_id=handle.subagent_id,
                        )
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "cascade teardown: sub-AppArmor teardown "
                        "failed for %s",
                        handle.isolated_session_id,
                    )

            # 4. Remove from registry.
            with self._lock:
                self._isolated_sub_runners.pop(
                    handle.isolated_session_id, None,
                )
            logger.info(
                "cascade teardown: completed for %s",
                handle.isolated_session_id,
            )

        # Phase 5 §5.3 — orphan sub-cgroup reaper.  After the known-
        # handles loop completes, scan for sub-cgroups under this
        # parent that exist on disk but aren't in the just-torn-down
        # set.  Sources of orphans: rollback failure mid-teardown
        # (e.g., transient EBUSY before _rollback_isolated_resources'
        # teardown_cgroup call) and mid-spawn crashes that left
        # kernel state without a corresponding handle.  Reap via the
        # existing teardown_cgroup path so behaviour matches known
        # handles exactly.  Best-effort: failures log WARNING but
        # don't change the cascade return value (which is the count
        # of HANDLES torn down, not orphans reaped).  See
        # docs/design/phase5_5_3_cgroup_leak_audit_audit.md.
        torn_down_ids = {h.isolated_session_id for h in owned_handles}
        try:
            cgroups_manager = self._resolve_cgroups_manager()
            if cgroups_manager is not None:
                orphans = cgroups_manager.list_orphan_sub_cgroups(
                    parent_session_id, torn_down_ids,
                )
                for orphan_id in orphans:
                    logger.warning(
                        "cascade teardown: reaped orphaned sub-cgroup "
                        "%s — likely cause: rollback or mid-spawn crash",
                        orphan_id,
                    )
                    try:
                        cgroups_manager.teardown_cgroup(orphan_id)
                    except Exception:  # noqa: BLE001
                        logger.exception(
                            "cascade teardown: orphan reap failed "
                            "for %s — manual cleanup may be required",
                            orphan_id,
                        )
        except Exception:  # noqa: BLE001
            logger.exception(
                "cascade teardown: orphan scan failed for "
                "parent_session=%s — no orphans reaped",
                parent_session_id,
            )

        return len(owned_handles)

    def _rollback_isolated_resources(
        self,
        *,
        parent_session_id: str,
        subagent_id: str,
        isolated_session_id: str,
        cgroup_path: str,
    ) -> None:
        """Tear down sub-cgroup + sub-AppArmor on §4.3.6a spawn
        failure (Phase 4 §4.3.6a).

        Best-effort: each teardown wrapped in try/except.  Rollback
        failures log but don't propagate — the helper's return
        shape is what matters to callers.
        """
        # Cgroup first (innermost resource).
        if cgroup_path:
            try:
                cgroups_manager = self._resolve_cgroups_manager()
                if cgroups_manager is not None:
                    cgroups_manager.teardown_cgroup(isolated_session_id)
            except Exception:  # noqa: BLE001 — best-effort
                logger.exception(
                    "_rollback_isolated_resources: sub-cgroup "
                    "teardown failed for parent=%s subagent=%s",
                    parent_session_id, subagent_id,
                )

        # AppArmor next.
        try:
            apparmor_manager = self._resolve_apparmor_manager()
            if apparmor_manager is not None:
                apparmor_manager.teardown_sub_profile(
                    parent_session_id=parent_session_id,
                    subagent_id=subagent_id,
                )
        except Exception:  # noqa: BLE001 — best-effort
            logger.exception(
                "_rollback_isolated_resources: sub-AppArmor "
                "teardown failed for parent=%s subagent=%s",
                parent_session_id, subagent_id,
            )

    def _log_nesting_visibility(
        self,
        *,
        parent_session_id: str,
        subagent_id: str,
        sub_limits: RuntimeLimits,
    ) -> None:
        """Phase 5 §5.2 — log when sub-cgroup bounds would have been
        capped by parent under a nested-layout migration.

        Today's sub-cgroup is sibling to the parent's (per §4.3.5
        + cgroup v2 "no internal processes" rule); their kernel
        limits don't compose.  When the supervisor declares an
        isolated sub with `memory_max_mb` or `pids_max` greater
        than the parent's, true nesting WOULD have capped the sub
        at parent's value.  This helper surfaces those cases at
        `INFO` so operators can see the evidence base for a Phase 6
        nested-layout migration.

        Pure observability — no behavior change.  Sub-cgroup limits
        are already applied as declared by the time this helper
        runs.

        `cpu_weight` is intentionally omitted: it's a share within
        a cgroup hierarchy, not an absolute cap, so a parent-vs-sub
        comparison doesn't represent composition semantics.  When
        Phase 6 ships the restructure, cpu_weight composition
        becomes meaningful; until then, omitting it from the log
        avoids misleading operators.

        See `docs/design/phase5_5_2_nested_cgroup_deferral_audit.md`.
        """
        parent_session = self._sessions.get(parent_session_id)
        parent_server = getattr(parent_session, "server", None) if parent_session else None
        parent_profile = getattr(parent_server, "_profile", None) if parent_server else None
        parent_limits = getattr(parent_profile, "runtime_limits", None) if parent_profile else None

        if parent_limits is None:
            logger.info(
                "_spawn_isolated_runner: nesting-visibility — "
                "parent=%s has no kernel runtime_limits; "
                "sub %s bounds are independent "
                "(sub.memory_max_mb=%s, sub.pids_max=%s).  "
                "Phase 6 nested layout would have nothing to cap.",
                parent_session_id, subagent_id,
                sub_limits.memory_max_mb, sub_limits.pids_max,
            )
            return

        fields_exceeded: List[str] = []
        if (parent_limits.memory_max_mb is not None
                and sub_limits.memory_max_mb is not None
                and sub_limits.memory_max_mb > parent_limits.memory_max_mb):
            fields_exceeded.append(
                f"memory_max_mb (sub={sub_limits.memory_max_mb}, "
                f"parent={parent_limits.memory_max_mb})"
            )
        if (parent_limits.pids_max is not None
                and sub_limits.pids_max is not None
                and sub_limits.pids_max > parent_limits.pids_max):
            fields_exceeded.append(
                f"pids_max (sub={sub_limits.pids_max}, "
                f"parent={parent_limits.pids_max})"
            )

        if fields_exceeded:
            logger.info(
                "_spawn_isolated_runner: nesting-visibility — "
                "parent=%s subagent=%s sub-cgroup bounds EXCEED "
                "parent's on: %s.  Today (sibling layout per §4.3.5): "
                "sub gets its declared bounds independently.  Phase 6 "
                "nested layout would cap at parent's value.  See "
                "Phase 5 §5.2.",
                parent_session_id, subagent_id, "; ".join(fields_exceeded),
            )

    def _resolve_cgroups_manager(self) -> Optional[Any]:
        """Return the daemon's :class:`CgroupsManager` instance, or
        ``None`` if not wired.

        Phase 4 §4.3.5: mirrors :meth:`_resolve_apparmor_manager`.
        ``_cgroups_manager`` lives on session managers wired via the
        IPC apparmor opt-in pre-init hook.  Test fakes that bypass
        the wire-up won't have the attribute; ``getattr`` returns
        ``None`` in that case so the helper falls back to the
        skip-cgroup-creation branch.
        """
        return getattr(self, "_cgroups_manager", None)

    def _resolve_apparmor_manager(self) -> Optional[Any]:
        """Return the daemon's :class:`AppArmorManager` instance, or
        ``None`` if not wired.

        Phase 4 §4.3.4: ``_spawn_isolated_runner`` needs the
        AppArmorManager to provision sub-profiles.  The manager
        instance lives at ``self._apparmor_manager`` on session
        managers wired via the IPC AppArmor opt-in path (set in the
        pre-initialize hook).  Test fakes that bypass the wire-up
        (``SessionManager.__new__`` without the hook) won't have the
        attribute; ``getattr`` returns ``None`` in that case so the
        helper falls back to the unavailable-message branch.

        Returns:
            The wired :class:`AppArmorManager`, or ``None`` if
            unavailable (test fake, host without AppArmor, etc.).
        """
        return getattr(self, "_apparmor_manager", None)

    def add_pre_initialize_hook(self, hook: Callable) -> None:
        """Register a callback invoked BEFORE ``server.initialize()`` runs
        (server 0.6.49+).

        The hook is called with four arguments:

        1. ``server`` — the just-constructed ``JaatoServer`` instance.
           Plugin registry NOT YET populated; provider NOT YET connected.
           Hook can stash the reference but must not call methods that
           depend on init state.
        2. ``session_id`` — the unique session identifier string.
        3. ``workspace_path`` — the session's workspace dir (or ``None``).
        4. ``client_id`` — the requesting client's id, or ``None`` for
           non-client-driven session creation paths (currently
           ``_load_session_impl``).  Phase 2 task 2.3 added this so the
           IPC AppArmor pre-init hook can look up the creator's
           ``ClientConfigRequest.apparmor`` opt-in (the lookup via
           ``_client_to_session`` doesn't work pre-init because that
           mapping isn't populated yet).

        Pre-initialize hooks exist so transports (notably the WS server's
        AppArmor wiring) can provision per-session kernel resources
        BEFORE ``server.initialize()`` runs the agent's
        ``configure()`` — including dynamic-instructions expansion and
        any prefetch scripts.  Without this hook, the AppArmor profile
        would not exist yet at prefetch time, leaving prefetch scripts
        unconfined and able to write to ``.jaato`` (or anywhere else
        the unconfined daemon can reach) before the deny rules apply.

        Distinct from ``add_session_hook`` which fires AFTER ``initialize()``
        completes (used for set_apparmor_confinement, cgroup attach
        callback, sandbox wiring — anything that depends on the
        executor existing).

        Hooks are called in registration order.  Exceptions are logged
        and subsequent hooks still run.

        Args:
            hook: A callable with signature
                ``(server: JaatoServer, session_id: str,
                workspace_path: Optional[str],
                client_id: Optional[str]) -> None``.

        Backwards-compat: hooks accepting only the first three args
        (the pre-Phase-2 signature) are still called with positional
        args 1-3; the ``client_id`` is dropped via ``inspect.signature``
        introspection inside :meth:`_run_pre_initialize_hooks`.
        """
        self._pre_initialize_hooks.append(hook)

    def _run_session_hooks(self, server: JaatoServer, session_id: str) -> None:
        """Invoke all registered session hooks for a newly set-up session.

        Args:
            server: The JaatoServer instance to pass to each hook.
            session_id: The session identifier.
        """
        for hook in self._session_hooks:
            try:
                hook(server, session_id)
            except Exception as exc:
                logger.warning("Session hook failed: %s", exc, exc_info=True)

    def _bootstrap_session(
        self,
        envelope: 'BootstrapEnvelope',
    ) -> Tuple[Optional[JaatoServer], Optional['Session']]:
        """Construct + initialize a session from a BootstrapEnvelope.

        Phase 3 §3.12.0.  This is the single helper every
        session-creation path funnels through, replacing the
        ad-hoc kwarg-bag previously inlined in each call site
        (``_create_session_impl``, ``_load_session_impl``,
        ``run_ephemeral_session``, ``JaatoWSServer`` standalone).

        Body matches the §3.12.0 spec:

        1. Construct ``JaatoServer`` from the envelope's
           construction fields.
        2. Push ``config_root`` onto the server BEFORE
           ``_run_pre_initialize_hooks`` so plugins discovered
           during ``initialize`` see the override.
        3. Run pre-initialize hooks (legacy 3-arg + 4-arg both
           still supported via :meth:`_run_pre_initialize_hooks`'s
           ``inspect.signature`` introspection).
        4. ``server.initialize()`` — return ``(None, None)`` on
           failure (the underlying error already emitted to the
           client).
        5. Build the :class:`Session` record with ``sandbox_mode``
           resolved per the priority chain:
           a. ``envelope.sandbox_mode`` — disk-restore's pre-known
              value (the saved Session record's mode).
           b. Return value of
              :meth:`_provision_ipc_apparmor_and_spawn_runner`
              (§3.13's inline call) — apparmor opt-in result for
              the IPC creation path.
           c. ``None`` — no opt-in / non-confined session.

           Phase 3 §3.13 removed the legacy
           ``server._planned_sandbox_mode`` stash slot; the
           apparmor opt-in is now read directly inside the
           inline provisioning call (whose return value flows
           into priority chain step (b)) rather than via a
           transient attribute on JaatoServer.

        The helper does NOT do:

        - Spawn-payload validation — that's profile-resolution
          time, before the envelope is built.
        - Session storage / event-callback rewiring / TODO
          configuration / client-config application — those are
          post-bootstrap concerns the caller handles after this
          returns.
        - Auth-complete callback registration — caller-specific.

        Args:
            envelope: The bootstrap payload aggregating every
                input the JaatoServer construction +
                Session-record path needs.

        Returns:
            ``(JaatoServer, Session)`` on success;
            ``(None, None)`` if ``server.initialize()`` failed
            (error already reported via the in-init event sink).

        Phase 3 §3.12.0 migrated the IPC path.  Phase 3 §3.12
        disk-restore migration routes ``_load_session_impl`` through
        the inner :meth:`_construct_and_initialize_server` helper —
        the helper splits the JaatoServer-construction-and-init from
        the Session-record assembly so the disk-restore path can
        share the construction logic while building its own
        record (with ``last_activity`` / ``user_inputs`` / etc.
        from the saved state).  The ephemeral path follows in a
        future commit.

        An AST partition test (``test_bootstrap_partition.py``)
        tracks remaining direct ``JaatoServer(...)`` construction
        sites with an explicit allow-list so a contributor adding
        a NEW path is forced to either funnel through here (or the
        sub-helper) or extend the allow-list.
        """
        server, planned_sandbox = self._construct_and_initialize_server(envelope)
        if server is None:
            return None, None

        session = Session(
            session_id=envelope.session_id,
            name=envelope.name,
            server=server,
            created_at=(
                envelope.timestamp.isoformat()
                if envelope.timestamp is not None
                else datetime.now(timezone.utc).isoformat()
            ),
            description=envelope.description,
            is_dirty=True,  # New session needs saving
            workspace_path=envelope.workspace_path,
            config_root=envelope.config_root,
            provisioned=envelope.provisioned,
            created_by=envelope.created_by,
            sandbox_mode=planned_sandbox,
        )

        return server, session

    def _construct_and_initialize_server(
        self,
        envelope: 'BootstrapEnvelope',
    ) -> Tuple[Optional[JaatoServer], Optional[str]]:
        """JaatoServer construction + pre-init + initialize, shared
        across IPC and disk-restore bootstrap paths (Phase 3 §3.12).

        Splits out from :meth:`_bootstrap_session` so the
        disk-restore path (which assembles a different Session
        record from the saved state — ``last_activity``,
        ``user_inputs``, restored ``is_dirty``) can share the
        construction logic without inheriting the create-session
        Session-record shape.

        Body matches the §3.12.0 spec:

        1. Construct ``JaatoServer`` from envelope construction
           fields.
        2. Push ``config_root`` onto the server BEFORE pre-init
           hooks so plugins see the override.
        3. Call ``_provision_ipc_apparmor_and_spawn_runner`` (§3.13
           inline relocation) — clean no-op when ``client_id`` is
           ``None`` or no apparmor opt-in.
        4. Run remaining pre-init hooks (WS + third-party).
        5. ``server.initialize()`` — return ``(None, None)`` on
           failure.
        6. Resolve sandbox_mode: ``envelope.sandbox_mode`` wins
           (disk-restore's pre-known value); else IPC method
           result; else None.

        Returns:
            ``(JaatoServer, sandbox_mode)`` on success;
            ``(None, None)`` on init failure.
        """
        server = JaatoServer(
            env_file=envelope.env_file,
            provider=None,  # Let env_file determine provider
            on_event=envelope.on_event_during_init,
            workspace_path=envelope.workspace_path,
            session_id=envelope.session_id,
            env_overrides=envelope.env_overrides,
            instruction_token_cache=envelope.instruction_token_cache,
            profile=envelope.profile,
            system_instruction_override=envelope.system_instruction_override,
            suppress_base_instructions=envelope.suppress_base_instructions,
            agent_name=envelope.agent_name,
        )

        # Push config_root BEFORE initialize so plugins discovered
        # during init see the override on their first
        # set_config_root notification.
        if envelope.config_root:
            server.config_root = envelope.config_root

        # Phase 4 §D: stash agent_params transiently on the per-session
        # JaatoServer so build_session_envelope can pick them up and
        # forward them on the SessionInitEnvelope.  The JaatoServer is
        # per-session (one instance per session, GC'd at session end),
        # so this is not centralized daemon-state — it's just the
        # per-session daemon-side handle holding session data during
        # the envelope-build window.
        server._agent_params = dict(envelope.agent_params or {})

        # Phase 4 §B: resolve workspace .env + profile.env + overrides
        # BEFORE the runner-spawn fork so secret URIs (pass://,
        # vault://, awssm://, sops://, keyring://) reach the runner
        # subprocess via inherited os.environ.  Pre-fix the resolution
        # ran inside server.initialize() step 1 which fires AFTER the
        # spawn — the fork inherited unresolved literal URIs, and the
        # runner-side resolver wasn't always able to re-resolve (entry-
        # point registration timing + GPG-agent priming concerns).
        # Resolving daemon-side is reliable: the daemon process has
        # premium's secret resolvers entry-point-discovered at startup
        # and the GPG-agent socket has been primed by the operator.
        server._resolve_session_env()

        # Phase 3 §3.13: IPC apparmor provisioning + runner spawn
        # used to live in a pre-initialize hook registered from
        # ``server/__main__.py``.  The hook indirection was a
        # transitional step from Phase 2 §2.3; Phase 3 inlines the
        # logic here so the call site is co-located with the rest
        # of the bootstrap helper.  The method is a clean no-op
        # when ``client_id`` is None or the client did not opt in
        # to apparmor; non-IPC paths continue unaffected.
        #
        # Phase 4 §B: wrapped in _with_session_env so the spawn's
        # fork() inherits the resolved env via os.environ.  See
        # docs/design/phase4_env_propagation_audit.md for the bug
        # this closes (workspace .env pass:// URIs not reaching the
        # runner post-§7c seat-flip).
        with server._with_session_env():
            ipc_sandbox_mode = self._provision_ipc_apparmor_and_spawn_runner(
                server,
                envelope.session_id,
                envelope.workspace_path,
                envelope.client_id,
                apparmor_override=envelope.apparmor,
                # 2026-05-14 unification: thread the envelope's
                # ``config_root`` + ``env_file`` so the AppArmor policy
                # generator receives them across all entry points (IPC,
                # disk-restore, reactor-spawned headless, ephemeral
                # subagent, WS standalone) — not just the IPC path
                # that historically populated ``client_config[client_id]``.
                config_root_override=envelope.config_root,
                env_file_override=envelope.env_file,
            )

        # Pre-initialize hooks fire BEFORE initialize() — gives
        # transports a window to provision kernel resources
        # (AppArmor profile, cgroup) so prefetch can run inside
        # the session's confinement instead of unconfined.  After
        # §3.13 the IPC apparmor hook is no longer registered
        # here (relocated to the inline call above); the WS hook
        # and any third-party hooks continue to use this surface.
        self._run_pre_initialize_hooks(
            server,
            envelope.session_id,
            envelope.workspace_path,
            envelope.client_id,
        )

        # Initialize.  On failure, core.py already emits a
        # ConfigurationError event via the in-init sink — no need
        # for a redundant SessionError here.
        if not server.initialize():
            return None, None

        # Resolve sandbox_mode.  Priority:
        # 1. ``envelope.sandbox_mode`` — authoritative pre-resolved
        #    value (disk-restore's saved mode).
        # 2. Result of inline IPC apparmor provisioning.
        # 3. None — no opt-in / non-confined.
        planned_sandbox = envelope.sandbox_mode
        if planned_sandbox is None:
            planned_sandbox = ipc_sandbox_mode
        return server, planned_sandbox

    def _run_pre_initialize_hooks(
        self,
        server: JaatoServer,
        session_id: str,
        workspace_path: Optional[str],
        client_id: Optional[str] = None,
    ) -> None:
        """Invoke pre-initialize hooks (server 0.6.49+).

        Called before ``server.initialize()`` so that transport-level
        kernel-resource provisioning (AppArmor profile, cgroup) can
        happen before the agent's configure() runs prefetch scripts.

        Args:
            server: The JaatoServer instance (constructed but not
                initialized).
            session_id: The session identifier.
            workspace_path: Session's workspace dir, or None.
            client_id: The requesting client id (or ``None`` for
                non-client-driven paths like ``_load_session_impl``).
                Phase 2 task 2.3+ extends the hook signature with this
                parameter so the IPC AppArmor pre-init hook can look up
                the creator's apparmor opt-in.

        Backwards-compat: hooks declared with the legacy 3-arg
        signature (server, session_id, workspace_path) keep working —
        we introspect the callable's parameter count and drop
        ``client_id`` for old-style hooks.
        """
        import inspect
        for hook in self._pre_initialize_hooks:
            try:
                # Count positional params on the hook (ignoring *args/**kwargs).
                # 4 = new-style; 3 = legacy.
                try:
                    sig = inspect.signature(hook)
                    n_positional = sum(
                        1 for p in sig.parameters.values()
                        if p.kind in (
                            inspect.Parameter.POSITIONAL_ONLY,
                            inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        )
                    )
                except (TypeError, ValueError):
                    n_positional = 4  # assume new-style on introspection failure
                if n_positional >= 4:
                    hook(server, session_id, workspace_path, client_id)
                else:
                    hook(server, session_id, workspace_path)
            except Exception as exc:
                logger.warning(
                    "Pre-initialize hook failed: %s", exc, exc_info=True,
                )

    def set_event_callback(
        self,
        callback: Callable[[str, Event], None],
    ) -> None:
        """Set callback for routing events to clients.

        Args:
            callback: Called with (client_id, event) for each event.
        """
        self._event_callback = callback

    def set_broadcast_callback(
        self,
        callback: Callable[[Event], None],
    ) -> None:
        """Set callback for broadcasting events to **all** connected clients.

        Used for daemon-wide events that don't belong to a specific
        session — currently the HandoffGate event family
        (``gate.announced`` / ``gate.released`` / ``gates.snapshot``)
        emitted by the jaato-premium reactor framework.  Wired in
        ``__main__.py`` to ``CompositeEventSink.broadcast_event``,
        which fans out across the IPC and WS transports.

        Args:
            callback: Called with (event,) for each broadcast.
        """
        self._broadcast_callback = callback

    def broadcast_event(self, event: Event) -> None:
        """Deliver an event to every connected client across all transports.

        Used by daemon extensions (notably the jaato-premium reactor
        framework's HandoffGate registry) to publish events that aren't
        tied to a specific session.  Per-session events should still go
        through the regular ``_emit_to_client`` / ``_emit_to_session``
        paths.

        No-op if no broadcast callback is wired (e.g. early daemon
        startup before transports are up).  Thread-safe: the underlying
        ``CompositeEventSink.broadcast_event`` snapshots its client
        registries before iterating.

        Args:
            event: The Event to broadcast.
        """
        callback = self._broadcast_callback
        if callback is None:
            logger.debug("broadcast_event: no broadcast callback wired; dropping %s", type(event).__name__)
            return
        callback(event)

    def _emit_to_client(self, client_id: str, event: Event) -> None:
        """Emit an event to a specific client."""
        logger.debug(f"_emit_to_client: {client_id} <- {type(event).__name__}")
        if self._event_callback:
            logger.debug(f"  calling event_callback")
            self._event_callback(client_id, event)
        else:
            logger.warning(f"  NO event_callback set!")

    def _emit_to_session(self, session_id: str, event: Event) -> None:
        """Emit an event to all clients attached to a session."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                # Handle session description updates - update in-memory Session
                if isinstance(event, SessionDescriptionUpdatedEvent):
                    if event.session_id == session_id:
                        session.description = event.description
                        session.is_dirty = True
                        logger.debug(f"Updated session {session_id} description: {event.description}")

                # Handle turn tracking for interrupted tool recovery
                self._handle_turn_tracking_event(session, event)

                for client_id in session.attached_clients:
                    self._emit_to_client(client_id, event)

    # ------------------------------------------------------------------
    # Workspace file monitoring
    # ------------------------------------------------------------------

    def _start_workspace_monitor(
        self,
        session_id: str,
        workspace_path: str,
        server: Optional[JaatoServer] = None,
    ) -> None:
        """Start (or restart) a workspace file monitor for a session.

        If a monitor already exists for this session it is stopped first.

        After starting the monitor, wires up the sandbox manager plugin
        so that readwrite sandbox paths are automatically watched.

        Args:
            session_id: The session to monitor.
            workspace_path: Absolute path to the workspace directory.
            server: The session's JaatoServer.  If not provided, looked up
                from ``self._sessions``.  Pass explicitly when the session
                is not yet registered (e.g., during ``_load_session``).
        """
        self._stop_workspace_monitor(session_id)

        def on_changed(changes: List[Dict[str, str]]) -> None:
            """Callback invoked by the debouncer on the timer thread."""
            # Mark session dirty so the tracked dict gets persisted.
            with self._lock:
                session = self._sessions.get(session_id)
                if session:
                    session.is_dirty = True

            self._emit_to_session(session_id, WorkspaceFilesChangedEvent(
                changes=changes,
            ))

        monitor = WorkspaceMonitor(workspace_path, on_changed=on_changed)
        monitor.start()
        self._workspace_monitors[session_id] = monitor
        logger.debug("Workspace monitor started for session %s", session_id)

        # Wire sandbox path monitoring: when the user runs
        # "sandbox add readwrite /some/path", the workspace monitor should
        # also watch that path for file changes.
        self._wire_sandbox_to_monitor(session_id, monitor, server=server)

    def _stop_workspace_monitor(self, session_id: str) -> None:
        """Stop the workspace monitor for a session if one exists.

        Also clears the sandbox manager callback so it doesn't reference
        a stale monitor.

        Args:
            session_id: The session whose monitor to stop.
        """
        monitor = self._workspace_monitors.pop(session_id, None)
        if monitor:
            monitor.stop()
            logger.debug("Workspace monitor stopped for session %s", session_id)

        # Clear the sandbox manager callback to avoid stale references
        with self._lock:
            session = self._sessions.get(session_id)
        if session and session.server:
            sandbox_plugin = session.server._find_plugin_for_command("sandbox")
            if sandbox_plugin and hasattr(sandbox_plugin, 'set_on_readwrite_paths_changed'):
                sandbox_plugin.set_on_readwrite_paths_changed(None)

    def _wire_sandbox_to_monitor(
        self,
        session_id: str,
        monitor: WorkspaceMonitor,
        server: Optional[JaatoServer] = None,
    ) -> None:
        """Connect the sandbox manager plugin to the workspace monitor.

        When the user adds or removes readwrite sandbox paths, the monitor
        should start or stop watching those directories.  This method:

        1. Finds the sandbox manager plugin in the session's registry.
        2. Seeds the monitor with current readwrite paths.
        3. Registers a callback so future changes are propagated.

        Args:
            session_id: The session whose sandbox plugin to wire.
            monitor: The workspace monitor to update.
            server: The session's JaatoServer.  If not provided, looked up
                from ``self._sessions``.
        """
        if server is None:
            with self._lock:
                session = self._sessions.get(session_id)
            if not session or not session.server:
                return
            server = session.server

        sandbox_plugin = server._find_plugin_for_command("sandbox")
        if not sandbox_plugin or not hasattr(sandbox_plugin, 'set_on_readwrite_paths_changed'):
            return

        # Seed the monitor with current readwrite paths
        if hasattr(sandbox_plugin, 'get_readwrite_paths'):
            current_paths = sandbox_plugin.get_readwrite_paths()
            if current_paths:
                monitor.update_sandbox_paths(current_paths)
                logger.debug(
                    "Seeded workspace monitor with %d sandbox readwrite paths for session %s",
                    len(current_paths), session_id,
                )

        # Register callback for future changes
        def on_readwrite_changed(paths: List[str]) -> None:
            """Called by sandbox manager when readwrite paths change."""
            monitor.update_sandbox_paths(paths)
            logger.debug(
                "Updated workspace monitor sandbox paths: %d paths for session %s",
                len(paths), session_id,
            )

        sandbox_plugin.set_on_readwrite_paths_changed(on_readwrite_changed)

    def _send_workspace_snapshot(self, session_id: str, client_id: str) -> None:
        """Send the full workspace file snapshot to a specific client.

        Used on reconnect / attach so the client can rebuild its mirror.

        Args:
            session_id: Session whose monitor state to send.
            client_id: Target client.
        """
        monitor = self._workspace_monitors.get(session_id)
        if not monitor:
            return

        snapshot = monitor.get_snapshot()
        if snapshot:
            self._emit_to_client(client_id, WorkspaceFilesSnapshotEvent(
                files=snapshot,
                total=monitor.active_file_count,
            ))

    def _apply_client_config(self, client_id: str, event: 'ClientConfigRequest') -> None:
        """Apply client configuration settings.

        Updates environment and plugin settings based on client's config.
        This allows clients to use their own .env settings (like JAATO_TRACE_LOG)
        even when connecting to a shared server.

        Args:
            client_id: The requesting client.
            event: The client config event with settings.
        """
        import os

        # Apply trace log paths if provided
        if event.trace_log_path:
            os.environ['JAATO_TRACE_LOG'] = event.trace_log_path
            logger.info(f"Client {client_id} set JAATO_TRACE_LOG={event.trace_log_path}")

        if event.provider_trace_log:
            os.environ['JAATO_PROVIDER_TRACE'] = event.provider_trace_log
            logger.info(f"Client {client_id} set JAATO_PROVIDER_TRACE={event.provider_trace_log}")

        # Initialize client config dict if needed
        if client_id not in self._client_config:
            self._client_config[client_id] = {}

        # Store presentation context (display capabilities)
        if event.presentation:
            self._client_config[client_id]['presentation'] = event.presentation
            logger.info(f"Client {client_id} set presentation context (client_type={event.presentation.get('client_type', 'unknown')})")

        # Store and apply working directory
        if event.working_dir:
            self._client_config[client_id]['working_dir'] = event.working_dir
            logger.info(f"Client {client_id} set working_dir={event.working_dir}")

        # Store config_root override (read-only framework config root).
        # When unset, the daemon falls back to ``working_dir/.jaato/`` per
        # ``shared.config_resolver.resolve_config_search_path``.
        if event.config_root:
            self._client_config[client_id]['config_root'] = event.config_root
            logger.info(f"Client {client_id} set config_root={event.config_root}")

        # Store client's env_file path for session creation
        if event.env_file:
            self._client_config[client_id]['env_file'] = event.env_file
            logger.info(f"Client {client_id} set env_file={event.env_file}")

        # Store permission timeout override
        if event.permission_timeout is not None:
            self._client_config[client_id]['permission_timeout'] = event.permission_timeout
            logger.info(f"Client {client_id} set permission_timeout={event.permission_timeout}")

        # Store opt-in AppArmor confinement flag.  Default is False —
        # IPC sessions historically run unconfined because the local
        # user already has full filesystem access.  Setting True asks
        # the daemon to provision a per-session AppArmor profile when
        # the client's next session is created (see session-creation
        # hook registered in ``__main__.py``).  When AppArmor is
        # unavailable the session falls back to running unconfined,
        # but the hook emits a ``SystemMessageEvent`` describing the
        # outcome — never a silent fallback.
        if getattr(event, 'apparmor', False):
            self._client_config[client_id]['apparmor'] = True
            logger.info(f"Client {client_id} set apparmor=True")

        # Apply to current session if client is attached to one
        session_id = self._client_to_session.get(client_id)
        if session_id:
            session = self._sessions.get(session_id)
            if session and session.server:
                self._apply_presentation_to_server(event, session.server)
                if event.working_dir:
                    session.server.workspace_path = event.working_dir
                    session.workspace_path = event.working_dir
                if event.config_root:
                    # Mirrors workspace_path propagation: stash on the
                    # JaatoServer so JaatoRuntime / JaatoSession can read
                    # it when threading through discovery sites.
                    session.server.config_root = event.config_root
                    session.config_root = event.config_root
                if event.permission_timeout is not None:
                    self._apply_permission_timeout(session.server, event.permission_timeout)

    def _apply_client_config_to_server(self, client_id: str, server: 'JaatoServer') -> None:
        """Apply stored client configuration to a server.

        Called when a client creates or attaches to a session.

        Args:
            client_id: The client whose config to apply.
            server: The server to configure.
        """
        config = self._client_config.get(client_id, {})
        if 'presentation' in config:
            from jaato_sdk.plugins.model_provider.types import PresentationContext
            ctx = PresentationContext.from_dict(config['presentation'])
            server.set_presentation_context(ctx)
            logger.debug(f"Applied presentation context to server for client {client_id}")
        elif client_id == self._HEADLESS_CLIENT_ID:
            # Reactor / extension-spawned headless sessions never send a
            # ClientConfigRequest, so without this branch their server's
            # presentation_context stays None.  Pre-server-0.6.67 that
            # was effectively API-equivalent (the lifecycle filter
            # exposes signal_completion when pctx is None), but default
            # to ClientType.API explicitly so the contract is declarative
            # and any future client_type-dependent code (telemetry,
            # rendering hints, plugin-level routing) can rely on a
            # known value rather than treating None as "unknown".
            from jaato_sdk.events import ClientType
            from jaato_sdk.plugins.model_provider.types import PresentationContext
            server.set_presentation_context(PresentationContext(
                client_type=ClientType.API,
            ))
            logger.debug(
                "Applied default API presentation context to headless server"
            )
        if 'working_dir' in config:
            server.workspace_path = config['working_dir']
            logger.debug(f"Applied working_dir={config['working_dir']} to server for client {client_id}")
        if 'config_root' in config:
            server.config_root = config['config_root']
            logger.debug(f"Applied config_root={config['config_root']} to server for client {client_id}")
        if 'permission_timeout' in config:
            self._apply_permission_timeout(server, config['permission_timeout'])

    @staticmethod
    def _apply_permission_timeout(server: 'JaatoServer', timeout: int) -> None:
        """Apply a permission timeout override to the session's permission plugin.

        Args:
            server: The server whose permission config to update.
            timeout: Timeout in seconds. 0 means wait forever.
        """
        if server.permission_plugin and hasattr(server.permission_plugin, '_config'):
            config = server.permission_plugin._config
            if config:
                config.channel_timeout = timeout
                logger.debug(f"Applied permission_timeout={timeout} to session {server.session_id}")

    @staticmethod
    def _apply_presentation_to_server(event: 'ClientConfigRequest', server: 'JaatoServer') -> None:
        """Construct and apply PresentationContext from a ClientConfigRequest.

        Args:
            event: The client config event.
            server: The server to configure.
        """
        if event.presentation:
            from jaato_sdk.plugins.model_provider.types import PresentationContext
            ctx = PresentationContext.from_dict(event.presentation)
            server.set_presentation_context(ctx)

    def _handle_turn_tracking_event(self, session: Session, event: Event) -> None:
        """Handle events for turn tracking (interrupted tool recovery).

        Tracks tool execution state so that if the server crashes during tool
        execution, we can recover by injecting synthetic error results.

        Args:
            session: The session being tracked.
            event: The event to process.
        """
        # Track when agent becomes active (turn starts).
        # Compare against the session's main_agent_id (typically "main",
        # but may be the ``--agent <name>`` value when one was supplied
        # at session creation).
        main_agent_id = (
            session.server.main_agent_id if session.server else "main"
        )
        if isinstance(event, AgentStatusChangedEvent):
            if event.status == "active" and event.agent_id == main_agent_id:
                # Main agent starting a turn - initialize tracking
                # Note: We don't have user_prompt here, but we can still track tool calls
                if not session.interrupted_turn:
                    session.interrupted_turn = {
                        "agent_id": event.agent_id,
                        "pending_tool_calls": [],
                        "user_prompt": "",  # Not available at this point
                        "started_at": datetime.now(timezone.utc).isoformat(),
                    }
                    session.is_dirty = True
                    logger.debug(f"Started turn tracking for session {session.session_id}")
            elif event.status == "done":
                # Agent finished - clear tracking
                if session.interrupted_turn:
                    session.interrupted_turn = None
                    session.is_dirty = True
                    logger.debug(f"Cleared turn tracking for session {session.session_id} (agent done)")

                # Re-check deferred unload: if all clients disconnected
                # while the model was running, the session was kept alive.
                # Now that the model is done, unload if still orphaned.
                if not session.attached_clients:
                    self._maybe_unload_session(session.session_id)

        # Track tool calls as they start
        elif isinstance(event, ToolCallStartEvent):
            if session.interrupted_turn and event.agent_id == session.interrupted_turn.get("agent_id"):
                # Add this tool call to pending list
                pending = session.interrupted_turn.get("pending_tool_calls", [])
                pending.append({
                    "id": event.call_id or "",
                    "name": event.tool_name,
                    "args": event.tool_args,
                })
                session.interrupted_turn["pending_tool_calls"] = pending
                session.is_dirty = True
                # Path H (cycle 10): incremental save deferred off
                # the synchronous _emit_to_session path.  Pre-Path-H
                # this called _save_session synchronously, which made
                # 2 blocking runner-RPCs that raced against the
                # runner's active send_message — 35s timeout starved
                # the permission-response window.  Async deferral
                # keeps the recovery contract (pending_tool_calls
                # still persisted) without blocking the emit path.
                self._save_session_async(session)
                logger.debug(
                    f"Updated pending tool calls for session {session.session_id}: "
                    f"{len(pending)} call(s), saving incrementally (async)"
                )

        # Remove completed tool calls from pending list
        elif isinstance(event, ToolCallEndEvent):
            if session.interrupted_turn and event.agent_id == session.interrupted_turn.get("agent_id"):
                pending = session.interrupted_turn.get("pending_tool_calls", [])
                # Remove the completed tool call by matching call_id
                original_count = len(pending)
                pending = [p for p in pending if p.get("id") != event.call_id]
                if len(pending) < original_count:
                    session.interrupted_turn["pending_tool_calls"] = pending
                    session.is_dirty = True
                    logger.debug(
                        f"Tool {event.tool_name} completed, {len(pending)} pending call(s) remain "
                        f"for session {session.session_id}"
                    )

        # Clear tracking when turn completes
        elif isinstance(event, TurnCompletedEvent):
            if session.interrupted_turn:
                session.interrupted_turn = None
                session.is_dirty = True
                logger.debug(f"Cleared turn tracking for session {session.session_id} (turn completed)")

    # =========================================================================
    # Session Lifecycle
    # =========================================================================

    def create_session(self, *args: Any, **kwargs: Any) -> str:
        """Create a new session and attach the client (server 0.6.71+ entry).

        Runs :meth:`_create_session_impl` via
        :func:`shared.session_context.run_in_fresh_session_context` so
        the bootstrap is ISOLATED from any ContextVar values inherited
        from the caller's task.  See the helper's module docstring for
        the full rationale (long-lived-daemon ContextVar leak class
        documented in
        ``project_backlog_workspace_root_contextvar_leak_long_lived_daemon.md``).

        Args:
            (forwarded verbatim to :meth:`_create_session_impl`)

        Returns:
            The new session ID, or empty string on failure.
        """
        from shared.session_context import run_in_fresh_session_context
        return run_in_fresh_session_context(
            self._create_session_impl, *args, **kwargs,
        )

    def _create_session_impl(
        self,
        client_id: str,
        session_name: Optional[str] = None,
        workspace_path: Optional[str] = None,
        env_overrides: Optional[Dict[str, str]] = None,
        profile_name: Optional[str] = None,
        provisioned: bool = False,
        created_by: Optional[str] = None,
        agent_name: Optional[str] = None,
        agent_params: Optional[Dict[str, str]] = None,
        system_instruction_override: Optional[str] = None,
        suppress_base_instructions: bool = False,
        initial_session_state: Optional[Dict[str, Any]] = None,
        inline_profile_data: Optional[Dict[str, Any]] = None,
        config_root: Optional[str] = None,
        apparmor: Optional[bool] = None,
    ) -> str:
        """Implementation of session creation, called via ``Context().run()``.

        See :meth:`create_session` for the isolation rationale and the
        public docstring.

        Args:
            client_id: The requesting client.
            session_name: Optional name (auto-generated if not provided).
            workspace_path: Client's working directory for file operations.
            env_overrides: Optional env vars that override the .env file
                          (e.g., JAATO_PROVIDER/MODEL_NAME from post-auth wizard).
            profile_name: Optional profile name for runtime config (model,
                provider, plugins, GC, env). Loaded from ``.jaato/profiles/``.
            provisioned: True if the workspace was auto-provisioned by the
                server (e.g., for WebSocket clients).  When True, the
                workspace_path is server-managed and should not be overridden
                by client config.
            created_by: Authenticated user who created the session.
            agent_name: Optional agent name. If provided, the agent's rendered
                markdown becomes the session's system instructions. Resolved
                from ``.jaato/agents/`` and ``.jaato/prompts/``.
            agent_params: Parameter values for the agent's ``{{param}}``
                placeholders.
            system_instruction_override: If provided, replaces the assembled
                system instruction passed to the model.  Use the empty string
                to send no system message at all.  Full replacement — the
                agent prompt and plugin instructions are also discarded.
            suppress_base_instructions: Partial suppression — drop only
                the BASE layer (``.jaato/instructions/*.md`` + any
                premium-provided baseline) while keeping the agent prompt,
                plugin instructions, and framework constants.  The usual
                choice for fitting a session into a small model's context
                window (the BASE layer is typically the single biggest
                token consumer).  Ignored when ``system_instruction_override``
                is also set.
            initial_session_state: Optional opaque dict seeded onto the
                new session's session-attached-state container BEFORE
                ``_run_session_hooks`` fires.  Consumer hooks read keys
                via ``session.get_session_state(...)`` to rebuild
                runtime structure (e.g. premium pseudonymization
                instantiates a ``PseudonymTable`` from the encrypted
                blob carried under ``"pseudonym_table"`` and registers
                a provider for it on the new session).  Forking the
                CURRENT state of an existing session is the caller's
                job: snapshot the source via
                ``source.get_all_session_state()`` before calling this
                method (the snapshot invokes registered providers so
                live values are captured, not stale set-state values).
                Values must be JSON-serialisable; encrypt before
                attach if confidentiality is needed (the framework
                treats values as opaque).
            inline_profile_data: Optional dict carrying the same shape
                as a profile JSON file on disk (model, provider,
                plugins, plugin_configs, system_instructions, gc, etc.).
                Lets SDK consumers create sessions with a custom
                runtime config without having to write a profile to
                ``.jaato/profiles/``.  Mutually exclusive with
                ``profile_name``.  The ``model`` key is required (no
                silent fallback) so caller intent is explicit.  The
                helper ``shared.plugins.subagent.config.build_inline_profile``
                does the parsing.

        Returns:
            The session ID (empty string on failure).
        """
        # Generate session ID (matches Session Plugin format)
        timestamp = datetime.now()
        session_id = timestamp.strftime("%Y%m%d_%H%M%S")
        name = session_name or f"Session {timestamp.strftime('%Y-%m-%d %H:%M')}"

        # Check for collision with existing session
        existing = self._get_persisted_sessions(workspace_path=workspace_path)
        existing_ids = {s.session_id for s in existing}
        counter = 0
        original_id = session_id
        while session_id in existing_ids or session_id in self._sessions:
            counter += 1
            session_id = f"{original_id}_{counter}"

        # Get env_file from client config or derive from workspace path
        # Sessions are workspace-bound: the workspace determines the .env file,
        # which in turn determines the provider.
        client_config = self._client_config.get(client_id, {})
        session_env_file = client_config.get('env_file')

        # Inherit config_root from the client's handshake (set via
        # ``ClientConfigRequest.config_root``) when the caller didn't
        # pass one explicitly.  An explicit per-session override (e.g.
        # from a future inline-spec field) takes precedence over the
        # client-level default.
        if config_root is None:
            config_root = client_config.get('config_root')
        import os
        if not session_env_file and workspace_path:
            # Default to workspace/.env
            workspace_env = os.path.join(workspace_path, '.env')
            if os.path.exists(workspace_env):
                session_env_file = workspace_env

        # Headless reactor-spawned sessions never get a client-config
        # entry (their ``client_id`` is the synthetic ``_headless`` id),
        # so the lookup above falls through.  When a ``config_root``
        # was supplied, the project's ``.env`` typically lives **next
        # to** the config_root dir (the orchestrator passes
        # ``<project>/.jaato`` as config_root, so ``<project>/.env``
        # is one level up).  Honoring that convention here means
        # reactor-spawned sessions inherit env-driven config — most
        # importantly the ``JAATO_REDACTION_ENABLED=off`` flag — so
        # PII pseudonymization stays disabled when the workspace owner
        # asked for it to be off.
        if not session_env_file and config_root:
            config_root_parent_env = os.path.join(
                os.path.dirname(os.path.abspath(config_root)), '.env',
            )
            if os.path.exists(config_root_parent_env):
                session_env_file = config_root_parent_env

        logger.info(f"Creating session for client {client_id}: env_file={session_env_file}")
        logger.info(f"  Client config: {client_config}")

        # Resolve agent profile if requested.  The two paths
        # (named-on-disk vs inline-spec from the SDK) are mutually
        # exclusive — a request that supplies both is rejected up
        # front rather than silently picking one.
        profile = None
        if profile_name and inline_profile_data:
            self._emit_to_client(client_id, ErrorEvent(
                error=(
                    "session.new: 'profile' name and inline 'spec' are "
                    "mutually exclusive — pass exactly one"
                ),
                error_type="InvalidSessionSpec",
                recoverable=True,
            ))
            return ""

        if profile_name:
            resolve_path = workspace_path or str(pathlib.Path.home())
            profile, error = self._resolve_profile(
                profile_name, resolve_path,
                config_root=config_root,
                env_file=session_env_file,
            )
            if profile is None:
                self._emit_to_client(client_id, ErrorEvent(
                    error=error,
                    error_type="ProfileNotFoundError",
                    recoverable=True,
                ))
                return ""
            logger.info(f"  Using profile: {profile_name}")
        elif inline_profile_data is not None:
            from shared.plugins.subagent.config import build_inline_profile
            if not inline_profile_data.get("model"):
                self._emit_to_client(client_id, ErrorEvent(
                    error=(
                        "session.new: inline spec requires a 'model' field "
                        "— defaults are not silently applied"
                    ),
                    error_type="InvalidSessionSpec",
                    recoverable=True,
                ))
                return ""
            try:
                profile = build_inline_profile(inline_profile_data)
            except ValueError as exc:
                self._emit_to_client(client_id, ErrorEvent(
                    error=f"session.new: {exc}",
                    error_type="InvalidSessionSpec",
                    recoverable=True,
                ))
                return ""
            logger.info(
                f"  Using inline profile spec (model={profile.model}, "
                f"plugins={profile.plugins})"
            )

        # Resolve agent if requested — the agent's rendered markdown
        # becomes the session's system instructions.
        agent_instructions = None
        if agent_name:
            agent_result = self._resolve_agent(
                agent_name, agent_params, workspace_path, config_root=config_root,
            )
            if agent_result is None:
                self._emit_to_client(client_id, ErrorEvent(
                    error=f"Agent '{agent_name}' not found in .jaato/agents/ or .jaato/prompts/",
                    error_type="AgentNotFoundError",
                    recoverable=True,
                ))
                return ""
            agent_instructions = agent_result["system_instructions"]
            # Use agent's default_profile if no explicit --profile was provided
            if not profile_name and agent_result.get("default_profile"):
                default_prof = agent_result["default_profile"]
                resolve_path = workspace_path or str(pathlib.Path.home())
                profile, error = self._resolve_profile(
                    default_prof, resolve_path,
                    config_root=config_root,
                    env_file=session_env_file,
                )
                if profile:
                    logger.info(f"  Using agent's default profile: {default_prof}")
                else:
                    logger.warning(f"  Agent's default_profile '{default_prof}' not found: {error}")
            if agent_result.get("missing_params"):
                logger.warning(f"  Agent has unresolved params: {agent_result['missing_params']}")
            logger.info(f"  Using agent: {agent_name}")

            # Set agent instructions on the profile (overrides deprecated
            # system_instructions if present).
            if profile:
                if profile.system_instructions:
                    logger.info("  Agent instructions override profile's system_instructions (deprecated)")
                profile.system_instructions = agent_instructions

        # ── Spawn-payload schema validation ──────────────────────────
        # Symmetric to the subagent plugin's check at the function-call
        # boundary: when the resolved profile declares
        # ``spawn_payload_schema``, validate the caller-supplied
        # ``agent_params`` dict against it BEFORE creating the session.
        # Catches missing-required-field bugs from BOTH spawn paths
        # (model-driven spawn_subagent AND reactor-side
        # create_headless_session) at a single chokepoint.
        if (
            profile is not None
            and getattr(profile, 'spawn_payload_schema', None) is not None
        ):
            try:
                from shared.spawn_schema_loader import resolve_spawn_schema
                resolved_schema = resolve_spawn_schema(
                    profile.spawn_payload_schema,
                    workspace_path=workspace_path,
                    config_root=config_root,
                )
                if resolved_schema is not None:
                    import jsonschema
                    try:
                        jsonschema.validate(
                            instance=agent_params or {},
                            schema=resolved_schema,
                        )
                    except jsonschema.ValidationError as exc:
                        required = list(resolved_schema.get('required') or [])
                        missing = [
                            f for f in required
                            if not agent_params or f not in agent_params
                        ]
                        details = (
                            f"missing required fields: {missing}. "
                            if missing
                            else f"first failure: {exc.message}. "
                        )
                        err_msg = (
                            f"create_session(profile={profile_name!r}) failed "
                            f"agent_params validation: {details}"
                            f"The profile requires agent_params matching its "
                            f"spawn_payload_schema "
                            f"({profile.spawn_payload_schema!r})."
                        )
                        logger.error(err_msg)
                        self._emit_to_client(client_id, ErrorEvent(
                            error=err_msg,
                            error_type="SpawnPayloadValidationError",
                            recoverable=True,
                        ))
                        return ""
            except Exception as exc:
                # Schema-loader bug or jsonschema crash — log and skip
                # validation rather than blocking session creation.
                logger.warning(
                    "spawn_payload_schema validation skipped for profile "
                    "%s: %s", profile_name, exc,
                )

        # Create JaatoServer for this session
        # Provider is determined by env_file, with optional overrides.
        # ``agent_name`` propagates to the main agent's ``agent_id`` so
        # reactor rules and event consumers can match on the agent's
        # logical identity (e.g. ``"coordinator"``).  Without this, all
        # AgentCompletedEvents would carry ``agent_id="main"`` regardless
        # of which agent the session was launched with.
        # Resolve effective suppress_base_instructions: explicit kwarg wins
        # if True; otherwise profile-level field controls.  Either source
        # asking for True is sufficient (OR semantics).
        effective_suppress_base = suppress_base_instructions or bool(
            profile and getattr(profile, "suppress_base_instructions", False)
        )

        # Phase 3 §3.12.0: route the construction +
        # pre-init-hooks + initialize + Session-record assembly
        # through the unified _bootstrap_session helper.  All four
        # session-creation paths (IPC, disk-restore, ephemeral, WS)
        # will eventually funnel through here; §3.12.0 ships only
        # the IPC migration as the focal commit.
        envelope = BootstrapEnvelope(
            session_id=session_id,
            workspace_path=workspace_path,
            name=name,
            description=None,
            client_id=client_id,
            env_file=session_env_file,
            profile=profile,
            agent_name=agent_name,
            system_instruction_override=system_instruction_override,
            suppress_base_instructions=effective_suppress_base,
            env_overrides=env_overrides,
            config_root=config_root,
            instruction_token_cache=self._instruction_token_cache,
            # Phase 4 §D: forward the originating create_session
            # ``agent_params`` so build_session_envelope can put them
            # on the wire envelope.  Without this, runner-side prefetch
            # scripts (e.g. tmux_pane in the documenter harness) see
            # empty agent_params and emit their "missing keys" error.
            agent_params=dict(agent_params or {}),
            provisioned=provisioned,
            created_by=created_by,
            timestamp=timestamp,
            # PR-A (2026-05-14): forward explicit AppArmor caller intent
            # to the bootstrap helper.  None = no override (consult
            # client_config then profile.apparmor); True/False short-
            # circuits that chain.
            apparmor=apparmor,
            # During init, emit directly to requesting client (not
            # yet attached to session).
            on_event_during_init=lambda e: self._emit_to_client(client_id, e),
        )
        server, session = self._bootstrap_session(envelope)
        if server is None or session is None:
            # server.initialize() failed; core.py already emitted a
            # detailed ConfigurationError to the in-init sink.
            return ""

        logger.info(f"Server initialized successfully for session {session_id}")

        # Wire SessionManager into the session_ops plugin so its
        # interrogate_session tool can fork_ask other daemon sessions.
        if server.registry:
            session_ops_plugin = server.registry.get_plugin("session_ops")
            if session_ops_plugin and hasattr(session_ops_plugin, 'set_session_manager'):
                session_ops_plugin.set_session_manager(self)

        # Switch to session-based event emission now that init is complete
        server.set_event_callback(lambda e: self._emit_to_session(session_id, e))

        # Configure TODO plugin with session-scoped storage
        if workspace_path:
            session_dir = self._session_storage_dir(workspace_path) / session_id
            self._configure_todo_storage(server, session_dir)

        # Apply client-specific config (e.g., presentation context)
        self._apply_client_config_to_server(client_id, server)

        # Register callback for when auth completes (if it was pending)
        def on_auth_complete():
            self._emit_to_session(session_id, self._build_session_info_event(session))
            self._emit_to_session(session_id, SystemMessageEvent(
                message=f"Session created: {name} ({session_id})",
                style="info",
            ))
        server.set_auth_complete_callback(on_auth_complete)

        with self._lock:
            self._sessions[session_id] = session
            session.attached_clients.add(client_id)
            self._client_to_session[client_id] = session_id

        # Seed session-attached state BEFORE hooks fire.  Consumer
        # hooks (e.g. premium pseudonymization) read these keys via
        # session.get_session_state(...) to rebuild runtime structure
        # and register providers for incrementally-mutated state.  Any
        # JSON-serialisability error surfaces here at the call site
        # (set_session_state validates the value at attach time).
        # Phase 3 §7c step 6.6.3.6: forward to runner-side via
        # the existing ``session.set_session_state`` RPC (§3.3c
        # precursor) instead of reaching into the daemon-side
        # session via ``server.get_session()``.
        if initial_session_state:
            rpc = getattr(server, "_runner_rpc", None)
            if rpc is not None:
                forwarder = getattr(
                    rpc, "session_set_state_threadsafe", None,
                )
                if callable(forwarder):
                    for key, value in initial_session_state.items():
                        try:
                            forwarder(key, value, timeout=2.0)
                        except Exception as exc:  # noqa: BLE001
                            logger.debug(
                                "set_session_state forward failed for "
                                "key=%r: %s", key, exc,
                            )

        # Run session hooks after the Session is stored so hooks can
        # call get_session() to modify session attributes (e.g. sandbox_mode).
        self._run_session_hooks(server, session_id)

        # Start workspace file monitor
        if workspace_path:
            self._start_workspace_monitor(session_id, workspace_path)

        # Save initial state to disk
        self._save_session(session)

        logger.info(f"Session created: {session_id} ({name})")

        # Note: We don't call emit_current_state() here because the client
        # already received all events during initialize() via direct emission.

        # Send SessionInfoEvent to confirm session creation.  When auth is
        # pending the tool/model lists may be incomplete, but the client
        # needs the session_id immediately.  on_auth_complete() will send
        # an updated SessionInfoEvent once the provider is fully ready.
        try:
            self._emit_to_client(client_id, self._build_session_info_event(session))
        except Exception as exc:
            logger.error("Failed to build SessionInfoEvent: %s", exc, exc_info=True)
            # Send a minimal SessionInfoEvent so the client can still proceed
            self._emit_to_client(client_id, SessionInfoEvent(
                session_id=session.session_id,
                session_name=session.name,
                model_provider=session.server.model_provider if session.server else "",
                model_name=session.server.model_name if session.server else "",
            ))

        if not server.auth_pending:
            self._emit_to_client(client_id, SystemMessageEvent(
                message=f"Session created: {name} ({session_id})",
                style="info",
            ))

        return session_id

    # ------------------------------------------------------------------
    # Headless session creation (for daemon extensions / reactors)
    # ------------------------------------------------------------------

    _HEADLESS_CLIENT_ID = "_headless"

    def create_headless_session(
        self,
        profile_name: Optional[str] = None,
        agent_name: Optional[str] = None,
        workspace_path: Optional[str] = None,
        initial_prompt: Optional[str] = None,
        initial_history: Optional[List[Any]] = None,
        initial_session_state: Optional[Dict[str, Any]] = None,
        session_name: Optional[str] = None,
        config_root: Optional[str] = None,
        agent_params: Optional[Dict[str, str]] = None,
        apparmor: Optional[bool] = None,
    ) -> str:
        """Create a top-level session not attached to any real client.

        Intended for daemon extensions (e.g., reactor rules) that need to
        spawn an independent session in response to an event.  The session
        is fully initialized and runs like any other, but its client-facing
        events are silently dropped — the transport layer ignores events
        addressed to the synthetic ``_headless`` client (IPC: unknown
        client → silent return; WS: same).

        The session **is** visible on the EventBus, so any reactor
        subscriptions (via the session hook) observe it normally.

        Args:
            profile_name: Optional profile to use (resolved from
                ``.jaato/profiles/``).  Controls model, plugins, GC.
            agent_name: Optional agent to use (resolved from
                ``.jaato/agents/``).  The agent's markdown becomes the
                session's system instructions.  This is typically the
                mandatory parameter — it defines *what* the session does.
            workspace_path: Workspace directory.  Defaults to the daemon's
                cwd if not provided.
            initial_prompt: If set, a ``SendMessageRequest`` is dispatched
                to the new session immediately after creation.
            initial_history: Optional list of ``Message`` objects to
                seed the new session's conversation history with.  Used
                by spawn-from-snapshot callers (premium handoff via
                ``fork_session_from_history``, waypoint fork-to-session)
                so the new agent picks up where another session left off.
                Loaded after ``server.initialize()`` succeeds and before
                any ``initial_prompt`` is dispatched.  Typed as ``Any``
                here to avoid a top-level SDK import; concrete type is
                ``List[jaato_sdk.plugins.model_provider.types.Message]``.
            initial_session_state: Optional opaque dict seeded onto the
                new session's session-attached-state container BEFORE
                its session-hook fires.  Threaded through to
                :meth:`create_session`; see that method's docstring for
                semantics and the fork-carry contract.  Forking the
                CURRENT state of an existing session is the caller's
                job — snapshot the source via
                ``source.get_all_session_state()`` first so the dict
                reflects live values (registered providers are
                invoked), not stale set-state.
            session_name: Optional human-readable name.

        Returns:
            The session ID (empty string on failure).
        """
        session_id = self.create_session(
            client_id=self._HEADLESS_CLIENT_ID,
            session_name=session_name,
            workspace_path=workspace_path,
            profile_name=profile_name,
            agent_name=agent_name,
            agent_params=agent_params,
            initial_session_state=initial_session_state,
            config_root=config_root,
            apparmor=apparmor,
        )
        if not session_id:
            # Server 0.6.50.1+: log at WARNING so reactor callers
            # (``ctx.create_session`` → ``create_headless_session``) get
            # a diagnostic trail when their headless spawn silently
            # fails.  ErrorEvents fired during create_session go to the
            # ``_HEADLESS_CLIENT_ID`` synthetic client which the
            # transport drops — without this log line, the reactor
            # caller sees only an empty session_id with no clue why
            # (cascade v8 finding from 7:3, 2026-05-06: a buggy
            # prefetch raised DynamicInstructionsError, the underlying
            # core.py:initialize correctly aborted, but the empty
            # session_id propagated up to the reactor handler without
            # any visible log entry).  The headless caller still has
            # to handle ``""`` as failure, but at least now there's a
            # log breadcrumb explaining WHY.
            logger.warning(
                "create_headless_session: session creation returned "
                "empty (profile=%s, agent=%s, workspace=%s).  Check "
                "earlier ERROR logs in this turn for the underlying "
                "cause (typically: prefetch failed, auth missing, "
                "provider connect failed).",
                profile_name, agent_name, workspace_path,
            )
            return ""

        if initial_history:
            # Seed history before any model turn fires.  Look up the
            # session record we just created and call the JaatoSession's
            # public initial-history primitive.  Any failure is
            # surfaced — partial state (session created, history not
            # loaded) would silently mislead the caller.
            with self._lock:
                session_record = self._sessions.get(session_id)
            if session_record is None:
                logger.error(
                    "create_headless_session: session %s vanished after "
                    "create; cannot seed initial_history",
                    session_id,
                )
                return ""
            # Phase 3 §7c step 6.6.3.6: forward to runner-side
            # via the new ``session.set_initial_history`` RPC
            # (§7c step 6.6.1.1, commit 3f859e3a) instead of
            # reaching into the daemon-side session.
            rpc = getattr(session_record.server, "_runner_rpc", None)
            if rpc is not None:
                forwarder = getattr(
                    rpc, "session_set_initial_history_threadsafe", None,
                )
                if callable(forwarder):
                    forwarder(initial_history, timeout=10.0)

        if initial_prompt:
            from jaato_sdk.events import SendMessageRequest
            self.handle_request(
                self._HEADLESS_CLIENT_ID,
                session_id,
                SendMessageRequest(text=initial_prompt),
            )

        return session_id

    def inject_prompt_to_session(
        self,
        target_session_id: str,
        text: str,
        source_id: Optional[str] = None,
        source_type: Optional[Any] = None,
    ) -> bool:
        """Deliver a prompt to a loaded session by ID.

        Routing primitive used by daemon extensions (reactor rules,
        webhook handlers, peer-clustering message routers) that need
        to deliver a prompt to a session other than the one whose
        event triggered them.  Generalises ``JaatoSession.inject_prompt``
        from "self-targeting" to "addressable by ID".

        Thread-safe.  ``inject_prompt`` itself queues mid-turn for
        busy sessions, so the caller doesn't need to check session
        state — this method only fails when the target isn't loaded.

        Args:
            target_session_id: The destination session.  Must be loaded
                in ``self._sessions`` (not just persisted on disk —
                attach first if needed).
            text: The prompt to inject.
            source_id: Identifier of the sender (e.g. ``"reactor"``,
                ``"webhook:github"``).  Defaults to ``"unknown"``
                downstream.
            source_type: ``SourceType`` enum value controlling priority
                (USER/PARENT/SYSTEM/EVENT/CHILD).  Defaults to USER
                downstream.  Typed as ``Any`` here to avoid a top-level
                import of the SDK enum.

        Returns:
            ``True`` if the prompt was delivered (queued or fired
            immediately), ``False`` if the session isn't loaded or
            doesn't yet have an active ``JaatoSession``.
        """
        with self._lock:
            session = self._sessions.get(target_session_id)
        if session is None:
            return False
        # Phase 3 §7c step 6.6.3.6: forward to runner-side via the
        # existing ``session.inject_prompt`` RPC (§7c step 6.1
        # (3/3) at commit 14e57709).  ``SourceType`` enum
        # serialized as its lowercase string value across the
        # wire.
        rpc = getattr(session.server, "_runner_rpc", None)
        if rpc is None:
            return False
        forwarder = getattr(rpc, "session_inject_prompt_threadsafe", None)
        if not callable(forwarder):
            return False
        try:
            forwarder(
                text,
                source_id=source_id,
                source_type=(
                    source_type.value if source_type is not None else None
                ),
                timeout=2.0,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("inject_prompt forward failed: %s", exc)
            return False
        return True

    def get_session_workspace(self, session_id: str) -> Optional[str]:
        """Get the workspace path of a session.

        Args:
            session_id: The session ID.

        Returns:
            The session's workspace path, or None if session not found.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                return session.workspace_path
        return None

    def check_workspace_mismatch(
        self,
        session_id: str,
        client_workspace: Optional[str],
    ) -> Optional[tuple]:
        """Check if there's a workspace mismatch between client and session.

        Args:
            session_id: The session to check.
            client_workspace: The client's workspace path.

        Returns:
            Tuple of (session_workspace, client_workspace) if there's a mismatch,
            None if no mismatch or session not found.
        """
        session_workspace: Optional[str] = None

        with self._lock:
            # First check in-memory sessions
            session = self._sessions.get(session_id)
            if session:
                session_workspace = session.workspace_path
            else:
                # Check persisted sessions on disk — try client's workspace first,
                # since that's the most likely location for the session file
                persisted = self._get_persisted_sessions(workspace_path=client_workspace)
                for s in persisted:
                    if s.session_id == session_id:
                        session_workspace = s.workspace_path
                        break

        if not session_workspace or not client_workspace:
            # No mismatch if either is not set
            return None

        # Use helper method to compare workspaces
        if not self._workspaces_match(session_workspace, client_workspace):
            return (session_workspace, client_workspace)

        return None

    def _count_pending_held_tool_calls(self, session: Session) -> int:
        """Phase 3 §3.12 + peer-review M5/N1: return the count of
        tool calls held by the runner-side permission plugin's
        defer-and-flush queue for *session*.

        The queue itself lives runner-side and is populated when a
        ``check_permission`` ASK lands while
        ``Session.restored_pending_attach`` is True (no client
        attached).  This commit lays the daemon-side foundation —
        the actual queue + drain logic ships in a §3.12 follow-on.
        For now the count is always 0; the helper exists so the
        ``SessionRestoredEvent`` field is wired end-to-end and the
        client side has a stable contract to integrate against.
        """
        # TODO §3.12 follow-on: query the runner-side permission
        # plugin via runner-RPC for the pending-call count.  Until
        # then, return 0.
        return 0

    def attach_session(
        self,
        client_id: str,
        session_id: str,
        workspace_path: Optional[str] = None,
    ) -> bool:
        """Attach a client to an existing session.

        If the session is not in memory, attempts to load from disk.

        Args:
            client_id: The requesting client.
            session_id: The session to attach to.
            workspace_path: Client's working directory for file operations.

        Returns:
            True if attached successfully.
        """
        # Track if session was already in memory (client missed init events)
        session_was_in_memory = False

        with self._lock:
            # Check if session is in memory
            session = self._sessions.get(session_id)
            session_was_in_memory = session is not None

            if not session:
                # Try to load from disk (pass client_id for init progress events)
                logger.debug(f"attach_session: session {session_id} not in memory, loading from disk...")
                try:
                    session = self._load_session(session_id, client_id=client_id, workspace_path=workspace_path)
                    logger.debug(f"attach_session: _load_session returned {session is not None}")
                except Exception as e:
                    logger.error(f"attach_session: _load_session raised: {type(e).__name__}: {e}")
                    import traceback
                    logger.error(f"attach_session: traceback:\n{traceback.format_exc()}")
                    session = None
                if session:
                    self._sessions[session_id] = session

            if not session:
                self._emit_to_client(client_id, ErrorEvent(
                    error=f"Session not found: {session_id}",
                    error_type="SessionError",
                ))
                return False

            # Detach from current session if any
            current = self._client_to_session.get(client_id)
            if current and current in self._sessions:
                old_session = self._sessions[current]
                old_session.attached_clients.discard(client_id)
                # Consider unloading if no clients
                self._maybe_unload_session(current)

            # Attach to new session
            #
            # Phase 3 §3.12 + peer-review M5/N1: detect first-attach
            # to a disk-restored session BEFORE adding the client to
            # ``attached_clients`` so the "no client previously
            # attached" precondition is unambiguous.  The actual
            # SessionRestoredEvent emission happens after the lock
            # is released (event sinks should not run under the
            # SessionManager lock).
            was_first_attach_to_restored = (
                session.restored_pending_attach
                and not session.attached_clients
            )
            session.attached_clients.add(client_id)
            self._client_to_session[client_id] = session_id

            if was_first_attach_to_restored:
                # Clear the flag now under the lock so a parallel
                # ``check_permission`` ASK observes the new state
                # (defer-and-flush off, normal denial / prompt
                # behaviour resumes).  The pending-tool-call queue
                # drain happens in the runner-side permission plugin
                # in a §3.12 follow-on commit; this commit lays the
                # foundation by surfacing the count of held calls
                # via the SessionRestoredEvent emitted below.
                session.restored_pending_attach = False

            # Only set workspace if session doesn't have one yet.
            # If session already has a workspace, it keeps it - clients are warned
            # about workspace mismatches before attach via check_workspace_mismatch().
            if workspace_path and not session.workspace_path:
                session.workspace_path = workspace_path
                session.server.workspace_path = workspace_path

        logger.info(f"Client {client_id} attached to session {session_id}")

        # M5/N1: emit SessionRestoredEvent for the first attach to a
        # disk-restored session.  The pending-tool-call count is 0
        # for this commit (the queue infrastructure lives in the
        # runner-side permission plugin and lands in a §3.12 follow-
        # on); the event still fires so the client can distinguish a
        # fresh-attach from a restored-attach for telemetry / UX.
        if was_first_attach_to_restored:
            self._emit_to_client(client_id, SessionRestoredEvent(
                session_id=session_id,
                pending_tool_call_count=self._count_pending_held_tool_calls(
                    session,
                ),
            ))

        # Apply client-specific config (e.g., presentation context)
        self._apply_client_config_to_server(client_id, session.server)

        # Only emit current state if session was already in memory.
        # If we just loaded it from disk, the client received all events during init.
        if session_was_in_memory:
            session.server.emit_current_state(
                lambda e: self._emit_to_client(client_id, e),
                skip_session_info=True
            )
        else:
            # Session was loaded from disk - clear any stale pending requests
            # the client might have from before the session was saved/restored
            session.server.emit_current_state(
                lambda e: self._emit_to_client(client_id, e),
                skip_session_info=True,
                clear_stale_pending_requests=True
            )

        # Send complete SessionInfoEvent with state snapshot
        self._emit_to_client(client_id, self._build_session_info_event(session))

        # Send workspace files snapshot so client can rebuild its mirror
        self._send_workspace_snapshot(session_id, client_id)

        # Build attach message with description if available
        desc_part = f" - {session.description}" if session.description else ""
        self._emit_to_client(client_id, SystemMessageEvent(
            message=f"Attached to session: {session_id}{desc_part}",
            style="info",
        ))

        return True

    def _load_session(
        self,
        session_id: str,
        client_id: Optional[str] = None,
        workspace_path: Optional[str] = None,
    ) -> Optional[Session]:
        """Load a session from disk (server 0.6.71+ entry).

        Wraps :meth:`_load_session_impl` in a fresh ContextVar context
        via :func:`shared.session_context.run_in_fresh_session_context`
        so the bootstrap is isolated from any ContextVar values
        inherited from the caller's task.  See the helper's docstring
        for the rationale.
        """
        from shared.session_context import run_in_fresh_session_context
        return run_in_fresh_session_context(
            self._load_session_impl, session_id, client_id, workspace_path,
        )

    def _load_session_impl(
        self,
        session_id: str,
        client_id: Optional[str] = None,
        workspace_path: Optional[str] = None,
    ) -> Optional[Session]:
        """Implementation of session loading, called via fresh-context wrap.

        See :meth:`_load_session` for the isolation rationale.

        Args:
            session_id: The session ID to load.
            client_id: Optional client ID to receive init progress events.
            workspace_path: Workspace directory for resolving the session
                storage path. Required so we know which workspace's
                ``.jaato/sessions/`` to look in.

        Returns:
            The loaded Session, or None if not found.
        """
        logger.debug(f"_load_session: attempting to load {session_id}")

        # Resolve storage directory from workspace
        storage_dir = self._session_storage_dir(workspace_path) if workspace_path else None

        try:
            state = self._session_plugin.load(session_id, storage_dir=storage_dir)
            logger.debug(f"_load_session: loaded state for {session_id}")
        except FileNotFoundError:
            logger.debug(f"_load_session: session {session_id} not found on disk")
            return None
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return None

        # Create JaatoServer and restore state
        logger.debug(f"_load_session: creating JaatoServer for {session_id}...")

        # During init, emit directly to requesting client if provided
        if client_id:
            init_callback = lambda e: self._emit_to_client(client_id, e)
        else:
            init_callback = lambda e: self._emit_to_session(session_id, e)

        # Determine which env_file to use for this session:
        # 1. If client_id is provided, use client's env_file from their config
        # 2. If session has workspace_path, try workspace/.env
        # Sessions are workspace-bound: the workspace determines the .env file,
        # which in turn determines the provider.
        session_env_file = None
        if client_id:
            client_config = self._client_config.get(client_id, {})
            if client_config.get('env_file'):
                session_env_file = client_config['env_file']
                logger.debug(f"_load_session: using client's env_file: {session_env_file}")
        if not session_env_file and state.workspace_path:
            import os
            workspace_env = os.path.join(state.workspace_path, '.env')
            if os.path.exists(workspace_env):
                session_env_file = workspace_env
                logger.debug(f"_load_session: using workspace env_file: {session_env_file}")

        # Phase 3 §3.12 disk-restore migration: route the JaatoServer
        # construction + pre-init hooks + initialize through the
        # unified ``_construct_and_initialize_server`` sub-helper that
        # the IPC path also uses.  The disk-restore path supplies the
        # saved ``state.sandbox_mode`` via ``envelope.sandbox_mode``
        # so the Session record below reflects the pre-restart value
        # rather than re-running the apparmor opt-in lookup (which
        # is a client-driven concept that doesn't apply to disk
        # restore).
        envelope = BootstrapEnvelope(
            session_id=session_id,
            workspace_path=state.workspace_path,
            name=state.description or f"Session {session_id}",
            description=state.description,
            client_id=None,  # disk-restore path; no client-driven opt-in
            sandbox_mode=getattr(state, "sandbox_mode", None),
            restore_state={"loaded_state": state},
            env_file=session_env_file,
            instruction_token_cache=self._instruction_token_cache,
            on_event_during_init=init_callback,
        )
        try:
            server, _restore_sandbox = self._construct_and_initialize_server(envelope)
        except Exception as e:
            logger.error(
                f"_load_session: initialize() raised exception: "
                f"{type(e).__name__}: {e}"
            )
            import traceback
            logger.error(f"_load_session: traceback:\n{traceback.format_exc()}")
            return None
        if server is None:
            logger.error(f"Failed to initialize server for session {session_id}")
            return None
        logger.debug(f"_load_session: initialize() returned True")

        logger.debug(f"_load_session: server initialized for {session_id}")

        # Switch to session-based event emission now that init is complete
        server.set_event_callback(lambda e: self._emit_to_session(session_id, e))

        # Configure TODO plugin with session-scoped storage
        effective_workspace = workspace_path or state.workspace_path
        if effective_workspace:
            session_dir = self._session_storage_dir(effective_workspace) / session_id
        else:
            session_dir = pathlib.Path(self._session_config.storage_path) / session_id
        self._configure_todo_storage(server, session_dir)

        # Restore history to the runner-side session.
        # Phase 3 §7c step 6.6.4.5b: route through the
        # ``session.set_initial_history`` RPC (added §7c step 6.6.1.1)
        # instead of ``server._jaato.reset_session(state.history)``.
        # Semantically equivalent at this site: the runner-side session
        # was just bootstrapped with empty history, which matches
        # ``set_initial_history``'s "session must be idle and history
        # must be empty" precondition.
        if state.history and server._runner_rpc is not None:
            server._runner_rpc.session_set_initial_history_threadsafe(state.history)
            logger.debug(f"Restored {len(state.history)} messages for session {session_id}")

            # Resolve the session's main agent id once — it may be the
            # default ``"main"`` or the ``--agent <name>`` value used at
            # session creation.  Used as the ``_agents`` dict key and the
            # emitted ``agent_id`` so consumers see consistent identity.
            main_agent_id = server.main_agent_id

            # Also populate AgentState.history so emit_current_state() can
            # replay conversation content for reconnecting clients.
            # on_agent_history_updated is only called during send_message(),
            # so we must set it explicitly after disk load.
            if main_agent_id in server._agents:
                server._agents[main_agent_id].history = list(state.history)

            # Restore turn accounting (reset_session clears it, so we restore after).
            # Phase 3 §7c step 6.6.3.6: forward to runner-side
            # via the new ``session.restore_turn_accounting``
            # RPC (§7c step 6.6.1.2 at commit 82b8da29) instead
            # of reaching into the daemon-side session.
            if state.turn_accounting:
                rpc = getattr(server, "_runner_rpc", None)
                if rpc is not None:
                    forwarder = getattr(
                        rpc, "session_restore_turn_accounting_threadsafe", None,
                    )
                    if callable(forwarder):
                        try:
                            forwarder(state.turn_accounting, timeout=5.0)
                            logger.debug(f"Restored {len(state.turn_accounting)} turn accounting entries for session {session_id}")
                        except Exception as exc:  # noqa: BLE001
                            logger.debug(
                                "restore_turn_accounting forward failed: %s",
                                exc,
                            )

                # Update server's agent state and emit context update.
                # Phase 3 §7c step 6.6.4.5b: route ``get_context_usage``
                # through the runner-RPC instead of the daemon-side
                # JaatoClient indirection.
                if main_agent_id in server._agents:
                    main_state = server._agents[main_agent_id]
                    main_state.turn_accounting = list(state.turn_accounting)
                    usage = server._runner_rpc.session_get_context_usage_threadsafe()
                    main_state.context_usage = {
                        'total_tokens': usage.get('total_tokens', 0),
                        'prompt_tokens': usage.get('prompt_tokens', 0),
                        'output_tokens': usage.get('output_tokens', 0),
                        'percent_used': usage.get('percent_used', 0.0),
                    }
                    # Emit context update so clients show correct usage
                    server.emit(ContextUpdatedEvent(
                        agent_id=main_agent_id,
                        usage=server._build_usage(
                            prompt_tokens=usage.get('prompt_tokens', 0),
                            output_tokens=usage.get('output_tokens', 0),
                            total_tokens=usage.get('total_tokens', 0),
                        ),
                        context_limit=usage.get('context_limit', 0),
                        percent_used=usage.get('percent_used', 0.0),
                        tokens_remaining=usage.get('tokens_remaining', 0),
                        turns=usage.get('turns', 0),
                    ))
                    logger.debug(f"Emitted ContextUpdatedEvent: {usage.get('percent_used', 0.0):.1f}% used")

        # Restore conversation budget if present (other budget sources are
        # automatically populated during session recreation).
        # Phase 3 §7c step 6.6.1.0: use the public
        # JaatoSession.restore_conversation_budget() method instead
        # of reaching through ``session.instruction_budget`` into
        # the underlying InstructionBudget's
        # ``restore_conversation_from_snapshot``.  The public
        # surface is the prerequisite for the upcoming
        # ``session.restore_conversation_budget`` runner-RPC
        # handler (§7c step 6.6.1.3).  The method is no-op when
        # the session's instruction_budget is None, so we drop
        # the explicit guard.
        # Phase 3 §7c step 6.6.3.6: forward to runner-side via the
        # new ``session.restore_conversation_budget`` RPC (§7c step
        # 6.6.1.3 at commit b40d2439); use the existing
        # ``session.snapshot_instruction_budget`` (§7c step 6.1
        # (2/3) at commit 1043bfde) for the post-restore emit.
        if state.budget_state:
            rpc = getattr(server, "_runner_rpc", None)
            if rpc is not None:
                restorer = getattr(
                    rpc, "session_restore_conversation_budget_threadsafe", None,
                )
                snapshotter = getattr(
                    rpc, "session_snapshot_instruction_budget_threadsafe", None,
                )
                if callable(restorer):
                    try:
                        restorer(state.budget_state, timeout=5.0)
                        logger.debug(
                            f"Restored conversation budget for session {session_id}",
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.debug(
                            "restore_conversation_budget forward failed: %s",
                            exc,
                        )
                if callable(snapshotter):
                    try:
                        snapshot = snapshotter(timeout=5.0)
                    except Exception as exc:  # noqa: BLE001
                        logger.debug(
                            "snapshot_instruction_budget forward failed: %s",
                            exc,
                        )
                        snapshot = None
                    if snapshot is not None:
                        # Emit budget event so clients show correct budget.
                        # ``agent_id`` is a top-level key in the snapshot
                        # dict (per InstructionBudget.snapshot() schema).
                        server.emit(InstructionBudgetEvent(
                            agent_id=snapshot.get('agent_id', server.main_agent_id),
                            budget_snapshot=snapshot,
                        ))

        # Restore subagent state if present in metadata.
        # Phase 3 §7c step 6.6.4.5e: truthiness check pivoted from
        # ``server._jaato`` (deleted) to ``server._runtime`` (the
        # daemon-side canonical handle since 5d).
        if state.metadata.get('subagents') and server._runtime is not None:
            self._restore_subagent_states(
                session_id,
                state.metadata['subagents'],
                server,
                workspace_path=effective_workspace,
            )

        # Restore TODO plugin state (agent-plan mapping, blocked steps)
        self._load_todo_state(server, session_dir)

        # Generic plugin state restoration: iterate plugin_states saved by
        # the generic persistence loop and call restore_persistence_state()
        # on each plugin that implements it.
        if state.metadata.get('plugin_states') and server.registry:
            for plugin_name, plugin_state in state.metadata['plugin_states'].items():
                plugin = server.registry.get_plugin(plugin_name)
                if plugin and hasattr(plugin, 'restore_persistence_state'):
                    try:
                        plugin.restore_persistence_state(plugin_state)
                        logger.debug(f"Restored persistence state for plugin: {plugin_name}")
                    except Exception as e:
                        logger.warning(
                            f"Failed to restore persistence state for plugin "
                            f"'{plugin_name}': {e}"
                        )

        # Check for and recover from interrupted turn
        recovered_count = 0
        if state.interrupted_turn:
            recovered_count = self._recover_interrupted_turn(
                session_id,
                state.interrupted_turn,
                server
            )
            if recovered_count > 0:
                logger.info(f"Recovered {recovered_count} interrupted tool calls for session {session_id}")

        session = Session(
            session_id=session_id,
            name=state.description or f"Session {session_id}",
            server=server,
            created_at=state.created_at.isoformat(),
            last_activity=state.updated_at.isoformat(),
            description=state.description,
            is_dirty=recovered_count > 0,  # Mark dirty if recovery happened
            workspace_path=state.workspace_path,
            user_inputs=state.user_inputs or [],  # Command history for prompt restoration
            provisioned=state.metadata.get('provisioned', False),
            sandbox_mode=getattr(state, "sandbox_mode", None),
            # Phase 3 §3.12 + peer-review M5/N1: mark this session as
            # awaiting first client-attach.  While set, the runner-
            # side permission plugin queues ASK prompts rather than
            # denying them (defer-and-flush posture).  Cleared in
            # ``attach_session`` after emitting SessionRestoredEvent.
            restored_pending_attach=True,
        )

        # Restore workspace file monitor with persisted tracked state
        if state.workspace_path:
            self._start_workspace_monitor(session_id, state.workspace_path, server=server)
            monitor = self._workspace_monitors.get(session_id)
            if monitor and state.workspace_files:
                monitor.restore(state.workspace_files)
                # Reconcile: detect changes that happened while server was down
                reconcile_changes = monitor.reconcile()
                if reconcile_changes:
                    session.is_dirty = True
                    logger.info(
                        "Workspace reconciliation found %d changes for session %s",
                        len(reconcile_changes),
                        session_id,
                    )

        # Store session before running hooks so hooks can call get_session().
        with self._lock:
            self._sessions[session_id] = session
        self._run_session_hooks(server, session_id)

        logger.info(f"Loaded session from disk: {session_id}")
        return session

    def _restore_subagent_states(
        self,
        session_id: str,
        subagent_registry: Dict[str, Any],
        server: JaatoServer,
        workspace_path: Optional[str] = None,
    ) -> int:
        """Restore subagent states from persisted data.

        Args:
            session_id: The parent session ID.
            subagent_registry: Registry dict from state.metadata["subagents"].
            server: The JaatoServer to restore subagents into.
            workspace_path: Workspace directory for resolving storage path.

        Returns:
            Number of subagents successfully restored.
        """
        if not server.registry:
            logger.warning("Cannot restore subagents: no registry available")
            return 0

        subagent_plugin = server.registry.get_plugin("subagent")
        if not subagent_plugin or not hasattr(subagent_plugin, 'restore_persistence_state'):
            logger.warning("Cannot restore subagents: subagent plugin not available")
            return 0

        # Load per-agent state files
        if workspace_path:
            subagents_dir = self._session_storage_dir(workspace_path) / session_id / "subagents"
        else:
            subagents_dir = pathlib.Path(
                self._session_config.storage_path
            ) / session_id / "subagents"

        agent_states: Dict[str, Dict[str, Any]] = {}
        if subagents_dir.exists():
            for agent_file in subagents_dir.glob("*.json"):
                agent_id = agent_file.stem
                try:
                    with open(agent_file, 'r', encoding='utf-8') as f:
                        agent_states[agent_id] = json.load(f)
                    logger.debug(f"Loaded subagent state file: {agent_file}")
                except Exception as e:
                    logger.error(f"Failed to load subagent state {agent_file}: {e}")

        # Phase 3 §7c step 6.6.4.5a: read ``server._runtime`` directly
        # instead of going through ``server._jaato.get_runtime()``.
        # ``self._runtime`` has been the daemon-side runtime field since
        # §7c step 4 first pass (commit 7c34f218); this site completes
        # the pattern.  Behavior-preserving: ``_runtime`` is non-None
        # iff ``_jaato`` was successfully connected.
        runtime = server._runtime
        if not runtime:
            logger.warning("Cannot restore subagents: no runtime available")
            return 0

        # Restore subagents
        restored = subagent_plugin.restore_persistence_state(
            subagent_registry,
            agent_states,
            runtime
        )

        # Emit AgentCreatedEvent for each restored subagent so clients see them
        for agent_id, info in subagent_plugin._active_sessions.items():
            profile = info.get('profile')
            created_at = info.get('created_at')
            if isinstance(created_at, datetime):
                created_at = created_at.isoformat()

            server.emit(AgentCreatedEvent(
                agent_id=agent_id,
                agent_name=profile.name if profile else agent_id,
                agent_type="subagent",
                profile_name=profile.name if profile else "",
                parent_agent_id=server.main_agent_id,
                created_at=created_at,
            ))

            # Emit context update for restored subagent
            session = info.get('session')
            if session:
                usage = session.get_context_usage()
                context_limit = session.get_context_limit()
                server.emit(ContextUpdatedEvent(
                    agent_id=agent_id,
                    usage=server._build_usage(
                        prompt_tokens=usage.get('prompt_tokens', 0),
                        output_tokens=usage.get('output_tokens', 0),
                        total_tokens=usage.get('total_tokens', 0),
                    ),
                    context_limit=context_limit,
                    percent_used=usage.get('percent_used', 0.0),
                    tokens_remaining=max(0, context_limit - usage.get('total_tokens', 0)),
                    turns=usage.get('turns', 0),
                ))

        logger.info(f"Restored {restored} subagents for session {session_id}")
        return restored

    def _recover_interrupted_turn(
        self,
        session_id: str,
        interrupted_state: Dict[str, Any],
        server: JaatoServer
    ) -> int:
        """Recover from an interrupted turn by injecting synthetic tool results.

        When a session is loaded with pending tool calls (from an interrupted turn),
        this method injects synthetic error results for each pending call. This
        completes the function_call/response pairs so the model sees what happened
        and can decide whether to retry.

        Args:
            session_id: The session ID being recovered.
            interrupted_state: The interrupted_turn dict from SessionState containing:
                - agent_id: Which agent was executing
                - pending_tool_calls: List of {id, name, args}
                - user_prompt: Original user prompt
                - started_at: When the turn started
            server: The JaatoServer to inject results into.

        Returns:
            Number of pending tool calls recovered.
        """
        from jaato_sdk.plugins.model_provider.types import Part, Message, Role, ToolResult

        pending_calls = interrupted_state.get('pending_tool_calls', [])
        if not pending_calls:
            logger.debug(f"No pending tool calls to recover for session {session_id}")
            return 0

        agent_id = interrupted_state.get('agent_id', 'main')

        # Build synthetic tool results for each pending call
        synthetic_parts = []
        for call in pending_calls:
            call_id = call.get('id', '')
            tool_name = call.get('name', 'unknown')

            synthetic_result = ToolResult(
                call_id=call_id,
                name=tool_name,
                result={
                    "error": "tool_interrupted",
                    "reason": "server_restart",
                    "message": f"Tool '{tool_name}' was interrupted by server restart. "
                               "You may retry this operation if appropriate."
                },
                is_error=True
            )
            synthetic_parts.append(Part.from_function_response(synthetic_result))

        # Create a TOOL message with all synthetic results
        synthetic_message = Message(role=Role.TOOL, parts=synthetic_parts)

        # Inject into history based on which agent was executing.
        # Phase 3 §7c step 6.6.3.6: forward the synthetic message
        # via the new ``session.append_history_message`` RPC
        # (§7c step 6.6.3.1 at commit aa9059ec) instead of the
        # daemon-side get-modify-reset dance.  The runner-side
        # JaatoSession.append_history_message wraps the same
        # get-history + append + reset_session flow internally
        # (preserves the ``_turn_accounting`` clear semantic).
        if agent_id == 'main':
            rpc = getattr(server, "_runner_rpc", None)
            if rpc is not None:
                forwarder = getattr(
                    rpc, "session_append_history_message_threadsafe", None,
                )
                if callable(forwarder):
                    try:
                        forwarder(synthetic_message, timeout=5.0)
                        logger.info(
                            f"Recovered {len(pending_calls)} interrupted tool call(s) "
                            f"for main agent in session {session_id}"
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.debug(
                            "append_history_message forward failed: %s",
                            exc,
                        )
        else:
            # Subagent recovery - find the subagent session
            if server.registry:
                subagent_plugin = server.registry.get_plugin("subagent")
                if subagent_plugin and hasattr(subagent_plugin, '_active_sessions'):
                    session_info = subagent_plugin._active_sessions.get(agent_id)
                    if session_info:
                        subagent_session = session_info.get('session')
                        if subagent_session:
                            # Append the synthetic tool message to history using proper API
                            current_history = subagent_session.get_history()
                            current_history.append(synthetic_message)
                            subagent_session.reset_session(current_history)
                            logger.info(
                                f"Recovered {len(pending_calls)} interrupted tool call(s) "
                                f"for subagent {agent_id} in session {session_id}"
                            )

        # Emit recovery event so clients know what happened
        server.emit(InterruptedTurnRecoveredEvent(
            session_id=session_id,
            agent_id=agent_id,
            recovered_calls=len(pending_calls),
            action_taken="synthetic_error",
        ))

        # Also emit a system message for user visibility
        tool_names = [call.get('name', 'unknown') for call in pending_calls]
        server.emit(SystemMessageEvent(
            message=f"Recovered from interrupted turn: {len(pending_calls)} tool call(s) "
                    f"({', '.join(tool_names)}) were interrupted by server restart.",
            style="warning",
        ))

        # Signal that the interrupted turn is now complete (agent is done)
        # This tells the client to stop showing the "thinking" spinner
        server.emit(AgentStatusChangedEvent(
            agent_id=agent_id,
            status="done",
        ))

        return len(pending_calls)

    def _save_session_async(self, session: Session) -> None:
        """Defer ``_save_session(session)`` to a background daemon thread.

        Path H (cycle 10).  Used by the ToolCallStartEvent branch in
        ``_handle_turn_tracking_event`` (line 1611) to persist
        ``pending_tool_calls`` for crash recovery WITHOUT blocking
        the synchronous ``_emit_to_session`` path.

        Pre-Path-H this site called ``_save_session(session)``
        synchronously, which made 2 blocking runner-RPCs
        (``session_get_history_threadsafe`` +
        ``session_snapshot_conversation_budget_threadsafe``) that
        raced against the runner's still-active ``send_message``.
        After 35s timeout the save failed silently AND the 35s
        delay starved the model loop's permission-response window.
        Architecturally same shape as Path E's Layer 5 race.

        Trade-off: the daemon-crash recovery window for IN-PROGRESS
        tool calls is narrowed (was synchronous fsync; becomes best-
        effort async fsync).  Recovery for COMPLETED turns is
        unaffected — the natural-boundary save still runs
        synchronously on the AgentStatusChangedEvent(status=done) path.

        Concurrent invocations for the same session serialize via
        ``self._async_save_lock`` so parallel ToolCallStartEvents
        produce consistent last-writer-wins ordering.

        Args:
            session: The session to save.
        """
        def _do_save() -> None:
            try:
                with self._async_save_lock:
                    self._save_session(session)
            except Exception as exc:  # noqa: BLE001 — best-effort
                logger.warning(
                    "async save for session %s failed: %s",
                    session.session_id, exc,
                )

        thread = threading.Thread(
            target=_do_save,
            name=f"async-save-{session.session_id}",
            daemon=True,
        )
        thread.start()

    def _save_session(self, session: Session) -> bool:
        """Save a session to disk.

        Args:
            session: The session to save.

        Returns:
            True if saved successfully.
        """
        try:
            # Get history directly from JaatoClient to ensure we capture
            # in-progress turns (the agent state cache is only updated at turn end)
            # Phase 3 §7c step 6.6.4.5b: fetch history via the
            # ``session.get_history`` RPC instead of the daemon-side
            # JaatoClient indirection.  Captures in-progress turns
            # (the agent state cache only updates at turn end).
            history = []
            if session.server and session.server._runner_rpc is not None:
                history = session.server._runner_rpc.session_get_history_threadsafe()
            turn_accounting = []

            if session.server:
                main_id = session.server.main_agent_id
                if main_id in session.server._agents:
                    turn_accounting = session.server._agents[main_id].turn_accounting

            # Resolve storage directory from workspace
            if session.workspace_path:
                storage_dir = self._session_storage_dir(session.workspace_path)
            else:
                storage_dir = pathlib.Path(self._session_config.storage_path)

            # Get subagent state if subagent plugin is available
            subagent_metadata = {}
            if session.server and session.server.registry:
                subagent_plugin = session.server.registry.get_plugin("subagent")
                if subagent_plugin and hasattr(subagent_plugin, 'get_persistence_state'):
                    subagent_registry = subagent_plugin.get_persistence_state()
                    if subagent_registry.get('agents'):
                        subagent_metadata['subagents'] = subagent_registry

                        # Save per-agent state files
                        self._save_subagent_states(
                            session.session_id,
                            subagent_plugin,
                            subagent_registry.get('agents', []),
                            storage_dir=storage_dir,
                        )

            # Save TODO plugin state
            session_dir = storage_dir / session.session_id
            if session.server:
                self._save_todo_state(session.server, session_dir)

            # Generic plugin state persistence: iterate all exposed plugins
            # and collect state from any that implement get_persistence_state().
            # Plugins with dedicated persistence (subagent, todo) are skipped
            # since they're handled above with their own file-based storage.
            plugin_states = {}
            _DEDICATED_PLUGINS = {'subagent', 'todo'}
            if session.server and session.server.registry:
                for plugin_name in session.server.registry.list_exposed():
                    if plugin_name in _DEDICATED_PLUGINS:
                        continue
                    plugin = session.server.registry.get_plugin(plugin_name)
                    if plugin and hasattr(plugin, 'get_persistence_state'):
                        try:
                            pstate = plugin.get_persistence_state()
                            if pstate:
                                plugin_states[plugin_name] = pstate
                        except Exception as e:
                            logger.warning(
                                f"Failed to get persistence state for plugin "
                                f"'{plugin_name}': {e}"
                            )
            if plugin_states:
                subagent_metadata['plugin_states'] = plugin_states

            # Get conversation budget for persistence (other budget sources are
            # automatically recreated when the session is restored).
            # Phase 3 §7c step 6.6.3.6: forward to runner-side via
            # the new ``session.snapshot_conversation_budget`` RPC
            # (§7c step 6.6.3.2 at commit abd7ec08) instead of
            # reaching into the daemon-side session's
            # instruction_budget.
            budget_state = None
            if session.server is not None:
                rpc = getattr(session.server, "_runner_rpc", None)
                if rpc is not None:
                    snapshotter = getattr(
                        rpc,
                        "session_snapshot_conversation_budget_threadsafe",
                        None,
                    )
                    if callable(snapshotter):
                        try:
                            budget_state = snapshotter(timeout=5.0)
                        except Exception as exc:  # noqa: BLE001
                            logger.debug(
                                "snapshot_conversation_budget forward failed: %s",
                                exc,
                            )

            # Get workspace file tracking state for persistence
            workspace_files = None
            monitor = self._workspace_monitors.get(session.session_id)
            if monitor:
                workspace_files = monitor.get_tracked_dict() or None

            # Persist provisioned flag in metadata so restored sessions
            # know their workspace is server-managed.
            if session.provisioned:
                subagent_metadata['provisioned'] = True

            # Create SessionState
            state = SessionState(
                session_id=session.session_id,
                history=history,
                created_at=datetime.fromisoformat(session.created_at),
                updated_at=datetime.now(),
                description=session.description or session.name,
                turn_count=len(history) // 2,  # Approximate
                turn_accounting=turn_accounting,
                user_inputs=session.user_inputs,  # Command history for prompt restoration
                model=session.server.model_name if session.server else None,
                workspace_path=session.workspace_path,
                metadata=subagent_metadata,
                budget_state=budget_state,
                interrupted_turn=session.interrupted_turn,  # For recovery on restart
                workspace_files=workspace_files,
            )

            self._session_plugin.save(state, storage_dir=storage_dir)
            session.is_dirty = False

            logger.debug(f"Saved session: {session.session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to save session {session.session_id}: {e}")
            return False

    def _get_todo_plugin(self, server: JaatoServer) -> Optional[Any]:
        """Get the TODO plugin from a server's registry.

        Args:
            server: The JaatoServer instance.

        Returns:
            The TodoPlugin instance, or None if not available.
        """
        if not server or not server.registry:
            return None
        return server.registry.get_plugin("todo")

    def _configure_todo_storage(self, server: JaatoServer, session_dir: pathlib.Path) -> None:
        """Configure TODO plugin with session-scoped file storage.

        Args:
            server: The JaatoServer instance.
            session_dir: The session's storage directory.
        """
        todo_plugin = self._get_todo_plugin(server)
        if not todo_plugin:
            return

        # Resolve to absolute path to avoid issues with CWD changes
        # (e.g., when subagents call os.chdir() in background threads)
        plans_dir = (session_dir / "plans").resolve()
        todo_plugin.initialize({
            "storage_type": "file",
            "storage_path": str(plans_dir),
            "storage_use_directory": True,  # One file per plan
        })
        logger.debug(f"Configured TODO storage at: {plans_dir}")

    def _save_todo_state(self, server: JaatoServer, session_dir: pathlib.Path) -> None:
        """Save TODO plugin state (agent-plan mapping, blocked steps).

        Args:
            server: The JaatoServer instance.
            session_dir: The session's storage directory.
        """
        todo_plugin = self._get_todo_plugin(server)
        if not todo_plugin or not hasattr(todo_plugin, 'get_persistence_state'):
            return

        state = todo_plugin.get_persistence_state()
        if not state.get('agent_plan_ids'):
            # No plans to save
            return

        # Resolve to absolute path to avoid CWD issues
        state_path = (session_dir / "plans" / "_state.json").resolve()
        try:
            from shared.atomic_write import atomic_write_json
            atomic_write_json(state_path, state)
            logger.debug(f"Saved TODO state: {state_path}")
        except Exception as e:
            logger.error(f"Failed to save TODO state: {e}")

    def _load_todo_state(self, server: JaatoServer, session_dir: pathlib.Path) -> None:
        """Load TODO plugin state from disk.

        Args:
            server: The JaatoServer instance.
            session_dir: The session's storage directory.
        """
        # Resolve to absolute path to avoid CWD issues
        state_path = (session_dir / "plans" / "_state.json").resolve()
        if not state_path.exists():
            return

        todo_plugin = self._get_todo_plugin(server)
        if not todo_plugin or not hasattr(todo_plugin, 'restore_persistence_state'):
            return

        try:
            with open(state_path, 'r', encoding='utf-8') as f:
                state = json.load(f)
            todo_plugin.restore_persistence_state(state)
            logger.debug(f"Loaded TODO state: {state_path}")
        except Exception as e:
            logger.error(f"Failed to load TODO state: {e}")

    def _save_subagent_states(
        self,
        session_id: str,
        subagent_plugin: Any,
        agents: List[Dict[str, Any]],
        storage_dir: Optional[pathlib.Path] = None,
    ) -> None:
        """Save per-agent state files for subagents.

        Args:
            session_id: The parent session ID.
            subagent_plugin: The SubagentPlugin instance.
            agents: List of agent info dicts from the registry.
            storage_dir: Workspace-resolved session storage directory.
        """
        # Create subagents directory
        base = storage_dir or pathlib.Path(self._session_config.storage_path)
        subagents_dir = base / session_id / "subagents"
        subagents_dir.mkdir(parents=True, exist_ok=True)

        for agent_info in agents:
            agent_id = agent_info.get('agent_id')
            if not agent_id:
                continue

            # Get full state from plugin
            full_state = subagent_plugin.get_agent_full_state(agent_id)
            if not full_state:
                continue

            # Write to file (atomically — Phase 3 §3.14).  A SIGTERM
            # mid-write would otherwise leave a corrupt subagent state
            # file that the disk-restore path can't parse.
            agent_file = subagents_dir / f"{agent_id}.json"
            try:
                from shared.atomic_write import atomic_write_json
                atomic_write_json(agent_file, full_state)
                logger.debug(f"Saved subagent state: {agent_file}")
            except Exception as e:
                logger.error(f"Failed to save subagent {agent_id}: {e}")

    def _maybe_unload_session(self, session_id: str) -> None:
        """Unload a session from memory if no clients attached.

        Saves to disk first if dirty.  The heavy cleanup work (save,
        workspace monitor stop, subagent teardown, server.shutdown)
        is deferred to a background thread because callers may
        invoke this from the asyncio loop's IPC disconnect handler.
        The deferred-to-thread work performs ``run_coroutine_threadsafe``
        RPCs to the runner; if those ran on the same loop they
        deadlock (the scheduled coro can't progress while the
        calling task holds the loop).  Pre-2026-05-14 the chain ran
        inline and froze the loop for ~55 s (30 s save + 10 s
        session.shutdown + 10 s runner-rpc close + 5 s buffer)
        per disconnect; the per-session unload error message logged
        as ``Failed to save session ...:`` (empty ``str(exc)``) was
        the timeout's tell.

        Args:
            session_id: The session to potentially unload.
        """
        session = self._sessions.get(session_id)
        if not session:
            return

        if session.attached_clients:
            return  # Still has clients

        # Don't unload while the model thread is still running — the
        # client may have disconnected (WS ping timeout, network blip)
        # but the model is mid-turn processing tools.  The session will
        # be unloaded when the model thread finishes and checks again.
        if session.server and session.server._model_running:
            logger.info(
                "Deferring unload of session %s — model thread still active",
                session_id,
            )
            return

        # Defer heavy cleanup to a background thread so the asyncio
        # loop (when this is called from an IPC disconnect handler)
        # doesn't deadlock on ``run_coroutine_threadsafe`` RPCs to
        # the runner.  Fire-and-forget — no caller awaits the unload
        # completion (verified against the 3 call sites).
        thread = threading.Thread(
            target=self._do_session_unload,
            args=(session_id,),
            name=f"unload-{session_id}",
            daemon=True,
        )
        thread.start()

    def _do_session_unload(self, session_id: str) -> None:
        """Background-thread body of session unload.

        Invoked by :meth:`_maybe_unload_session` after the pre-checks
        confirm no attached clients and no active model thread.  Does
        the actual save + plugin teardown + server.shutdown on a
        thread distinct from the daemon's asyncio loop so the
        ``run_coroutine_threadsafe`` RPCs to the runner can complete
        normally.

        Re-checks ``self._sessions`` under the lock — a new client
        might have attached between the pre-check and this thread's
        first action.  If so, abort the unload.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return
            # Re-check attached clients: a new client could have
            # attached between _maybe_unload_session's check and
            # this thread starting.
            if session.attached_clients:
                logger.info(
                    "Unload of session %s aborted — a client re-attached "
                    "while the unload thread was scheduling", session_id,
                )
                return
            if session.server and session.server._model_running:
                logger.info(
                    "Unload of session %s aborted — model thread became "
                    "active while the unload thread was scheduling",
                    session_id,
                )
                return

        # Save before unloading.  Now running on a real thread so
        # session_get_history_threadsafe + budget snapshot RPCs work
        # without deadlocking.
        if session.is_dirty:
            self._save_session(session)

        # Close session-specific log handlers
        handler = get_session_handler()
        if handler:
            handler.close_session(session_id)

        # Stop workspace monitor
        self._stop_workspace_monitor(session_id)

        # Phase 4 §4.3.6d: cascade-teardown any isolated subagents
        # owned by this parent.  Must run BEFORE the parent's
        # server.shutdown() so we have a chance to close sub-runner
        # RPCs cleanly; running after shutdown leaves orphaned
        # processes that the daemon loses track of.
        try:
            n_torn_down = self._cascade_teardown_isolated_subagents(
                parent_session_id=session_id,
            )
            if n_torn_down > 0:
                logger.info(
                    "Unload: cascade-teardown completed for %d isolated "
                    "subagent(s) of session %s",
                    n_torn_down, session_id,
                )
        except Exception:  # noqa: BLE001 — best-effort
            logger.exception(
                "Unload: cascade-teardown raised for session %s — "
                "continuing with server.shutdown",
                session_id,
            )

        # Shutdown server and remove from memory.
        try:
            session.server.shutdown()
        except Exception:  # noqa: BLE001 — best-effort
            logger.exception(
                "Unload: server.shutdown raised for session %s — "
                "removing from sessions anyway", session_id,
            )
        with self._lock:
            self._sessions.pop(session_id, None)
        logger.info(f"Unloaded session: {session_id}")

    # ------------------------------------------------------------------
    # Prompt skill expansion
    # ------------------------------------------------------------------

    _PROMPT_REF_PATTERN = re.compile(r'%([a-zA-Z][a-zA-Z0-9_-]*)')

    # ``%<name> --help`` / ``%<name> -h`` — the leading whitespace and the
    # flag are consumed together so the span can be excised cleanly from
    # the outgoing message without leaving a dangling ``--help`` token.
    _PROMPT_HELP_REF_PATTERN = re.compile(
        r'%([a-zA-Z][a-zA-Z0-9_-]*)[ \t]+(--help|-h)\b'
    )

    def _intercept_prompt_help_refs(
        self,
        text: str,
        server: 'JaatoServer',
        client_id: Optional[str],
    ) -> str:
        """Handle ``%<name> --help`` references before model dispatch.

        Scans *text* for ``%name --help`` (or ``-h``) tokens, resolves
        each via the prompt library plugin, emits the resulting help as
        a :class:`HelpTextEvent` (pager) directly to *client_id*, and
        returns *text* with those tokens excised.  This runs *before*
        :meth:`_expand_prompt_references` so the help content never
        reaches the model — the user types ``%foo --help`` and sees the
        help, without the model also being asked to respond.

        If the prompt library plugin is unavailable, or a referenced
        prompt doesn't exist, the reference is left untouched and
        flows through to the normal expansion path for graceful
        degradation.

        Args:
            text: Raw message text from the client.
            server: The session's :class:`JaatoServer` for plugin access.
            client_id: Target client for the emitted help events. When
                ``None`` (rare — only during offline replay), the
                matched spans are still stripped but no events fire.

        Returns:
            The message text with matched ``%<name> --help`` spans
            removed.  If the entire original message consisted only of
            such references, the return value is the surrounding
            whitespace — the caller should check ``.strip()`` and skip
            the model call.
        """
        if not text or '%' not in text:
            return text

        matches = list(self._PROMPT_HELP_REF_PATTERN.finditer(text))
        if not matches:
            return text

        # Phase 3 §7c step 6.6.4.5a: read ``server._runtime`` directly.
        prompt_plugin = None
        runtime = server._runtime
        if runtime and runtime.registry:
            prompt_plugin = runtime.registry.get_plugin("prompt_library")
        if prompt_plugin is None or not hasattr(
            prompt_plugin, '_execute_prompt_command'
        ):
            return text

        from jaato_sdk.plugins.base import HelpLines
        from jaato_sdk.events import HelpTextEvent, SystemMessageEvent

        result = text
        # Walk matches in reverse so earlier spans' positions remain
        # valid while we cut later spans out of ``result``.
        for match in reversed(matches):
            # ``%`` must stand alone — skip if preceded by an
            # alphanumeric (e.g. ``100%foo`` is not a prompt reference).
            if match.start() > 0 and text[match.start() - 1].isalnum():
                continue

            prompt_name = match.group(1)
            try:
                help_result = prompt_plugin._execute_prompt_command(
                    {'args': [prompt_name, '--help']}
                )
            except Exception:
                continue

            if isinstance(help_result, HelpLines):
                if client_id is not None:
                    self._emit_to_client(
                        client_id, HelpTextEvent(lines=help_result.lines)
                    )
            elif isinstance(help_result, str):
                # Fallback: plugin returned an error string (e.g.
                # "Prompt not found") — surface it as a system message
                # rather than pushing it to the model.
                if client_id is not None:
                    self._emit_to_client(
                        client_id, SystemMessageEvent(
                            message=help_result, style="error"
                            if help_result.startswith('Prompt not found')
                            else "info",
                        )
                    )
            else:
                # Unknown return type — leave reference in place and
                # let the normal expansion path handle it.
                continue

            result = result[:match.start()] + result[match.end():]

        return result

    def _expand_prompt_references(self, text: str, server: 'JaatoServer') -> str:
        """Expand ``%prompt-name`` references in a message.

        Scans *text* for ``%name`` tokens, resolves each via the prompt
        library plugin, strips the ``%`` prefix from the message body,
        and appends the prompt content in a ``--- Referenced Prompts ---``
        section (same format as the TUI client).

        This runs server-side so that **all** clients (TUI, WS, IPC)
        get automatic prompt skill injection regardless of client-side
        capabilities.

        If the prompt library plugin is not available or a prompt is not
        found, the ``%`` prefix is stripped but no content is appended
        (same graceful degradation as the TUI).

        Args:
            text: Raw message text from the client.
            server: The session's ``JaatoServer`` for plugin access.

        Returns:
            The message with prompt references expanded, or the
            original text if no references are found.
        """
        if not text or '%' not in text:
            return text

        matches = list(self._PROMPT_REF_PATTERN.finditer(text))
        if not matches:
            return text

        # Phase 3 §7c step 6.6.4.5a: read ``server._runtime`` directly.
        prompt_plugin = None
        runtime = server._runtime
        if runtime and runtime.registry:
            prompt_plugin = runtime.registry.get_plugin("prompt_library")

        # Process matches in reverse to preserve positions
        result = text
        expanded = []
        for match in reversed(matches):
            # Skip if preceded by alphanumeric (not a standalone reference)
            if match.start() > 0 and text[match.start() - 1].isalnum():
                continue

            prompt_name = match.group(1)

            # Capture args after the prompt name on the same line so the
            # expansion can substitute them.  Args run from the end of the
            # name match to the next newline (or end of text), supporting
            # both positional tokens and ``key=value`` named tokens.  The
            # original args are LEFT in the message body — they remain
            # visible to the user, while the expansion gets a fully
            # populated copy via _execute_prompt_command.
            args_start = match.end()
            newline_pos = text.find('\n', args_start)
            args_end = newline_pos if newline_pos != -1 else len(text)
            args_text = text[args_start:args_end].strip()
            # Tokenize via the prompt library's shared splitter so that
            # ``%name key="value with spaces"`` and ``%name "positional
            # value"`` work consistently across the text path and the
            # slash-command path.
            from shared.plugins.prompt_library.plugin import tokenize_prompt_args
            prompt_args = tokenize_prompt_args(args_text)

            # Strip the % prefix regardless of whether we find the prompt
            result = result[:match.start()] + prompt_name + result[match.end():]

            # Try to expand via prompt library
            if prompt_plugin and hasattr(prompt_plugin, '_execute_prompt_command'):
                try:
                    content = prompt_plugin._execute_prompt_command(
                        {'args': [prompt_name] + prompt_args}
                    )
                    # _execute_prompt_command usually returns strings,
                    # but ``help``-style calls can return HelpLines — the
                    # latter shouldn't be embedded in the outgoing message
                    # (both because it has no sensible f-string form and
                    # because it's meant for the user, not the model).
                    if isinstance(content, str) and content and not content.startswith('Prompt not found'):
                        expanded.append({'name': prompt_name, 'content': content})
                except Exception:
                    pass  # Graceful degradation

        if expanded:
            # Reverse to restore original order
            expanded.reverse()
            parts = [result, "\n\n--- Referenced Prompts ---\n"]
            for prompt in expanded:
                parts.append(f"\n[Prompt: {prompt['name']}]\n")
                parts.append(f"{prompt['content']}\n")
            return ''.join(parts)

        return result

    def detach_client(self, client_id: str) -> None:
        """Detach a client from its current session.

        Args:
            client_id: The client to detach.
        """
        with self._lock:
            session_id = self._client_to_session.pop(client_id, None)
            if session_id and session_id in self._sessions:
                session = self._sessions[session_id]
                session.attached_clients.discard(client_id)
                logger.info(f"Client {client_id} detached from session {session_id}")

                # Maybe unload if no more clients
                self._maybe_unload_session(session_id)

    def save_session(self, session_id: str) -> bool:
        """Explicitly save a session to disk.

        Args:
            session_id: The session to save.

        Returns:
            True if saved successfully.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return False
            return self._save_session(session)

    def delete_session(self, session_id: str) -> bool:
        """Delete a session from memory and disk.

        Args:
            session_id: The session to delete.

        Returns:
            True if deleted.
        """
        workspace_path = None
        with self._lock:
            # Remove from memory — capture workspace_path before popping
            session = self._sessions.pop(session_id, None)
            if session:
                workspace_path = session.workspace_path
                # Notify attached clients
                for client_id in session.attached_clients:
                    self._emit_to_client(client_id, SystemMessageEvent(
                        message=f"Session deleted: {session.name}",
                        style="warning",
                    ))
                    self._client_to_session.pop(client_id, None)

                # Shutdown the server
                session.server.shutdown()

        # Delete from disk
        storage_dir = self._session_storage_dir(workspace_path) if workspace_path else None
        deleted = self._session_plugin.delete(session_id, storage_dir=storage_dir)

        logger.info(f"Session deleted: {session_id}")
        return deleted or session is not None

    def _normalize_workspace(self, path: Optional[str]) -> Optional[str]:
        """Normalize a workspace path for comparison.

        Args:
            path: The path to normalize.

        Returns:
            Normalized absolute path, or None if path is None.
        """
        if not path:
            return None
        import os
        return os.path.normpath(os.path.abspath(path))

    def _workspaces_match(
        self,
        path1: Optional[str],
        path2: Optional[str],
    ) -> bool:
        """Check if two workspace paths match.

        Args:
            path1: First path.
            path2: Second path.

        Returns:
            True if both paths are set and point to the same directory.
        """
        norm1 = self._normalize_workspace(path1)
        norm2 = self._normalize_workspace(path2)
        if not norm1 or not norm2:
            return False
        return norm1 == norm2

    def get_or_create_default(
        self,
        client_id: str,
        workspace_path: Optional[str] = None,
    ) -> str:
        """Get the default session for a workspace, or create a new one.

        Finds the most recently used session for the given workspace.
        Creates a new session if no matching session exists.

        Args:
            client_id: The requesting client.
            workspace_path: Client's working directory for file operations.

        Returns:
            The session ID.
        """
        logger.debug(f"get_or_create_default called for client {client_id}, workspace={workspace_path}")

        # Check in-memory sessions first - find one matching the workspace
        with self._lock:
            if self._sessions and workspace_path:
                # Find sessions matching this workspace
                matching_sessions = [
                    s for s in self._sessions.values()
                    if self._workspaces_match(s.workspace_path, workspace_path)
                ]
                if matching_sessions:
                    # Use the first matching session (they're all for the same workspace)
                    session = matching_sessions[0]
                    logger.debug(f"  found in-memory session for workspace: {session.session_id}")
                    session.attached_clients.add(client_id)
                    self._client_to_session[client_id] = session.session_id
                    # Emit current agent state to the newly attached client
                    session.server.emit_current_state(
                        lambda e: self._emit_to_client(client_id, e),
                        skip_session_info=True
                    )
                    # Send complete SessionInfoEvent with state snapshot
                    self._emit_to_client(client_id, self._build_session_info_event(session))
                    return session.session_id

        # Check persisted sessions (already sorted by updated_at descending)
        logger.debug(f"  checking persisted sessions...")
        persisted = self._get_persisted_sessions(workspace_path=workspace_path)
        logger.debug(f"  found {len(persisted)} persisted session(s)")

        if persisted and workspace_path:
            # Find sessions matching this workspace
            matching_persisted = [
                s for s in persisted
                if self._workspaces_match(s.workspace_path, workspace_path)
            ]
            logger.debug(f"  found {len(matching_persisted)} session(s) for workspace")

            if matching_persisted:
                # Use the most recent one for this workspace
                most_recent = matching_persisted[0]
                logger.debug(f"  attaching to workspace session: {most_recent.session_id}")
                if self.attach_session(client_id, most_recent.session_id, workspace_path):
                    return most_recent.session_id

        # No matching sessions exist - create a new one for this workspace
        logger.debug(f"  creating new session for workspace...")
        return self.create_session(client_id, workspace_path=workspace_path)

    # =========================================================================
    # Session Queries
    # =========================================================================

    def _get_persisted_sessions(
        self,
        workspace_path: Optional[str] = None,
    ) -> List[PluginSessionInfo]:
        """Get list of sessions from disk.

        Args:
            workspace_path: Workspace directory to list sessions for.
                When provided, lists sessions from that workspace's storage.
        """
        storage_dir = self._session_storage_dir(workspace_path) if workspace_path else None
        try:
            return self._session_plugin.list_sessions(storage_dir=storage_dir)
        except Exception as e:
            logger.error(f"Failed to list persisted sessions: {e}")
            return []

    def list_profiles(
        self,
        workspace_path: Optional[str] = None,
        config_root: Optional[str] = None,
    ) -> Tuple[List["ProfileSummary"], List["ProfileParseError"]]:
        """List available agent profiles.

        Discovers profiles from three sources in decreasing precedence:
        workspace ``.jaato/profiles/``, user ``~/.jaato/profiles/``,
        and premium entry-point profiles.

        Profiles that fail to parse are returned in the second element
        (parse errors) so the caller can surface them distinctly from
        usable profiles — picker UIs typically render the two lists
        differently (badges vs. the main grid).

        Sensitive material is filtered: env *values* are dropped (only
        variable names survive); ``system_instructions`` and
        ``inherits`` are not exposed (deprecated or already resolved).

        Args:
            workspace_path: Workspace directory to discover profiles from.
                May be ``None`` (user-level and premium profiles are still
                returned).

        Returns:
            Tuple ``(profiles, parse_errors)``.  ``profiles`` is the
            list of typed ``ProfileSummary`` records; ``parse_errors``
            is the list of files that failed discovery.
        """
        from dataclasses import asdict
        from shared.plugins.subagent.config import discover_profiles
        from jaato_sdk.events import ProfileSummary, ProfileParseError

        discovery = discover_profiles(
            ".jaato/profiles",
            base_path=workspace_path or ".",
            config_root=config_root,
        )
        summaries: List[ProfileSummary] = []
        for name, profile in discovery.profiles.items():
            summaries.append(ProfileSummary(
                name=name,
                description=profile.description,
                plugins=profile.plugins,
                preloaded_plugins=sorted(profile.preloaded_plugins),
                plugin_configs=profile.plugin_configs,
                model=profile.model,
                provider=profile.provider,
                max_turns=profile.max_turns,
                model_tiers=profile.model_tiers,
                gc=asdict(profile.gc) if profile.gc else None,
                runtime_limits=(
                    asdict(profile.runtime_limits)
                    if profile.runtime_limits else None
                ),
                completion_payload_schema=profile.completion_payload_schema,
                env_var_names=sorted(profile.env.keys()),
            ))

        parse_errors: List[ProfileParseError] = [
            ProfileParseError(name=stem, error=error)
            for stem, error in discovery.errors.items()
        ]
        return summaries, parse_errors

    def list_sessions(self) -> List[RuntimeSessionInfo]:
        """List all sessions (in-memory and on-disk).

        Returns merged view with runtime status for loaded sessions.
        Collects persisted sessions from all known workspaces (in-memory
        sessions + client configs).
        """
        result: Dict[str, RuntimeSessionInfo] = {}

        # Collect all known workspace paths from in-memory sessions and client configs
        known_workspaces: Set[str] = set()
        with self._lock:
            for session in self._sessions.values():
                if session.workspace_path:
                    norm = self._normalize_workspace(session.workspace_path)
                    if norm:
                        known_workspaces.add(norm)
            for config in self._client_config.values():
                wp = config.get('working_dir')
                if wp:
                    norm = self._normalize_workspace(wp)
                    if norm:
                        known_workspaces.add(norm)

        # Add persisted sessions from all known workspaces
        for wp in known_workspaces:
            for info in self._get_persisted_sessions(workspace_path=wp):
                result[info.session_id] = RuntimeSessionInfo(
                    session_id=info.session_id,
                    name=info.description or info.session_id,
                    description=info.description,
                    created_at=info.created_at.isoformat(),
                    last_activity=info.updated_at.isoformat(),
                    model_provider="",
                    model_name=info.model or "",
                    is_processing=False,
                    is_loaded=False,
                    client_count=0,
                    turn_count=info.turn_count,
                    workspace_path=info.workspace_path,
                )

        # Overlay in-memory sessions (have more current info)
        with self._lock:
            for session in self._sessions.values():
                result[session.session_id] = RuntimeSessionInfo(
                    session_id=session.session_id,
                    name=session.name,
                    description=session.description,
                    created_at=session.created_at,
                    last_activity=session.last_activity,
                    model_provider=session.server.model_provider,
                    model_name=session.server.model_name,
                    is_processing=session.server.is_processing,
                    is_loaded=True,
                    client_count=len(session.attached_clients),
                    turn_count=len(session.server.get_history()) // 2,
                    workspace_path=session.workspace_path,
                    created_by=session.created_by,
                )

        # Sort by last activity
        sessions = list(result.values())
        sessions.sort(key=lambda s: s.last_activity, reverse=True)
        return sessions

    def _build_session_info_event(self, session: "Session") -> SessionInfoEvent:
        """Build a complete SessionInfoEvent with state snapshot.

        Includes current session info plus:
        - sessions: All available sessions for completion/display
        - tools: All tools with enabled status
        - models: Available model names
        """
        # Get sessions list
        # Build sessions list. Enrich with sandbox_mode from Session objects
        # (sandbox_mode is set by the WS server during workspace provisioning).
        session_lookup = {s.session_id: s for s in self._sessions.values()}
        sessions_data = []
        for s in self.list_sessions():
            entry = {
                "id": s.session_id,
                "name": s.name or "",
                "description": s.description or "",
                "model_provider": s.model_provider or "",
                "model_name": s.model_name or "",
                "is_loaded": s.is_loaded,
                "client_count": s.client_count,
                "turn_count": s.turn_count,
                "workspace_path": s.workspace_path or "",
            }
            sess = session_lookup.get(s.session_id)
            if sess and sess.sandbox_mode:
                entry["sandbox_mode"] = sess.sandbox_mode
            sessions_data.append(entry)

        # Get tools list from the session's server
        tools_data = []
        if session.server:
            tools_data = session.server.get_tool_status()

        # Models list is lazy-loaded on demand to avoid API calls during init
        # Client fetches models when user requests completions
        models_data = []

        # Get memory metadata from the session's server for completion cache
        memories_data = []
        if session.server:
            mem_plugin = session.server._find_plugin_for_command("memory")
            if mem_plugin and hasattr(mem_plugin, 'get_memory_metadata'):
                memories_data = mem_plugin.get_memory_metadata()

        # Get sandbox paths from the session's server for @@ completion cache
        sandbox_paths_data = []
        if session.server:
            sandbox_paths_data = session.server._get_sandbox_paths()

        # Get service metadata from the session's server for completion cache
        services_data = []
        if session.server:
            svc_plugin = session.server._find_plugin_for_command("services")
            if svc_plugin and hasattr(svc_plugin, 'get_service_metadata'):
                services_data = svc_plugin.get_service_metadata()

        # Build tool ID mappings for client-side display resolution
        tool_id_mappings = {}
        if session.server:
            tool_id_mappings = session.server._build_tool_id_mappings()

        return SessionInfoEvent(
            session_id=session.session_id,
            session_name=session.name,
            model_provider=session.server.model_provider if session.server else "",
            model_name=session.server.model_name if session.server else "",
            profile_name=session.server.profile_name if session.server else None,
            sessions=sessions_data,
            tools=tools_data,
            models=models_data,
            user_inputs=session.user_inputs,  # Command history for prompt restoration
            memories=memories_data,
            sandbox_paths=sandbox_paths_data,
            services=services_data,
            tool_id_mappings=tool_id_mappings,
        )

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID (in-memory only)."""
        with self._lock:
            return self._sessions.get(session_id)

    def get_client_session(self, client_id: str) -> Optional[Session]:
        """Get the session a client is attached to."""
        with self._lock:
            session_id = self._client_to_session.get(client_id)
            if session_id:
                return self._sessions.get(session_id)
        return None

    # =========================================================================
    # Workspace Snapshot
    # =========================================================================

    def snapshot_workspace(
        self,
        target_session_id: str,
        requester_workspace: str,
    ) -> Dict[str, Any]:
        """Copy a target session's workspace to the requester's replay area.

        For git-managed workspaces, uses ``git archive`` for committed
        content and manually copies untracked files.  For non-git
        workspaces, uses ``shutil.copytree``.

        The snapshot is created inside the requester's workspace at
        ``<requester_workspace>/.jaato/replay/<uuid>/`` so the
        requester's AppArmor profile can access it without any
        confinement changes.

        The target session is NOT paused during the copy.  The files
        the fine-tuner cares about (profiles, agent markdown, prompts,
        service configs) are written once at session creation and never
        modified during a turn, so a consistent snapshot does not
        require pausing.

        Args:
            target_session_id: The session whose workspace to snapshot.
            requester_workspace: The requesting session's workspace
                path (destination parent).

        Returns:
            Dict with ``snapshot_path``, ``source_session_id``, and
            ``source_commit`` (``None`` if not a git repo).

        Raises:
            ValueError: If the target session is not found, has no
                workspace, or if the workspace doesn't exist.
            OSError: If file operations fail.
        """
        import shutil
        import subprocess
        import uuid as _uuid

        session = self.get_session(target_session_id)
        if session is None:
            raise ValueError(f"Session '{target_session_id}' not found")
        workspace = session.workspace_path
        if not workspace:
            raise ValueError(
                f"Session '{target_session_id}' has no workspace"
            )
        if not os.path.isdir(workspace):
            raise ValueError(f"Workspace does not exist: {workspace}")

        dest_dir = os.path.join(
            requester_workspace, ".jaato", "replay", str(_uuid.uuid4()),
        )
        os.makedirs(dest_dir, exist_ok=True)

        source_commit: Optional[str] = None
        is_git = os.path.isdir(os.path.join(workspace, ".git"))

        try:
            if is_git:
                # Committed files via git archive
                archive = subprocess.run(
                    ["git", "-C", workspace, "archive", "HEAD"],
                    capture_output=True,
                    check=True,
                )
                subprocess.run(
                    ["tar", "-x", "-C", dest_dir],
                    input=archive.stdout,
                    check=True,
                )
                # Capture current commit
                rev = subprocess.run(
                    ["git", "-C", workspace, "rev-parse", "HEAD"],
                    capture_output=True, text=True,
                )
                source_commit = rev.stdout.strip() if rev.returncode == 0 else None

                # Copy untracked files
                status = subprocess.run(
                    ["git", "-C", workspace, "status", "--porcelain"],
                    capture_output=True, text=True,
                )
                if status.returncode == 0:
                    for line in status.stdout.splitlines():
                        if line.startswith("??"):
                            rel_path = line[3:].strip()
                            src = os.path.join(workspace, rel_path)
                            dst = os.path.join(dest_dir, rel_path)
                            if os.path.isdir(src):
                                shutil.copytree(
                                    src, dst,
                                    ignore=shutil.ignore_patterns(
                                        "__pycache__", "*.pyc",
                                    ),
                                )
                            elif os.path.isfile(src):
                                os.makedirs(os.path.dirname(dst), exist_ok=True)
                                shutil.copy2(src, dst)
            else:
                # Non-git: plain copytree
                shutil.copytree(
                    workspace, dest_dir,
                    dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns(
                        ".git", ".venv", "__pycache__", "node_modules",
                        "*.pyc",
                    ),
                )
        except Exception:
            # Clean up partial snapshot on failure
            if os.path.isdir(dest_dir):
                shutil.rmtree(dest_dir, ignore_errors=True)
            raise

        logger.info(
            "Snapshot workspace '%s' → '%s' (commit=%s)",
            workspace, dest_dir, source_commit,
        )
        return {
            "snapshot_path": dest_dir,
            "source_session_id": target_session_id,
            "source_commit": source_commit,
        }

    # =========================================================================
    # Turn Tracking for Recovery
    # =========================================================================

    def start_turn_tracking(
        self,
        session_id: str,
        user_prompt: str,
        agent_id: str = "main"
    ) -> None:
        """Mark a turn as in-progress for recovery purposes.

        Call this when a turn starts (user sends message) to enable recovery
        if the server crashes during tool execution.

        Args:
            session_id: The session ID.
            user_prompt: The user's original prompt.
            agent_id: Which agent is executing ("main" or subagent ID).
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.interrupted_turn = {
                    "agent_id": agent_id,
                    "pending_tool_calls": [],
                    "user_prompt": user_prompt,
                    "started_at": datetime.now(timezone.utc).isoformat(),
                }
                session.is_dirty = True
                logger.debug(f"Started turn tracking for session {session_id}, agent {agent_id}")

    def update_pending_tool_calls(
        self,
        session_id: str,
        function_calls: List[Dict[str, Any]]
    ) -> None:
        """Update pending tool calls after model response.

        Call this after the model returns function calls, before tool execution.
        This triggers an incremental save so the pending calls are persisted.

        Args:
            session_id: The session ID.
            function_calls: List of {id, name, args} dicts from model response.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session and session.interrupted_turn:
                session.interrupted_turn["pending_tool_calls"] = function_calls
                session.is_dirty = True
                # Incremental save to persist pending tool calls before execution
                self._save_session(session)
                logger.debug(
                    f"Updated pending tool calls for session {session_id}: "
                    f"{len(function_calls)} call(s)"
                )

    def clear_turn_tracking(self, session_id: str) -> None:
        """Clear turn tracking on successful completion.

        Call this when a turn completes successfully (no more function calls).

        Args:
            session_id: The session ID.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.interrupted_turn = None
                session.is_dirty = True
                logger.debug(f"Cleared turn tracking for session {session_id}")

    # =========================================================================
    # Request Routing
    # =========================================================================

    def handle_request(
        self,
        client_id: str,
        session_id: str,
        event: Event,
    ) -> None:
        """Route a request to the appropriate session.

        Args:
            client_id: The requesting client.
            session_id: The target session.
            event: The request event.
        """
        from jaato_sdk.events import ClientConfigRequest

        # Handle client config before session lookup (doesn't require session)
        if isinstance(event, ClientConfigRequest):
            self._apply_client_config(client_id, event)
            return

        session = self.get_session(session_id)
        if not session:
            self._emit_to_client(client_id, ErrorEvent(
                error=f"Session not found: {session_id}",
                error_type="SessionError",
            ))
            return

        # Update activity timestamp
        session.last_activity = datetime.now(timezone.utc).isoformat()
        session.is_dirty = True

        # Route to session's server
        server = session.server

        # Set logging context for session-specific log routing
        workspace_path = session.workspace_path
        session_env = server.get_all_session_env() if server else {}
        set_logging_context(
            session_id=session_id,
            client_id=client_id,
            workspace_path=workspace_path,
            session_env=session_env,
        )

        from jaato_sdk.events import (
            SendMessageRequest,
            PermissionResponseRequest,
            ClarificationResponseRequest,
            ClarificationBatchResponseEvent,
            ReferenceSelectionResponseRequest,
            StopRequest,
            CommandRequest,
            GetInstructionBudgetRequest,
            InstructionBudgetEvent,
            InjectPromptRequest,
            ReplayMessagesRequest,
            ReplayMessagesResultEvent,
            ResolveForkPointRequest,
            ResolveForkPointResultEvent,
            PermissionAddWhitelistRequest,
            PermissionAddBlacklistRequest,
            PermissionRemoveRequest,
            PermissionClearRequest,
            PermissionSetDefaultRequest,
            PermissionPolicySnapshotRequest,
            PermissionPolicySnapshotEvent,
        )

        if isinstance(event, SendMessageRequest):
            # Per-call parallel-tools override.  When the request
            # specifies parallel_tools (True/False), stash it on the
            # session so the next turn's tool-execution branch
            # consults it instead of the JAATO_PARALLEL_TOOLS env var.
            # The session clears the override after the turn.  None
            # leaves env-driven behaviour unchanged.
            if event.parallel_tools is not None:
                # Phase 3 §7c step 6.6.3.6: forward to runner-side
                # via the new ``session.set_parallel_tools_override``
                # RPC (§7c step 6.6.3.3 at commit b678ce2c) instead
                # of the private-attr write on the daemon-side
                # session.
                rpc = getattr(server, "_runner_rpc", None)
                if rpc is not None:
                    forwarder = getattr(
                        rpc,
                        "session_set_parallel_tools_override_threadsafe",
                        None,
                    )
                    if callable(forwarder):
                        try:
                            forwarder(event.parallel_tools, timeout=2.0)
                        except Exception as exc:  # noqa: BLE001
                            logger.debug(
                                "set_parallel_tools_override forward failed: %s",
                                exc,
                            )

            # Track user input for command history restoration
            if event.text and event.text.strip():
                session.user_inputs.append(event.text)
                session.is_dirty = True

            # ``%<name> --help`` is intercepted first: the help is
            # emitted straight to the client via a HelpTextEvent and the
            # reference is stripped so it never reaches the model.
            message_text = self._intercept_prompt_help_refs(
                event.text, server, client_id
            )

            # If the message was purely help requests, the user just
            # wanted documentation — don't dispatch anything to the
            # model.  Persist the (already-appended) user input and
            # return early.
            if not (message_text and message_text.strip()):
                self._save_session(session)
                return

            # Server-side prompt skill expansion: expand %prompt-name
            # references so all clients (TUI, WS, IPC) get automatic
            # prompt content injection regardless of client capabilities.
            message_text = self._expand_prompt_references(
                message_text, server
            )

            # Capture context for thread (ContextVars don't propagate to threads)
            ctx_session_id = session_id
            ctx_client_id = client_id
            ctx_workspace = workspace_path
            ctx_session_env = session_env

            # Run in thread to not block
            def run_message():
                # Set logging context in thread
                set_logging_context(
                    session_id=ctx_session_id,
                    client_id=ctx_client_id,
                    workspace_path=ctx_workspace,
                    session_env=ctx_session_env,
                )
                # Ensure provider traces from the main agent go to the
                # base provider_trace.log (not a subagent-specific file).
                try:
                    from jaato_sdk.trace import set_trace_agent_context
                    set_trace_agent_context("main")
                except ImportError:
                    pass  # Older jaato_sdk without per-agent trace routing
                try:
                    server.send_message(
                        message_text,
                        event.attachments if event.attachments else None
                    )
                    # Auto-save after turn
                    self._save_session(session)
                finally:
                    clear_logging_context()

            threading.Thread(target=run_message, daemon=True).start()

        elif isinstance(event, PermissionResponseRequest):
            server.respond_to_permission(
                event.request_id, event.response,
                edited_arguments=event.edited_arguments,
            )

        elif isinstance(event, ClarificationResponseRequest):
            server.respond_to_clarification(event.request_id, event.response)

        elif isinstance(event, ClarificationBatchResponseEvent):
            server.respond_to_clarification_batch(event.request_id, event.answers)

        elif isinstance(event, ReferenceSelectionResponseRequest):
            server.respond_to_reference_selection(event.request_id, event.response)

        elif isinstance(event, StopRequest):
            server.stop()

        elif isinstance(event, CommandRequest):
            # Intercept workspace command — it needs the monitor from
            # session manager, not from the plugin system.
            if event.command == "workspace":
                from .workspace_command import handle_workspace_command

                monitor = self._workspace_monitors.get(session_id)
                result = handle_workspace_command(monitor, event.args or [])
            else:
                result = server.execute_command(event.command, event.args)
            # Format result properly
            if isinstance(result, dict):
                if "_pager" in result:
                    # HelpLines result already emitted via HelpTextEvent, skip
                    pass
                elif "error" in result:
                    # Error result
                    self._emit_to_client(client_id, SystemMessageEvent(
                        message=result["error"],
                        style="error",
                    ))
                elif "result" in result:
                    # Simple result - show the text directly
                    self._emit_to_client(client_id, SystemMessageEvent(
                        message=result["result"],
                        style="info",
                    ))
                else:
                    # Dict result with multiple keys - format each
                    lines = []
                    for key, value in result.items():
                        if not key.startswith('_'):
                            if isinstance(value, list):
                                # Format lists nicely
                                if value:
                                    lines.append(f"{key}:")
                                    for item in value:
                                        # Extract short name for model paths
                                        if isinstance(item, str) and '/' in item:
                                            item = item.split('/')[-1]
                                        lines.append(f"  • {item}")
                                else:
                                    lines.append(f"{key}: (none)")
                            else:
                                lines.append(f"{key}: {value}")
                    self._emit_to_client(client_id, SystemMessageEvent(
                        message="\n".join(lines) if lines else str(result),
                        style="info",
                    ))
            else:
                self._emit_to_client(client_id, SystemMessageEvent(
                    message=str(result),
                    style="info",
                ))

        elif isinstance(event, GetInstructionBudgetRequest):
            # Get instruction budget for the requested agent.  ``None`` (or
            # the legacy default ``"main"``) targets this server's main
            # agent, whose actual id may be a custom ``--agent <name>``.
            main_id = server.main_agent_id
            agent_id = event.agent_id or main_id

            if agent_id == main_id or agent_id == "main":
                # Main agent budget — Phase 3 §7c step 6.6.3.6:
                # forward to runner-side via the existing
                # ``session.snapshot_instruction_budget`` RPC (§7c
                # step 6.1 (2/3) at commit 1043bfde).
                snapshot = None
                rpc = getattr(server, "_runner_rpc", None)
                if rpc is not None:
                    snapshotter = getattr(
                        rpc,
                        "session_snapshot_instruction_budget_threadsafe",
                        None,
                    )
                    if callable(snapshotter):
                        try:
                            snapshot = snapshotter(timeout=5.0)
                        except Exception as exc:  # noqa: BLE001
                            logger.debug(
                                "snapshot_instruction_budget forward failed: %s",
                                exc,
                            )
                if snapshot is not None:
                    self._emit_to_client(client_id, InstructionBudgetEvent(
                        agent_id=agent_id,
                        budget_snapshot=snapshot,
                    ))
                else:
                    self._emit_to_client(client_id, ErrorEvent(
                        error="No instruction budget available for main agent",
                        error_type="BudgetNotFound",
                    ))
            else:
                # Subagent budget from SubagentPlugin
                subagent_plugin = server.registry.get_plugin("subagent") if server.registry else None
                if subagent_plugin and hasattr(subagent_plugin, '_active_sessions'):
                    session_info = subagent_plugin._active_sessions.get(agent_id)
                    if session_info:
                        subagent_session = session_info.get('session')
                        if subagent_session and hasattr(subagent_session, 'instruction_budget') and subagent_session.instruction_budget:
                            self._emit_to_client(client_id, InstructionBudgetEvent(
                                agent_id=agent_id,
                                budget_snapshot=subagent_session.instruction_budget.snapshot(),
                            ))
                        else:
                            self._emit_to_client(client_id, ErrorEvent(
                                error=f"No instruction budget available for agent {agent_id}",
                                error_type="BudgetNotFound",
                            ))
                    else:
                        self._emit_to_client(client_id, ErrorEvent(
                            error=f"Agent not found: {agent_id}",
                            error_type="AgentNotFound",
                        ))
                else:
                    self._emit_to_client(client_id, ErrorEvent(
                        error=f"Subagent plugin not available",
                        error_type="PluginNotFound",
                    ))

        # ─── SDK feature parity — session-primitive verbs ───────────────
        # Typed WS verbs over JaatoSession's public primitives so SDK
        # consumers can drive inject_prompt / replay_messages /
        # resolve_fork_point without going through the model loop.
        # See ``project_backlog_sdk_feature_parity.md``.

        elif isinstance(event, InjectPromptRequest):
            # Phase 3 §7c step 6.6.3.6: forward to runner-side via
            # the existing ``session.inject_prompt`` RPC (§7c step
            # 6.1 (3/3) at commit 14e57709) instead of reaching
            # into the daemon-side session.  The runner-side
            # handler validates source_type itself, but we pre-
            # validate daemon-side to surface the typed error
            # to clients without an RPC round-trip.
            from shared.message_queue import SourceType
            try:
                source_type = SourceType(event.source_type)
            except ValueError:
                self._emit_to_client(client_id, ErrorEvent(
                    error=(
                        f"Invalid source_type: {event.source_type!r}. "
                        f"Valid values: {[s.value for s in SourceType]}"
                    ),
                    error_type="ValidationError",
                ))
                return
            rpc = getattr(server, "_runner_rpc", None)
            if rpc is None:
                self._emit_to_client(client_id, ErrorEvent(
                    error="No active JaatoSession",
                    error_type="SessionError",
                ))
            else:
                forwarder = getattr(
                    rpc, "session_inject_prompt_threadsafe", None,
                )
                if callable(forwarder):
                    try:
                        forwarder(
                            event.text,
                            source_id=event.source_id,
                            source_type=source_type.value,
                            timeout=5.0,
                        )
                    except Exception as exc:  # noqa: BLE001
                        self._emit_to_client(client_id, ErrorEvent(
                            error=f"inject_prompt forward failed: {exc}",
                            error_type="SessionError",
                        ))
                else:
                    self._emit_to_client(client_id, ErrorEvent(
                        error="No active JaatoSession",
                        error_type="SessionError",
                    ))

        elif isinstance(event, ReplayMessagesRequest):
            # Phase 3 §7c step 6.6.3.6: forward to runner-side via
            # the new ``session.replay_messages`` RPC (§7c step
            # 6.6.3.4 at commit 24ed6c0f) instead of reaching
            # into the daemon-side session.  When ``event.messages``
            # is None, the runner's handler falls back to the
            # session's history — but the wrapper requires a
            # messages list, so we read history first via the
            # existing ``session.get_history`` RPC (§3.3c precursor).
            rpc = getattr(server, "_runner_rpc", None)
            if rpc is None:
                self._emit_to_client(client_id, ReplayMessagesResultEvent(
                    request_id=event.request_id,
                    error="No active JaatoSession",
                ))
            else:
                replayer = getattr(
                    rpc, "session_replay_messages_threadsafe", None,
                )
                if not callable(replayer):
                    self._emit_to_client(client_id, ReplayMessagesResultEvent(
                        request_id=event.request_id,
                        error="No active JaatoSession",
                    ))
                else:
                    # Resolve messages: use the request's if provided;
                    # else fall back to the runner-side session's
                    # current history via the existing get_history
                    # RPC.  Deserialize daemon-side first when the
                    # request supplied a list (the runner handler
                    # also does this; we deserialize early to surface
                    # malformed input as a typed daemon error).
                    if event.messages is not None:
                        from shared.plugins.session.serializer import deserialize_history
                        try:
                            messages = deserialize_history(event.messages)
                        except Exception as exc:  # noqa: BLE001
                            self._emit_to_client(client_id, ReplayMessagesResultEvent(
                                request_id=event.request_id,
                                error=f"deserialize failed: {exc}",
                            ))
                            return
                    else:
                        history_getter = getattr(
                            rpc, "session_get_history_threadsafe", None,
                        )
                        if not callable(history_getter):
                            self._emit_to_client(client_id, ReplayMessagesResultEvent(
                                request_id=event.request_id,
                                error="No active JaatoSession",
                            ))
                            return
                        try:
                            messages = history_getter(timeout=5.0)
                        except Exception as exc:  # noqa: BLE001
                            self._emit_to_client(client_id, ReplayMessagesResultEvent(
                                request_id=event.request_id,
                                error=f"get_history failed: {exc}",
                            ))
                            return

                    # Run in a worker thread — replay_messages
                    # blocks until the provider call completes.
                    def run_replay():
                        try:
                            response_text = replayer(
                                messages,
                                replay_timeout=event.timeout_seconds,
                                timeout=event.timeout_seconds + 60.0,
                            )
                            self._emit_to_client(client_id, ReplayMessagesResultEvent(
                                request_id=event.request_id,
                                response_text=response_text,
                            ))
                        except Exception as exc:
                            self._emit_to_client(client_id, ReplayMessagesResultEvent(
                                request_id=event.request_id,
                                error=f"{type(exc).__name__}: {exc}",
                            ))
                    threading.Thread(target=run_replay, daemon=True).start()

        elif isinstance(event, ResolveForkPointRequest):
            # Phase 3 §7c step 6.6.3.6: forward to runner-side
            # via the new ``session.resolve_fork_point`` RPC (§7c
            # step 6.6.3.5 at commit e4eddc0e).  The runner-side
            # handler defaults ``history`` to its own
            # ``session.get_history()`` when omitted (matches the
            # pre-§7c daemon-side pattern at line 4573).
            rpc = getattr(server, "_runner_rpc", None)
            if rpc is None:
                self._emit_to_client(client_id, ResolveForkPointResultEvent(
                    request_id=event.request_id,
                    fork_index=-1,
                    error="No active JaatoSession",
                ))
            else:
                resolver = getattr(
                    rpc, "session_resolve_fork_point_threadsafe", None,
                )
                if not callable(resolver):
                    self._emit_to_client(client_id, ResolveForkPointResultEvent(
                        request_id=event.request_id,
                        fork_index=-1,
                        error="No active JaatoSession",
                    ))
                else:
                    try:
                        fork_index = resolver(
                            after_message=event.after_message,
                            after_tool_call=event.after_tool_call,
                            after_timestamp=event.after_timestamp,
                            timeout=5.0,
                        )
                        self._emit_to_client(client_id, ResolveForkPointResultEvent(
                            request_id=event.request_id,
                            fork_index=fork_index,
                        ))
                    except Exception as exc:
                        self._emit_to_client(client_id, ResolveForkPointResultEvent(
                            request_id=event.request_id,
                            fork_index=-1,
                            error=f"{type(exc).__name__}: {exc}",
                        ))

        # ─── SDK feature parity — permission policy verbs ───────────────
        # Typed verbs replacing stringly-typed CommandRequest("permissions",
        # [...]) for SDK consumers.  CLI command path stays for users.

        elif isinstance(event, PermissionAddWhitelistRequest):
            permission_plugin = (
                server.registry.get_plugin("permission")
                if server.registry else None
            )
            if permission_plugin is None or permission_plugin._policy is None:
                self._emit_to_client(client_id, ErrorEvent(
                    error="Permission plugin not available",
                    error_type="PluginNotFound",
                ))
            else:
                if event.tools:
                    permission_plugin.add_whitelist_tools(event.tools)
                for pattern in event.patterns:
                    permission_plugin._policy.add_session_whitelist(pattern)

        elif isinstance(event, PermissionAddBlacklistRequest):
            permission_plugin = (
                server.registry.get_plugin("permission")
                if server.registry else None
            )
            if permission_plugin is None or permission_plugin._policy is None:
                self._emit_to_client(client_id, ErrorEvent(
                    error="Permission plugin not available",
                    error_type="PluginNotFound",
                ))
            else:
                # PermissionPolicy tracks blacklist rules in a single
                # session_blacklist set — tools and patterns live in
                # the same set (the matcher checks exact match first,
                # then pattern match).  Both add through the same call.
                for tool in event.tools:
                    permission_plugin._policy.add_session_blacklist(tool)
                for pattern in event.patterns:
                    permission_plugin._policy.add_session_blacklist(pattern)

        elif isinstance(event, PermissionRemoveRequest):
            permission_plugin = (
                server.registry.get_plugin("permission")
                if server.registry else None
            )
            if permission_plugin is None or permission_plugin._policy is None:
                self._emit_to_client(client_id, ErrorEvent(
                    error="Permission plugin not available",
                    error_type="PluginNotFound",
                ))
            elif event.target == "whitelist":
                # Direct set mutation — no remove method on the
                # policy, but the sets are public attributes.  discard
                # is a no-op for missing items so the call is idempotent.
                policy = permission_plugin._policy
                for tool in event.tools:
                    policy.whitelist_tools.discard(tool)
                    policy.session_whitelist.discard(tool)
                for pattern in event.patterns:
                    policy.session_whitelist.discard(pattern)
            elif event.target == "blacklist":
                policy = permission_plugin._policy
                for tool in event.tools:
                    policy.blacklist_tools.discard(tool)
                    policy.session_blacklist.discard(tool)
                for pattern in event.patterns:
                    policy.session_blacklist.discard(pattern)
            else:
                self._emit_to_client(client_id, ErrorEvent(
                    error=(
                        f"Invalid target: {event.target!r}. "
                        f"Valid values: 'whitelist', 'blacklist'"
                    ),
                    error_type="ValidationError",
                ))

        elif isinstance(event, PermissionClearRequest):
            permission_plugin = (
                server.registry.get_plugin("permission")
                if server.registry else None
            )
            if permission_plugin is None or permission_plugin._policy is None:
                self._emit_to_client(client_id, ErrorEvent(
                    error="Permission plugin not available",
                    error_type="PluginNotFound",
                ))
            else:
                policy = permission_plugin._policy
                # Clear targets the SESSION-level overrides only
                # (matches the semantics of `permissions clear`).
                # Base policy from permissions.json is unaffected.
                if event.target in ("whitelist", "all"):
                    policy.session_whitelist.clear()
                if event.target in ("blacklist", "all"):
                    policy.session_blacklist.clear()
                if event.target == "all":
                    policy.session_default_policy = None

        elif isinstance(event, PermissionSetDefaultRequest):
            permission_plugin = (
                server.registry.get_plugin("permission")
                if server.registry else None
            )
            if permission_plugin is None or permission_plugin._policy is None:
                self._emit_to_client(client_id, ErrorEvent(
                    error="Permission plugin not available",
                    error_type="PluginNotFound",
                ))
            else:
                try:
                    permission_plugin._policy.set_session_default_policy(
                        event.policy
                    )
                except ValueError as exc:
                    self._emit_to_client(client_id, ErrorEvent(
                        error=str(exc),
                        error_type="ValidationError",
                    ))

        elif isinstance(event, PermissionPolicySnapshotRequest):
            permission_plugin = (
                server.registry.get_plugin("permission")
                if server.registry else None
            )
            if permission_plugin is None or permission_plugin._policy is None:
                self._emit_to_client(client_id, PermissionPolicySnapshotEvent(
                    request_id=event.request_id,
                    default_policy="ask",
                ))
            else:
                policy = permission_plugin._policy
                self._emit_to_client(client_id, PermissionPolicySnapshotEvent(
                    request_id=event.request_id,
                    default_policy=policy.default_policy,
                    session_default_policy=policy.session_default_policy,
                    whitelist_tools=sorted(policy.whitelist_tools),
                    whitelist_patterns=list(policy.whitelist_patterns),
                    blacklist_tools=sorted(policy.blacklist_tools),
                    blacklist_patterns=list(policy.blacklist_patterns),
                    session_whitelist=sorted(policy.session_whitelist),
                    session_blacklist=sorted(policy.session_blacklist),
                ))

        else:
            self._emit_to_client(client_id, ErrorEvent(
                error=f"Unknown request type: {type(event).__name__}",
                error_type="RequestError",
            ))

    # =========================================================================
    # Cleanup
    # =========================================================================

    def save_all(self) -> int:
        """Save all dirty sessions to disk.

        Returns:
            Number of sessions saved.
        """
        saved = 0
        with self._lock:
            for session in self._sessions.values():
                if session.is_dirty:
                    if self._save_session(session):
                        saved += 1
        return saved

    # ------------------------------------------------------------------
    # Ephemeral sessions (Phase 3 — remote subagent delegation)
    # ------------------------------------------------------------------

    def run_ephemeral_session(self, *args: Any, **kwargs: Any) -> str:
        """Run an ephemeral session for a remote subagent (server 0.6.71+ entry).

        Wraps :meth:`_run_ephemeral_session_impl` in a fresh ContextVar
        context via :func:`shared.session_context.run_in_fresh_session_context`
        so the bootstrap is isolated from any ContextVar values
        inherited from the caller's task.  See the helper's docstring
        for the rationale.
        """
        from shared.session_context import run_in_fresh_session_context
        return run_in_fresh_session_context(
            self._run_ephemeral_session_impl, *args, **kwargs,
        )

    def _run_ephemeral_session_impl(
        self,
        profile_json: str,
        inline_config_json: str,
        prompt: str,
        agent_name: str,
        on_output: Any,
        workspace_path: Optional[str] = None,
    ) -> str:
        """Implementation of ephemeral session run, called via fresh-context wrap.

        See :meth:`run_ephemeral_session` for the isolation rationale.

        This is a blocking call intended to be run from a background thread
        (via ``asyncio.to_thread``).  The session is not persisted to disk
        and is not visible in the session list.

        When ``workspace_path`` is provided (Phase 5 workspace replication),
        the ephemeral session runs with that directory as its working
        directory, and ``JAATO_WORKSPACE_ROOT`` is set accordingly.  The
        server's plugin registry also gets the workspace path so plugins
        can discover project files.

        Args:
            profile_json: JSON-serialized SubagentProfile from the origin.
            inline_config_json: JSON-serialized inline config (empty if profile-based).
            prompt: The full prompt to send.
            agent_name: Display name for the subagent.
            on_output: Callback ``(source: str, text: str, mode: str) -> None``
                invoked for each output chunk.
            workspace_path: Optional workspace directory for the session.
                When provided, CWD and ``JAATO_WORKSPACE_ROOT`` are set to
                this path for the duration of the session.

        Returns:
            Summary string from the model's final response.
        """
        import json as _json
        import uuid
        from shared.plugins.subagent.config import build_inline_profile

        # Parse profile/config
        profile_data = _json.loads(profile_json) if profile_json else {}
        inline_data = _json.loads(inline_config_json) if inline_config_json else {}

        # Phase 3 §3.12 ephemeral migration: route through the unified
        # ``_construct_and_initialize_server`` sub-helper.  Compose the
        # ephemeral inputs (model/provider/plugins/system_instructions
        # /max_turns) into a single inline ``SubagentProfile`` so the
        # construction shape matches the IPC + disk-restore paths
        # (env_file-driven JaatoServer construction with a profile
        # override) rather than the pre-§3.12 broken
        # ``JaatoServer().initialize(model=, provider_name=, tools=)``
        # signature (the underlying ``JaatoServer.initialize`` takes
        # no kwargs — that call would have raised TypeError; the
        # ephemeral path was effectively unreachable in production).
        merged: Dict[str, Any] = {}
        for source in (profile_data, inline_data):
            for key, value in source.items():
                merged.setdefault(key, value)
        profile = build_inline_profile(
            merged,
            name="<ephemeral>",
            description=f"Ephemeral subagent: {agent_name}",
        )

        # Save and set workspace context if provided (Phase 5).
        # The env-driven workspace root + chdir survive untouched —
        # they're a separate concern from the bootstrap helper.
        prev_cwd = None
        prev_workspace_root = None
        if workspace_path:
            prev_cwd = os.getcwd()
            prev_workspace_root = os.environ.get("JAATO_WORKSPACE_ROOT")
            os.chdir(workspace_path)
            os.environ["JAATO_WORKSPACE_ROOT"] = workspace_path

        try:
            # Build the bootstrap envelope.  ``client_id=None`` because
            # ephemeral subagent fan-out has no client-driven apparmor
            # opt-in (per §3.12 spec; runner-sharing semantics for
            # default-share / opt-in-isolation come with §3.11 + the
            # seat-flip).  ``parent_runner_handle`` stays None for now
            # — Phase 3's spec for ephemeral migration adds the
            # parent-runner reference for shared-runner default once
            # the seat-flip lands.
            envelope = BootstrapEnvelope(
                session_id=f"ephemeral-{uuid.uuid4().hex[:12]}",
                workspace_path=workspace_path,
                name=agent_name or "<ephemeral>",
                description="Ephemeral subagent fan-out",
                client_id=None,
                profile=profile,
                agent_name=agent_name or "main",
            )
            server, _sandbox = self._construct_and_initialize_server(envelope)
            if server is None:
                # Init failure already emitted via the in-init sink;
                # surface a minimal hint to the caller so the
                # subagent fan-out doesn't silently swallow.
                return "Ephemeral session initialization failed."

            # Set workspace path on the registry so plugins discover it
            if workspace_path and hasattr(server, 'registry') and server.registry:
                server.registry.set_workspace_path(workspace_path)

            # Run the prompt and collect output
            collected_output = []

            def _output_callback(source: str, text: str, mode: str = "") -> None:
                collected_output.append(text)
                if on_output:
                    on_output(source, text, mode)

            response = server.send_message(prompt, on_output=_output_callback)

            # Cleanup
            server.shutdown()

            return response or "".join(collected_output) or "Task completed."

        finally:
            # Restore previous workspace context
            if prev_cwd is not None:
                os.chdir(prev_cwd)
            if prev_workspace_root is not None:
                os.environ["JAATO_WORKSPACE_ROOT"] = prev_workspace_root
            elif workspace_path:
                os.environ.pop("JAATO_WORKSPACE_ROOT", None)

    def shutdown(self) -> None:
        """Shutdown all sessions, saving to disk first."""
        logger.info("SessionManager shutting down...")

        # Stop all workspace monitors
        for sid in list(self._workspace_monitors):
            self._stop_workspace_monitor(sid)

        with self._lock:
            # Save all sessions
            for session in self._sessions.values():
                self._save_session(session)
                session.server.shutdown()

            self._sessions.clear()
            self._client_to_session.clear()

        # Close all session log handlers
        handler = get_session_handler()
        if handler:
            handler.close()

        self._session_plugin.shutdown()
        logger.info("SessionManager shutdown complete")
