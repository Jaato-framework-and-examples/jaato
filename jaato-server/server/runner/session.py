"""Runner-side session host (Phase 3 §3.3b).

Hosts the live :class:`JaatoSession` instance on the runner side.
Receives a :class:`SessionInitEnvelope` over the RPC channel at
runner startup, constructs the session, runs ``configure()``, and
exposes the resulting handle for downstream RPC dispatch.

Lifecycle relationship to §7c:

This module ships the host SHAPE — the bootstrap function, the
envelope-to-runtime wiring, the test scaffold.  As of §7c step 1
the daemon dispatches the ``session.bootstrap`` RPC unconditionally
(was previously gated on ``JAATO_RUNNER_HOSTS_SESSION`` — flag
removed in §7c step 1).  The daemon-side :class:`JaatoSession`
still instantiates in-process at this point; the runner-side host
coexists with it under the §7c rollout window.

Subsequent §7c steps flip the authoritative seat: the daemon's
session lifecycle moves to dispatching against the runner's host;
the in-process JaatoSession reference disappears from
:class:`JaatoServer`.

The bootstrap is testable in isolation via the ``runtime_factory``
constructor argument — tests inject a stub runtime that bypasses
the provider connect + plugin discovery dance.  Production callers
get the real :class:`JaatoRuntime`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Protocol, TYPE_CHECKING

from shared.session_envelope import SessionInitEnvelope


if TYPE_CHECKING:  # pragma: no cover — types only
    from shared.jaato_runtime import JaatoRuntime
    from shared.jaato_session import JaatoSession


logger = logging.getLogger(__name__)


class RuntimeFactory(Protocol):
    """Callable shape for constructing a runtime from an envelope.

    Production: defaults to :func:`_default_runtime_factory` which
    builds a real :class:`JaatoRuntime`.  Tests inject a stub that
    returns a pre-configured runtime (or a mock).
    """

    def __call__(self, envelope: SessionInitEnvelope) -> "JaatoRuntime": ...


# Sentinel marker used by ``bootstrap_session`` to signal "let
# the function pick the default factory".  We can't use ``None``
# because the test-injection path explicitly accepts ``None`` to
# mean "no runtime — use the test-only no-provider path".
_USE_DEFAULT = object()


@dataclass
class RunnerSessionHost:
    """Holds the runner-side session lifecycle artifacts.

    Attributes:
        envelope: The :class:`SessionInitEnvelope` the daemon sent
            at bootstrap.  Stored so downstream RPC handlers can
            inspect session metadata (session_id, profile_name,
            etc.) without re-walking the JaatoSession's internals.
        runtime: The :class:`JaatoRuntime` the host built.  ``None``
            in test-stub mode where the bootstrap deliberately
            skips runtime construction.
        session: The live :class:`JaatoSession`.  Populated after
            ``configure()`` returns successfully.  ``None`` until
            then (or in test-stub mode).
    """

    envelope: SessionInitEnvelope
    runtime: Optional["JaatoRuntime"] = None
    session: Optional["JaatoSession"] = None

    @property
    def session_id(self) -> str:
        return self.envelope.session_id

    @property
    def workspace_path(self) -> Optional[str]:
        return self.envelope.workspace_path

    @property
    def is_ready(self) -> bool:
        """True iff ``session`` has been constructed + configured."""
        return self.session is not None


class BootstrapError(RuntimeError):
    """Raised when ``bootstrap_session`` fails.

    Carries the failure stage so callers can decide whether to
    retry / log / propagate to ``SessionFailedEvent``.
    """

    def __init__(self, stage: str, message: str) -> None:
        super().__init__(f"runner-session bootstrap failed at {stage}: {message}")
        self.stage = stage
        self.message = message


def _default_runtime_factory(envelope: SessionInitEnvelope) -> "JaatoRuntime":
    """Build a :class:`JaatoRuntime` from the envelope.

    Production path; tests inject a stub.  Lives at module level
    rather than inline in ``bootstrap_session`` so the import-time
    cost of pulling in the heavy ``JaatoRuntime`` is paid only when
    actually needed (a runner that never receives a
    ``session.bootstrap`` RPC skips this).
    """
    # Import inside the factory so the runner's import surface
    # doesn't force the JaatoRuntime import (and its provider plugin
    # transitive imports) at module load.  Runners that receive a
    # session.bootstrap RPC pay this once per process; the cost is
    # amortized against the spawn cost.
    from pathlib import Path
    from shared.jaato_runtime import JaatoRuntime

    workspace_path = (
        Path(envelope.workspace_path)
        if envelope.workspace_path
        else None
    )
    return JaatoRuntime(
        provider_name=envelope.provider_name or "anthropic",
        workspace_path=workspace_path,
        config_root=envelope.config_root,
    )


def _configure_runtime_plugins(
    runtime: "JaatoRuntime", envelope: SessionInitEnvelope,
) -> None:
    """Mirror daemon-side ``_run_load_plugins`` on the runner.

    Phase 3 post-Step-7 Path D.

    Daemon-side ``server/core.py:1615-1777`` constructs a
    :class:`PluginRegistry`, discovers plugins (no tier filter — the
    daemon hosts everything), expands per-plugin configs with
    ``workspace_path`` + ``session_id``, calls ``expose_all``,
    broadcasts ``set_workspace_path`` / ``set_config_root``,
    constructs a :class:`PermissionPlugin`, and finally calls
    ``runtime.configure_plugins(registry, permission_plugin,
    ledger)``.  All nine steps must happen runner-side before
    ``runtime.create_session(...)`` because ``create_session`` guards
    on both ``_connected`` (Path C) AND ``_registry`` (this Path D).

    Differences from daemon-side:

    1. ``registry.discover(tier_filter="runner")`` — runner-tier
       plugins only (per §3.3.5).  Daemon-tier plugins (auth, gc_*,
       cache_*, session, background) must NOT load runner-side; they
       live on the daemon and any session.* RPC that needs them
       crosses the wire.
    2. ``ledger=None`` passed to ``configure_plugins`` — token
       accounting is daemon-tier per §4.2.
    3. No ``on_progress`` callback on ``expose_all`` — runner has no
       client event sink for per-plugin init progress (the daemon's
       ``_emit_init_progress`` doesn't apply).
    4. ``permission_plugin`` initialized with the daemon-side default
       policy (``defaultPolicy: "ask"``).  Profile-supplied
       ``plugin_configs["permission"]`` overrides aren't currently in
       the envelope (filed: backlog §3.3c.X).

    Raises:
        Any exception from registry discovery, plugin
        initialization, or ``configure_plugins`` propagates to the
        caller, which wraps it as ``BootstrapError("plugins", ...)``.
    """
    from shared.plugins.permission.plugin import PermissionPlugin
    from shared.plugins.registry import PluginRegistry

    # Step 1-2: construct + discover (runner-tier only).
    registry = PluginRegistry(model_name=envelope.model_name)
    registry.discover(tier_filter="runner")

    # Step 3: assemble plugin_configs.  Defaults mirror daemon-side
    # `core.py:1621-1675` for the 6 runner-tier entries.  Auth plugin
    # entries are skipped — they're daemon-tier and the tier filter
    # already excluded them from the registry.  Envelope-supplied
    # per-plugin configs (resolved daemon-side from the profile) layer
    # on top: same precedence as daemon-side which merges profile
    # overrides into the default dict.
    workspace_path = envelope.workspace_path
    session_id = envelope.session_id
    plugin_configs: dict = {
        "todo": {
            "reporter_type": "memory",
            "storage_type": "memory",
        },
        "references": {
            "channel_type": "queue",
            "workspace_path": workspace_path,
        },
        "clarification": {
            "channel_type": "queue",
        },
        "lsp": {
            "workspace_path": workspace_path,
            "session_id": session_id,
        },
        "mcp": {
            "workspace_path": workspace_path,
            "session_id": session_id,
        },
        "file_edit": {
            "session_id": session_id,
        },
        "waypoint": {
            "session_id": session_id,
        },
        "sandbox_manager": {
            "session_id": session_id,
        },
    }
    # Merge envelope-supplied per-plugin configs from
    # build_session_envelope (runner_spawn.py:200-208).
    for entry in envelope.plugins:
        name = entry.get("name")
        cfg = entry.get("config")
        if isinstance(name, str) and name and isinstance(cfg, dict) and cfg:
            existing = plugin_configs.get(name, {})
            plugin_configs[name] = {**existing, **dict(cfg)}

    # Step 4: expose_all — initializes each plugin.  No on_progress
    # callback runner-side.
    registry.expose_all(plugin_configs)

    # Step 5 (`self.todo_plugin = ...`): N/A runner-side — no
    # runner-resident code path needs the cached reference.

    # Step 6-7: workspace + config_root broadcast.
    if workspace_path:
        registry.set_workspace_path(workspace_path)
    if envelope.config_root:
        registry.set_config_root(envelope.config_root)

    # Step 8: permission plugin.  Default policy mirrors daemon-side
    # `core.py:1712-1721` baseline; profile overrides not yet
    # propagated through the envelope (backlog §3.3c.X).
    permission_plugin = PermissionPlugin()
    permission_plugin.initialize({
        "channel_type": "queue",
        "channel_config": {"use_colors": False},
        "workspace_path": workspace_path,
        "policy": {
            "defaultPolicy": "ask",
            "whitelist": {"tools": [], "patterns": []},
            "blacklist": {"tools": [], "patterns": []},
        },
    })

    # Step 9: wire onto the runtime.  ``ledger=None`` because token
    # accounting is daemon-tier per §4.2.
    runtime.configure_plugins(registry, permission_plugin, None)

    logger.info(
        "runner-session bootstrap: configured %d plugins runner-tier "
        "(session_id=%s workspace=%s)",
        len(registry._exposed), session_id, workspace_path or "(none)",
    )


def bootstrap_session(
    envelope: SessionInitEnvelope,
    *,
    runtime_factory: Any = _USE_DEFAULT,
) -> RunnerSessionHost:
    """Construct a runner-side session from a daemon-supplied envelope.

    Phase 3 §3.3b.

    Args:
        envelope: The bootstrap payload the daemon sent over RPC.
            Carries session_id, workspace_path, profile_name,
            provider_name, model_name, plugins (resolved), system
            instructions, agent_id, gc, completion_payload_schema,
            agent_params, etc.
        runtime_factory: Optional override for runtime construction.
            Defaults to :func:`_default_runtime_factory` which
            builds a real :class:`JaatoRuntime`.  Tests inject a
            stub.  Pass ``None`` explicitly for the test-only
            "skip runtime construction entirely" path (the host
            is returned with ``runtime=None`` + ``session=None``).

    Returns:
        A :class:`RunnerSessionHost` wrapping the bootstrap
        artifacts.  When the envelope was valid AND a runtime
        was constructed, the host's ``is_ready`` is True and
        ``session`` carries the live JaatoSession.

    Raises:
        BootstrapError: when the envelope fails validation or any
            stage of construction throws.  ``BootstrapError.stage``
            identifies where the failure occurred:
            ``"validate"``, ``"runtime"``, ``"connect"`` (Path C),
            ``"plugins"`` (Path D), ``"configure"``, ``"unknown"``.

    Notes on §3.3b vs §3.3c scope:

    This module's bootstrap goes through ``runtime.create_session``
    via the runtime factory.  The runner's RPC dispatch routes
    session.* calls against ``host.session`` (see
    ``server/runner/rpc.py`` — ``_session_host`` field).  As of
    §7c step 1 the daemon dispatches the ``session.bootstrap`` RPC
    unconditionally; the daemon-side ``JaatoSession`` still
    coexists and is authoritative until the seat-flip steps land.
    """
    # ---- 1. Validate ----
    try:
        _validate_envelope(envelope)
    except ValueError as exc:
        logger.error("runner-session bootstrap: validation failed: %s", exc)
        raise BootstrapError("validate", str(exc)) from exc

    # ---- 2. Optionally construct the runtime ----
    if runtime_factory is None:
        # Test-only path: caller explicitly wants no runtime.  Useful
        # for tests that inspect the envelope-validation half without
        # depending on JaatoRuntime's import surface or provider
        # connect.
        logger.info(
            "runner-session bootstrap: skipping runtime construction "
            "(runtime_factory=None — test path)"
        )
        return RunnerSessionHost(envelope=envelope, runtime=None, session=None)

    factory = (
        _default_runtime_factory if runtime_factory is _USE_DEFAULT
        else runtime_factory
    )

    try:
        runtime = factory(envelope)
    except Exception as exc:  # noqa: BLE001 — boundary surface
        logger.exception(
            "runner-session bootstrap: runtime construction crashed",
        )
        raise BootstrapError("runtime", str(exc)) from exc

    # ---- 2b. Connect the runtime ----
    # Phase 3 post-Step-7 Path C: ``JaatoRuntime.create_session``
    # guards on ``self._connected`` (jaato_runtime.py:964) and
    # raises ``RuntimeError("Runtime not connected. Call connect()
    # first.")`` if invoked on a fresh runtime.  The envelope now
    # carries ``project`` + ``location`` (added alongside this fix)
    # so the runner can self-connect without daemon involvement.
    # Non-Vertex providers tolerate empty strings; Vertex AI uses
    # the values daemon-side read from ``PROJECT_ID`` / ``LOCATION``
    # env.
    #
    # ``connect()`` is idempotent on the project/location side (just
    # sets ``_project`` / ``_location`` / ``_provider_config`` /
    # ``_connected = True``); no provider-network call here.  The
    # real network connect happens inside ``create_session`` →
    # provider plugin ``initialize()``.
    try:
        if hasattr(runtime, "connect") and not getattr(
            runtime, "is_connected", False,
        ):
            runtime.connect(envelope.project, envelope.location)
    except Exception as exc:  # noqa: BLE001 — boundary surface
        logger.exception(
            "runner-session bootstrap: runtime.connect crashed",
        )
        raise BootstrapError("connect", str(exc)) from exc

    # ---- 2c. Configure plugins on the runtime ----
    # Phase 3 post-Step-7 Path D: ``runtime.create_session`` guards
    # on ``self._registry`` (jaato_runtime.py:966-967) in addition to
    # ``self._connected``.  Path C closed the _connected guard; Path
    # D closes _registry by mirroring daemon-side
    # ``_run_load_plugins`` (core.py:1615-1728) + the post-threadpool
    # ``runtime.configure_plugins`` call (core.py:1773-1777) here.
    #
    # Skipped when the test path injects an already-configured
    # runtime (``runtime._registry`` truthy on entry — stub runtimes
    # in tests pre-wire their own minimal registry).  The
    # idempotent guard mirrors the Path C connect guard.
    if getattr(runtime, "_registry", None) is None:
        try:
            _configure_runtime_plugins(runtime, envelope)
        except Exception as exc:  # noqa: BLE001 — boundary surface
            logger.exception(
                "runner-session bootstrap: plugin configuration crashed",
            )
            raise BootstrapError("plugins", str(exc)) from exc

    # ---- 3. Construct + configure the session ----
    try:
        session = _build_session(runtime, envelope)
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "runner-session bootstrap: session construct/configure crashed",
        )
        raise BootstrapError("configure", str(exc)) from exc

    logger.info(
        "runner-session bootstrap ready: session_id=%s profile=%s "
        "model=%s plugins=%d",
        envelope.session_id, envelope.profile_name, envelope.model_name,
        len(envelope.plugins),
    )
    return RunnerSessionHost(envelope=envelope, runtime=runtime, session=session)


def _validate_envelope(envelope: SessionInitEnvelope) -> None:
    """Stage-1 envelope checks.

    Catches obvious misconstructions before we sink import + provider
    cost into runtime/session construction.  Raises ``ValueError``
    on any failure; ``bootstrap_session`` translates that into
    ``BootstrapError("validate", ...)``.
    """
    if not envelope.session_id:
        raise ValueError("envelope.session_id is empty")
    if not envelope.model_name:
        raise ValueError("envelope.model_name is empty")
    if not envelope.provider_name:
        raise ValueError("envelope.provider_name is empty")
    # workspace_path may be None (headless / no-workspace sessions
    # are legitimate per parent design §3); profile_name may be None
    # (inline-spec sessions don't carry a profile name).


def _build_session(
    runtime: "JaatoRuntime", envelope: SessionInitEnvelope,
) -> "JaatoSession":
    """Stage-3 session construction.

    For Phase 3 §3.3b, calls ``runtime.create_session(...)`` with
    the envelope-derived args.  Provider-connect + plugin discovery
    happen inside ``create_session`` per the runtime's existing
    contract; failures bubble up as ``BootstrapError("configure",
    ...)`` from the caller.

    Plugin spec extraction: each entry in ``envelope.plugins`` is a
    dict ``{"name": "...", "preload": bool, "config": dict?}``.
    The plugin-list (just names) feeds ``tools=...`` so the runtime
    exposes them; the per-plugin configs feed
    ``plugin_configs=...``; the preload set feeds
    ``preloaded_plugins=...``.
    """
    tool_names: List[str] = []
    plugin_configs: dict = {}
    preloaded: set = set()
    for entry in envelope.plugins:
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"plugin entry missing 'name': {entry!r}")
        tool_names.append(name)
        cfg = entry.get("config")
        if isinstance(cfg, dict) and cfg:
            plugin_configs[name] = dict(cfg)
        if entry.get("preload"):
            preloaded.add(name)

    return runtime.create_session(
        model=envelope.model_name,
        tools=tool_names or None,
        system_instructions=envelope.system_instructions,
        plugin_configs=plugin_configs or None,
        provider_name=envelope.provider_name or None,
        preloaded_plugins=preloaded or None,
        completion_payload_schema=envelope.completion_payload_schema,
        agent_params=envelope.agent_params or None,
        completion_artifacts=(
            envelope.completion_artifacts or None
        ),
    )
