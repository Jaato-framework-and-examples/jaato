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
import os
from dataclasses import dataclass
from typing import (Any, Callable, Dict, List, Optional, Protocol, Tuple,
                    TYPE_CHECKING)

from shared.apparmor_label import (
    AppArmorLabel,
    COMPLAIN_ENV_VAR,
    profile_name_ignoring_mode,
)
from shared.session_envelope import SessionInitEnvelope


if TYPE_CHECKING:  # pragma: no cover — types only
    from shared.jaato_runtime import JaatoRuntime
    from shared.jaato_session import JaatoSession
    from shared.plugins.registry import PluginRegistry


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

    The envelope's ``plugin_configs.telemetry`` block is forwarded to the
    runtime because telemetry is runtime-scoped and is therefore built
    here, before any session exists — the profile block reaches it on this
    argument or not at all (#858).  Step 8's per-plugin config merge cannot
    serve it: telemetry is not a registry plugin.
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
        telemetry_config=(envelope.plugin_configs or {}).get("telemetry"),
    )


def _set_runner_workspace_context(
    workspace_path: Optional[str], config_root: Optional[str]
) -> None:
    """Seed the runner process's workspace/config root for runner-tier plugins.

    Runner-tier path plugins (filesystem_query / file_edit / cli / notebook) read
    the session workspace from ``session_context.get_workspace_root()`` (a
    ContextVar, with an ``os.environ['JAATO_WORKSPACE_ROOT']`` fallback) when
    their per-plugin config carries no explicit root.  The runner — unlike the
    daemon's ``JaatoServer._in_workspace`` (core.py:913-948) — historically set
    NEITHER, so a pool-/cold-served session initialized with ``workspace=none``
    and path tools were denied, UNLESS ``JAATO_WORKSPACE_ROOT`` happened to be
    exported globally (the env mask that hid this build-wide regression).

    Both surfaces are seeded: the ContextVar (read + cached by each plugin's
    ``initialize()`` in the bootstrap context) and ``os.environ`` (for
    cross-thread / ``os.environ``-reading consumers).  There is NO reset — a
    runner process serves exactly one session for its whole lifetime, so the
    root is constant.  Must be called BEFORE ``expose_all``.
    """
    from shared.session_context import set_workspace_root, set_config_root
    if workspace_path:
        set_workspace_root(workspace_path)
        os.environ["JAATO_WORKSPACE_ROOT"] = workspace_path
    if config_root:
        set_config_root(config_root)
        os.environ["JAATO_CONFIG_ROOT"] = config_root


# Sentinel plugin name routed daemon-side by the daemon.plugin_execute handler
# to the session registry's client-tool proxy executor.
_CLIENT_TOOL_PLUGIN = "__client_tools__"


def _make_client_tool_forwarder(registry, tool_name):
    """Runner-side executor for a client-provided ("host") tool.

    Forwards the call to the daemon via ``daemon.plugin_execute`` (under the
    ``__client_tools__`` sentinel plugin name), which the daemon-side handler
    routes to the session registry's existing proxy executor →
    ``ToolExecuteRequestEvent`` → the ws client → ``ToolExecuteResultEvent``.
    ``runner_rpc_client`` is resolved per-call (the registry attribute is set
    runner-side after bootstrap completes, before any tool call).
    """
    def _executor(args):
        rpc = getattr(registry, "runner_rpc_client", None)
        if rpc is None:
            return {"error": f"client tool '{tool_name}': runner->daemon channel "
                             "unavailable (bootstrap incomplete?)"}
        return rpc.daemon_plugin_execute(
            plugin_name=_CLIENT_TOOL_PLUGIN, tool_name=tool_name, args=args)
    return _executor


def _register_client_tools_on_runner(registry, client_tools) -> None:
    """Register client-provided tool SCHEMAS on the runner registry as core
    tools (so the model sees them in list_tools), each with a daemon-forwarding
    executor.  Entries are schema dicts (name/description/parameters/category)
    ferried in ``envelope.client_tools``.
    """
    from jaato_sdk.plugins.model_provider.types import ToolSchema, DISCOVERABILITY_EAGER
    for ct in client_tools:
        name = ct.get("name")
        if not name:
            continue
        schema = ToolSchema(
            name=name,
            description=ct.get("description", ""),
            parameters=ct.get("parameters", {}),
            category=ct.get("category") or None,
            # Client-provided tools default to 'core' (EAGER): the client
            # explicitly provided them, so the model should see them in its
            # initial schema and use them on INTENT — not only after a
            # list_tools discovery or a persona that names them.  Honor an
            # explicit "discoverability" from the dict to opt a tool back to
            # "discoverable" (a self-extending agent with many tools that
            # would bloat the eager surface).
            discoverability=ct.get("discoverability", DISCOVERABILITY_EAGER),
        )
        registry.register_core_tool(
            schema, _make_client_tool_forwarder(registry, name),
            auto_approved=True,
        )


def _adopt_then_discover(
    registry: "PluginRegistry",
    envelope: SessionInitEnvelope,
    plugin_configs: Dict[str, Any],
) -> List[str]:
    """Adopt carried-over plugins, then discover the rest (#890).

    The ORDER is the mechanism, which is why the two calls live together in
    one named step.  Both of ``discover()``'s paths skip a name that is
    already registered, so adopting first means discovery never constructs a
    rival to a warm instance; adopting after would leave the carried plugin
    registered nowhere and the leak intact.

    A cold slot, a standalone session, a slot recycled onto another
    workspace, or a profile that changed the plugin's config all adopt
    nothing — in which case this is exactly the pre-#890 ``discover()`` call
    and the full plugin set is built fresh.

    Args:
        registry: The new session's registry, freshly constructed.
        envelope: The arriving session envelope, whose cascade / workspace
            identity gates reuse.
        plugin_configs: The effective per-plugin config map this bootstrap
            will pass to ``expose_all`` — compared against the config each
            parked instance was initialized under.

    Returns:
        Names adopted, for the caller's logging.
    """
    from . import slot_plugins

    adopted = slot_plugins.adopt_into(registry, envelope, plugin_configs)
    registry.discover(tier_filter="runner")
    if adopted:
        logger.info(
            "runner-session bootstrap: reusing warm plugin instance(s) %s "
            "carried over from the previous session on this slot — no "
            "re-initialize, no re-connect",
            ", ".join(adopted),
        )
    return adopted


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

    0. Slot-scoped adoption runs between construction and discovery.
       A pool slot serving several stages of one cascade parks the
       plugin instances that declare
       :data:`~jaato_sdk.plugins.base.TRAIT_SLOT_SCOPED` at the
       previous ``session.end``; this function adopts them into the new
       registry so discovery skips those names and the warm resources
       (a connected language server, a cascade's plan map) are reused
       rather than rebuilt beside an abandoned copy.  See
       :mod:`server.runner.slot_plugins` and #890.  The daemon has no
       counterpart — its registry is per-``JaatoServer``, which is
       per-session.
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
    4. ``permission_plugin`` is constructed here rather than taken
       from the registry, seeded with the same daemon-side default
       policy (``defaultPolicy: "ask"``) and then updated from
       ``envelope.plugin_configs["permission"]`` — which Phase 4 §C
       put on the wire (schema v2), closing backlog §3.3c.X.  The
       profile's permission block therefore applies whether or not
       ``permission`` appears in ``profile.plugins``; see Step 8.

    Bootstrap timing (2026-05-14): wraps every step with a sibling
    :class:`BootstrapTimer` instance.  When
    ``JAATO_BOOTSTRAP_TIMING=true`` is set in the runner subprocess'
    environment (inherited from the daemon's launch env via fork
    from the template), the per-stage + per-plugin breakdown lands
    in the runner-side log.  Closes the gap documented in
    ``project_backlog_runner_side_bootstrap_timer``: the daemon-side
    timer in ``server/core.py`` measures only the ~2.3s of
    ``JaatoServer.initialize()``; the ~10.4s of runner-tier plugin
    configure happens in this function and was previously unmeasured.

    Raises:
        Any exception from registry discovery, plugin
        initialization, or ``configure_plugins`` propagates to the
        caller, which wraps it as ``BootstrapError("plugins", ...)``.
    """
    from shared.bootstrap_timing import BootstrapTimer
    from shared.plugins.permission.plugin import PermissionPlugin
    from shared.plugins.registry import PluginRegistry

    timer = BootstrapTimer()
    timing_enabled = os.environ.get(
        "JAATO_BOOTSTRAP_TIMING", "",
    ).lower() in ("1", "true", "yes")

    # Step 1: construct.  Discovery no longer happens here — it moved
    # below Step 3, because #890's slot-scoped adoption has to run
    # between the two: both discovery paths skip a name that is already
    # registered, and that skip is what stops discovery constructing a
    # rival to a carried-over warm instance.  Adoption in turn needs the
    # plugin_configs map (it compares each parked instance's config
    # against the arriving session's).  Hoisting Step 3 costs nothing —
    # that block never touches the registry.
    registry = PluginRegistry(model_name=envelope.model_name)

    # Step 3 (hoisted, see Step 1): assemble plugin_configs.
    # Defaults mirror daemon-side
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
    # Phase 4 §C: merge profile.plugin_configs into the runner-side
    # per-plugin init dict.  Reads from envelope.plugin_configs (the
    # full top-level map) instead of the per-entry plugins[i].config
    # that pre-§C only carried configs for plugins named in
    # profile.plugins.  This is what lets auto-loaded plugins like
    # ``permission`` (loaded below by name even when not in
    # profile.plugins) pick up their profile overrides.
    for name, cfg in envelope.plugin_configs.items():
        if isinstance(name, str) and name and isinstance(cfg, dict) and cfg:
            existing = plugin_configs.get(name, {})
            plugin_configs[name] = {**existing, **dict(cfg)}

    # Step 2 (runs here, see Step 1): adopt the slot-scoped plugins the
    # previous session on this pool slot parked, then discover the rest.
    with timer.stage("discover"):
        _adopt_then_discover(registry, envelope, plugin_configs)

    # Server 0.6.129+ structural fix: register framework-known values
    # on the registry BEFORE ``expose_all`` fires so each plugin's
    # ``initialize`` sees them in config.  Mirrors the daemon-side
    # reorder at ``core.py`` (same change shipped in the same PR).
    # Pre-fix the runner-side had the same wire-gap: file_edit
    # initialized BEFORE the broadcast lit up workspace/config_root,
    # so its ``_detect_*`` fallbacks fired the cosmetic WARN every
    # bootstrap.  See ``shared/plugins/registry.py:_augment_plugin_config``.
    if workspace_path:
        registry.set_workspace_path(workspace_path)
    if envelope.config_root:
        registry.set_config_root(envelope.config_root)
    # Seed the runner PROCESS's workspace/config context (session_context
    # ContextVar + os.environ fallback) so runner-tier path plugins resolve the
    # session workspace at ``initialize()`` — BEFORE ``expose_all`` below.  See
    # the helper for why the runner must do this and why both surfaces.
    _set_runner_workspace_context(workspace_path, envelope.config_root)
    if session_id:
        registry.set_session_id(session_id)
    if envelope.agent_id:
        registry.set_agent_name(envelope.agent_id)

    # Step 4: expose_all — initializes each plugin.  No on_progress
    # callback runner-side.  Initializes ALL discovered runner-tier
    # plugins; tool exposure to the model is gated separately at the
    # ``runtime.create_session(plugins=...)`` layer.
    #
    # **PR-112 / option C disabled at the call site (2026-05-15).**
    # PR-112 introduced ``requested_plugins=`` gating on this call
    # so plugins not in ``profile.plugins`` would skip
    # ``initialize()`` (saving ~7.6s of references-plugin
    # SentenceTransformer model load for sessions that don't use
    # references).  The gating broke peer's cascade in a way we
    # couldn't fully diagnose within the available debug budget:
    # discovery agent received the correct tool list (including
    # ``signal_completion`` with its typed payload schema) but
    # consistently produced prose-only responses for three turns,
    # never invoking any function call.  Two interventions
    # (PR-113's introspection-guidance reword and a controlled
    # script that reproduced the working tool list) both
    # falsified the working hypotheses.  Rolling back the gate
    # restores the full plugin init set so peer's cascade
    # unblocks; registry-side framework support
    # (:meth:`PluginRegistry.expose_all`'s ``requested_plugins``
    # kwarg) stays in place so option C can be re-enabled once
    # the actual failure mode is understood.
    with timer.stage("expose_all"):
        registry.expose_all(plugin_configs)

    # Step 5 (`self.todo_plugin = ...`): N/A runner-side — no
    # runner-resident code path needs the cached reference.

    # Step 6-7: workspace + config_root broadcast.  Post-init refresh
    # — idempotent given the pre-init injection above, but still
    # needed to fire the ``set_workspace_path`` / ``set_config_root``
    # hooks on plugins that update derived state (e.g.
    # ``file_edit._reinit_backup_manager`` per PR-144) and to
    # propagate any mid-session changes.
    with timer.stage("set_workspace_path"):
        if workspace_path:
            registry.set_workspace_path(workspace_path)
        if envelope.config_root:
            registry.set_config_root(envelope.config_root)

    # Step 7.5: client-provided ("host") tools.  Register the schemas the client
    # registered BEFORE session.new (carried in envelope.client_tools) on the
    # RUNNER registry as core tools, so the model SEES them in list_tools.
    # Execution forwards back to the daemon's existing proxy executor via
    # daemon.plugin_execute (sentinel plugin name) → ToolExecuteRequestEvent →
    # ws client.  Pre-fix these registered only on the daemon registry, so the
    # runner-tier model was blind (the #344-sibling daemon-vs-runner split).
    if envelope.client_tools:
        _register_client_tools_on_runner(registry, envelope.client_tools)

    # Step 8: permission plugin.  Default policy mirrors daemon-side
    # `core.py:1778-1794` baseline; profile-supplied
    # ``plugin_configs.permission`` overrides are now applied via the
    # Phase 4 §C envelope.plugin_configs field (schema v2).  Shallow
    # merge: top-level keys from the profile (most commonly ``policy``)
    # replace defaults.  Mirrors daemon-side ``permission_init_config.update(...)``.
    with timer.stage("permission_init"):
        permission_init_config: Dict[str, Any] = {
            "channel_type": "queue",
            "channel_config": {"use_colors": False},
            "workspace_path": workspace_path,
            "policy": {
                "defaultPolicy": "ask",
                "whitelist": {"tools": [], "patterns": []},
                "blacklist": {"tools": [], "patterns": []},
            },
        }
        profile_perm_config = envelope.plugin_configs.get("permission")
        if profile_perm_config:
            permission_init_config.update(profile_perm_config)
        permission_plugin = PermissionPlugin()
        permission_plugin.initialize(permission_init_config)

    # Step 9: wire onto the runtime.  ``ledger=None`` because token
    # accounting is daemon-tier per §4.2.
    with timer.stage("configure_plugins"):
        runtime.configure_plugins(registry, permission_plugin, None)

    # Bootstrap timing report (when enabled).  Mirrors daemon-side
    # `server/core.py:2213-2245` format so operators see the same
    # shape across both halves of bootstrap.  The per-plugin
    # breakdown reuses `registry.get_bootstrap_timings()` — the
    # registry has been tracking import_ms / create_ms / init_ms
    # since commit eb0ca640, just nobody read those numbers
    # runner-side.
    timer.finish()
    if timing_enabled:
        import io as _io
        _buf = _io.StringIO()
        _buf.write("Runner-side bootstrap timing report:\n")
        timer.report(file=_buf)
        plugin_timings = registry.get_bootstrap_timings()
        if plugin_timings:
            _buf.write("\n  PER-PLUGIN BREAKDOWN (sorted by total time):\n")
            _buf.write("  " + "-" * 68 + "\n")
            sorted_plugins = sorted(
                plugin_timings.items(),
                key=lambda x: x[1].get("total_ms", 0),
                reverse=True,
            )
            for pname, ptiming in sorted_plugins:
                total = ptiming.get("total_ms", 0)
                if total < 1.0:
                    continue
                imp = ptiming.get("import_ms", 0)
                create = ptiming.get("create_ms", 0)
                init = ptiming.get("init_ms", 0)
                _buf.write(
                    f"    {pname:<30} total={total:>7.1f}ms  "
                    f"import={imp:>6.1f}  create={create:>6.1f}  init={init:>7.1f}\n"
                )
            _buf.write("\n")
        logger.info("%s", _buf.getvalue())
    else:
        logger.debug(
            "Runner-side bootstrap completed in %.0f ms",
            timer.total_elapsed * 1000,
        )

    logger.info(
        "runner-session bootstrap: configured %d plugins runner-tier "
        "(session_id=%s workspace=%s)",
        len(registry._exposed), session_id, workspace_path or "(none)",
    )


#: The slot's environment as inherited from the template, captured before
#: the FIRST session's ``session_env`` is applied.  A reused slot is restored
#: to this between sessions so one session's resolved secrets cannot reach
#: the next.  ``None`` until the first apply.
_PRISTINE_ENVIRON: Optional[Dict[str, str]] = None


def _apply_envelope_session_env(envelope: SessionInitEnvelope) -> Dict[str, str]:
    """Apply ``envelope.session_env`` to the runner's ``os.environ``.

    PR #91 Y fix: the daemon (unconfined) resolves workspace ``.env``
    + profile.env + env_overrides — including secret URIs via local
    :class:`SecretResolver` plugins — and ships the fully-resolved
    dict via the envelope's ``session_env`` field.  The runner
    applies the dict verbatim, **never** running its own resolver
    discovery (which would fail under AppArmor confinement: the
    runner can't exec ``pass`` / ``vault`` / etc.).

    Returns:
        The dict that was applied (copy of ``envelope.session_env``),
        so the caller can attach it to ``JaatoSession._session_env``
        for the :meth:`JaatoSession.get_session_env` accessor.

    History: an earlier iteration (Shape 3 PR 1, PR #91) had the
    runner read ``<workspace>/.env`` directly + call
    ``_resolve_secret_uri`` runner-side.  That broke under AppArmor
    confinement when ``PassResolver.__init__`` shelled to
    ``pass version`` and got exit 126 (AppArmor-blocked exec) →
    resolver registration failed → ``pass://`` URIs survived as
    literals into ``os.environ`` → provider 401s.  The audited Y
    shape (this method) puts secret resolution back where the
    process is unconfined.
    """
    # RESTORE FIRST, THEN APPLY.  This ran set-only, and a pool slot serves
    # MORE THAN ONE SESSION -- so a key session A declared and session B does
    # not was left exactly as A left it.  ``session_env`` carries values
    # resolved from ``pass://`` / ``vault://`` (that resolution exists because
    # the literal URI reaching a runner produced provider 401s), so what
    # persisted was A's DECODED credentials, in the environment of a runner
    # now serving B, readable by any tool B runs.
    #
    # Absent-versus-empty again: "B does not mention this key" was read as
    # "leave it alone" when it means "B must not have it".
    #
    # The snapshot is taken ONCE, before the first session's env is applied,
    # so it is the slot's pristine inherited environment -- not session A's.
    # Taking it per-session would snapshot A's leak and faithfully restore it.
    return apply_session_env(envelope.session_env)


def apply_session_env(session_env: Optional[Dict[str, str]]) -> Dict[str, str]:
    """Restore the slot's pristine environment, then lay *session_env* over it.

    The one writer of the runner's session-scoped environment, shared by
    bootstrap (:func:`_apply_envelope_session_env`) and by the
    ``session.reload_env`` RPC, so the two cannot disagree about what a
    re-application means: it is a REPLACEMENT of the previous session env,
    never a merge onto it.  A key the previous dict set and the new one
    does not is gone afterwards, which is what lets a reload retract a
    credential as well as supply one.

    Args:
        session_env: The daemon-resolved dict (workspace ``.env`` + profile
            ``env:`` + overrides, secret URIs already decoded).  ``None`` or
            empty restores the pristine environment and applies nothing.

    Returns:
        A copy of what was applied (empty when nothing was).
    """
    global _PRISTINE_ENVIRON
    if _PRISTINE_ENVIRON is None:
        _PRISTINE_ENVIRON = dict(os.environ)
    else:
        # Drop keys no session before B set, and restore any the previous
        # session overwrote.  ``clear()`` + ``update()`` rather than a diff:
        # a diff has to enumerate what changed, and anything it fails to
        # enumerate survives -- which is the bug being fixed.
        os.environ.clear()
        os.environ.update(_PRISTINE_ENVIRON)

    if not session_env:
        return {}
    applied: Dict[str, str] = dict(session_env)
    for key, value in applied.items():
        if value is not None:
            os.environ[key] = value
    return applied


def _maybe_self_confine(
    envelope: SessionInitEnvelope,
    recycle_pools: Optional[Callable[[str], Any]] = None,
) -> None:
    """Transition the runner to ``envelope.profile_name`` if needed.

    Pool PR 5a (initial).  Pool slots fork from the template unconfined;
    this step transitions them to the session's AppArmor profile BEFORE
    runtime construction, plugin initialize, and prefetch run — so
    workspace access + tool execution honor the per-session confinement.

    Phase 3 cascade-sharing (server 0.6.146+).  When a pool slot is
    REUSED across sessions of a cascade, this step also handles the
    cross-session transition (P_N → P_{N+1}).  The per-session profile
    template carries ``change_profile -> jaato-ws-*,`` in the main
    runner scope (apparmor.py template v28) authorising the
    inter-session transition.  The transition space is closed — every
    ``jaato-ws-*`` profile is framework-composed, so this rule does NOT
    open an escape vector.  See docs/design/runner-cascade-sharing.md
    §4.4 for the full lifecycle.

    Path by initial state:
      - ``unconfined`` → P1  (cold-spawn, or the first session ever
                             served by this pool slot)
      - P → P  (idempotent skip — same profile, no transition).  **This
               is the reuse path now** (#1033): a slot's reuse key
               carries the profile its threads wear, so a slot is only
               ever handed to a session that wants the profile it has.
      - P_N → P_{N+1}  (requires the v28 template rule).  Reachable only
                       where a daemon hands a confined runner a
                       different profile, which the pool no longer does
                       — because ``aa_change_profile`` is per-task and
                       the threads created under P_N could not follow
                       (#1023).  Kept because the transition itself is
                       still legal and a non-pool caller may use it.
      - empty profile_name  (operator opted out; unconfined session)

    Cold-spawn runners self-confined in ``__main__.py`` step 2 BEFORE
    ``bootstrap_session`` was called, so the kernel already reports
    the target profile in ``/proc/self/attr/current``.  This function
    detects that and skips the redundant transition (``aa_change_profile``
    from ``P → P`` requires ``change_profile -> P`` in P itself, which
    the per-session profiles deliberately omit per §6.1 escape-vector
    hardening; only the glob rule covers cross-session targets).

    No-op cases:
      - ``envelope.profile_name`` is empty (operator opted out of
        confinement; runner runs unconfined).
      - The kernel already reports the target profile (idempotency) —
        note this still recycles and verifies, see below.

    **Per-thread confinement (#1023).**  ``aa_change_profile`` confines
    the CALLING TASK, not the process, and every check in this tree reads
    ``/proc/self/attr/current`` — which resolves to ``/proc/<pid>/`` and
    therefore reports the MAIN THREAD's label.  A worker thread created
    before the transition keeps its own cred for the life of the slot and
    is invisible to all of them.  So once the process is confined, two
    more things happen on every path where a profile is expected:

    1. *recycle* — ``recycle_pools`` retires the RPC worker lanes so every
       later RPC runs on a thread spawned under the confined cred;
    2. *verify* — every thread's own ``attr/current`` is read and compared
       against the target profile.

    Both run on the IDEMPOTENT path too, and deliberately.  That path is
    reached by a slot serving its second session of a cascade under the
    same profile, where a worker created before the slot's FIRST bootstrap
    is still ``unconfined`` — precisely the durable population #1023
    reports.  Skipping the check there would leave the commonest case
    unexamined.

    Args:
        envelope: The bootstrap payload; ``profile_name`` is read.
        recycle_pools: Optional ``(reason) -> Any`` supplied by
            :class:`server.runner.rpc.RunnerRPC`, which owns the worker
            lanes.  ``None`` on the paths that have no RPC lanes (the
            cold-spawn ``__main__`` sequence confines before any executor
            exists; tests) — verification still runs, because a process
            with no lanes can still carry another thread.

    Raises:
        BootstrapError: confinement attempt failed (kernel refused
            the transition, libapparmor unavailable, or
            ``/proc/self/attr/current`` disagrees post-transition).
            Daemon-side spawn helper translates this into a session
            failure via the bootstrap RPC's error envelope.  Phase 3
            ``P_N → P_{N+1}`` denial almost always means the kernel
            template was loaded pre-v28 (no glob rule) — surface this
            in the error message so operators know to restart the
            daemon to pick up the new template.
    """
    target_profile = envelope.profile_name or ""
    if not target_profile:
        logger.info(
            "runner-session bootstrap: envelope.profile_name empty; "
            "skipping AppArmor self-confine (unconfined session)",
        )
        return

    # Check kernel-reported current profile.  Idempotency: if we're
    # already in the target profile (cold-spawn already self-confined),
    # skip the no-op-but-error-prone re-transition.
    try:
        from .bootstrap import (
            ConfinementMismatchError,
            confine_to_profile,
            current_confinement,
        )
    except ImportError as exc:  # noqa: BLE001 — boundary surface
        # AppArmor module not importable (test path or Windows host
        # where the module guards platform).  Log + skip; the daemon's
        # session-spawn flow detects the lack of confinement via the
        # apparmor.is_available() probe and chooses the path
        # accordingly.
        logger.warning(
            "runner-session bootstrap: AppArmor module unavailable "
            "(%s); skipping self-confine for profile=%s",
            exc, target_profile,
        )
        return

    try:
        label = current_confinement()
        actual = label.raw
    except OSError as exc:
        # ``/proc/self/attr/current`` not readable — non-Linux or
        # apparmor-less host.  Daemon shouldn't have set
        # ``profile_name`` in this case; surface the inconsistency.
        raise BootstrapError(
            "confine",
            f"cannot read /proc/self/attr/current ({exc}) but "
            f"envelope.profile_name={target_profile!r} indicates "
            f"confinement was expected — likely a non-Linux host "
            f"running a profile-bearing envelope",
        ) from exc

    # ``current_confinement`` parses e.g. ``jaato-ws-<sid> (enforce)``
    # post-transition, or ``unconfined`` pre-transition.  Match on the
    # NAME only, ignoring the enforcement mode: this is an IDEMPOTENCY
    # question -- "have we already transitioned, so skip the no-op
    # re-transition" -- and re-entering the same profile would not change
    # its mode, so mode-tolerance here is correct and deliberate (#1014).
    #
    # What it must NOT do is let "already confined" stand in for "there is
    # a boundary".  The mode check below is that separation.
    if profile_name_ignoring_mode(actual) == target_profile:
        logger.info(
            "runner-session bootstrap: already in AppArmor profile %s "
            "(kernel reports: %s); skipping redundant self-confine",
            target_profile, actual,
        )
        _announce_unenforced_profile(label, target_profile)
        # NOT a skip of the #1023 work: this is the path a reused pool
        # slot takes when the next session of its cascade carries the
        # same profile, and a worker created before the slot's FIRST
        # bootstrap is unconfined on it.
        _retire_and_verify_threads(target_profile, recycle_pools)
        return

    # Need to transition.  ``confine_to_profile`` does the
    # ``aa_change_profile`` syscall + verifies the kernel agrees.
    try:
        confine_to_profile(target_profile)
    except ConfinementMismatchError as exc:
        # Phase 3 cascade-sharing diagnostic: an empty/non-jaato-ws
        # current profile usually means daemon-side provisioning
        # didn't fire; a jaato-ws-* current profile means the
        # transition was denied — almost always because the kernel
        # has an old template loaded (pre-v28, no
        # ``change_profile -> jaato-ws-*,`` rule).  Restart the
        # daemon to pick up the new template.
        current = profile_name_ignoring_mode(actual)
        likely_cause: str
        if current.startswith("jaato-ws-"):
            likely_cause = (
                f"current profile {current!r} doesn't permit "
                f"``change_profile -> {exc.expected}``.  Almost "
                f"always means the kernel has a pre-v28 template "
                f"loaded (no cascade-sharing glob rule).  Restart "
                f"the daemon to reload all per-session profiles "
                f"against the current template."
            )
        else:
            likely_cause = (
                f"pool slot's current profile ({current!r}) "
                f"doesn't permit ``change_profile -> {exc.expected}``.  "
                f"Verify daemon-side ``AppArmorManager.provision_profile`` "
                f"loaded {exc.expected} before the bootstrap RPC was "
                f"dispatched."
            )
        raise BootstrapError(
            "confine",
            f"AppArmor confinement mismatch — kernel reports "
            f"{exc.actual!r} but we requested {exc.expected!r}.  "
            f"Likely cause: {likely_cause}",
        ) from exc
    except RuntimeError as exc:
        raise BootstrapError(
            "confine",
            f"AppArmor self-confine failed for profile={target_profile}: "
            f"{exc}",
        ) from exc

    _retire_and_verify_threads(target_profile, recycle_pools)


def _announce_unenforced_profile(
    label: AppArmorLabel,
    target_profile: str,
) -> None:
    """WARN when a profile is attached and the kernel is not enforcing it.

    The idempotent bootstrap path (a pool slot serving its second session
    of a cascade under the same profile) never calls
    :func:`~server.runner.bootstrap.confine_to_profile`, so it never
    reached that function's mode announcement.  Left silent, the commonest
    path in a cascade would be the one that says nothing — which is how
    #1014's posture stayed invisible: the record claims a boundary, the log
    claims confinement, and no line anywhere names the mode.

    No-op in the enforcing case, so an ordinary cascade gains no noise.
    """
    if label.enforced:
        return
    remedy = (
        f"Unset {COMPLAIN_ENV_VAR} to enforce."
        if label.complaining
        else "No enforcement mode was reported, which is not evidence of "
             "a boundary and is not treated as one."
    )
    logger.warning(
        "runner-session bootstrap: AppArmor profile %s is attached WITHOUT "
        "a kernel boundary: %s.  This session's tools are NOT confined.  %s",
        target_profile, label.describe(), remedy,
    )


def _retire_and_verify_threads(
    target_profile: str,
    recycle_pools: Optional[Callable[[str], Any]],
) -> None:
    """Retire pre-transition worker threads, then verify every thread (#1023).

    Order is load-bearing: recycling FIRST removes the population the
    framework created and can remove, so anything the verification still
    finds is a thread reached by neither lane — the unknown that must not
    be certified silently.

    **Divergence fails the bootstrap.**  Argued rather than assumed, since
    failing closed on a false positive would take down every session on a
    host whose ``/proc`` were misread:

    - the check acts only on POSITIVE evidence — a label read successfully
      that names a different profile.  A ``/proc`` that cannot be read
      (``hidepid``, an unusual container, a profile template predating the
      task-dir grant) yields ``unreadable`` and is logged, never raised;
    - the alternative — log an ERROR and proceed — reproduces exactly the
      incident state: a session whose record asserts
      ``sandbox_mode: apparmor`` while in-process tools run outside the
      kernel boundary.  A log line is not a boundary, and #1013's proposed
      ``require`` mode would certify such a runner;
    - an operator who cannot tolerate the failure already has an honest
      opt-out: an empty ``profile_name`` runs the session unconfined and
      the record then claims nothing.  "Confined, but tolerating threads
      that are not" is not a posture anyone needs — it is the defect.

    A failed bootstrap is also the right remedy in kind: the slot is
    poisoned for the life of the process, and discarding it is what the
    operator did by hand (SIGTERM on the two divergent slots).
    """
    if recycle_pools is not None:
        try:
            recycle_pools(f"confined to {target_profile}")
        except Exception as exc:  # noqa: BLE001 — boundary surface
            # Recycling is a remedy, not the verdict.  If it fails, the
            # verification below is still run and still fails the
            # bootstrap on any divergence it finds.
            logger.error(
                "runner-session bootstrap: worker-pool recycle failed "
                "(%s); per-thread verification still applies", exc,
            )

    try:
        from .bootstrap import (
            ThreadConfinementDivergence,
            verify_thread_confinement,
        )
    except ImportError as exc:  # noqa: BLE001 — boundary surface
        logger.warning(
            "runner-session bootstrap: per-thread confinement check "
            "unavailable (%s)", exc,
        )
        return

    try:
        scan = verify_thread_confinement(target_profile)
    except ThreadConfinementDivergence as exc:
        raise BootstrapError(
            "confine",
            f"{exc}  The runner refuses this session rather than report "
            f"sandbox_mode=apparmor for a process carrying threads "
            f"outside the profile.",
        ) from exc

    if scan.unreadable:
        logger.warning(
            "runner-session bootstrap: per-thread confinement "
            "UNVERIFIED for %d of %d threads (%s) — %s.  Absence of "
            "evidence is not divergence, so the session continues; a "
            "confined runner needs `/proc/*/task/ r,` (AppArmor "
            "template v32+) for the complete walk.",
            len(scan.unreadable), scan.scanned,
            "; ".join(f"tid={tid}: {why}" for tid, why in scan.unreadable),
            scan.summary(),
        )
    else:
        logger.info(
            "runner-session bootstrap: per-thread confinement verified "
            "(%s)", scan.summary(),
        )


def _stamp_daemon_identity(envelope: SessionInitEnvelope, session: Any) -> None:
    """Stamp the daemon's session id and the authenticated creator onto
    the runner-side session (bootstrap steps 3b / 3c).

    ``session_id`` is this session's daemon id — every runner-tier
    consumer of the per-session id (memory ``source_session``, telemetry
    ``jaato.session_id``, ``{{session_id}}``) reads it from here rather
    than from shared registry state.

    ``created_by`` (#859) is the user the daemon authenticated for the
    creating client.  Until this stamp nothing called
    ``set_client_user_id`` on the runner-side session, so the telemetry
    ``user.id`` attribute and the ledger's ``user_id`` stayed empty on
    every runner-tier session.  Absent on IPC sessions and on envelopes
    from older daemons — then nothing is stamped, as before.
    """
    if envelope.session_id:
        session.set_daemon_session_id(envelope.session_id)
    created_by = getattr(envelope, "created_by", None)
    if created_by:
        session.set_client_user_id(created_by)


def _maybe_install_child_callback(
    envelope: SessionInitEnvelope, session: Any,
) -> None:
    """Install the AppArmor //child transition callback on the
    session's executor (Phase 5 §5.10c).

    The callback hardens subprocess spawns (cli, interactive_shell)
    by transitioning forked children to the ``jaato-ws-<sid>//child``
    sub-profile between fork() and exec(), where the escape rules
    (``change_profile -> unconfined,``, writable ``attr/current``)
    are dropped.  Without the install, subprocesses inherit the
    runner's base profile and can re-issue the apparmor.py:413-449
    escape via the kernel-level ``changeprofile unconfined`` write.

    Source of truth (PR 102, 2026-05-13): ``envelope.profile_name``.
    Pre-PR-102 this read ``os.environ.get("JAATO_RUNNER_PROFILE")``
    which was set by ``RunnerSpawner._build_env`` on cold-spawn but
    never set on pool slots (template inherited daemon's
    ``os.environ.copy()`` without the var).  Pool-slot runners
    confined via PR 5a's ``_maybe_self_confine`` were nevertheless
    hitting matrix case-1 (env empty → "unconfined" log → skip) and
    silently leaving the //child install uninstalled.  Reading from
    the envelope makes the install decision consistent with the
    kernel-reported confinement state.

    Three-case matrix (post-§5.10e):

    1. ``envelope.profile_name`` empty → operator opted out of
       kernel confinement (``JAATO_RUNNER_DISABLE_CONFINE=1`` OR
       profile-less unconfined session).  Skip silently with an
       INFO log.  No escape vector when runner is unconfined.

    2. ``envelope.profile_name`` contains ``//`` → sub-runner under
       an isolated-subagent sub-profile (``jaato-ws-{parent}//{subagent}``
       per Audit 6).  Skip with INFO log.  Per the v15 author's
       sign-off on §5.10e, the sub-profile already drops the
       escape primitive by deliberate design — subprocesses inherit
       the no-escape posture.  Installing a //child transition
       would itself EACCES at preexec_fn (sub-profile lacks
       writable attr/current).

    3. ``envelope.profile_name`` set + lacks ``//`` → main runner
       under a per-session AppArmor profile.  Operator opted INTO
       kernel confinement.  Install MUST succeed or this function
       MUST raise ``BootstrapError("configure", ...)`` (peer review
       of e805e4d0, same audible-failure rule that fixed Phase 4
       §4.3 PR #57 silent-isolation-downgrade).

    Raises:
        BootstrapError: case 3 hit but executor lacks
            ``set_apparmor_child_transition_callback`` OR the setter
            raised.  Bubbles up unchanged through ``bootstrap_session``.
    """
    runner_profile = (envelope.profile_name or "").strip()
    if not runner_profile:
        logger.info(
            "runner-session bootstrap: envelope.profile_name empty; "
            "skipping AppArmor //child transition callback "
            "install (runner is unconfined)",
        )
        return
    if "//" in runner_profile:
        logger.info(
            "runner-session bootstrap: envelope.profile_name is a "
            "sub-profile (%s); skipping AppArmor //child transition "
            "install — sub-profile already drops the escape "
            "primitive per v15 design intent (no writable "
            "attr/current, no change_profile -> unconfined).  "
            "Subprocesses inherit the sub-profile by construction. "
            "See docs/design/phase5_5_10e_sub_runner_skip_audit.md.",
            runner_profile,
        )
        return

    # Case 3: main runner, install required + audibly failing.
    try:
        from server.apparmor import make_child_transition_callback
        child_cb = make_child_transition_callback(runner_profile)
        executor = getattr(session, "_executor", None)
        if executor is None or not hasattr(
            executor, "set_apparmor_child_transition_callback",
        ):
            raise BootstrapError(
                "configure",
                "AppArmor //child transition install failed: "
                "session has no executor with "
                "set_apparmor_child_transition_callback.  "
                "envelope.profile_name is set "
                f"({runner_profile!r}) so the operator opted "
                "into kernel confinement — failing audibly "
                "rather than running with the escape vector "
                "open.  Operator escape hatch: "
                "JAATO_RUNNER_DISABLE_CONFINE=1.",
            )
        executor.set_apparmor_child_transition_callback(child_cb)
        logger.info(
            "runner-session bootstrap: installed AppArmor "
            "//child transition callback for profile=%s",
            runner_profile,
        )
    except BootstrapError:
        raise  # already classified
    except Exception as exc:  # noqa: BLE001 — boundary surface
        logger.exception(
            "runner-session bootstrap: AppArmor //child "
            "transition install crashed for profile=%s",
            runner_profile,
        )
        raise BootstrapError(
            "configure",
            f"AppArmor //child transition install crashed: "
            f"{type(exc).__name__}: {exc}.  envelope.profile_name "
            f"is set ({runner_profile!r}) so the operator opted "
            "into kernel confinement — failing audibly rather "
            "than running with the escape vector open.  Operator "
            "escape hatch: JAATO_RUNNER_DISABLE_CONFINE=1.",
        ) from exc


def bootstrap_session(
    envelope: SessionInitEnvelope,
    *,
    runtime_factory: Any = _USE_DEFAULT,
    recycle_pools: Optional[Callable[[str], Any]] = None,
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
        recycle_pools: Optional ``(reason) -> Any`` the RPC dispatcher
            supplies so step 1c can retire worker threads that predate
            its AppArmor transition (#1023).  ``None`` leaves the
            per-thread VERIFICATION in place and skips only the
            retirement — see :func:`_retire_and_verify_threads`.

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

    # ---- 1b. Apply daemon-resolved session env (PR #91 Y fix) ----
    # The daemon resolved workspace ``.env`` + profile.env +
    # env_overrides daemon-side (secret URIs decoded via the local
    # SecretResolver entry points the daemon has access to as an
    # unconfined process) and shipped the fully-resolved dict via
    # ``envelope.session_env``.  Apply verbatim to ``os.environ``
    # BEFORE plugin discovery + plugin.initialize() runs so:
    #
    #   - Plugin ``initialize(config)`` calls see the resolved env.
    #   - Prefetch scripts that read ``os.environ[...]`` see it.
    #   - Provider clients constructed inside ``runtime.create_session``
    #     pick up the env-resolved API keys / endpoints.
    #
    # Trust posture: the runner-rpc socketpair (daemon ↔ runner) is
    # FD-pass only, so resolved secrets transit a channel as private
    # as pre-PR-91's fork-inherited ``os.environ`` overlay.  See
    # ``shared/session_envelope.py:SessionInitEnvelope.session_env``
    # docstring for the full security contract.
    resolved_session_env: Dict[str, str] = _apply_envelope_session_env(envelope)
    if resolved_session_env:
        logger.info(
            "runner-session bootstrap: applied %d session env keys "
            "to os.environ (session_id=%s)",
            len(resolved_session_env), envelope.session_id,
        )

    # ---- 1c. Per-slot AppArmor self-confine (pool PR 5a) ----
    # Pool slots fork from the (unconfined) template — they need to
    # transition to the session's AppArmor profile BEFORE plugin
    # initialize / prefetch runs, so workspace + tool execution
    # happen under the per-session confinement.  Cold-spawn runners
    # already self-confined in ``__main__.py`` step 2 before
    # ``bootstrap_session`` was called; this step is a NO-OP for
    # them (the kernel already reports the target profile in
    # ``/proc/self/attr/current``).
    #
    # Idempotency invariant:
    #   - Cold-spawn: __main__.py confined → ``proc/self/attr/current``
    #     starts with ``<profile> (enforce)`` → we detect + skip.
    #   - Pool slot: template was unconfined → ``proc/self/attr/current``
    #     reads ``unconfined`` → we call ``confine_to_profile``.
    #
    # When ``envelope.profile_name`` is empty (operator-side
    # ``disable_confine`` opt-out or no AppArmor opt-in), the step
    # is also a no-op — runner runs unconfined.
    _maybe_self_confine(envelope, recycle_pools)

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

    # Shape 3 PR 1: attach the resolved session env onto the
    # JaatoSession so plugin code can read per-session env via
    # ``session.get_session_env(key)``.  Mirrors the daemon's
    # ``JaatoServer._session_env`` attribute — the runner-side analog
    # for the same surface.  Tools needing the raw process env
    # continue reading ``os.environ`` directly, which step 1b already
    # populated.
    if resolved_session_env:
        try:
            session._session_env = dict(resolved_session_env)
        except Exception:  # noqa: BLE001 — best-effort attribute set
            logger.debug(
                "runner-session bootstrap: failed to attach _session_env "
                "to session; plugin reads will fall back to os.environ",
            )

    # ---- 3b. Stamp the daemon session_id onto the runner-side session.
    # Pre-this, ``_daemon_session_id`` was set ONLY daemon-side (by
    # ``JaatoClient``), so the runner-side JaatoSession carried None.
    # That forced any runner-tier consumer of the per-session id (the
    # ``memory`` plugin's ``source_session``, telemetry span
    # ``jaato.session_id``, dynamic-instructions ``{{session_id}}``) to
    # fall back to SHARED state — ``registry._session_id`` — which is
    # overwritten by whichever sibling subagent bootstrapped last,
    # leaking one sibling's id into another's records.  Each sibling has
    # its OWN JaatoSession, so stamping the id here (envelope.session_id
    # is this session's daemon id) gives every consumer a per-execution,
    # per-sibling-correct value via ``get_current_session()``.
    _stamp_daemon_identity(envelope, session)

    # ---- 4. Phase 5 §5.10c — install AppArmor child-profile
    # transition callback on subprocess-spawning plugins.
    _maybe_install_child_callback(envelope, session)

    logger.info(
        "runner-session bootstrap ready: session_id=%s profile=%s "
        "model=%s plugins=%s",
        envelope.session_id, envelope.profile_name, envelope.model_name,
        _describe_plugin_selection(envelope.plugins),
    )
    return RunnerSessionHost(envelope=envelope, runtime=runtime, session=session)


def _describe_plugin_selection(
    plugins: Optional[List[Dict[str, Any]]],
) -> str:
    """Render ``envelope.plugins`` for the bootstrap-ready log line.

    The None/empty distinction is the whole point of the field, so the log
    must not flatten it: a bare ``len()`` would print ``0`` for both "no
    profile" (all exposed plugins) and "profile asked for the minimal set",
    which is exactly the confusion that let a profile-less session ship with
    an empty tool wire unnoticed.

    Lives at module level rather than inline so ``bootstrap_session`` keeps
    its cyclomatic score at the ceiling (15) instead of crossing it.
    """
    if plugins is None:
        return "none (no profile — all exposed)"
    return str(len(plugins))


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


def _runtime_limits_from_envelope(
    envelope: SessionInitEnvelope,
) -> Optional[Any]:
    """Rebuild ``RuntimeLimits`` from the v7 wire block, or ``None``.

    The envelope carries the block as a plain dict (the same shape
    ``profile_to_snapshot`` persists) so the daemon and the runner do
    not have to agree on a dataclass across the socketpair.  The runner
    re-parses it here, which is also where it is re-validated:
    ``RuntimeLimits.__post_init__`` raises on a bad value.

    A parse failure degrades to ``None`` — "nobody declared limits" —
    rather than aborting the bootstrap.  The block was already
    validated at profile-load time daemon-side, so anything that fails
    here came from a NEWER daemon whose vocabulary this runner does not
    share; refusing to bootstrap over it would turn a forward-compat
    skew into a session that cannot start at all.  The failure is
    logged at WARNING because an unarmed cap must never be silent
    (#735).

    Args:
        envelope: The bootstrap envelope.

    Returns:
        The parsed limits, or ``None`` when the envelope carried none.
    """
    raw = getattr(envelope, "runtime_limits", None)
    if not raw:
        return None
    from shared.runtime_limits import RuntimeLimits
    try:
        return RuntimeLimits.from_dict(raw)
    except (ValueError, TypeError) as exc:
        logging.getLogger(__name__).warning(
            "session.bootstrap: envelope runtime_limits %r is not "
            "parseable by this runner (%s); the session will run with "
            "framework defaults and NO tool wall-clock or output cap",
            raw, exc,
        )
        return None


def _processors_from_envelope(
    envelope: SessionInitEnvelope,
) -> List[Any]:
    """Reconstruct ``CompletionProcessor`` instances from wire-dict shape.

    The wire envelope carries processors as a list of plain dicts
    (``{"script": ..., "output": ..., "on_error": ..., "description": ...}``)
    so the daemon ↔ runner socketpair doesn't require importing
    ``shared.plugins.subagent.config.CompletionProcessor`` on both
    sides of the boundary.  The runner reconstructs the dataclass
    here because ``LifecycleTools`` reads attributes (``.script``,
    ``.output``, ``.on_error``) not dict keys.

    Server 0.6.125+ — replaces the prior ``completion_artifacts`` +
    ``completion_validators`` envelope fields, both of which used
    the same wire-dict-reconstruction pattern.
    """
    from shared.plugins.subagent.config import (
        completion_processors_from_wire,
    )

    # The SAME parser a profile file's completion_processors: block goes
    # through, so the wire and the profile cannot disagree about what an
    # entry means.  The hand-rolled reconstruction this replaces named five
    # fields and defaulted the rest, so `max_refusals` was discarded on
    # arrival even once the daemon started sending it (jaato #770).
    return completion_processors_from_wire(envelope.completion_processors)


def _extract_plugin_specs(
    specs: Optional[List[Dict[str, Any]]],
) -> Tuple[Optional[List[str]], set, Dict[str, List[str]]]:
    """Split ``envelope.plugins`` into the three args ``create_session`` wants.

    Returns ``(tool_names, preloaded, tool_scopes)``:

    - ``tool_names`` feeds ``plugins=`` — the plugin allow-list.  **``None``
      is preserved, not flattened to ``[]``.**  ``None`` means the session
      was created WITHOUT a profile, so nothing declared a plugin set and
      ``JaatoRuntime.create_session`` expands it to every exposed plugin;
      ``[]`` means a profile explicitly asked for the minimal set.  Both are
      reachable and they are different answers — collapsing them hands a
      profile-less session an empty tool wire (the empty-wire gate in
      ``jaato_session._should_drop_introspection`` then strips list_tools /
      get_tool_schemas too, so the model gets no tools at all while the
      system prompt still tells it to discover them).
    - ``preloaded`` feeds ``preloaded_plugins=`` — names carrying
      ``(preload)``, which bypass deferred tool loading.
    - ``tool_scopes`` feeds the per-plugin ``tools:[...]`` allow-list.

    Lives outside ``_build_session`` because that function is over the
    cyclomatic ceiling and frozen in the complexity baseline at its current
    size; new branching has to go in a helper.

    Raises:
        ValueError: an entry is missing a usable ``name``.
    """
    if specs is None:
        return None, set(), {}

    tool_names: List[str] = []
    preloaded: set = set()
    tool_scopes: Dict[str, List[str]] = {}
    for entry in specs:
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"plugin entry missing 'name': {entry!r}")
        tool_names.append(name)
        if entry.get("preload"):
            preloaded.add(name)
        # Per-plugin tool allow-list (profile ``tools:[...]`` modifier),
        # carried alongside ``name`` / ``preload`` on the envelope entry.
        scope = entry.get("tools")
        if scope:
            tool_scopes[name] = list(scope)
    return tool_names, preloaded, tool_scopes


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
    The plugin-list (just names) feeds ``plugins=...`` so the runtime
    exposes them; the per-plugin configs feed
    ``plugin_configs=...``; the preload set feeds
    ``preloaded_plugins=...``.
    """
    tool_names, preloaded, tool_scopes = _extract_plugin_specs(
        envelope.plugins)
    # Phase 4 §C: per-plugin configs come from the top-level
    # envelope.plugin_configs map (schema v2); shallow-copy so the
    # callee can't mutate the envelope's dict.
    plugin_configs: dict = {
        k: dict(v) for k, v in envelope.plugin_configs.items()
    }

    # v3 (2026-05-14): resolve per-turn model-tier config from
    # ``envelope.model_tiers`` (carried from profile.model_tiers
    # daemon-side).  Profile-level config wins; an absent / empty dict
    # falls through to the env-var path (``JAATO_TIER_*``).  A failed
    # resolve degrades to single-model mode with a warning rather than
    # aborting the runner — operators get an enter_tier-less session
    # instead of a hard bootstrap failure.
    tier_config = None
    try:
        from shared.model_tiers import ModelTierConfig
        tier_config = ModelTierConfig.resolve(
            profile_model_tiers=envelope.model_tiers or None,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "runner-session bootstrap: tier config rejected "
            "(falling back to single-model mode): %s", exc,
        )

    # Budget control: the envelope carries the profile's re-serialised
    # ``budget_control`` (v5).  Re-parse + validate on this side of the
    # process boundary; a rejected block degrades to UNBUDGETED with a
    # warning rather than aborting the runner — same posture as tier
    # config above (an operator gets an unbudgeted session, not a hard
    # bootstrap failure).
    budget_control = None
    try:
        from shared.budget_control import BudgetControlConfig
        budget_control = BudgetControlConfig.from_dict(
            envelope.budget_control or None)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            'runner-session bootstrap: budget_control rejected '
            '(session runs unbudgeted): %s', exc,
        )

    return runtime.create_session(
        model=envelope.model_name,
        # PR #240 (2026-06-07) attempted to fix the same bug at
        # ``server/core.py:_build_profile_session_kwargs`` — but that
        # function turned out to be dead code (only referenced by
        # tests; no production caller).  The LIVE production path
        # routes through this runner-side bootstrap, which had the
        # exact same falsy bug: ``tool_names or None`` swallows the
        # explicit-empty-list semantic.  When ``profile.plugins: []``
        # was declared (canonical "minimal framework set"),
        # ``tool_names`` was ``[]`` → ``[] or None`` → ``None`` →
        # ``runtime.create_session`` interprets None as "load all
        # exposed plugins", producing the ~22-tool wire surface that
        # confused Llama-3.1-8B-AWQ in the vLLM smoke.
        #
        # Pass the list verbatim — empty means empty (minimal set:
        # introspection + lifecycle + framework infra like stream /
        # event_bus that are registered as core tools regardless).
        #
        # ``None`` is the THIRD case and is equally verbatim: no profile
        # was supplied, so no one asked for a subset and the runtime
        # expands it to every exposed plugin.  Before the envelope could
        # carry None, a profile-less session arrived here as ``[]`` and
        # got the minimal set; the empty-wire gate in
        # ``jaato_session._should_drop_introspection`` then dropped
        # list_tools/get_tool_schemas too, leaving no tools at all while
        # the system prompt still told the model to discover them.
        plugins=tool_names,
        system_instructions=envelope.system_instructions,
        plugin_configs=plugin_configs or None,
        provider_name=envelope.provider_name or None,
        preloaded_plugins=preloaded or None,
        completion_payload_schema=envelope.completion_payload_schema,
        agent_params=envelope.agent_params or None,
        completion_processors=(
            _processors_from_envelope(envelope) or None
        ),
        tier_config=tier_config,
        budget_control=budget_control,
        # Envelope v6 (#862): the profile's tool-pool ceiling.  ``None``
        # from an older daemon, or from a profile that declares no
        # ``runtime_limits``, leaves the framework default in charge.
        max_parallel_tools=envelope.max_parallel_tools,
        # Envelope v7 (#735): the whole resolved ``runtime_limits``.
        # ``JaatoSession.configure`` is the ONE place that arms the
        # subprocess plugins with ``tool_timeout_seconds`` /
        # ``max_output_bytes``; before this the caps reached the runner
        # only as process-startup env, which configures the Phase-2
        # cli-only executor a bootstrapped session never dispatches
        # through -- so the caps were inert on the pool path AND on
        # cold-spawn.  Built here rather than in the envelope so a
        # malformed block from a future daemon degrades to "nobody
        # declared limits" instead of refusing the bootstrap.
        runtime_limits=_runtime_limits_from_envelope(envelope),
        # Per-plugin tool allow-lists (profile ``tools:[...]`` modifier),
        # threaded from the envelope so scoped-out tools are absent from
        # this runner session's wire body + grammar surface.
        tool_scopes=tool_scopes or None,
        # Thread the envelope's resolved agent_id into the runner-
        # side JaatoSession so AgentCompletedEvent.agent_id carries
        # the daemon's ``--agent <name>`` resolution (the envelope
        # carries it correctly post PR #79; pre-thread it was
        # silently discarded here because create_session didn't
        # accept the kwarg).
        agent_id=envelope.agent_id,
        # 2026-06-06: forward the two daemon-resolved system-instruction
        # knobs to the runner-side JaatoSession.  Pre-fix these were
        # silently dropped here — JaatoSession.configure defaulted to
        # ``suppress_base_instructions=False`` and
        # ``system_instruction_override=None`` regardless of what the
        # profile / IPC client requested.  See SessionInitEnvelope field
        # docstrings + ``project_backlog_suppress_base_instructions_not_honored``
        # memory for the bug history.
        suppress_base_instructions=envelope.suppress_base_instructions,
        system_instruction_override=envelope.system_instruction_override,
    )
