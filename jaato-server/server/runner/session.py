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
from typing import Any, Callable, Dict, List, Optional, Protocol, TYPE_CHECKING

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
    # `core.py:1778-1794` baseline; profile-supplied
    # ``plugin_configs.permission`` overrides are now applied via the
    # Phase 4 §C envelope.plugin_configs field (schema v2).  Shallow
    # merge: top-level keys from the profile (most commonly ``policy``)
    # replace defaults.  Mirrors daemon-side ``permission_init_config.update(...)``.
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
    runtime.configure_plugins(registry, permission_plugin, None)

    logger.info(
        "runner-session bootstrap: configured %d plugins runner-tier "
        "(session_id=%s workspace=%s)",
        len(registry._exposed), session_id, workspace_path or "(none)",
    )


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
    if not envelope.session_env:
        return {}
    applied: Dict[str, str] = dict(envelope.session_env)
    for key, value in applied.items():
        if value is not None:
            os.environ[key] = value
    return applied


def _maybe_self_confine(envelope: SessionInitEnvelope) -> None:
    """Transition the runner to ``envelope.profile_name`` if needed.

    Pool PR 5a.  Pool slots fork from the template unconfined; this
    step transitions them to the session's AppArmor profile BEFORE
    runtime construction, plugin initialize, and prefetch run — so
    workspace access + tool execution honor the per-session
    confinement.

    Cold-spawn runners self-confined in ``__main__.py`` step 2 BEFORE
    ``bootstrap_session`` was called, so the kernel already reports
    the target profile in ``/proc/self/attr/current``.  This function
    detects that and skips the redundant transition (it would also
    fail anyway — ``aa_change_profile`` from ``P → P`` requires
    ``change_profile -> P`` in P itself, which the per-session
    profiles deliberately omit per §6.1 escape-vector hardening).

    No-op cases:
      - ``envelope.profile_name`` is empty (operator opted out of
        confinement; runner runs unconfined).
      - The kernel already reports the target profile (cold-spawn
        idempotency).

    Raises:
        BootstrapError: confinement attempt failed (kernel refused
            the transition, libapparmor unavailable, or
            ``/proc/self/attr/current`` disagrees post-transition).
            Daemon-side spawn helper translates this into a session
            failure via the bootstrap RPC's error envelope.
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
            read_current_profile,
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
        actual = read_current_profile()
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

    # ``read_current_profile`` returns e.g. ``jaato-ws-<sid> (enforce)``
    # post-transition, or ``unconfined`` pre-transition.  Match by
    # prefix so the enforcement-mode suffix doesn't trip the check.
    expected_prefix = f"{target_profile} "
    if actual.startswith(expected_prefix) or actual == target_profile:
        logger.info(
            "runner-session bootstrap: already confined to %s "
            "(kernel reports: %s); skipping redundant self-confine",
            target_profile, actual,
        )
        return

    # Need to transition.  ``confine_to_profile`` does the
    # ``aa_change_profile`` syscall + verifies the kernel agrees.
    try:
        confine_to_profile(target_profile)
    except ConfinementMismatchError as exc:
        raise BootstrapError(
            "confine",
            f"AppArmor confinement mismatch — kernel reports "
            f"{exc.actual!r} but we requested {exc.expected!r}.  "
            f"Likely cause: pool slot's current profile (typically "
            f"``unconfined``) doesn't permit ``change_profile -> "
            f"{exc.expected}``.  Verify daemon-side "
            f"``AppArmorManager.provision_profile`` loaded "
            f"{exc.expected} before the bootstrap RPC was dispatched.",
        ) from exc
    except RuntimeError as exc:
        raise BootstrapError(
            "confine",
            f"AppArmor self-confine failed for profile={target_profile}: "
            f"{exc}",
        ) from exc


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
    _maybe_self_confine(envelope)

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

    # ---- 4. Phase 5 §5.10c — install AppArmor child-profile
    # transition callback on subprocess-spawning plugins.
    _maybe_install_child_callback(envelope, session)

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
    preloaded: set = set()
    for entry in envelope.plugins:
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"plugin entry missing 'name': {entry!r}")
        tool_names.append(name)
        if entry.get("preload"):
            preloaded.add(name)
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
        tier_config=tier_config,
        # Thread the envelope's resolved agent_id into the runner-
        # side JaatoSession so AgentCompletedEvent.agent_id carries
        # the daemon's ``--agent <name>`` resolution (the envelope
        # carries it correctly post PR #79; pre-thread it was
        # silently discarded here because create_session didn't
        # accept the kwarg).
        agent_id=envelope.agent_id,
    )
