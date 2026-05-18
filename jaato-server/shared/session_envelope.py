"""Session-init envelope for the daemon → runner handshake +
SessionManager-level bootstrap envelope.

Phase 3 §3.3a + §3.12.0.

When the daemon spawns a runner subprocess (Phase 2 task 2.3) and
the runner reports ready (``RunnerReadyEvent``), the daemon sends a
:class:`SessionInitEnvelope` as the first frame after ready.  The
runner's ``runner.session.bootstrap_session(envelope)`` (§3.3b)
constructs a live :class:`JaatoSession`, runs ``configure()``, and
hosts the session for the duration of the runner's lifetime.

The :class:`BootstrapEnvelope` (Phase 3 §3.12.0) is the
SessionManager-level envelope above the JaatoSession-level
``SessionInitEnvelope``.  It aggregates every input the per-session
``SessionManager._bootstrap_session`` helper needs across the four
session-creation paths (IPC, disk-restore, ephemeral subagent
fan-out, WS standalone) into a single typed payload, replacing the
ad-hoc kwarg-bag previously inlined in each call site.

This module defines ONLY the schemas.  The serialization
(``to_dict`` / ``from_dict``) is plain JSON-friendly — wraps primitive
types + dicts + lists.  Anything richer (callable references, file
descriptors, etc.) is NOT permitted in the envelope; all session
state needed runner-side must be reducible to JSON.

Versioning: ``schema_version`` is incremented when fields are added
or semantics change.  Phase 3 ships v1.  Phase 4+ may bump.  The
runner reads the version on receipt and refuses to bootstrap if the
daemon advertises a higher version than the runner supports — this
catches mid-deploy version skew (operator restarted the daemon to
0.6.X but a long-running runner is still 0.6.X-1).

:class:`BootstrapEnvelope` is daemon-internal — it never crosses the
RPC wire — so it doesn't carry a ``schema_version`` and may hold
non-JSON-serializable fields (Callable references, plugin instances,
profile objects).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


# Bumped per schema change.  Runners refuse a higher-version
# envelope from the daemon (forward-compat is opt-in, not free).
#
# v3 (2026-05-14): added ``model_tiers``.  Phase 3 §3.3a/§3.12 moved
# session state from daemon-tier to runner-tier; the
# ``profile.model_tiers`` field (server 0.5.20) was never migrated and
# the runner therefore left ``JaatoSession._tier_config = None``,
# suppressing ``enter_tier`` tool registration for every pool-served
# session.  v3 carries the tier mapping on the envelope so the runner
# can resolve ``ModelTierConfig`` and register the tool.
SESSION_ENVELOPE_VERSION = 3


@dataclass
class SessionInitEnvelope:
    """Daemon → runner session-bootstrap payload.

    Carries everything the runner needs to construct a live
    :class:`JaatoSession` against the configured profile + plugin
    set.  Daemon-resolved fields (plugin list, plugin configs, agent
    instructions) are sent as already-resolved dicts so the runner
    doesn't re-walk profile / agent / config_root paths
    independently — single source of truth, single resolution path.

    Attributes:
        schema_version: Echoes :data:`SESSION_ENVELOPE_VERSION`.
            Runner refuses higher-than-known versions.
        session_id: Stable session identifier; matches the AppArmor
            profile name suffix (``jaato-ws-{session_id}``).
        workspace_path: Absolute path to the session's workspace
            root, or ``None`` for headless / no-workspace sessions.
        profile_name: Name of the profile JSON the daemon resolved
            (e.g. ``"cli_test"``).  Informational; the resolved
            plugin list + configs are authoritative.
        provider_name: Model provider this session uses
            (``"anthropic"``, ``"openrouter"``, etc.).  The runner
            doesn't talk to the provider directly — daemon-tier per
            §4.2 — but knows the name for telemetry attribution.
        model_name: Model identifier (e.g.
            ``"claude-sonnet-4-6"``).
        plugins: Ordered list of plugin specifications the runner
            should instantiate.  Each entry is a dict carrying
            ``name`` (str) + ``preload`` (bool — Phase 2 carries
            this via ``"name(preload)"`` syntax; Phase 3 normalizes
            to typed dict).  Per-plugin configs live in
            ``plugin_configs`` (Phase 4 §C) — a top-level dict that
            carries the full ``profile.plugin_configs`` map so
            auto-loaded plugins (``permission``, ``gc_*``, etc.) that
            aren't in this list still receive their profile overrides.
        plugin_configs: Map of plugin name → config dict, mirroring
            ``profile.plugin_configs``.  Carries configs for **all**
            plugins the profile names — including ones the runner
            auto-loads without them appearing in ``plugins``.  Pre-§C
            the per-plugin config lived in ``plugins[i].config`` and
            was dropped for non-listed plugins; this field closes that
            gap (backlog §3.3c.X).  Schema v2.
        system_instructions: Resolved system-instructions text the
            runner installs onto the JaatoSession.  ``None`` for
            sessions that compose instructions on-the-fly via
            dynamic-instructions render scripts; non-None for the
            simple opaque-string path.
        agent_id: Logical agent identifier (defaults to ``"main"``
            for top-level sessions).  Carried in event emissions
            for cascade attribution.
        gc: Optional GC strategy config dict (``{"type": "budget",
            "threshold_percent": 80.0}`` etc.).  ``None`` falls back
            to the runtime default.
        completion_payload_schema: Profile-declared JSON Schema for
            ``signal_completion``'s ``payload`` parameter.  Inline
            dict or string path resolved via
            ``.jaato/completion_schemas/``.  ``None`` = legacy
            untyped completion.
        completion_artifacts: Profile-declared output artefacts
            (renderer / output / on_error specs) the runner's
            ``LifecycleTools`` runs after a validated
            ``signal_completion``.  Empty list = legacy behaviour.
        agent_params: Spawn-time parameters from the parent caller
            (``agent_params={...}`` on ``spawn_subagent``).  Carried
            into dynamic-instructions render-context.  Empty for
            top-level sessions whose prompt carries case data
            inline.
        config_root: Override for the framework config root
            (``.jaato/`` typically).  Set when a client passed
            ``ClientConfigRequest.config_root`` at handshake.
            ``None`` = use the workspace's ``.jaato/``.
        env_overrides: Environment-variable overrides applied during
            session-init (e.g. provider env from a post-auth wizard
            response).  Layered atop the workspace's ``.env``.

            **Deprecated:** post-Y the runner consumes ``session_env``
            (which already includes the layered + resolved overrides).
            Carried for backward compat with older runner versions
            until schema_version bumps; new code should not rely on
            this field reaching the runner.
        session_env: Fully-resolved per-session environment (workspace
            ``.env`` + profile.env + env_overrides, all ``${VAR}`` and
            secret-URI resolved daemon-side).  **Carries plaintext
            secrets** (decoded ``pass://`` / ``vault://`` values).
            Wire-only — never persisted, never logged, never forwarded
            to clients.  Runner applies these to ``os.environ``
            verbatim during ``bootstrap_session`` without further
            resolution.

            This is the load-bearing channel for confined-runner
            secret access: the daemon (unconfined) does the resolver
            exec; the runner (AppArmor-confined and unable to exec
            ``pass``) consumes pre-resolved literals.  See
            ``project_backlog_env_propagation_seat_flip_gap`` history
            + the PR #91 retrospective for context.
        model_tiers: Profile-declared per-turn model-tier mapping
            (``{"planner": "...", "dispatcher": "...", "executor":
            "...", "initial": "...", "fallback": "..."}``) or ``None``
            when the session runs in single-model mode.  Carried on the
            envelope so the runner can resolve a
            :class:`shared.model_tiers.ModelTierConfig` and pass it to
            ``runtime.create_session(tier_config=...)``; that in turn
            registers the ``enter_tier`` lifecycle tool so the model
            can switch tiers mid-turn.  Schema v3.
    """

    session_id: str
    workspace_path: Optional[str]
    profile_name: Optional[str]
    provider_name: str
    model_name: str
    plugins: List[Dict[str, Any]] = field(default_factory=list)
    # Phase 4 §C: top-level plugin configs map (replaces per-entry
    # ``plugins[i].config`` which only carried configs for plugins
    # named in ``plugins``).  Carries the full ``profile.plugin_configs``
    # so auto-loaded plugins like ``permission`` receive their profile
    # overrides too.  Schema v2 — old runners refuse v2 envelopes.
    plugin_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    system_instructions: Optional[str] = None
    agent_id: str = "main"
    gc: Optional[Dict[str, Any]] = None
    completion_payload_schema: Optional[Any] = None
    completion_artifacts: List[Dict[str, Any]] = field(default_factory=list)
    # Profile-declared completion validators (server 0.6.121+).  Each
    # entry is a path string (absolute, ``<config_root>/<path>``, or
    # ``~/.jaato/<path>``) to a kb-authored Python module exposing
    # ``validate(payload, tool_calls, workspace_path, ctx) -> list[str]``.
    # Runner-side ``LifecycleTools._execute_signal_completion`` invokes
    # them AFTER ``jsonschema.validate`` passes; non-empty error list
    # returns the same ``validation_failed`` shape as a schema failure.
    # Empty list = no semantic checks (legacy behaviour).  See
    # ``shared/completion_validators.py`` for the loader + ledger
    # builder.
    completion_validators: List[str] = field(default_factory=list)
    agent_params: Dict[str, str] = field(default_factory=dict)
    config_root: Optional[str] = None
    env_overrides: Dict[str, str] = field(default_factory=dict)
    # PR #91 Y fix: fully-resolved per-session env carrying plaintext
    # secrets.  Wire-only — daemon → runner over the socketpair, never
    # persisted, logged, or forwarded to clients.  See field docstring
    # above for the full security contract.
    session_env: Dict[str, str] = field(default_factory=dict)
    # Phase 3 post-Step-7 Path C: provider-connect args.  Carried in
    # the envelope so the runner-side ``bootstrap_session`` can call
    # ``runtime.connect(project, location)`` before
    # ``runtime.create_session`` (which guards on ``_connected``).
    # Non-Vertex providers (anthropic, openrouter, ollama, etc.)
    # leave these as empty strings — the provider plugin's
    # ``initialize()`` ignores them.  Vertex AI / Google GenAI
    # sessions populate from ``PROJECT_ID`` / ``LOCATION`` env
    # daemon-side.  Defaults preserve backward compat with earlier
    # callers and the envelope schema_version stays unchanged.
    project: str = ""
    location: str = ""
    # v3 (2026-05-14): per-turn model-tier mapping forwarded from
    # ``profile.model_tiers``.  Empty dict / None means single-model
    # mode (no ``enter_tier`` tool, no per-tier system-prompt line).
    # See ``shared/model_tiers.py`` for the resolver and schema.
    model_tiers: Optional[Dict[str, Any]] = None
    schema_version: int = SESSION_ENVELOPE_VERSION

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-friendly dict for the wire.

        Field-order is fixed (matches the dataclass declaration) so
        log output across versions is comparable.
        """
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "workspace_path": self.workspace_path,
            "profile_name": self.profile_name,
            "provider_name": self.provider_name,
            "model_name": self.model_name,
            "plugins": [dict(p) for p in self.plugins],
            "plugin_configs": {k: dict(v) for k, v in self.plugin_configs.items()},
            "system_instructions": self.system_instructions,
            "agent_id": self.agent_id,
            "gc": dict(self.gc) if self.gc is not None else None,
            "completion_payload_schema": self.completion_payload_schema,
            "completion_artifacts": [dict(a) for a in self.completion_artifacts],
            "completion_validators": list(self.completion_validators),
            "agent_params": dict(self.agent_params),
            "config_root": self.config_root,
            "env_overrides": dict(self.env_overrides),
            "session_env": dict(self.session_env),
            "project": self.project,
            "location": self.location,
            "model_tiers": (
                dict(self.model_tiers) if self.model_tiers else None
            ),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SessionInitEnvelope":
        """Deserialize from a wire dict.

        Raises:
            ValueError: when ``schema_version`` is missing OR exceeds
                :data:`SESSION_ENVELOPE_VERSION`.  Forward-compat is
                opt-in — runners refuse newer envelopes.
            KeyError: when required fields are missing.
        """
        version = d.get("schema_version")
        if version is None:
            raise ValueError(
                "SessionInitEnvelope: missing 'schema_version' — "
                "are you decoding a Phase 2 frame against the "
                "Phase 3 schema?"
            )
        if version > SESSION_ENVELOPE_VERSION:
            raise ValueError(
                f"SessionInitEnvelope: envelope schema_version "
                f"{version} > runner-supported "
                f"{SESSION_ENVELOPE_VERSION}; runner is older than "
                f"daemon (mid-deploy skew?)"
            )

        return cls(
            schema_version=int(version),
            session_id=str(d["session_id"]),
            workspace_path=d.get("workspace_path"),
            profile_name=d.get("profile_name"),
            provider_name=str(d.get("provider_name", "")),
            model_name=str(d.get("model_name", "")),
            plugins=[dict(p) for p in (d.get("plugins") or [])],
            plugin_configs={
                k: dict(v)
                for k, v in (d.get("plugin_configs") or {}).items()
            },
            system_instructions=d.get("system_instructions"),
            agent_id=str(d.get("agent_id", "main")),
            gc=dict(d["gc"]) if d.get("gc") else None,
            completion_payload_schema=d.get("completion_payload_schema"),
            completion_validators=[
                str(v) for v in (d.get("completion_validators") or [])
                if isinstance(v, str)
            ],
            completion_artifacts=[
                dict(a) for a in (d.get("completion_artifacts") or [])
            ],
            agent_params=dict(d.get("agent_params") or {}),
            config_root=d.get("config_root"),
            env_overrides=dict(d.get("env_overrides") or {}),
            session_env=dict(d.get("session_env") or {}),
            project=str(d.get("project", "")),
            location=str(d.get("location", "")),
            model_tiers=(
                dict(d["model_tiers"]) if d.get("model_tiers") else None
            ),
        )


# ----------------------------------------------------------------------
# §3.12.0 — SessionManager-level bootstrap envelope
# ----------------------------------------------------------------------


@dataclass
class BootstrapEnvelope:
    """SessionManager-level bootstrap envelope (Phase 3 §3.12.0).

    Aggregates every input the per-session
    :meth:`SessionManager._bootstrap_session` helper needs across
    the four session-creation paths (IPC, disk-restore, ephemeral
    subagent fan-out, WS standalone) into a single typed payload.

    Fields are grouped by purpose:

    1. **Identity** — ``session_id``, ``workspace_path``, ``name``,
       ``description``.
    2. **Path discriminators** (per the §3.12.0 spec):

       - ``client_id`` — ``None`` for disk-restore + ephemeral.
       - ``parent_runner_handle`` — set only on ephemeral subagent
         fan-out per §4.3 default share; else ``None``.
       - ``sandbox_mode`` — the planned-sandbox-mode value the IPC
         apparmor pre-init hook stashed into Phase 2's
         ``_planned_sandbox_mode``.  Pre-resolved by the caller (the
         IPC path's pre-init hook runs to completion before the
         envelope is built); ``None`` for paths without an apparmor
         opt-in.
       - ``restore_state`` — populated only on disk-restore.

    3. **JaatoServer construction** — ``env_file``, ``profile``,
       ``agent_name``, ``system_instruction_override``,
       ``env_overrides``, ``suppress_base_instructions``,
       ``config_root``, ``instruction_token_cache``.
    4. **Session record** — ``provisioned``, ``created_by``,
       ``timestamp``.
    5. **Bootstrap-time event sink** — ``on_event_during_init`` for
       error reporting BEFORE the client is attached to the
       session.

    Daemon-internal only — never crosses the RPC wire.  Holds
    non-JSON-serializable fields (Callable references, profile
    objects, plugin instances) and therefore exposes no
    ``to_dict`` / ``from_dict`` serializer.

    Subsequent §3.12 commits extend this dataclass with path-
    specific fields as the disk-restore / ephemeral / WS-standalone
    migrations land.  New fields default to ``None`` / empty so the
    existing IPC migration stays byte-identical.
    """

    # -- Identity ---------------------------------------------------------
    session_id: str
    workspace_path: Optional[str]
    name: str
    description: Optional[str] = None

    # -- Path discriminators ---------------------------------------------
    client_id: Optional[str] = None
    parent_runner_handle: Optional[Any] = None
    sandbox_mode: Optional[str] = None
    restore_state: Optional[Dict[str, Any]] = None

    # -- JaatoServer construction ----------------------------------------
    env_file: Optional[str] = None
    profile: Optional[Any] = None
    agent_name: str = "main"
    system_instruction_override: Optional[str] = None
    suppress_base_instructions: bool = False
    env_overrides: Dict[str, str] = field(default_factory=dict)
    config_root: Optional[str] = None
    instruction_token_cache: Optional[Any] = None
    # Phase 4 §D: agent_params from the originating IPC ``create_session``
    # request (or ``spawn_subagent`` fan-out).  Carried daemon-internal
    # so ``build_session_envelope`` can forward them through the
    # SessionInitEnvelope to the runner, where the JaatoSession applies
    # them to its ``{{!py:...}}`` prefetch render context.  Pre-§D this
    # field was missing and ``runner_spawn.build_session_envelope``
    # hard-coded ``agent_params={}`` on the wire envelope — prefetch
    # scripts reading ``context.agent_params`` on the runner saw empty
    # dicts and emitted their "missing required keys" error block,
    # which caused the documenter agent to hallucinate tmux pane ids.
    agent_params: Dict[str, str] = field(default_factory=dict)

    # PR-A (2026-05-14): explicit AppArmor confinement override from the
    # session-creation caller (``SessionManager.create_headless_session``
    # ``apparmor=`` kwarg; reactor surface through ``ActionContext``).
    # ``None`` means "no caller override — consult the IPC
    # client_config, then the profile field, then the legacy default".
    # ``True`` / ``False`` short-circuit that chain.  See backlog
    # ``project_backlog_apparmor_kwarg_for_headless_sessions`` for the
    # two-PR migration plan (PR-A: surface + back-compat False default;
    # PR-B: flip profile default to True).
    apparmor: Optional[bool] = None

    # -- Session record --------------------------------------------------
    provisioned: bool = False
    created_by: Optional[str] = None
    timestamp: Optional[Any] = None

    # -- Bootstrap-time event sink ---------------------------------------
    on_event_during_init: Optional[Callable[[Any], None]] = None
