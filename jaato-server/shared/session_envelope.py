"""Session-init envelope for the daemon → runner handshake.

Phase 3 §3.3a.

When the daemon spawns a runner subprocess (Phase 2 task 2.3) and
the runner reports ready (``RunnerReadyEvent``), the daemon sends a
:class:`SessionInitEnvelope` as the first frame after ready.  The
runner's ``runner.session.bootstrap_session(envelope)`` (§3.3b)
constructs a live :class:`JaatoSession`, runs ``configure()``, and
hosts the session for the duration of the runner's lifetime.

This module defines ONLY the schema.  The serialization
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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# Bumped per schema change.  Runners refuse a higher-version
# envelope from the daemon (forward-compat is opt-in, not free).
SESSION_ENVELOPE_VERSION = 1


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
            to typed dict) + optional ``config`` (dict).  The
            daemon's profile resolver expands ``signal_completion(preload)``
            shorthand into the typed form before sending.
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
    """

    session_id: str
    workspace_path: Optional[str]
    profile_name: Optional[str]
    provider_name: str
    model_name: str
    plugins: List[Dict[str, Any]] = field(default_factory=list)
    system_instructions: Optional[str] = None
    agent_id: str = "main"
    gc: Optional[Dict[str, Any]] = None
    completion_payload_schema: Optional[Any] = None
    completion_artifacts: List[Dict[str, Any]] = field(default_factory=list)
    agent_params: Dict[str, str] = field(default_factory=dict)
    config_root: Optional[str] = None
    env_overrides: Dict[str, str] = field(default_factory=dict)
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
            "system_instructions": self.system_instructions,
            "agent_id": self.agent_id,
            "gc": dict(self.gc) if self.gc is not None else None,
            "completion_payload_schema": self.completion_payload_schema,
            "completion_artifacts": [dict(a) for a in self.completion_artifacts],
            "agent_params": dict(self.agent_params),
            "config_root": self.config_root,
            "env_overrides": dict(self.env_overrides),
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
            system_instructions=d.get("system_instructions"),
            agent_id=str(d.get("agent_id", "main")),
            gc=dict(d["gc"]) if d.get("gc") else None,
            completion_payload_schema=d.get("completion_payload_schema"),
            completion_artifacts=[
                dict(a) for a in (d.get("completion_artifacts") or [])
            ],
            agent_params=dict(d.get("agent_params") or {}),
            config_root=d.get("config_root"),
            env_overrides=dict(d.get("env_overrides") or {}),
        )
