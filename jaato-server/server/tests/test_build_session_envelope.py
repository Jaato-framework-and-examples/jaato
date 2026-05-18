"""Tests for ``_build_session_envelope`` (Phase 3 §3.3c part 2).

The helper reads the resolved profile from a pre-init JaatoServer
and constructs a :class:`SessionInitEnvelope` for the runner-side
host.  Covers the no-profile fallback, profile field extraction,
plugin spec construction, preload set translation, and GC config
flattening.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from server.__main__ import _build_session_envelope
from shared.session_envelope import SessionInitEnvelope


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _stub_server(
    *,
    profile=None,
    config_root=None,
    agent_params=None,
    main_agent_id=None,
) -> Any:
    """Build a minimal JaatoServer-shaped stub.  ``_build_session_envelope``
    reads ``_profile`` + ``config_root`` + (Phase 4 §D) ``_agent_params``
    + (regression fix 2026-05-12) ``_main_agent_id`` so SimpleNamespace
    suffices.

    ``main_agent_id=None`` means the stub omits ``_main_agent_id``
    entirely so the getattr-with-default fallback can be exercised."""
    fields: dict = {
        "_profile": profile,
        "config_root": config_root,
        "_agent_params": dict(agent_params or {}),
    }
    if main_agent_id is not None:
        fields["_main_agent_id"] = main_agent_id
    return SimpleNamespace(**fields)


def _stub_profile(**overrides: Any) -> Any:
    """Build a SubagentProfile-shaped stub.  ``_build_session_envelope``
    reads ``provider``, ``model``, ``plugins``, ``preloaded_plugins``,
    ``plugin_configs``, ``system_instructions``, ``gc``, ``env``."""
    base = dict(
        provider=None,
        model=None,
        plugins=[],
        preloaded_plugins=set(),
        plugin_configs={},
        system_instructions=None,
        gc=None,
        env={},
        # v3 (2026-05-14): default to no tier mode so tests not
        # exercising tier propagation see the envelope's ``None``.
        model_tiers=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


# ----------------------------------------------------------------------
# No-profile fallback
# ----------------------------------------------------------------------


def test_no_profile_no_env_leaves_provider_empty() -> None:
    """PR 101: when the server has no resolved profile AND no
    session_env (workspace .env), the envelope's provider_name +
    model_name stay empty.  Pre-PR-101 there was a hardcoded
    ``"anthropic"`` provider fallback that papered over the
    configuration gap; PR 101 removed it per the user's standing
    rule against hardcoded fallbacks (global CLAUDE.md).

    The runner-side ``_validate_envelope`` raises
    ``BootstrapError(stage="validate")`` for empty model_name —
    loud failure beats silent guess."""
    server = _stub_server(profile=None)
    env = _build_session_envelope(
        server=server,
        session_id="s",
        workspace_path="/tmp/ws",
        profile_name="auto",
    )
    assert env.provider_name == ""
    assert env.model_name == ""  # validate stage will reject this loudly
    assert env.plugins == []
    assert env.system_instructions is None
    assert env.gc is None


def test_no_profile_workspace_and_session_passthrough() -> None:
    server = _stub_server(profile=None)
    env = _build_session_envelope(
        server=server,
        session_id="sess-42",
        workspace_path="/tmp/ws-42",
        profile_name="auto",
    )
    assert env.session_id == "sess-42"
    assert env.workspace_path == "/tmp/ws-42"
    assert env.profile_name == "auto"
    assert env.agent_id == "main"


def test_no_workspace_path_is_none() -> None:
    """Headless / no-workspace sessions surface as workspace_path=None."""
    env = _build_session_envelope(
        server=_stub_server(profile=None),
        session_id="s",
        workspace_path=None,
        profile_name="auto",
    )
    assert env.workspace_path is None


# ----------------------------------------------------------------------
# Profile field extraction
# ----------------------------------------------------------------------


def test_profile_provider_and_model_pass_through() -> None:
    profile = _stub_profile(
        provider="openrouter",
        model="anthropic/claude-3.5",
    )
    env = _build_session_envelope(
        server=_stub_server(profile=profile),
        session_id="s",
        workspace_path="/tmp/ws",
        profile_name="cli_test",
    )
    assert env.provider_name == "openrouter"
    assert env.model_name == "anthropic/claude-3.5"


def test_profile_provider_empty_stays_empty_post_pr101() -> None:
    """PR 101: profile with ``provider=None`` no longer triggers a
    hardcoded ``"anthropic"`` fallback.  The envelope's provider_name
    stays empty (or picks up from session_env JAATO_PROVIDER if
    workspace .env supplied one — covered by
    ``test_build_session_envelope_profileless_fallback.py``).  Pre-
    PR-101 the silent default could mask Vertex / OpenRouter / etc.
    sessions slipping through with wrong provider."""
    profile = _stub_profile(provider=None, model="claude-3.5")
    env = _build_session_envelope(
        server=_stub_server(profile=profile),
        session_id="s",
        workspace_path="/tmp/ws",
        profile_name="cli_test",
    )
    assert env.provider_name == ""


def test_profile_system_instructions_pass_through() -> None:
    profile = _stub_profile(
        provider="anthropic",
        model="claude-3.5",
        system_instructions="Be concise.",
    )
    env = _build_session_envelope(
        server=_stub_server(profile=profile),
        session_id="s",
        workspace_path="/tmp/ws",
        profile_name="x",
    )
    assert env.system_instructions == "Be concise."


# ----------------------------------------------------------------------
# Plugin spec construction
# ----------------------------------------------------------------------


def test_plugin_list_translated_to_envelope_specs() -> None:
    """The profile's ``plugins`` (list of names) becomes envelope
    plugin specs ({"name": ..., "preload": bool})."""
    profile = _stub_profile(
        provider="anthropic",
        model="m",
        plugins=["cli", "todo", "permission"],
    )
    env = _build_session_envelope(
        server=_stub_server(profile=profile),
        session_id="s",
        workspace_path="/tmp/ws",
        profile_name="x",
    )
    assert env.plugins == [
        {"name": "cli", "preload": False},
        {"name": "todo", "preload": False},
        {"name": "permission", "preload": False},
    ]


def test_preloaded_plugins_marked_in_envelope() -> None:
    profile = _stub_profile(
        provider="anthropic",
        model="m",
        plugins=["cli", "signal_completion", "todo"],
        preloaded_plugins={"signal_completion"},
    )
    env = _build_session_envelope(
        server=_stub_server(profile=profile),
        session_id="s",
        workspace_path="/tmp/ws",
        profile_name="x",
    )
    preload_map = {p["name"]: p["preload"] for p in env.plugins}
    assert preload_map == {
        "cli": False,
        "signal_completion": True,
        "todo": False,
    }


def test_plugin_configs_attached_to_envelope_specs() -> None:
    profile = _stub_profile(
        provider="anthropic",
        model="m",
        plugins=["cli", "todo"],
        plugin_configs={
            "cli": {"max_workers": 4},
            "todo": {"storage_path": "/tmp/todos"},
        },
    )
    env = _build_session_envelope(
        server=_stub_server(profile=profile),
        session_id="s",
        workspace_path="/tmp/ws",
        profile_name="x",
    )
    # Phase 4 §C: plugin configs live in the top-level
    # envelope.plugin_configs map; per-entry ``config`` key is gone.
    assert env.plugin_configs == {
        "cli": {"max_workers": 4},
        "todo": {"storage_path": "/tmp/todos"},
    }
    for p in env.plugins:
        assert "config" not in p, (
            f"plugins[i] should not carry per-entry config post-§C: {p!r}"
        )


def test_plugin_configs_for_unlisted_plugin_carried_through() -> None:
    """Phase 4 §C: plugin_configs entries for plugins NOT in the
    profile.plugins list are now carried in the envelope (closes
    backlog §3.3c.X).  Auto-loaded plugins like ``permission`` and
    ``gc_*`` need their profile overrides to reach the runner even
    when they aren't named in profile.plugins."""
    profile = _stub_profile(
        provider="anthropic",
        model="m",
        plugins=["cli"],
        plugin_configs={
            "cli": {"x": 1},
            "memory": {"orphan": True},  # not in plugins list — now carried
            "permission": {"policy": {"defaultPolicy": "allow"}},
        },
    )
    env = _build_session_envelope(
        server=_stub_server(profile=profile),
        session_id="s",
        workspace_path="/tmp/ws",
        profile_name="x",
    )
    names_in_envelope = [p["name"] for p in env.plugins]
    assert "memory" not in names_in_envelope
    # But its config IS in the top-level plugin_configs map.
    assert env.plugin_configs == {
        "cli": {"x": 1},
        "memory": {"orphan": True},
        "permission": {"policy": {"defaultPolicy": "allow"}},
    }


# ----------------------------------------------------------------------
# GC config flattening
# ----------------------------------------------------------------------


def test_gc_config_flattened_into_envelope() -> None:
    """GCProfileConfig is structurally ``{type, config: {...}}``;
    the envelope flattens to a single dict ``{type, ...config}``."""
    gc = SimpleNamespace(type="budget", config={"threshold_percent": 80.0})
    profile = _stub_profile(
        provider="anthropic",
        model="m",
        gc=gc,
    )
    env = _build_session_envelope(
        server=_stub_server(profile=profile),
        session_id="s",
        workspace_path="/tmp/ws",
        profile_name="x",
    )
    assert env.gc == {"type": "budget", "threshold_percent": 80.0}


def test_gc_with_no_type_skipped() -> None:
    """A GC config object without a type field doesn't synthesize
    a partial envelope.gc — it stays None."""
    gc = SimpleNamespace(type=None, config={"k": "v"})
    profile = _stub_profile(provider="anthropic", model="m", gc=gc)
    env = _build_session_envelope(
        server=_stub_server(profile=profile),
        session_id="s",
        workspace_path="/tmp/ws",
        profile_name="x",
    )
    assert env.gc is None


# ----------------------------------------------------------------------
# Env overrides + config_root
# ----------------------------------------------------------------------


def test_env_overrides_pass_through() -> None:
    profile = _stub_profile(
        provider="anthropic",
        model="m",
        env={"JAATO_PROVIDER": "anthropic", "DEBUG": "1"},
    )
    env = _build_session_envelope(
        server=_stub_server(profile=profile),
        session_id="s",
        workspace_path="/tmp/ws",
        profile_name="x",
    )
    assert env.env_overrides == {
        "JAATO_PROVIDER": "anthropic",
        "DEBUG": "1",
    }


def test_config_root_pass_through() -> None:
    server = _stub_server(profile=None, config_root="/srv/operator/.jaato")
    env = _build_session_envelope(
        server=server,
        session_id="s",
        workspace_path="/tmp/ws",
        profile_name="x",
    )
    assert env.config_root == "/srv/operator/.jaato"


# ----------------------------------------------------------------------
# Regression fix 2026-05-12: agent_id propagation + completion_payload_schema
# ----------------------------------------------------------------------
#
# Two bugs in this builder discovered via the kb-enablement-2.0
# cascade smoke test:
#
# 1. ``agent_id="main"`` was HARDCODED.  Result: runner-side
#    ``JaatoSession._agent_id`` stayed ``"main"`` regardless of the
#    daemon's ``--agent <name>`` resolution.  Downstream
#    ``AgentCompletedEvent.agent_id`` always carried ``"main"``;
#    reactor rules keying on logical agent identity (e.g.
#    ``agent_id == "discovery"``) silently missed.
#
# 2. ``completion_payload_schema`` was MISSING from the
#    ``SessionInitEnvelope`` constructor call entirely.  Even though
#    the dataclass field existed (default None), the builder never
#    populated it.  Result: profiles declaring
#    ``completion_payload_schema`` saw the runner-side
#    ``LifecycleTools._payload_schema`` stay ``None`` → legacy
#    summary path fired instead of typed payload validation.
#
# Both were introduced when build_session_envelope was relocated in
# §7c step 2 (copied verbatim with the bugs intact).  IPC test
# fixtures used agent_name=None + no schema, masking both.


class TestAgentIdPropagation:
    """Pin: envelope.agent_id reflects the daemon-resolved
    ``--agent <name>`` instead of the hardcoded ``"main"``."""

    def test_agent_id_reads_server_main_agent_id(self) -> None:
        """Pin: when the JaatoServer's ``_main_agent_id`` is set
        (the daemon resolved ``--agent discovery`` at construction),
        the envelope carries the logical identity verbatim."""
        env = _build_session_envelope(
            server=_stub_server(main_agent_id="discovery"),
            session_id="s",
            workspace_path="/tmp/ws",
            profile_name="x",
        )
        assert env.agent_id == "discovery", (
            f"envelope must carry the daemon-resolved agent_id "
            f"'discovery', got {env.agent_id!r}"
        )

    def test_agent_id_defaults_to_main_when_attr_missing(self) -> None:
        """Pin: when the server stub omits ``_main_agent_id``
        entirely (legacy server / no ``--agent`` flag), the
        envelope falls back to the literal ``"main"``.  Preserves
        backward compat with the pre-fix behavior."""
        env = _build_session_envelope(
            server=_stub_server(main_agent_id=None),
            session_id="s",
            workspace_path="/tmp/ws",
            profile_name="x",
        )
        assert env.agent_id == "main"

    def test_agent_id_with_dashes_and_underscores(self) -> None:
        """Pin: any valid agent name flows through verbatim — no
        sanitization at the envelope layer (that lives in the
        daemon's agent-name resolver)."""
        env = _build_session_envelope(
            server=_stub_server(main_agent_id="kb-enablement_2_dispatcher"),
            session_id="s",
            workspace_path="/tmp/ws",
            profile_name="x",
        )
        assert env.agent_id == "kb-enablement_2_dispatcher"


class TestCompletionPayloadSchema:
    """Pin: envelope.completion_payload_schema is sourced from
    ``profile.completion_payload_schema`` instead of being silently
    omitted from the constructor."""

    def test_inline_schema_dict_passes_through(self) -> None:
        """Pin: an inline JSON Schema dict declared on the profile
        flows verbatim to the envelope (no flattening, no string
        coercion).  Runner-side
        ``completion_schema_loader.resolve_completion_schema``
        accepts dicts directly."""
        inline = {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
        }
        profile = _stub_profile(completion_payload_schema=inline)
        env = _build_session_envelope(
            server=_stub_server(profile=profile),
            session_id="s",
            workspace_path="/tmp/ws",
            profile_name="x",
        )
        assert env.completion_payload_schema == inline

    def test_string_path_passes_through(self) -> None:
        """Pin: a string path (resolved by
        ``completion_schema_loader`` against the workspace's
        ``.jaato/completion_schemas/``) flows verbatim — resolution
        is the runner's job, not the envelope builder's."""
        profile = _stub_profile(
            completion_payload_schema="completion_schemas/discovery_result.schema.json",
        )
        env = _build_session_envelope(
            server=_stub_server(profile=profile),
            session_id="s",
            workspace_path="/tmp/ws",
            profile_name="x",
        )
        assert (
            env.completion_payload_schema
            == "completion_schemas/discovery_result.schema.json"
        )

    def test_none_when_profile_has_no_schema(self) -> None:
        """Pin: profile without ``completion_payload_schema``
        attribute → envelope carries None.  Legacy untyped
        completion path on the runner side."""
        profile = _stub_profile()  # no completion_payload_schema set
        env = _build_session_envelope(
            server=_stub_server(profile=profile),
            session_id="s",
            workspace_path="/tmp/ws",
            profile_name="x",
        )
        assert env.completion_payload_schema is None

    def test_none_when_no_profile_at_all(self) -> None:
        """Pin: server without a resolved profile (auto-spawn path)
        → envelope carries None.  Symmetric to the no-profile
        provider/model fallback above."""
        env = _build_session_envelope(
            server=_stub_server(profile=None),
            session_id="s",
            workspace_path="/tmp/ws",
            profile_name="auto",
        )
        assert env.completion_payload_schema is None


class TestCompletionValidators:
    """Pin: envelope.completion_validators is sourced from
    ``profile.completion_validators`` instead of being silently dropped.

    Regression guard (server 0.6.122+): pre-0.6.122 the envelope
    dataclass declared the field but ``build_session_envelope`` never
    populated it.  Runner-side ``runtime.create_session`` always got
    an empty list, so the kb-authored semantic validators silently
    didn't fire.  kb-enablement-2.0 v118 evidence: validator module
    loaded fine in isolation but never invoked during cascade runs.
    Same bug class as the 2026-05-12 ``completion_payload_schema``
    omission.
    """

    def test_validators_pass_through(self) -> None:
        profile = _stub_profile(
            completion_validators=[
                "scripts/validators/codegen_files_exist.py",
                "scripts/validators/codegen_render_succeeded.py",
            ],
        )
        env = _build_session_envelope(
            server=_stub_server(profile=profile),
            session_id="s",
            workspace_path="/tmp/ws",
            profile_name="x",
        )
        assert env.completion_validators == [
            "scripts/validators/codegen_files_exist.py",
            "scripts/validators/codegen_render_succeeded.py",
        ]

    def test_empty_list_when_profile_has_no_validators(self) -> None:
        """Profile without the attribute → envelope carries empty
        list (default).  Runner-side LifecycleTools skips invocation."""
        profile = _stub_profile()  # no completion_validators set
        env = _build_session_envelope(
            server=_stub_server(profile=profile),
            session_id="s",
            workspace_path="/tmp/ws",
            profile_name="x",
        )
        assert env.completion_validators == []

    def test_empty_list_when_no_profile_at_all(self) -> None:
        env = _build_session_envelope(
            server=_stub_server(profile=None),
            session_id="s",
            workspace_path="/tmp/ws",
            profile_name="auto",
        )
        assert env.completion_validators == []


class TestCompletionArtifactsPassdown:
    """Pin: ``profile.completion_artifacts`` flow through to the
    envelope.  Same wire-gap class as the validators above —
    discovered during the v118 diagnostic on 2026-05-18."""

    def test_artifacts_pass_through_as_dict(self) -> None:
        from types import SimpleNamespace
        artifact = SimpleNamespace(
            renderer="markdown",
            output="report.md",
            on_error="warn",
            description="Final report",
        )
        profile = _stub_profile(completion_artifacts=[artifact])
        env = _build_session_envelope(
            server=_stub_server(profile=profile),
            session_id="s",
            workspace_path="/tmp/ws",
            profile_name="x",
        )
        assert len(env.completion_artifacts) == 1
        assert env.completion_artifacts[0]["renderer"] == "markdown"
        assert env.completion_artifacts[0]["output"] == "report.md"


class TestModelTiersPassdown:
    """v3 (2026-05-14): ``profile.model_tiers`` must be forwarded
    on the envelope so the runner-side ``_build_session`` can resolve
    a :class:`ModelTierConfig` and register the ``enter_tier``
    lifecycle tool.

    Before v3 the runner never received the tier map; pool-served
    sessions silently dropped tier mode regardless of profile
    config.  Discovery: 2026-05-14 in-pane TUI smoke test.
    """

    def test_profile_model_tiers_pass_through_to_envelope(self) -> None:
        tiers = {
            "planner": "glm-5",
            "dispatcher": "glm-5-turbo",
            "executor": "glm-4.7-flash",
            "initial": "dispatcher",
            "fallback": "dispatcher",
        }
        profile = _stub_profile(
            provider="zhipuai",
            model="glm-5-turbo",
            model_tiers=tiers,
        )
        env = _build_session_envelope(
            server=_stub_server(profile=profile),
            session_id="s",
            workspace_path="/tmp/ws",
            profile_name="tier-test",
        )
        assert env.model_tiers == tiers

    def test_profile_without_model_tiers_leaves_envelope_none(self) -> None:
        """Single-model profiles emit ``model_tiers=None`` (not ``{}``)
        so the runner's ``ModelTierConfig.resolve`` short-circuits and
        ``enter_tier`` stays unregistered."""
        profile = _stub_profile(provider="anthropic", model="claude-sonnet-4-6")
        env = _build_session_envelope(
            server=_stub_server(profile=profile),
            session_id="s",
            workspace_path="/tmp/ws",
            profile_name="cli_test",
        )
        assert env.model_tiers is None

    def test_no_profile_leaves_envelope_model_tiers_none(self) -> None:
        env = _build_session_envelope(
            server=_stub_server(profile=None),
            session_id="s",
            workspace_path="/tmp/ws",
            profile_name="auto",
        )
        assert env.model_tiers is None


def test_envelope_round_trip_via_dict() -> None:
    """The constructed envelope is wire-serializable (the dict
    survives a round-trip through SessionInitEnvelope.from_dict)."""
    profile = _stub_profile(
        provider="anthropic",
        model="m",
        plugins=["cli"],
        plugin_configs={"cli": {"max_output_chars": 1000}},
        preloaded_plugins={"cli"},
        system_instructions="hi",
    )
    env = _build_session_envelope(
        server=_stub_server(profile=profile, config_root="/cfg"),
        session_id="s",
        workspace_path="/tmp/ws",
        profile_name="cli_test",
    )
    back = SessionInitEnvelope.from_dict(env.to_dict())
    assert back == env
