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


def _stub_server(*, profile=None, config_root=None, agent_params=None) -> Any:
    """Build a minimal JaatoServer-shaped stub.  ``_build_session_envelope``
    reads ``_profile`` + ``config_root`` + (Phase 4 §D) ``_agent_params``
    so SimpleNamespace suffices."""
    return SimpleNamespace(
        _profile=profile,
        config_root=config_root,
        _agent_params=dict(agent_params or {}),
    )


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
    )
    base.update(overrides)
    return SimpleNamespace(**base)


# ----------------------------------------------------------------------
# No-profile fallback
# ----------------------------------------------------------------------


def test_no_profile_uses_provider_default() -> None:
    """When the server has no resolved profile, the envelope falls
    back to the framework's default provider (``anthropic``)."""
    server = _stub_server(profile=None)
    env = _build_session_envelope(
        server=server,
        session_id="s",
        workspace_path="/tmp/ws",
        profile_name="auto",
    )
    assert env.provider_name == "anthropic"
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


def test_profile_provider_empty_falls_back_to_anthropic() -> None:
    """An explicit empty/None ``provider`` on the profile triggers
    the default fallback."""
    profile = _stub_profile(provider=None, model="claude-3.5")
    env = _build_session_envelope(
        server=_stub_server(profile=profile),
        session_id="s",
        workspace_path="/tmp/ws",
        profile_name="cli_test",
    )
    assert env.provider_name == "anthropic"


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
