"""Phase 4 §C regression pins: profile.plugin_configs → runner propagation.

Closes the §3.3c.X backlog item: pre-§C the bootstrap envelope only
carried per-plugin config for plugins named in ``profile.plugins`` (via
the per-entry ``plugins[i].config`` field), so auto-loaded plugins like
``permission``, ``gc_*``, and any framework-provided plugin that isn't
listed in the profile silently lost their configuration.

The §C fix promotes per-plugin configs to a top-level
``envelope.plugin_configs: Dict[str, Dict]`` map that carries the full
``profile.plugin_configs`` dict.  The per-entry ``config`` field is
removed; envelope schema bumps to v2.

These tests pin:

1. The schema-level round-trip with the new field + v2 version.
2. ``build_session_envelope`` emits the new shape from a stub profile.
3. The runner-side ``_configure_runtime_plugins`` applies
   ``envelope.plugin_configs.get("permission")`` to the PermissionPlugin
   init, mirroring the daemon-side path at ``core.py:1788-1793``.

The Discipline-#9 real-provider integration counterpart (documenter
agent making it past the permission check on a profile with
defaultPolicy=allow) lives in the harness workspace; see
``docs/design/phase4_profile_plugin_configs_passdown_audit.md``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from shared.session_envelope import (
    SESSION_ENVELOPE_VERSION,
    SessionInitEnvelope,
)
from server.runner_spawn import build_session_envelope


# ----------------------------------------------------------------------
# Schema round-trip
# ----------------------------------------------------------------------


def test_schema_version_is_at_least_2() -> None:
    """§C bumps schema_version 1 → 2.  Old runners refuse v≥2 envelopes
    (per ``SessionInitEnvelope.from_dict``'s ``version > known`` guard);
    new runners accept v1 envelopes with an empty plugin_configs default.

    The assertion is monotonic — later schema bumps (v3 added model_tiers
    on 2026-05-14) keep this contract intact, so the version-floor is
    what we pin here rather than a hard equality."""
    assert SESSION_ENVELOPE_VERSION >= 2


def test_envelope_plugin_configs_round_trip() -> None:
    """``to_dict`` includes ``plugin_configs``; ``from_dict`` restores
    it intact.  Per-plugin keys carry arbitrary dict values."""
    env = SessionInitEnvelope(
        session_id="s",
        workspace_path=None,
        profile_name="p",
        provider_name="anthropic",
        model_name="m",
        plugins=[{"name": "cli", "preload": True}],
        plugin_configs={
            "permission": {"policy": {"defaultPolicy": "allow"}},
            "cli": {"max_workers": 4},
        },
    )
    d = env.to_dict()
    assert d["schema_version"] == SESSION_ENVELOPE_VERSION
    assert d["plugin_configs"] == {
        "permission": {"policy": {"defaultPolicy": "allow"}},
        "cli": {"max_workers": 4},
    }
    assert d["plugins"] == [{"name": "cli", "preload": True}]

    env2 = SessionInitEnvelope.from_dict(d)
    assert env2.plugin_configs == env.plugin_configs


def test_envelope_plugins_entries_have_no_config_key() -> None:
    """§C drops the per-entry ``config`` key.  Entries are
    {name, preload} only.  Carrying both fields would create a
    precedence ambiguity; the top-level map is the single source."""
    env = SessionInitEnvelope(
        session_id="s",
        workspace_path=None,
        profile_name="p",
        provider_name="anthropic",
        model_name="m",
        plugins=[{"name": "cli", "preload": True}],
        plugin_configs={"cli": {"max_workers": 4}},
    )
    d = env.to_dict()
    for p in d["plugins"]:
        assert "config" not in p, (
            f"plugins[i] must not carry per-entry config post-§C: {p!r}"
        )


def test_v1_envelope_back_compat_empty_plugin_configs() -> None:
    """A v1 envelope (pre-§C) deserializes with empty plugin_configs
    via the dataclass default_factory; new code reads zero entries
    and proceeds with the runner-side defaults."""
    d_v1 = {
        "schema_version": 1,
        "session_id": "old",
        "provider_name": "anthropic",
        "model_name": "m",
        "plugins": [],
    }
    env = SessionInitEnvelope.from_dict(d_v1)
    assert env.plugin_configs == {}


# ----------------------------------------------------------------------
# build_session_envelope writes the new shape
# ----------------------------------------------------------------------


def _stub_profile(**fields: Any) -> SimpleNamespace:
    """Minimal SubagentProfile stand-in matching what runner_spawn reads."""
    base = dict(
        provider="anthropic",
        model="m",
        plugins=[],
        plugin_configs={},
        preloaded_plugins=set(),
        system_instructions=None,
        gc=None,
        env={},
    )
    base.update(fields)
    return SimpleNamespace(**base)


def _stub_server(profile: Any) -> MagicMock:
    server = MagicMock()
    server._profile = profile
    server.config_root = None
    return server


def test_build_envelope_carries_full_profile_plugin_configs() -> None:
    """build_session_envelope copies the whole profile.plugin_configs
    map into envelope.plugin_configs — including entries for plugins
    NOT in profile.plugins (the §3.3c.X gap).

    This is what unblocks the documenter (whose permission config sits
    under plugin_configs.permission but ``permission`` isn't in
    profile.plugins, so pre-§C the override was dropped)."""
    profile = _stub_profile(
        plugins=["cli", "file_edit"],
        plugin_configs={
            "cli": {"max_workers": 4},
            "file_edit": {"session_id": "x"},
            "permission": {"policy": {"defaultPolicy": "allow"}},  # ← unlisted
            "openrouter": {"api_params": {"temperature": 0.0}},   # ← unlisted
        },
    )
    env = build_session_envelope(
        server=_stub_server(profile),
        session_id="s",
        workspace_path="/tmp/ws",
        profile_name="p",
    )
    assert env.plugin_configs == {
        "cli": {"max_workers": 4},
        "file_edit": {"session_id": "x"},
        "permission": {"policy": {"defaultPolicy": "allow"}},
        "openrouter": {"api_params": {"temperature": 0.0}},
    }


def test_build_envelope_plugins_entries_have_no_config_key() -> None:
    """Per-entry ``config`` is gone from build_session_envelope's
    output.  Entries are {name, preload}."""
    profile = _stub_profile(
        plugins=["cli"],
        plugin_configs={"cli": {"max_workers": 4}},
    )
    env = build_session_envelope(
        server=_stub_server(profile),
        session_id="s",
        workspace_path="/tmp/ws",
        profile_name="p",
    )
    for p in env.plugins:
        assert set(p.keys()) <= {"name", "preload"}, (
            f"unexpected key in plugins entry post-§C: {p!r}"
        )


# ----------------------------------------------------------------------
# Runner-side permission init reads from envelope.plugin_configs
# ----------------------------------------------------------------------


def _bootstrap_envelope(
    plugin_configs: Dict[str, Dict[str, Any]],
) -> SessionInitEnvelope:
    """Minimal envelope for exercising _configure_runtime_plugins."""
    return SessionInitEnvelope(
        session_id="s-perm",
        workspace_path="/tmp/ws",
        profile_name="p",
        provider_name="anthropic",
        model_name="m",
        plugins=[],  # no listed plugins — permission is auto-loaded
        plugin_configs=plugin_configs,
    )


def _capture_permission_init(envelope: SessionInitEnvelope) -> Dict[str, Any]:
    """Run _configure_runtime_plugins against stubs that capture the
    PermissionPlugin.initialize() call.

    Returns the captured init dict so tests can assert the merge
    semantics directly.
    """
    from server.runner.session import _configure_runtime_plugins

    captured: Dict[str, Any] = {}

    class _StubPerm:
        def initialize(self, cfg: Dict[str, Any]) -> None:
            captured["init"] = cfg

    class _StubRegistry:
        def __init__(self) -> None:
            self._exposed: Dict[str, Any] = {}

        def discover(self, **kwargs: Any) -> None:
            pass

        def expose_all(self, configs: Dict[str, Any]) -> None:
            pass

        def set_workspace_path(self, p: Any) -> None:
            pass

        def set_config_root(self, c: Any) -> None:
            pass

    class _StubRuntime:
        def configure_plugins(self, *a: Any, **kw: Any) -> None:
            pass

    import server.runner.session as session_mod
    monkey_patches = []
    try:
        # Patch PermissionPlugin + PluginRegistry inside the import scope of
        # the function-under-test.
        import shared.plugins.permission.plugin as perm_mod
        import shared.plugins.registry as reg_mod
        monkey_patches.append((perm_mod, "PermissionPlugin", perm_mod.PermissionPlugin))
        monkey_patches.append((reg_mod, "PluginRegistry", reg_mod.PluginRegistry))
        perm_mod.PermissionPlugin = lambda: _StubPerm()  # type: ignore[assignment]
        reg_mod.PluginRegistry = lambda **kw: _StubRegistry()  # type: ignore[assignment]

        _configure_runtime_plugins(_StubRuntime(), envelope)
    finally:
        for mod, attr, orig in monkey_patches:
            setattr(mod, attr, orig)

    return captured["init"]


def test_permission_init_uses_default_when_no_profile_override() -> None:
    """No plugin_configs.permission → PermissionPlugin gets the runner-
    side fallback ``defaultPolicy: "ask"``.  This is the pre-§C behaviour
    preserved for profiles that don't customize permission."""
    env = _bootstrap_envelope(plugin_configs={})
    init_cfg = _capture_permission_init(env)
    assert init_cfg["policy"]["defaultPolicy"] == "ask"


def test_permission_init_merges_profile_override() -> None:
    """plugin_configs.permission.policy.defaultPolicy = "allow" wins
    over the runner-side fallback.  This is the documenter-fix case:
    the agent's tools auto-approve and don't trip the
    ``ASK → wait forever`` hang."""
    env = _bootstrap_envelope(plugin_configs={
        "permission": {
            "policy": {
                "defaultPolicy": "allow",
                "blacklist": {
                    "tools": ["list_tools", "get_tool_schemas"],
                    "patterns": [],
                },
            },
        },
    })
    init_cfg = _capture_permission_init(env)
    assert init_cfg["policy"]["defaultPolicy"] == "allow"
    assert init_cfg["policy"]["blacklist"]["tools"] == [
        "list_tools", "get_tool_schemas",
    ]


def test_permission_init_preserves_channel_keys_from_default() -> None:
    """Shallow merge: profile's top-level keys (``policy``) replace
    defaults; keys the profile doesn't mention (``channel_type``,
    ``workspace_path``) stay from the default dict.  Mirrors daemon-
    side ``permission_init_config.update(...)`` semantics."""
    env = _bootstrap_envelope(plugin_configs={
        "permission": {"policy": {"defaultPolicy": "allow"}},
    })
    init_cfg = _capture_permission_init(env)
    # policy.defaultPolicy comes from profile
    assert init_cfg["policy"]["defaultPolicy"] == "allow"
    # channel_type came from runner-side default and survived
    assert init_cfg["channel_type"] == "queue"
    assert init_cfg["workspace_path"] == "/tmp/ws"
