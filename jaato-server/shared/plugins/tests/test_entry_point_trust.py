"""Built-in plugin names are reserved against entry-point shadowing.

Issue #684: ``PluginRegistry.discover()`` ran entry-point discovery
before the directory scan, and the directory scan skipped any name
already registered — silently.  Entry-point loading had no name
reservation and no allowlist, so any distribution sharing the venv
could declare::

    [project.entry-points."jaato.plugins"]
    permission = "anything:create_plugin"

and replace the built-in ``permission`` plugin.  The real one was never
imported and nothing was logged.  Because ``ep.load()`` imports the
target module, being installed was enough to execute code.

These tests pin the post-fix contract:

* a reserved name claimed by a foreign distribution is refused **before
  ``ep.load()``** — the module is never imported;
* the security-critical set is refused even with the operator opt-in;
* every refusal, honoured shadow and duplicate claim emits a WARNING
  naming the distribution involved;
* the framework declaring its own plugins through the same groups stays
  silent and keeps working;
* provenance is recorded so a shadow is visible without reading logs.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from shared.plugins.entry_point_trust import (
    ENV_ALLOW_SHADOW,
    ENV_ENTRY_POINT_ALLOWLIST,
    NEVER_SHADOWABLE,
    PluginOrigin,
    builtin_plugin_names,
    evaluate_entry_point,
    normalize_distribution,
)
from shared.plugins.registry import PluginRegistry


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


def _plugin_class(plugin_name: str) -> type:
    """A Protocol-complete ToolPlugin reporting ``plugin_name``."""

    class _Plugin:
        @property
        def name(self) -> str:
            return plugin_name

        def initialize(self, config: Any = None) -> None: ...
        def shutdown(self) -> None: ...
        def get_tool_schemas(self) -> List: return []
        def get_executors(self) -> Dict[str, Any]: return {}
        def get_user_commands(self) -> List: return []
        def get_auto_approved_tools(self) -> List[str]: return []
        def get_system_instructions(self) -> Any: return None
        def reset_for_next_session(self) -> None: ...

    return _Plugin


def _make_ep(
    name: str,
    value: str,
    distribution: str | None = "evil-dist",
    plugin_name: str | None = None,
) -> Any:
    """Stand-in ``importlib.metadata.EntryPoint``.

    Mirrors the surface the registry touches: ``name``, ``value``,
    ``dist.name`` and ``load()``.  ``load`` is a ``MagicMock`` so tests
    can assert it was *never called* for a refused claim — the property
    that keeps a hostile distribution's module unimported.
    """
    ep = MagicMock()
    ep.name = name
    ep.value = value
    if distribution is None:
        ep.dist = None
    else:
        ep.dist = MagicMock()
        ep.dist.name = distribution
    ep.load.return_value = _plugin_class(plugin_name or name)
    return ep


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """No operator knobs unless the test sets them explicitly."""
    monkeypatch.delenv(ENV_ALLOW_SHADOW, raising=False)
    monkeypatch.delenv(ENV_ENTRY_POINT_ALLOWLIST, raising=False)


def _discover(registry: PluginRegistry, *eps: Any) -> List[str]:
    """Run entry-point discovery over exactly *eps*."""
    with patch(
        "shared.plugins.registry.importlib.metadata.entry_points",
        return_value=list(eps),
    ):
        return registry._discover_via_entry_points(plugin_kind="tool")


def _warnings(caplog: pytest.LogCaptureFixture) -> List[str]:
    return [r.message for r in caplog.records if r.levelno >= logging.WARNING]


# ----------------------------------------------------------------------
# Reserved names
# ----------------------------------------------------------------------


def test_builtin_names_are_read_from_the_plugin_package() -> None:
    """The reserved set is the built-in package's own module listing —
    not a hand-maintained literal that drifts as plugins are added."""
    names = builtin_plugin_names(refresh=True)
    for expected in ("cli", "permission", "file_edit", "mcp", "todo"):
        assert expected in names
    assert NEVER_SHADOWABLE <= names, (
        "Every never-shadowable name must correspond to a real built-in; "
        "a typo there would silently reserve nothing."
    )


def test_foreign_entry_point_claiming_builtin_name_is_refused_before_load(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The #684 headline: a third-party distribution declaring
    ``permission`` neither replaces the built-in nor gets imported."""
    ep = _make_ep("permission", "evil_pkg.plugins:create_plugin")
    registry = PluginRegistry()

    with caplog.at_level(logging.WARNING, logger="shared.plugins.registry"):
        discovered = _discover(registry, ep)

    assert discovered == []
    assert "permission" not in registry._plugins
    ep.load.assert_not_called(), (
        "A refused entry point must never be loaded — ep.load() imports "
        "the target module, which is code execution on the strength of "
        "being installed."
    )
    assert any(
        "permission" in m and "evil-dist" in m for m in _warnings(caplog)
    ), f"Refusal must name the plugin and the distribution. Got {_warnings(caplog)}"


def test_security_critical_names_refuse_even_with_operator_opt_in(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``JAATO_PLUGIN_ALLOW_SHADOW`` deliberately does not reach the
    plugins that mediate command execution, file writes and the
    permission prompt itself."""
    monkeypatch.setenv(ENV_ALLOW_SHADOW, "permission,cli,file_edit")
    registry = PluginRegistry()
    eps = [
        _make_ep(name, f"evil_pkg.{name}:create_plugin")
        for name in ("permission", "cli", "file_edit")
    ]

    with caplog.at_level(logging.WARNING, logger="shared.plugins.registry"):
        discovered = _discover(registry, *eps)

    assert discovered == []
    for ep in eps:
        ep.load.assert_not_called()
    assert all(
        "security-critical" in m for m in _warnings(caplog)
    ), _warnings(caplog)


def test_non_critical_builtin_may_be_shadowed_with_opt_in(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An operator can still replace a non-critical built-in — but the
    substitution is announced, not silent."""
    monkeypatch.setenv(ENV_ALLOW_SHADOW, "todo")
    ep = _make_ep("todo", "vendor_pkg.todo:create_plugin", distribution="vendor-pkg")
    registry = PluginRegistry()

    with caplog.at_level(logging.WARNING, logger="shared.plugins.registry"):
        discovered = _discover(registry, ep)

    assert discovered == ["todo"]
    assert any(
        "SHADOWS" in m and "vendor-pkg" in m for m in _warnings(caplog)
    ), _warnings(caplog)


def test_framework_declaring_its_own_plugins_is_silent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """jaato-server publishes its built-ins through the very same entry
    point groups.  The reservation must not fire on the framework's own
    declaration, and must not add noise to a normal boot."""
    ep = _make_ep(
        "permission",
        "shared.plugins.permission:create_plugin",
        distribution="jaato-server",
    )
    registry = PluginRegistry()

    with caplog.at_level(logging.WARNING, logger="shared.plugins.registry"):
        discovered = _discover(registry, ep)

    assert discovered == ["permission"]
    assert _warnings(caplog) == []


def test_out_of_tree_plugin_with_a_novel_name_loads_silently(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Entry points are the documented out-of-tree mechanism; only
    *shadowing* is restricted.  A plugin claiming a name the framework
    does not use is unaffected."""
    ep = _make_ep(
        "moon_phase",
        "moon_phase.plugin:create_plugin",
        distribution="jaato-moon-phase",
    )
    registry = PluginRegistry()

    with caplog.at_level(logging.WARNING, logger="shared.plugins.registry"):
        discovered = _discover(registry, ep)

    assert discovered == ["moon_phase"]
    assert _warnings(caplog) == []


def test_plugin_claiming_a_reserved_name_after_load_is_refused(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A plugin's ``name`` property need not match its entry-point name.
    Vetting only the declared name would let ``harmless`` register
    itself as ``permission``."""
    ep = _make_ep(
        "harmless",
        "evil_pkg.plugins:create_plugin",
        plugin_name="permission",
    )
    registry = PluginRegistry()

    with caplog.at_level(logging.WARNING, logger="shared.plugins.registry"):
        discovered = _discover(registry, ep)

    assert discovered == []
    assert "permission" not in registry._plugins
    assert any("permission" in m for m in _warnings(caplog))


# ----------------------------------------------------------------------
# Distribution allowlist
# ----------------------------------------------------------------------


def test_allowlist_refuses_distributions_not_named(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With an allowlist configured, a transitive dependency nobody
    chose no longer participates in plugin discovery at all."""
    monkeypatch.setenv(ENV_ENTRY_POINT_ALLOWLIST, "jaato-premium")
    ep = _make_ep(
        "moon_phase",
        "moon_phase.plugin:create_plugin",
        distribution="some-transitive-dep",
    )
    registry = PluginRegistry()

    with caplog.at_level(logging.WARNING, logger="shared.plugins.registry"):
        discovered = _discover(registry, ep)

    assert discovered == []
    ep.load.assert_not_called()
    assert any(ENV_ENTRY_POINT_ALLOWLIST in m for m in _warnings(caplog))


def test_allowlist_admits_named_distribution_regardless_of_spelling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Distribution names compare under PEP 503 normalisation, so
    ``Jaato_Premium`` and ``jaato-premium`` are the same entry."""
    monkeypatch.setenv(ENV_ENTRY_POINT_ALLOWLIST, "Jaato_Premium")
    ep = _make_ep(
        "session_ops",
        "jaato_premium.session_ops:create_plugin",
        distribution="jaato-premium",
    )
    registry = PluginRegistry()

    assert _discover(registry, ep) == ["session_ops"]


def test_allowlist_never_blocks_the_builtin_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An allowlist narrows third parties; it must not be able to
    switch off the framework's own plugins."""
    monkeypatch.setenv(ENV_ENTRY_POINT_ALLOWLIST, "nothing-real")
    ep = _make_ep("cli", "shared.plugins.cli:create_plugin", distribution="jaato-server")
    registry = PluginRegistry()

    assert _discover(registry, ep) == ["cli"]


def test_normalize_distribution_tolerates_missing_metadata() -> None:
    """``ep.dist`` is absent on some metadata backends; the policy must
    treat that as 'unknown', not crash mid-discovery."""
    assert normalize_distribution(None) == ""
    assert normalize_distribution("Jaato.Server") == "jaato-server"


# ----------------------------------------------------------------------
# Collisions and provenance
# ----------------------------------------------------------------------


def test_two_distributions_claiming_one_name_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """First writer still wins — but the loser is now named."""
    first = _make_ep(
        "moon_phase", "moon_a.plugin:create_plugin", distribution="moon-a",
    )
    second = _make_ep(
        "moon_phase", "moon_b.plugin:create_plugin", distribution="moon-b",
    )
    registry = PluginRegistry()

    with caplog.at_level(logging.WARNING, logger="shared.plugins.registry"):
        discovered = _discover(registry, first, second)

    assert discovered == ["moon_phase"]
    second.load.assert_not_called()
    assert any(
        "moon-a" in m and "moon-b" in m for m in _warnings(caplog)
    ), _warnings(caplog)


def test_rediscovery_by_the_same_provider_is_quiet(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``discover()`` runs more than once per process; a name held by
    the very module now re-offering it is not a shadow."""
    registry = PluginRegistry()
    ep = _make_ep(
        "moon_phase", "moon_a.plugin:create_plugin", distribution="moon-a",
    )
    _discover(registry, ep)

    with caplog.at_level(logging.WARNING, logger="shared.plugins.registry"):
        again = _discover(
            registry,
            _make_ep(
                "moon_phase",
                "moon_a.plugin:create_plugin",
                distribution="moon-a",
            ),
        )

    assert again == []
    assert _warnings(caplog) == []


def test_directory_scan_reports_a_foreign_incumbent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The other half of the silence: when a built-in module is skipped
    because an out-of-tree plugin already holds its name, say so."""
    registry = PluginRegistry()
    registry._plugins["todo"] = object()
    registry._plugin_sources["todo"] = PluginOrigin(
        name="todo",
        via="entry_point",
        module="evil_pkg.todo",
        distribution="evil-dist",
        entry_point="todo",
    )

    with caplog.at_level(logging.WARNING, logger="shared.plugins.registry"):
        registry._discover_via_directory(plugin_kind="tool")

    shadow = [m for m in _warnings(caplog) if "'todo'" in m]
    assert shadow, f"Expected a shadow warning for todo. Got {_warnings(caplog)}"
    assert "evil-dist" in shadow[0]


def test_directory_scan_is_quiet_for_the_frameworks_own_entry_points(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Built-ins are declared as entry points AND live in the scanned
    directory, so every normal boot hits the skip.  That must not warn."""
    registry = PluginRegistry()
    registry.discover()  # entry points first, then the directory scan

    with caplog.at_level(logging.WARNING, logger="shared.plugins.registry"):
        registry._discover_via_directory(plugin_kind="tool")

    assert [m for m in _warnings(caplog) if "already provided by" in m] == []


def test_a_builtin_reaching_the_registry_twice_is_quiet(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The framework reaches the registry by three routes — entry point,
    directory scan, and ``register_plugin`` for the session plugin, whose
    module (``shared.plugins.session.file_session``) is not the one the
    directory scan would import.  A built-in losing to a built-in is not
    the shadow this warning is for, and must not add boot noise."""
    registry = PluginRegistry()
    registry._plugins["session"] = object()
    registry._plugin_sources["session"] = PluginOrigin(
        name="session",
        via="registered",
        module="shared.plugins.session.file_session",
    )

    with caplog.at_level(logging.WARNING, logger="shared.plugins.registry"):
        registry._discover_via_directory(plugin_kind="tool")

    assert [m for m in _warnings(caplog) if "already provided by" in m] == []


def test_provenance_is_recorded_for_every_registered_plugin() -> None:
    """Item 4 of #684: a shadow should be visible without reading logs."""
    registry = PluginRegistry()
    registry.discover()

    source = registry.get_plugin_source("cli")
    assert source is not None
    assert source.builtin is True
    assert source.module.startswith("shared.plugins.cli")
    assert "shared.plugins.cli" in source.describe()
    assert set(registry.get_plugin_sources()) >= set(registry.list_available())


def test_provenance_names_an_out_of_tree_distribution() -> None:
    registry = PluginRegistry()
    _discover(
        registry,
        _make_ep(
            "moon_phase",
            "moon_phase.plugin:create_plugin",
            distribution="jaato-moon-phase",
        ),
    )

    source = registry.get_plugin_source("moon_phase")
    assert source is not None
    assert source.builtin is False
    assert "jaato-moon-phase" in source.describe()


# ----------------------------------------------------------------------
# Policy unit
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "claim,module,expected",
    [
        ("cli", "shared.plugins.cli", "builtin"),
        ("permission", "evil.pkg", "never_shadowable"),
        ("todo", "evil.pkg", "reserved"),
        ("moon_phase", "moon.pkg", "external"),
    ],
)
def test_decision_reasons_are_stable(
    claim: str, module: str, expected: str,
) -> None:
    """Callers and tests branch on ``reason``, so it is part of the
    contract rather than an incidental label."""
    assert evaluate_entry_point(claim, module, "evil-dist").reason == expected


# ----------------------------------------------------------------------
# Reporting (jaato-scaffold plugins)
# ----------------------------------------------------------------------


def test_scaffold_report_marks_out_of_tree_plugins() -> None:
    """Item 4 of #684: the plugin listing distinguishes a built-in from
    a plugin an installed distribution supplied, so a shadow is visible
    without reading daemon logs."""
    from shared.scaffold import explain, introspect

    builtin = introspect.PluginInfo(name="cli")
    introspect._stamp_origin(
        builtin,
        PluginOrigin(name="cli", via="entry_point", module="shared.plugins.cli"),
    )
    foreign = introspect.PluginInfo(name="todo")
    introspect._stamp_origin(
        foreign,
        PluginOrigin(
            name="todo",
            via="entry_point",
            module="evil_pkg.todo",
            distribution="evil-dist",
        ),
    )
    assert builtin.builtin is True
    assert foreign.builtin is False

    with patch.object(
        introspect, "plugins", return_value={"cli": builtin, "todo": foreign},
    ):
        data, text = explain.plugins()

    assert data["todo"]["builtin"] is False
    assert "evil-dist" in text, text
    # The built-in row stays bare — the marker is a signal, not decoration.
    cli_row = [ln for ln in text.splitlines() if ln.strip().startswith("cli ")]
    assert cli_row and "<-" not in cli_row[0], cli_row


def test_stamp_origin_leaves_defaults_when_provenance_is_missing() -> None:
    """A plugin registered by a path that never stamped an origin must
    not be misreported as foreign."""
    from shared.scaffold import introspect

    info = introspect.PluginInfo(name="mystery")
    introspect._stamp_origin(info, None)
    assert info.builtin is True
    assert info.source == ""
