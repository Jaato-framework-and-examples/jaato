"""A plugin that declares no ``PLUGIN_TIER`` is skipped LOUDLY (issue #917).

Background
----------
Tier-filtered discovery excludes an unannotated plugin under **every**
filter — that rule is deliberate (§3.3.5) and is not what this test
pins.  What it pins is the diagnostic.

``_tier_filter_matches`` conflates two different events, and until this
fix both were recorded by one debug ``_trace``:

- ``PLUGIN_TIER="daemon"`` under ``tier_filter="runner"`` is the
  partition working.  Announcing it would print a line per daemon-tier
  plugin on every runner bootstrap.
- ``PLUGIN_TIER`` **absent** is an authoring mistake, and the only one
  an out-of-tree author can make without knowing the concept exists.
  ``test_plugin_tier_partition`` fails the build on it — but that walk
  is an AST scan of ``shared/plugins/``, so it cannot see a third-party
  distribution at all.

The cost of the silence, as reported: the author installs the
distribution, runs ``jaato-scaffold plugins``, sees the plugin listed
with its provenance line, writes ``plugins: [m365]`` in a profile — and
the session comes up without the tools, with no error, no warning, and
one debug trace.

Note the asymmetry that makes this worse than a trust refusal:
``ep.load()`` has already run by the time the tier is read, so the
module is imported and *then* discarded.

This is the same promotion the protocol-gap check got at PR #171, for
the same audience.
"""

from __future__ import annotations

import logging
import sys
import types
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from shared.plugins.registry import PluginRegistry, _report_tier_skip


class _FakePlugin:
    """Protocol-complete tool plugin, so the ONLY gate under test is tier."""

    @property
    def name(self) -> str:
        return "m365"

    def initialize(self, config: Any = None) -> None: ...
    def shutdown(self) -> None: ...
    def get_tool_schemas(self) -> List: return []
    def get_executors(self) -> Dict[str, Any]: return {}
    def get_user_commands(self) -> List: return []
    def get_auto_approved_tools(self) -> List[str]: return []
    def get_system_instructions(self) -> Any: return None
    def reset_for_next_session(self) -> None: ...


@pytest.fixture
def out_of_tree_package(request):
    """Register a fake out-of-tree distribution's modules in ``sys.modules``.

    ``_lookup_module_tier`` reads ``PLUGIN_TIER`` off the loaded
    factory's module or its parent package, both looked up in
    ``sys.modules`` — so a realistic fixture is two module objects, not a
    patched lookup.  Patching the lookup would test the warning against
    a stub of the very mechanism whose contract is at issue.

    Yields a factory ``make(tier) -> create_plugin`` whose
    ``__module__`` is ``jaato_m365.plugin`` and whose parent package
    declares (or omits) *tier*.
    """
    created: List[str] = []

    def make(tier: Optional[str]):
        pkg = types.ModuleType("jaato_m365")
        pkg.PLUGIN_KIND = "tool"
        if tier is not None:
            pkg.PLUGIN_TIER = tier
        mod = types.ModuleType("jaato_m365.plugin")
        sys.modules["jaato_m365"] = pkg
        sys.modules["jaato_m365.plugin"] = mod
        created.extend(["jaato_m365", "jaato_m365.plugin"])

        def create_plugin() -> _FakePlugin:
            return _FakePlugin()

        create_plugin.__module__ = "jaato_m365.plugin"
        return create_plugin

    yield make

    for name in created:
        sys.modules.pop(name, None)


def _make_ep(factory: Any) -> Any:
    ep = MagicMock()
    ep.name = "m365"
    ep.value = "jaato_m365.plugin:create_plugin"
    ep.load.return_value = factory
    # ``ep.dist.name`` is what the trust gate resolves into
    # ``PluginOrigin.distribution``; a MagicMock would otherwise supply
    # a repr-shaped stand-in and the assertion below would pass on it.
    ep.dist.name = "jaato-m365"
    return ep


def _discover(factory: Any, *, tier_filter: Optional[str], caplog) -> List[str]:
    registry = PluginRegistry()
    with patch(
        "shared.plugins.registry.importlib.metadata.entry_points",
        return_value=[_make_ep(factory)],
    ):
        with caplog.at_level(logging.WARNING, logger="shared.plugins.registry"):
            return registry._discover_via_entry_points(
                plugin_kind="tool", tier_filter=tier_filter,
            )


def _warnings(caplog) -> List[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


# ---------------------------------------------------------------- the defect


def test_missing_annotation_warns_and_says_how_to_fix(
    out_of_tree_package, caplog: pytest.LogCaptureFixture,
) -> None:
    """The reported case: no ``PLUGIN_TIER``, runner filter, silent drop.

    The skip itself is correct and unchanged — what must not stay quiet
    is that it happened.
    """
    discovered = _discover(
        out_of_tree_package(None), tier_filter="runner", caplog=caplog,
    )

    assert discovered == [], "the skip is the contract; only the diagnostic changed"

    warnings = _warnings(caplog)
    assert warnings, (
        "an unannotated plugin dropped by the runner MUST be announced — "
        "a debug trace is what made this cost an afternoon"
    )
    msg = warnings[0]
    # Everything the author needs to act, without reading registry.py.
    assert "m365" in msg, "name the entry point"
    assert "jaato_m365.plugin:create_plugin" in msg, "name what declared it"
    assert "jaato-m365" in msg, (
        "name the DISTRIBUTION — on a machine with several plugin "
        "packages installed, that is what turns 'a plugin is broken' "
        "into 'this dependency is'"
    )
    assert "PLUGIN_TIER" in msg, "name the annotation"
    assert "jaato_m365" in msg and "__init__.py" in msg, (
        f"name the file to edit, not just the concept. Got: {msg!r}"
    )
    assert '"runner"' in msg, "name the value that would fix it"


def test_warning_names_the_entry_point_path_not_the_directory_path(
    out_of_tree_package, caplog: pytest.LogCaptureFixture,
) -> None:
    """Two discovery paths can produce this skip and the fix differs.

    Mirrors ``test_registry_entry_point_protocol_warning``'s equivalent
    check: an operator must know whether to edit a distribution's
    ``pyproject.toml`` package or an in-tree plugin module.
    """
    _discover(out_of_tree_package(None), tier_filter="runner", caplog=caplog)
    assert "Entry point" in _warnings(caplog)[0]


# ------------------------------------------------------- the non-defect cases


def test_mismatched_annotation_stays_quiet(
    out_of_tree_package, caplog: pytest.LogCaptureFixture,
) -> None:
    """A daemon-tier plugin not loading in the runner is the partition
    working, not a mistake — warning on it would print a line per
    daemon plugin per runner bootstrap and train operators to ignore
    the one that matters."""
    discovered = _discover(
        out_of_tree_package("daemon"), tier_filter="runner", caplog=caplog,
    )

    assert discovered == []
    assert _warnings(caplog) == [], (
        f"correct partitioning must not warn. Got: {_warnings(caplog)}"
    )


def test_annotated_runner_plugin_loads_silently(
    out_of_tree_package, caplog: pytest.LogCaptureFixture,
) -> None:
    """The happy path stays silent — a diagnostic that fires on success
    is noise, and noise is how the real signal gets filtered out."""
    discovered = _discover(
        out_of_tree_package("runner"), tier_filter="runner", caplog=caplog,
    )

    assert discovered == ["m365"]
    assert _warnings(caplog) == []


def test_daemon_callable_is_admitted_by_the_runner_filter(
    out_of_tree_package, caplog: pytest.LogCaptureFixture,
) -> None:
    """Cross-tier plugins load under BOTH filters — the promotion must
    not have narrowed ``_TIER_FILTER_ACCEPTS``."""
    discovered = _discover(
        out_of_tree_package("daemon_callable"), tier_filter="runner",
        caplog=caplog,
    )

    assert discovered == ["m365"]
    assert _warnings(caplog) == []


def test_unfiltered_discovery_still_loads_an_unannotated_plugin(
    out_of_tree_package, caplog: pytest.LogCaptureFixture,
) -> None:
    """``tier_filter=None`` admits everything (the daemon-side and
    introspection behaviour).  The warning belongs to the filtered
    paths only: nothing was skipped here, so there is nothing to
    announce.

    This asymmetry IS the reported bug's shape — ``jaato-scaffold
    plugins`` discovers unfiltered and therefore lists a plugin the
    runner drops.  The registry cannot fix that half; ``explain
    plugins`` marks the row instead.
    """
    discovered = _discover(
        out_of_tree_package(None), tier_filter=None, caplog=caplog,
    )

    assert discovered == ["m365"]
    assert _warnings(caplog) == []


# ------------------------------------------------------- the directory wording


def test_directory_path_wording_points_at_the_in_tree_plugin(caplog) -> None:
    """The shared reporter is used by both discovery paths, and each
    must name a fix location that exists for it.

    Exercised through the helper rather than through
    ``_discover_via_directory``: that scan hardcodes
    ``importlib.import_module(f".{name}", package="shared.plugins")``,
    so a temp directory cannot stand in for a plugin package, and every
    in-tree plugin is annotated (``test_plugin_tier_partition`` fails
    the build otherwise) so there is no live case to observe.
    """
    with caplog.at_level(logging.WARNING, logger="shared.plugins.registry"):
        _report_tier_skip(
            what="Plugin 'weather' (shared.plugins.weather)",
            plugin_tier=None,
            tier_filter="runner",
            fix_location="weather/__init__.py",
        )

    msg = _warnings(caplog)[0]
    assert "weather/__init__.py" in msg
    assert "Entry point" not in msg
