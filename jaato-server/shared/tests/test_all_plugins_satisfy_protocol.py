"""Regression test for the silently-skipped-plugins class.

Phase 1a (PR #160) added ``reset_for_next_session`` to the
``ToolPlugin`` + ``EnrichmentPlugin`` protocols at
``jaato-sdk/jaato_sdk/plugins/base.py`` but only added the method to
13 plugins.  The other ~40 plugins were silently dropped from
``PluginRegistry.discover()`` because the registry's
``isinstance(plugin, ToolPlugin)`` check at ``registry.py:667``
fails for any plugin missing a Protocol method.

This test pins the contract by enumerating every plugin discovered
from disk + asserting it satisfies its declared Protocol.  Adding
a new Protocol method without rolling it out to every plugin will
fail this test loudly instead of silently disabling 40+ plugins.

Hotfix landed 2026-05-20 (server 0.6.148) — see
``project_backlog_cascade_sharing_silently_skipped_plugins.md``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from jaato_sdk.plugins.base import ToolPlugin, EnrichmentPlugin
from shared.plugins.registry import PluginRegistry


def _plugin_directories() -> list[Path]:
    """Locate every ``plugin.py`` file under ``shared/plugins/``."""
    plugins_root = (
        Path(__file__).resolve()
        .parents[2]  # jaato-server/
        / "shared" / "plugins"
    )
    return sorted(
        p.parent
        for p in plugins_root.rglob("plugin.py")
        if "build/" not in str(p)
    )


@pytest.mark.parametrize("plugin_dir", _plugin_directories(), ids=lambda p: p.name)
def test_plugin_file_has_reset_for_next_session_method(plugin_dir: Path) -> None:
    """Pin: every ``plugin.py`` under ``shared/plugins/`` declares
    ``def reset_for_next_session`` in source.

    Source-level check (not import-time) — catches the class even
    when the plugin's runtime dependencies aren't installed in the
    test env.  Mermaid + hidden_content_filter formatters are
    streaming-protocol plugins (no ``PLUGIN_KIND``); they skip the
    ToolPlugin discovery path entirely and don't need the method.
    """
    init_py = plugin_dir / "__init__.py"
    plugin_py = plugin_dir / "plugin.py"
    if not init_py.exists() or not plugin_py.exists():
        pytest.skip(f"{plugin_dir.name}: missing __init__.py or plugin.py")
    init_src = init_py.read_text()
    if "PLUGIN_KIND" not in init_src:
        # No PLUGIN_KIND → not a ToolPlugin/EnrichmentPlugin discovery
        # path; doesn't need the method.
        pytest.skip(
            f"{plugin_dir.name}: no PLUGIN_KIND declared "
            f"(streaming protocol or formatter)"
        )
    src = plugin_py.read_text()
    assert "def reset_for_next_session" in src, (
        f"{plugin_dir.name}/plugin.py is missing "
        f"``def reset_for_next_session`` — will be silently skipped "
        f"by ``PluginRegistry.discover()`` per the Protocol "
        f"structural-typing check.  Add a NO-OP method (or a real "
        f"one if the litmus test calls for it) — see "
        f"docs/design/runner-cascade-sharing.md §4.3."
    )


def test_registry_discovers_full_plugin_set() -> None:
    """End-to-end pin: a fresh ``PluginRegistry.discover()`` finds
    at least 30 plugins.  Pre-hotfix it found ~13 (only the Phase 1
    plugins that had the method).  Pins the count to catch a
    future Protocol regression that silently drops large swaths
    of the framework."""
    reg = PluginRegistry()
    reg.discover()
    discovered = sorted(reg._plugins.keys())
    # Must include the canonical core plugins that were silently
    # disabled pre-hotfix (these are the user-visible regression
    # surface — agents lose their tool surface when these go missing).
    canonical_must_present = {
        "cli", "file_edit", "template", "references", "web_search",
        "web_fetch", "introspection", "filesystem_query", "todo",
        "permission", "lsp",
    }
    missing = canonical_must_present - set(discovered)
    assert not missing, (
        f"PluginRegistry.discover() didn't surface canonical "
        f"plugins: {sorted(missing)}.  Total discovered: "
        f"{len(discovered)} ({discovered})"
    )
    assert len(discovered) >= 30, (
        f"PluginRegistry.discover() found only {len(discovered)} "
        f"plugins — expected 30+.  Likely cause: another Protocol "
        f"method was added without rolling out to every plugin "
        f"(same shape as the Phase 1a regression).  Discovered: "
        f"{discovered}"
    )
