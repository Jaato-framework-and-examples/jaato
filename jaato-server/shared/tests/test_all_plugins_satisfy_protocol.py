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

import ast
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


def _base_is_protocol(node: ast.ClassDef) -> bool:
    """True if the class derives from ``Protocol`` (``Name`` or
    ``typing.Protocol`` attribute form)."""
    for b in node.bases:
        if isinstance(b, ast.Name) and b.id == "Protocol":
            return True
        if isinstance(b, ast.Attribute) and b.attr == "Protocol":
            return True
    return False


def _concrete_classes_defining(py_path: Path, marker_method: str) -> dict[str, set[str]]:
    """AST-scan *py_path* for top-level classes that define
    *marker_method* and are NOT a ``Protocol`` definition.

    Returns ``{class_name: {method names}}`` — env-free (no import), so
    it works in the contract-guards job where optional backends (otel,
    etc.) aren't installed.
    """
    out: dict[str, set[str]] = {}
    tree = ast.parse(py_path.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or _base_is_protocol(node):
            continue
        methods = {
            n.name for n in node.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        if marker_method in methods:
            out[node.name] = methods
    return out


def test_telemetry_plugins_have_reset_for_next_session() -> None:
    """Pin: every concrete ``TelemetryPlugin`` impl declares
    ``reset_for_next_session``.

    Complements the per-dir ``plugin.py`` scan above — telemetry is NOT
    registry-discovered (no ``PLUGIN_KIND``) and its concrete impls live
    in sibling files (``otel_plugin.py``, ``null_plugin.py``), so that
    scan never reaches them.  ``OTelPlugin`` shipped without the method
    (added to the ``TelemetryPlugin`` protocol in the cascade-sharing
    arc), failing ``isinstance`` against the protocol it's typed as and
    risking an ``AttributeError`` if the framework wires the per-session
    reset to telemetry.  Concrete impls are detected by the
    telemetry-specific ``begin_session`` method (excludes the ``Protocol``
    definition itself).  AST-based — env-free under the minimal
    contract-guards install.
    """
    tele_dir = (
        Path(__file__).resolve().parents[2]
        / "shared" / "plugins" / "telemetry"
    )
    found: dict[str, set[str]] = {}
    for py in sorted(tele_dir.glob("*.py")):
        found.update(_concrete_classes_defining(py, "begin_session"))
    assert found, (
        "no concrete TelemetryPlugin impls detected (marker method "
        "'begin_session') — the guard is mis-scoped; verify telemetry "
        "plugin layout."
    )
    missing = sorted(c for c, m in found.items() if "reset_for_next_session" not in m)
    assert not missing, (
        f"concrete TelemetryPlugin impls missing reset_for_next_session: "
        f"{missing}.  The TelemetryPlugin protocol requires it (cascade-"
        f"sharing arc); without it the class fails isinstance against the "
        f"protocol and crashes if the per-session reset is wired.  Add a "
        f"method (no-op if it holds no per-session state) — see "
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


@pytest.mark.parametrize(
    "plugin_dir", _plugin_directories(), ids=lambda p: p.name
)
def test_plugin_does_not_import_from_bare_jaato_facade(plugin_dir: Path) -> None:
    """Pin: no ``plugin.py`` imports from the bare ``jaato`` facade.

    The ``jaato`` package is a convenience re-export for EXTERNAL SDK
    consumers.  Internal framework plugins must import framework types
    from the canonical ``jaato_sdk...`` paths (as all other plugins do).

    The bare facade import (``from jaato import ToolSchema``) failed to
    resolve inside the live runner's forked plugin-discovery context —
    ``jaato`` resolved to a namespace ("unknown location") while
    ``jaato_sdk`` stayed warm in ``sys.modules`` post-fork — so the two
    plugins that used it (``environment``, ``calculator``) were silently
    dropped at every session bootstrap.  A plain unconfined import
    succeeds, so only a source-level gate catches a re-introduction.

    Source-level AST check (not import-time) so it runs even when a
    plugin's runtime deps aren't installed.
    """
    plugin_py = plugin_dir / "plugin.py"
    if not plugin_py.exists():
        pytest.skip(f"{plugin_dir.name}: missing plugin.py")
    tree = ast.parse(plugin_py.read_text())
    offenders = [
        f"line {node.lineno}: from {node.module} import "
        + ", ".join(a.name for a in node.names)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.level == 0
        and node.module == "jaato"  # exactly the facade, not jaato_sdk / jaato.x
    ]
    assert not offenders, (
        f"{plugin_dir.name}/plugin.py imports from the bare 'jaato' "
        f"facade (fails in the forked runner). Repoint to the canonical "
        f"jaato_sdk path like every other plugin. Offending:\n  "
        + "\n  ".join(offenders)
    )
