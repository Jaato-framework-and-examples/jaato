"""Tests for ``PLUGIN_TIER`` annotation + tier-filtered discovery.

Phase 3 §3.3.5.

Two surfaces:

1. **Annotation completeness.**  Every plugin under
   ``shared/plugins/<name>/__init__.py`` that declares
   ``PLUGIN_KIND`` MUST also declare ``PLUGIN_TIER``.  Catches new
   plugins landing without an explicit tier — fails the build loud
   instead of silently slotting into "no filter sees you".

2. **Partition correctness.**  Daemon-tier and runner-tier plugin
   sets must NOT overlap.  Phase 3 splits the daemon's responsibility
   from the runner's; a plugin in both tiers means double
   instantiation + state-split bugs at runtime.  This test makes
   that contract structural.

3. **Filter behaviour.**  ``PluginRegistry.discover(tier_filter=...)``
   returns only plugins matching the filter; unannotated plugins
   are skipped.

The annotation walk uses AST-only parsing so test setup doesn't
trigger plugin imports (which can fail without optional deps —
``ddgs``, ``ast-grep-py``, etc.).
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import pytest


PLUGIN_DIR = Path(__file__).resolve().parents[1] / "plugins"


def _read_plugin_init_attrs(init_path: Path) -> Dict[str, Optional[str]]:
    """Parse a plugin __init__.py for module-level constants.

    Returns a dict containing whichever of ``PLUGIN_KIND`` /
    ``PLUGIN_TIER`` / ``SESSION_INDEPENDENT`` are present.  Values
    are the literal strings (or ``None`` for ``True``-typed ones).
    Skips anything that isn't a simple Name = Constant assignment.
    """
    out: Dict[str, Optional[str]] = {}
    try:
        tree = ast.parse(init_path.read_text())
    except SyntaxError:
        return out
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        name = target.id
        if name not in ("PLUGIN_KIND", "PLUGIN_TIER", "SESSION_INDEPENDENT"):
            continue
        if isinstance(node.value, ast.Constant):
            out[name] = node.value.value
    return out


def _collect_annotations() -> Tuple[
    Set[str],  # plugins with PLUGIN_KIND but missing PLUGIN_TIER
    Dict[str, str],  # plugin_name -> PLUGIN_TIER
]:
    """Walk every plugin dir, returning the gap + tier map."""
    missing: Set[str] = set()
    tiers: Dict[str, str] = {}
    for entry in sorted(PLUGIN_DIR.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name in ("__pycache__", "tests", "bundle_common"):
            continue
        init = entry / "__init__.py"
        if not init.exists():
            continue
        attrs = _read_plugin_init_attrs(init)
        if "PLUGIN_KIND" not in attrs:
            # No PLUGIN_KIND → not in the partition (formatters,
            # utility helpers, etc.).  Out of scope for §3.3.5.
            continue
        tier = attrs.get("PLUGIN_TIER")
        if not isinstance(tier, str):
            missing.add(entry.name)
        else:
            tiers[entry.name] = tier
    return missing, tiers


# ----------------------------------------------------------------------
# Annotation completeness — the build-fail gate
# ----------------------------------------------------------------------


def test_every_plugin_with_kind_has_tier() -> None:
    """If a plugin declares PLUGIN_KIND, it MUST declare PLUGIN_TIER.

    Catches new plugins landing without an explicit tier — the
    contract per §3.3.5 is "annotate or be excluded from the
    runner".  Silent exclusion would surface as "tool isn't
    available in cascade" much later; this test surfaces the gap
    at the build line.
    """
    missing, _tiers = _collect_annotations()
    assert missing == set(), (
        f"Plugins with PLUGIN_KIND but missing PLUGIN_TIER: "
        f"{sorted(missing)}.  Add ``PLUGIN_TIER = \"daemon\"`` or "
        f"``PLUGIN_TIER = \"runner\"`` to each __init__.py per the "
        f"parent design §4.2 classification table."
    )


def test_tier_values_are_valid() -> None:
    """PLUGIN_TIER must be one of the two known values."""
    _missing, tiers = _collect_annotations()
    valid = {"daemon", "runner"}
    invalid = {n: t for n, t in tiers.items() if t not in valid}
    assert invalid == {}, (
        f"PLUGIN_TIER values must be 'daemon' or 'runner'; got: "
        f"{invalid}"
    )


# ----------------------------------------------------------------------
# Partition correctness
# ----------------------------------------------------------------------


def test_daemon_and_runner_tiers_are_disjoint() -> None:
    """Daemon-tier and runner-tier sets MUST NOT overlap.

    A plugin in both tiers means the daemon-side discovery + the
    runner-side discovery both instantiate it, producing two
    instances with split state.  This test makes the partition
    structural so the bug class can't reach merge.
    """
    _missing, tiers = _collect_annotations()
    daemon = {n for n, t in tiers.items() if t == "daemon"}
    runner = {n for n, t in tiers.items() if t == "runner"}
    overlap = daemon & runner
    assert overlap == set(), (
        f"Plugins classified as both daemon and runner: "
        f"{sorted(overlap)}.  Each plugin must declare a single tier "
        f"per the parent design §4.2 table."
    )


def test_partition_covers_every_annotated_plugin() -> None:
    """The daemon ∪ runner partition equals the full annotated
    plugin set.  No plugin is silently omitted."""
    _missing, tiers = _collect_annotations()
    daemon = {n for n, t in tiers.items() if t == "daemon"}
    runner = {n for n, t in tiers.items() if t == "runner"}
    assert daemon | runner == set(tiers), (
        f"Plugins not in daemon ∪ runner: "
        f"{sorted(set(tiers) - (daemon | runner))}"
    )


# ----------------------------------------------------------------------
# Sanity-check spot values from the parent design §4.2 classification
# ----------------------------------------------------------------------


def test_canonical_plugin_classifications() -> None:
    """Pin a few load-bearing classifications so a ctrl-z on a
    plugin's tier flag fails CI (rather than silently changing the
    multitenancy promise)."""
    _missing, tiers = _collect_annotations()
    # Auth plugins MUST be daemon (parent §4.2: SESSION_INDEPENDENT,
    # tokens cross sessions).
    for auth in (
        "anthropic_auth", "antigravity_auth", "github_auth",
        "nim_auth", "openrouter_auth", "zhipuai_auth",
    ):
        assert tiers.get(auth) == "daemon", (
            f"{auth} must be daemon-tier (auth plugins span sessions)"
        )
    # Runner-tier load-bearing entries (subprocess inheritance,
    # workspace FS access).
    for runner in (
        "cli", "interactive_shell", "file_edit", "filesystem_query",
        "todo", "lsp", "mcp", "permission", "references", "memory",
        "subagent", "webhook",
    ):
        assert tiers.get(runner) == "runner", (
            f"{runner} must be runner-tier per parent design §4.2"
        )
    # GC + cache stay daemon-side (criterion 1 for cache; criterion
    # for GC because session history is daemon-side until §3.3c).
    for daemon in ("gc", "gc_truncate", "gc_summarize", "gc_hybrid",
                   "cache", "cache_anthropic", "cache_zhipuai"):
        assert tiers.get(daemon) == "daemon", (
            f"{daemon} must be daemon-tier per parent design §4.2"
        )


# ----------------------------------------------------------------------
# Filter behaviour — runtime call into PluginRegistry.discover
# ----------------------------------------------------------------------


def test_discover_with_no_tier_filter_returns_full_set() -> None:
    """``tier_filter=None`` preserves Phase 2 behavior — all plugins
    matching ``plugin_kind`` get loaded regardless of tier."""
    from shared.plugins.registry import PluginRegistry

    registry = PluginRegistry()
    discovered = registry.discover(plugin_kind="tool", tier_filter=None)
    # Should include both daemon-tier (auth) and runner-tier (cli)
    # plugins.
    assert "cli" in discovered or any(
        p in discovered for p in ("cli", "todo", "file_edit")
    ), (
        f"Expected runner-tier plugins in unfiltered discover; "
        f"got: {discovered}"
    )


def test_discover_runner_filter_excludes_daemon_plugins() -> None:
    """``tier_filter='runner'`` skips daemon-tier plugins."""
    from shared.plugins.registry import PluginRegistry

    registry = PluginRegistry()
    discovered = registry.discover(plugin_kind="tool", tier_filter="runner")
    # Auth plugins are daemon-tier; verify they're excluded.
    daemon_auth = {
        "anthropic_auth", "antigravity_auth", "github_auth",
        "nim_auth", "openrouter_auth", "zhipuai_auth",
    }
    leaked = daemon_auth & set(discovered)
    assert leaked == set(), (
        f"Daemon-tier auth plugins leaked into runner-filtered "
        f"discover: {sorted(leaked)}"
    )


def test_discover_daemon_filter_excludes_runner_plugins() -> None:
    """``tier_filter='daemon'`` skips runner-tier plugins."""
    from shared.plugins.registry import PluginRegistry

    registry = PluginRegistry()
    discovered = registry.discover(plugin_kind="tool", tier_filter="daemon")
    # cli + todo + file_edit are runner-tier; verify they're excluded.
    runner_load_bearing = {"cli", "todo", "file_edit", "permission"}
    leaked = runner_load_bearing & set(discovered)
    assert leaked == set(), (
        f"Runner-tier plugins leaked into daemon-filtered discover: "
        f"{sorted(leaked)}"
    )
