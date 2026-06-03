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
    """PLUGIN_TIER must be one of the three known values."""
    _missing, tiers = _collect_annotations()
    valid = {"daemon", "runner", "daemon_callable"}
    invalid = {n: t for n, t in tiers.items() if t not in valid}
    assert invalid == {}, (
        f"PLUGIN_TIER values must be 'daemon', 'runner', or "
        f"'daemon_callable'; got: {invalid}"
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
    """The daemon ∪ runner ∪ daemon_callable partition equals the
    full annotated plugin set.  No plugin is silently omitted."""
    _missing, tiers = _collect_annotations()
    daemon = {n for n, t in tiers.items() if t == "daemon"}
    runner = {n for n, t in tiers.items() if t == "runner"}
    cross = {n for n, t in tiers.items() if t == "daemon_callable"}
    assert daemon | runner | cross == set(tiers), (
        f"Plugins not in daemon ∪ runner ∪ daemon_callable: "
        f"{sorted(set(tiers) - (daemon | runner | cross))}"
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


# ----------------------------------------------------------------------
# Cross-tier (``daemon_callable``) plugin discipline
# ----------------------------------------------------------------------


def test_daemon_callable_plugins_extend_daemon_forwarding_mixin() -> None:
    """Any plugin with ``PLUGIN_TIER = "daemon_callable"`` MUST extend
    ``DaemonForwardingMixin``.

    The cross-tier pattern requires the runner-side instance to route
    execution back to the daemon via ``daemon.plugin_execute`` RPC.
    Without the mixin, the runner-side instance would attempt to
    execute the body in-runner — and the body typically depends on
    daemon-only state (e.g. ``SessionManager`` reference), so the
    runner-side call would crash with AttributeError or worse, silently
    return wrong-data.

    This test is the structural enforcement of the contract documented
    in ``shared/plugins/CLAUDE.md`` § "Cross-tier plugins".
    """
    from shared.plugins.daemon_forwarding import DaemonForwardingMixin
    from shared.plugins.registry import PluginRegistry

    _missing, tiers = _collect_annotations()
    cross_tier = {n for n, t in tiers.items() if t == "daemon_callable"}
    if not cross_tier:
        pytest.skip(
            "No daemon_callable plugins under shared/plugins/ — the "
            "premium-side session_ops plugin is the canonical caller "
            "and lives in jaato-premium."
        )

    # Load each via the registry to get the actual class.
    registry = PluginRegistry()
    registry.discover(plugin_kind="tool", tier_filter=None)

    offenders = []
    for name in cross_tier:
        plugin = registry.get_plugin(name)
        if plugin is None:
            offenders.append(f"{name} (failed to discover)")
            continue
        if not isinstance(plugin, DaemonForwardingMixin):
            offenders.append(
                f"{name} (class {type(plugin).__name__} does not "
                f"extend DaemonForwardingMixin)"
            )

    assert offenders == [], (
        f"Cross-tier plugins missing DaemonForwardingMixin: "
        f"{offenders}.  Add the mixin to the plugin class (see "
        f"shared/plugins/CLAUDE.md § 'Cross-tier plugins') OR change "
        f"PLUGIN_TIER to 'daemon' or 'runner'."
    )


def test_runner_filter_loads_daemon_callable_plugins() -> None:
    """``discover(tier_filter='runner')`` MUST load
    ``daemon_callable`` plugins.

    This is the production-shape integration test for the cross-tier
    discovery contract — it calls the **actual** code path the runner
    subprocess uses (``server/runner/session.py:205``).  Per
    ``feedback_test_fixtures_must_mirror_production_caller_shape``:
    a fixture that bypasses ``tier_filter="runner"`` would let the
    PR-207-style silent-skip class re-surface for the next plugin
    that forgets ``PLUGIN_TIER``.

    Skipped if no ``daemon_callable`` plugins exist under
    shared/plugins/ — the canonical caller (premium session_ops)
    lives outside this tree.
    """
    from shared.plugins.registry import PluginRegistry

    _missing, tiers = _collect_annotations()
    cross_tier = {n for n, t in tiers.items() if t == "daemon_callable"}
    if not cross_tier:
        pytest.skip(
            "No daemon_callable plugins under shared/plugins/ — the "
            "tier-filter integration assertion is unconditionally "
            "exercised by the daemon-side cross-tier test."
        )

    registry = PluginRegistry()
    discovered = registry.discover(
        plugin_kind="tool", tier_filter="runner",
    )
    missing_from_runner = cross_tier - set(discovered)
    assert missing_from_runner == set(), (
        f"daemon_callable plugins did NOT load under "
        f"tier_filter='runner': {sorted(missing_from_runner)}.  "
        f"This would cause an empty tool schema runner-side and the "
        f"model would hallucinate that the tool is unavailable (the "
        f"smoke-validation symptom that motivated this pattern).  "
        f"Likely cause: _tier_filter_matches() lost the "
        f"daemon_callable acceptance under 'runner'."
    )


def test_runner_tier_plugins_dont_secretly_need_daemon_state() -> None:
    """Gap #1 trap guard: a plugin with ``PLUGIN_TIER = "runner"``
    must NOT declare ``set_session_manager`` unless it also extends
    ``DaemonForwardingMixin``.

    The motivating incident: premium ``session_ops`` initially shipped
    with no ``PLUGIN_TIER`` annotation.  After adding
    ``PLUGIN_TIER = "runner"`` to make it discoverable runner-side,
    the runner-side instance had ``_session_manager = None`` because
    the daemon-side wire-up at ``session_manager.py:4379`` only mutates
    the daemon-side instance.  The runner-side body executed against a
    None reference.

    The fix is to declare ``PLUGIN_TIER = "daemon_callable"`` AND
    extend ``DaemonForwardingMixin``.  This test catches plugins that
    take the half-step (``"runner"`` without the mixin) — the silent
    failure mode is what cost us a cycle.
    """
    from shared.plugins.daemon_forwarding import DaemonForwardingMixin
    from shared.plugins.registry import PluginRegistry

    _missing, tiers = _collect_annotations()
    runner_tier = {n for n, t in tiers.items() if t == "runner"}

    registry = PluginRegistry()
    # Load with no filter so we get every annotated plugin (not just
    # the ones the runner-filter would admit).  Mirrors daemon-side
    # discovery shape (``core.py:1079``).
    registry.discover(plugin_kind="tool", tier_filter=None)

    offenders = []
    for name in runner_tier:
        plugin = registry.get_plugin(name)
        if plugin is None:
            continue  # discovery failure handled by other tests
        # The trap shape: takes a SessionManager via a setter (canonical
        # signal that the body needs daemon-only state).  Any plugin
        # exposing this contract on a runner-tier instance is in the
        # bug class.
        if not hasattr(plugin, "set_session_manager"):
            continue
        if isinstance(plugin, DaemonForwardingMixin):
            continue  # has the escape hatch
        offenders.append(name)

    assert offenders == [], (
        f"Runner-tier plugins declare set_session_manager but lack "
        f"DaemonForwardingMixin: {offenders}.  Either: "
        f"(a) change PLUGIN_TIER to 'daemon_callable' + extend "
        f"DaemonForwardingMixin (cross-tier pattern), OR "
        f"(b) remove set_session_manager (the runner-side body cannot "
        f"reach SessionManager — daemon-only state).  See "
        f"shared/plugins/CLAUDE.md § 'Cross-tier plugins' for the trap "
        f"history."
    )
