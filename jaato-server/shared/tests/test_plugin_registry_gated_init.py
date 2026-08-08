"""Tests for ``PluginRegistry.expose_all(requested_plugins=...)``
— option C of the references-init investigation (2026-05-15).

Pre-2026-05-15 the runner-tier ``expose_all`` initialized EVERY
discovered runner-tier plugin regardless of whether the session's
``profile.plugins`` listed it.  The references plugin's
``SentenceTransformer`` model load (~7.6s) ran on every session
even for profiles that never used references.

This PR adds a ``requested_plugins`` kwarg.  When set, the
registry initializes ONLY:
1. plugins whose name is in the requested list,
2. plus the unconditionally-essential set
   (``_ALWAYS_INITIALIZE_PLUGINS`` — currently
   ``{"introspection", "permission"}``),
3. plus all enrichment-only plugins (their enrichment fires for
   every session regardless of profile.plugins).

These tests pin the contract — both back-compat (None preserves
original behaviour) and the new gated behaviour.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from shared.plugins.registry import PluginRegistry


class _StubPlugin:
    """Minimal plugin stub that records initialize() calls.

    Idempotent on initialize() — mirrors real plugin behaviour
    (references checks ``if self._model is not None: return``;
    mcp checks ``if self._initialized: return``; etc.).  Without
    this idempotency the framework's PARALLEL_INIT path calls
    initialize() twice — once in the background thread, once via
    ``expose_tool``'s normal flow — and the test would observe
    spurious double-init.

    The registry uses ``hasattr(plugin, '_initialized')`` as the
    sentinel for enrichment-only plugin initialization
    (registry.py:897), so the attribute is set ONLY after the
    first initialize() call (not preemptively in __init__).
    """

    PARALLEL_INIT = False

    def __init__(self, name: str) -> None:
        self.name = name
        self.initialize_calls: List[Optional[Dict[str, Any]]] = []
        # ``_initialized`` is INTENTIONALLY not set here — the
        # registry's enrichment-only path uses ``hasattr`` as the
        # idempotency check.  Setting it here would make the
        # registry skip initialize on enrichment-only stubs.

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        if getattr(self, '_initialized', False):
            return  # idempotent, like real plugins
        self.initialize_calls.append(config)
        self._initialized = True

    def get_tool_schemas(self):
        return []

    def get_executors(self):
        return {}

    def get_user_commands(self):
        return []

    def get_auto_approved_tools(self):
        return []

    def shutdown(self) -> None:
        pass


def _make_registry(plugin_names: List[str], enrichment_only: set = None) -> PluginRegistry:
    """Construct a registry pre-populated with named stub plugins.

    Bypasses the discovery code path (which scans filesystem
    paths) and inserts stubs directly so tests can pin
    ``expose_all`` behaviour without filesystem dependencies.
    """
    reg = PluginRegistry()
    for name in plugin_names:
        reg._plugins[name] = _StubPlugin(name)
    if enrichment_only:
        reg._enrichment_only |= set(enrichment_only)
    return reg


class TestExposeAllBackCompat:
    """When ``requested_plugins`` is None (default), every plugin
    gets initialized — preserves pre-2026-05-15 behaviour for the
    daemon-side ``server/core.py`` caller + tests that don't pass
    the kwarg."""

    def test_none_initializes_all_plugins(self):
        reg = _make_registry(["alpha", "beta", "gamma"])

        reg.expose_all()

        for name in ("alpha", "beta", "gamma"):
            plugin = reg._plugins[name]
            assert len(plugin.initialize_calls) == 1, (
                f"plugin {name!r} should have been initialized when "
                f"requested_plugins is None (back-compat)"
            )

    def test_omitted_kwarg_initializes_all(self):
        """The signature defaults ``requested_plugins=None`` — calling
        without the kwarg must keep back-compat."""
        reg = _make_registry(["alpha", "beta"])

        reg.expose_all({})

        for name in ("alpha", "beta"):
            assert len(reg._plugins[name].initialize_calls) == 1


class TestExposeAllGated:
    """When ``requested_plugins`` is non-None, ``expose_all``
    initializes only the requested set ∪ always-initialize ∪
    enrichment-only."""

    def test_only_requested_plugins_initialized(self):
        reg = _make_registry(["alpha", "beta", "gamma", "delta"])

        reg.expose_all(requested_plugins=["alpha", "gamma"])

        assert len(reg._plugins["alpha"].initialize_calls) == 1
        assert len(reg._plugins["gamma"].initialize_calls) == 1
        assert reg._plugins["beta"].initialize_calls == [], (
            "beta was not in requested set; must NOT be initialized"
        )
        assert reg._plugins["delta"].initialize_calls == [], (
            "delta was not in requested set; must NOT be initialized"
        )

    def test_empty_requested_list_initializes_only_essentials(self):
        """``requested_plugins=[]`` is distinct from ``None``:
        no caller-requested plugins, only the unconditionally-
        essential set fires."""
        reg = _make_registry(["alpha", "introspection", "permission", "beta"])

        reg.expose_all(requested_plugins=[])

        # introspection + permission are essential — always init.
        assert len(reg._plugins["introspection"].initialize_calls) == 1
        assert len(reg._plugins["permission"].initialize_calls) == 1
        # alpha + beta were not requested → no init.
        assert reg._plugins["alpha"].initialize_calls == []
        assert reg._plugins["beta"].initialize_calls == []

    def test_essentials_initialized_even_when_not_requested(self):
        """Caller may forget to list ``introspection`` /
        ``permission`` — they still initialize because the
        registry's always-init set covers them."""
        reg = _make_registry(["alpha", "introspection", "permission"])

        # Request only alpha; essentials must still fire.
        reg.expose_all(requested_plugins=["alpha"])

        assert len(reg._plugins["alpha"].initialize_calls) == 1
        assert len(reg._plugins["introspection"].initialize_calls) == 1
        assert len(reg._plugins["permission"].initialize_calls) == 1

    def test_enrichment_only_plugins_always_initialized(self):
        """Enrichment-only plugins (PLUGIN_KIND="enrichment") fire
        enrichment for every session regardless of profile.plugins;
        their initialize() must run even when not in
        requested_plugins."""
        reg = _make_registry(
            ["alpha", "beta", "enricher1", "enricher2"],
            enrichment_only={"enricher1", "enricher2"},
        )

        reg.expose_all(requested_plugins=["alpha"])

        # alpha requested → initialized.
        assert len(reg._plugins["alpha"].initialize_calls) == 1
        # beta not requested → skipped.
        assert reg._plugins["beta"].initialize_calls == []
        # enrichment-only → always initialized.
        assert len(reg._plugins["enricher1"].initialize_calls) == 1
        assert len(reg._plugins["enricher2"].initialize_calls) == 1

    def test_skipped_plugin_remains_registered(self):
        """A plugin that ``expose_all`` skipped stays in
        ``self._plugins`` (registered) but not in ``self._exposed``.
        Introspection still sees it; its tools just don't get
        registered."""
        reg = _make_registry(["alpha", "beta"])

        reg.expose_all(requested_plugins=["alpha"])

        assert "beta" in reg._plugins, (
            "skipped plugin must remain registered for introspection"
        )
        assert "beta" not in reg._exposed, (
            "skipped plugin must NOT be in _exposed set"
        )
        assert "alpha" in reg._exposed

    def test_config_only_reaches_initialized_plugins(self):
        """Plugin configs in the ``config`` dict should only flow to
        plugins that actually initialize — skipping a plugin must
        also skip its config dispatch (otherwise the plugin would
        appear to "remember" config from a session that didn't use
        it)."""
        reg = _make_registry(["alpha", "beta"])
        configs = {"alpha": {"k": "v_alpha"}, "beta": {"k": "v_beta"}}

        reg.expose_all(configs, requested_plugins=["alpha"])

        assert reg._plugins["alpha"].initialize_calls == [{"k": "v_alpha"}]
        assert reg._plugins["beta"].initialize_calls == []


class TestExposeAllParallelInitInteraction:
    """``PARALLEL_INIT`` opt-in plugins respect the
    ``requested_plugins`` gate too — they don't get backgrounded
    if they're not in the requested set."""

    def test_parallel_init_plugin_skipped_when_not_requested(self):
        reg = _make_registry(["alpha", "heavy"])
        reg._plugins["heavy"].PARALLEL_INIT = True

        reg.expose_all(requested_plugins=["alpha"])

        # heavy was PARALLEL_INIT-eligible but not in requested →
        # must not initialize.
        assert reg._plugins["heavy"].initialize_calls == []
        assert len(reg._plugins["alpha"].initialize_calls) == 1

    def test_parallel_init_plugin_runs_when_requested(self):
        """Sanity check: parallel-init still works when the plugin
        IS in the requested set."""
        reg = _make_registry(["alpha", "heavy"])
        reg._plugins["heavy"].PARALLEL_INIT = True

        reg.expose_all(requested_plugins=["alpha", "heavy"])

        # Both initialized (heavy via parallel path, alpha sequentially).
        assert len(reg._plugins["alpha"].initialize_calls) == 1
        assert len(reg._plugins["heavy"].initialize_calls) == 1
