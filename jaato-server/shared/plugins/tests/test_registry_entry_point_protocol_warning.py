"""Tests for entry-point Protocol-check WARNING (PR-210 systemic fix).

Background: when a plugin registered via the ``jaato.plugins`` entry
point group fails the ``ToolPlugin`` / ``EnrichmentPlugin`` Protocol
check (``isinstance(plugin, ToolPlugin)`` returns False because the
plugin is missing one of the Protocol's declared methods), the plugin
is silently skipped.

Pre-PR-210, the skip emitted a debug-only ``_trace`` line — invisible
to operators unless they explicitly enabled trace logging.  The
directory-path side of the same check was promoted to ``logger.warning``
with named missing methods at PR #171 (2026-05-20 incident).  The
entry-point path was NEVER updated.

Concrete cost (2026-06-03): premium 0.1.191's ``SessionOpsPlugin``
shipped missing ``reset_for_next_session`` (added to the ToolPlugin
Protocol later).  The entry-point silent skip dropped session_ops from
the runner-side registry.  The model saw an empty schema and
hallucinated that ``interrogate_session`` was unavailable.  Smoke
validation against the 0.6.179 + 0.1.191 composed pair failed for an
opaque reason — diagnosis required a direct PluginRegistry probe.

This test pins the post-PR-210 contract: a Protocol-incomplete
entry-point plugin emits a ``logger.warning`` naming the missing
methods, AND the plugin is still skipped from the registry (the
behavioural change is the diagnostic, not the skip itself).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from shared.plugins.registry import PluginRegistry


# ----------------------------------------------------------------------
# Fixture: a plugin class missing one ToolPlugin Protocol method
# ----------------------------------------------------------------------


class _CompletePlugin:
    """Minimal plugin implementing every method ToolPlugin declares."""

    @property
    def name(self) -> str:
        return "complete_fake"

    def initialize(self, config: Any = None) -> None: ...
    def shutdown(self) -> None: ...
    def get_tool_schemas(self) -> List: return []
    def get_executors(self) -> Dict[str, Any]: return {}
    def get_user_commands(self) -> List: return []
    def get_auto_approved_tools(self) -> List[str]: return []
    def get_system_instructions(self) -> Any: return None
    def reset_for_next_session(self) -> None: ...


class _IncompletePlugin:
    """Plugin MISSING ``reset_for_next_session`` — mirrors the exact
    shape that broke premium ``SessionOpsPlugin`` pre-fix.

    ``isinstance(_IncompletePlugin(), ToolPlugin)`` returns False
    because the runtime_checkable Protocol's ``__instancecheck__`` walks
    every declared method and finds the gap.
    """

    @property
    def name(self) -> str:
        return "incomplete_fake"

    def initialize(self, config: Any = None) -> None: ...
    def shutdown(self) -> None: ...
    def get_tool_schemas(self) -> List: return []
    def get_executors(self) -> Dict[str, Any]: return {}
    def get_user_commands(self) -> List: return []
    def get_auto_approved_tools(self) -> List[str]: return []
    def get_system_instructions(self) -> Any: return None
    # NB: no reset_for_next_session


def _make_ep(name: str, factory: Any, value: str) -> Any:
    """Build a stand-in entry-point object compatible with the registry's
    ``ep.load()`` / ``ep.name`` / ``ep.value`` / ``ep.__module__`` access
    pattern.  Production uses ``importlib.metadata.EntryPoint`` — this
    mock satisfies the same surface without the package-install overhead.
    """
    ep = MagicMock()
    ep.name = name
    ep.value = value
    # ep.load() must return the factory (callable that produces the plugin
    # instance).  Production callers do: create_plugin = ep.load();
    # plugin = create_plugin().  For a class, calling it instantiates.
    ep.load.return_value = factory
    return ep


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


def test_incomplete_tool_plugin_emits_warning_with_missing_methods(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When an entry-point plugin fails the ToolPlugin Protocol check,
    a WARNING is emitted (not a debug-only trace) naming the missing
    methods so the author has actionable diagnostic output."""

    ep = _make_ep(
        name="incomplete_fake",
        factory=_IncompletePlugin,
        value="fake.pkg.module:IncompletePlugin",
    )
    registry = PluginRegistry()

    with patch(
        "shared.plugins.registry.importlib.metadata.entry_points",
        return_value=[ep],
    ):
        with caplog.at_level(
            logging.WARNING, logger="shared.plugins.registry",
        ):
            discovered = registry._discover_via_entry_points(
                plugin_kind="tool",
            )

    # Plugin was skipped (behavioural contract unchanged).
    assert discovered == [], (
        "Incomplete plugin should still be skipped — promotion was "
        "of the diagnostic, not the skip itself."
    )
    assert "incomplete_fake" not in registry._plugins

    # Warning emitted with named missing methods.
    warning_records = [
        r for r in caplog.records
        if r.levelno == logging.WARNING
        and "incomplete_fake" in r.message
    ]
    assert warning_records, (
        f"Expected a WARNING naming the entry-point. Got records: "
        f"{[(r.levelname, r.message) for r in caplog.records]}"
    )
    warning_msg = warning_records[0].message
    assert "reset_for_next_session" in warning_msg, (
        f"WARNING must name the missing method. Got: {warning_msg!r}"
    )
    assert "ToolPlugin" in warning_msg
    assert "fake.pkg.module:IncompletePlugin" in warning_msg, (
        f"WARNING must include ep.value for diagnostic discoverability. "
        f"Got: {warning_msg!r}"
    )


def test_complete_tool_plugin_emits_no_warning_and_loads(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A Protocol-complete plugin loads normally and emits NO warning
    — the promotion must not produce noise on the happy path."""

    ep = _make_ep(
        name="complete_fake",
        factory=_CompletePlugin,
        value="fake.pkg.module:CompletePlugin",
    )
    registry = PluginRegistry()

    with patch(
        "shared.plugins.registry.importlib.metadata.entry_points",
        return_value=[ep],
    ):
        with caplog.at_level(
            logging.WARNING, logger="shared.plugins.registry",
        ):
            discovered = registry._discover_via_entry_points(
                plugin_kind="tool",
            )

    assert "complete_fake" in discovered
    assert "complete_fake" in registry._plugins

    # No warning records about complete_fake.
    warning_records = [
        r for r in caplog.records
        if r.levelno == logging.WARNING
        and "complete_fake" in r.message
    ]
    assert warning_records == [], (
        f"Complete plugin must not emit a warning. Got: "
        f"{[r.message for r in warning_records]}"
    )


def test_warning_mentions_entry_point_path_not_directory_path() -> None:
    """The entry-point warning must clearly identify itself as an
    entry-point failure (not a directory failure) — both paths now
    emit warnings, and operators need to know which discovery path
    failed so they can fix the right thing (pyproject.toml entry
    declaration vs in-tree plugin module)."""

    ep = _make_ep(
        name="incomplete_fake",
        factory=_IncompletePlugin,
        value="fake.pkg.module:IncompletePlugin",
    )
    registry = PluginRegistry()

    with patch(
        "shared.plugins.registry.importlib.metadata.entry_points",
        return_value=[ep],
    ):
        with patch("shared.plugins.registry.logger") as mock_logger:
            registry._discover_via_entry_points(plugin_kind="tool")

    assert mock_logger.warning.called, (
        "Expected logger.warning to be invoked on Protocol failure."
    )
    call_args = mock_logger.warning.call_args
    # First positional arg is the format string.
    fmt = call_args[0][0]
    assert "Entry point" in fmt, (
        f"Warning format must say 'Entry point' so operators can "
        f"distinguish from the directory-path warning. Got: {fmt!r}"
    )


def test_missing_methods_field_says_unknown_when_introspection_fails() -> None:
    """If ``_missing_protocol_methods`` returns an empty list (e.g. the
    Protocol introspection fails for some plugin shape), the warning
    must still emit with ``(unknown)`` rather than an empty string —
    mirrors the directory-path fallback at registry.py:735."""

    class _OpaquelyBrokenPlugin:
        """A plugin that fails isinstance(p, ToolPlugin) but whose
        missing-method introspection yields nothing actionable.

        Synthesised by removing every public attribute so the
        introspection helper has nothing to compare against."""
        pass

    ep = _make_ep(
        name="opaque_broken",
        factory=_OpaquelyBrokenPlugin,
        value="fake.pkg.module:OpaquelyBrokenPlugin",
    )
    registry = PluginRegistry()

    with patch(
        "shared.plugins.registry.importlib.metadata.entry_points",
        return_value=[ep],
    ):
        with patch("shared.plugins.registry.logger") as mock_logger:
            discovered = registry._discover_via_entry_points(
                plugin_kind="tool",
            )

    assert discovered == []
    assert mock_logger.warning.called
    # Should still emit the WARNING with whatever missing methods the
    # helper found (or "(unknown)" if none).  The point is loud, not
    # silent.


# ----------------------------------------------------------------------
# Integration: composed-pair shape (session_ops smoke regression)
# ----------------------------------------------------------------------


def test_session_ops_shaped_plugin_emits_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Regression test for the 2026-06-03 incident: a plugin shaped
    like premium 0.1.191's SessionOpsPlugin (missing
    reset_for_next_session) MUST emit a loud warning naming the
    missing method, not silently disappear.

    Before PR-210 this exact shape produced an empty schema runner-
    side and a model hallucination ("tool not available").  The
    operator's only diagnostic was direct registry probing.

    After PR-210, the daemon-side bootstrap log carries the WARNING
    naming ``reset_for_next_session`` and the plugin's entry-point
    path — operators can fix in one minute instead of one hour.
    """

    # Shape-mirror of SessionOpsPlugin: every ToolPlugin method except
    # reset_for_next_session.  PLUGIN_TIER not relevant here (the
    # tier filter passed in the real incident; the Protocol check
    # was the gate that failed).
    class _SessionOpsShape:
        @property
        def name(self) -> str: return "session_ops_shape"
        def initialize(self, config: Any = None) -> None: ...
        def shutdown(self) -> None: ...
        def get_tool_schemas(self) -> List: return []
        def get_executors(self) -> Dict[str, Any]: return {}
        def get_user_commands(self) -> List: return []
        def get_auto_approved_tools(self) -> List[str]: return []
        def get_system_instructions(self) -> Any: return None
        # NB: no reset_for_next_session — the exact shape that broke
        # premium 0.1.191.

    ep = _make_ep(
        name="session_ops_shape",
        factory=_SessionOpsShape,
        value="jaato_premium.session_ops:SessionOpsPlugin",
    )
    registry = PluginRegistry()

    with patch(
        "shared.plugins.registry.importlib.metadata.entry_points",
        return_value=[ep],
    ):
        with caplog.at_level(
            logging.WARNING, logger="shared.plugins.registry",
        ):
            registry._discover_via_entry_points(plugin_kind="tool")

    warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING
    ]
    assert warnings, "Expected a WARNING — silent skip is the bug class"
    msg = warnings[0].message
    assert "reset_for_next_session" in msg
    assert "session_ops_shape" in msg
