"""Tests for the directory-scan missing-dependency WARNING.

Background: during directory-based plugin discovery
(``_discover_via_directory``), each candidate module is imported to
enumerate it.  When a plugin's *declared* backend dependency is not
installed in the current environment — the canonical case is
``interactive_shell`` importing ``pexpect`` at module top on Unix — the
import raises ``ModuleNotFoundError``.

Pre-fix, that was caught by the generic ``except Exception`` handler and
logged via ``_trace(..., include_traceback=True)`` → a full traceback at
ERROR level.  During a scaffold ``compile`` / introspection run in a
minimal venv this dumps scary multi-line stacks mid-output for what is
actually a benign, actionable environment-provisioning gap (the plugin
correctly does NOT load; the operator just needs to ``pip install`` the
dep).

This test pins the contract: a ``ModuleNotFoundError`` during directory
discovery emits a concise one-line WARNING naming the missing module and
the plugin, the plugin is still skipped, and NO full-traceback ERROR is
produced.  Any *other* exception (a genuine plugin bug) must still get
the full-traceback ERROR treatment.
"""

from __future__ import annotations

import importlib
import logging
from unittest.mock import patch

import pytest

from shared.plugins.registry import PluginRegistry

_REAL_IMPORT_MODULE = importlib.import_module


def _one_fake_module(name: str):
    """A ``pkgutil.iter_modules`` stand-in yielding exactly one candidate."""
    return [(None, name, False)]


def _selective_import(candidate: str, exc: Exception):
    """Build an ``import_module`` side-effect that raises ``exc`` only for
    the candidate plugin module (``.<candidate>`` under ``shared.plugins``)
    and delegates every other import to the real ``import_module`` — so the
    global patch doesn't disturb unrelated imports during the with-block.
    """

    def _side_effect(modname, package=None):
        if modname == f".{candidate}":
            raise exc
        return _REAL_IMPORT_MODULE(modname, package)

    return _side_effect


def test_module_not_found_emits_concise_warning_not_traceback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    registry = PluginRegistry()

    # Mirror ``import pexpect`` failing on a pexpect-less Unix venv.
    missing = ModuleNotFoundError("No module named 'pexpect'", name="pexpect")

    with patch(
        "shared.plugins.registry.pkgutil.iter_modules",
        return_value=_one_fake_module("interactive_shell"),
    ), patch(
        "shared.plugins.registry.importlib.import_module",
        side_effect=_selective_import("interactive_shell", missing),
    ):
        with caplog.at_level(logging.DEBUG, logger="shared.plugins.registry"):
            discovered = registry._discover_via_directory(plugin_kind="tool")

    # Plugin skipped (behavioural contract unchanged).
    assert discovered == []
    assert "interactive_shell" not in registry._plugins

    # A WARNING naming BOTH the plugin and the missing module.
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings, (
        f"Expected a WARNING for the missing dep. Got: "
        f"{[(r.levelname, r.message) for r in caplog.records]}"
    )
    msg = warnings[0].message
    assert "interactive_shell" in msg
    assert "pexpect" in msg

    # And NO full-traceback ERROR (the scary-stack regression).
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors == [], (
        f"A missing dependency must not log at ERROR / with a traceback. "
        f"Got: {[(r.levelname, r.message) for r in errors]}"
    )


def test_non_import_error_still_logs_full_traceback_error() -> None:
    """A genuine plugin bug (any non-ModuleNotFoundError) must still be
    logged at ERROR with a traceback — the concise path is scoped to
    missing-dependency skips only, not a blanket softening."""
    registry = PluginRegistry()

    bug = RuntimeError("plugin body is broken")

    with patch(
        "shared.plugins.registry.pkgutil.iter_modules",
        return_value=_one_fake_module("buggy_plugin"),
    ), patch(
        "shared.plugins.registry.importlib.import_module",
        side_effect=_selective_import("buggy_plugin", bug),
    ):
        with patch("shared.plugins.registry.logger") as mock_logger:
            discovered = registry._discover_via_directory(plugin_kind="tool")

    assert discovered == []
    # Generic handler path: logger.error(..., exc_info=True).
    assert mock_logger.error.called, "A real bug must still log at ERROR."
    _, kwargs = mock_logger.error.call_args
    assert kwargs.get("exc_info") is True
    assert not mock_logger.warning.called, (
        "A non-ModuleNotFoundError must not be softened to a WARNING."
    )
