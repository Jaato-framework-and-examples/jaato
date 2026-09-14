"""A plugin the session ASKED FOR that did not initialize must be named.

`expose_tool` deliberately refuses to let one broken plugin take the session
down: it logs the failure, records it in `_failed_plugins`, and carries on.
That recovery had no audience — `_failed_plugins` was written in four places
and read in none — so the session came up looking healthy with a plugin the
PROFILE named simply absent.  The model finds out by calling `list_tools` and
not seeing what it was told to use; the driver sees a task that quietly did
nothing.

`file_edit` is the case that surfaced it: it writes backups under
`config_root` and raises at `initialize()` without one, so a session whose
client supplied no `config_root` lost `writeNewFile` and every other file tool
with nothing in the transcript saying why.
"""

import logging

import pytest

from shared.plugins.registry import PluginRegistry


class _Boom:
    """A plugin whose initialize() always fails."""
    PLUGIN_KIND = "tool"
    PLUGIN_TIER = "runner"

    def initialize(self, config=None):
        raise RuntimeError("no config_root")

    def get_tool_schemas(self):
        return []

    def get_executors(self):
        return {}


class _Fine(_Boom):
    def initialize(self, config=None):
        return None


def _registry():
    r = PluginRegistry()
    r._plugins["boom"] = _Boom()
    r._plugins["fine"] = _Fine()
    return r


def test_failed_plugins_are_readable():
    r = _registry()
    r.expose_all(config={}, requested_plugins=["boom", "fine"])
    failed = r.get_failed_plugins()
    assert "boom" in failed and failed["boom"][0] == "initialize"
    assert "no config_root" in failed["boom"][1]
    assert "fine" not in failed


def test_get_failed_plugins_returns_a_copy():
    """Callers iterate it while another thread may be spawning a subagent."""
    r = _registry()
    r.expose_all(config={}, requested_plugins=["boom"])
    snapshot = r.get_failed_plugins()
    snapshot["injected"] = ("initialize", "x")
    assert "injected" not in r.get_failed_plugins()


def test_a_requested_failure_is_announced_at_warning(caplog):
    r = _registry()
    with caplog.at_level(logging.WARNING):
        r.expose_all(config={}, requested_plugins=["boom", "fine"])
    text = caplog.text
    assert "boom" in text
    assert "REQUESTED" in text
    assert "no config_root" in text


def test_an_unrequested_failure_is_not_announced(caplog):
    """Only the profile's own list is a promise to the author.

    The failure is recorded first (a direct expose_tool, as an earlier
    session on a reused registry would have done), so this exercises the
    FILTER rather than the absence of a failure.
    """
    r = _registry()
    r.expose_tool("boom")                      # records the failure
    assert "boom" in r.get_failed_plugins()
    with caplog.at_level(logging.WARNING):
        r.expose_all(config={}, requested_plugins=["fine"])
    assert "REQUESTED" not in caplog.text


def test_a_clean_session_says_nothing(caplog):
    r = PluginRegistry()
    r._plugins["fine"] = _Fine()
    with caplog.at_level(logging.WARNING):
        r.expose_all(config={}, requested_plugins=["fine"])
    assert "REQUESTED" not in caplog.text
    assert r.get_failed_plugins() == {}


def test_the_broken_plugin_is_not_exposed_and_the_rest_survive():
    """The recovery itself must be unchanged — this is a diagnostic."""
    r = _registry()
    r.expose_all(config={}, requested_plugins=["boom", "fine"])
    assert "fine" in r._exposed
    assert "boom" not in r._exposed
