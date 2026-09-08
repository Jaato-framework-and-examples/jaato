"""interactive_shell scrubs the inherited env of spawned sessions (#503 / #863).

The cli and MCP subprocess paths had the secrets-broker scrub since #10;
interactive_shell spawns inherited the runner env untouched, so a model-driven
REPL could ``echo $GITHUB_TOKEN``.  The session applies whatever the plugin
resolves; the plugin resolves the same grammar as the other surfaces, with the
framework set as the default.
"""

import logging
import sys

import pytest

from shared.plugins.interactive_shell.plugin import InteractiveShellPlugin
from shared.plugins.interactive_shell.session import ShellSession
from shared.secret_scrub import DEFAULT_SECRET_ENV_PATTERNS

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="sh -c")

_ECHO = 'sh -c \'echo "${MYAPP_TOKEN:-EMPTY}"\''


def _spawn_output(monkeypatch, **kwargs):
    monkeypatch.setenv("MYAPP_TOKEN", "tok-secret")
    session = ShellSession(command=_ECHO, session_id="scrub", idle_timeout=0.3,
                           **kwargs)
    try:
        return session.read_initial_output()
    finally:
        session.close()


def test_session_without_patterns_inherits_everything(monkeypatch):
    # The SESSION is policy-free — the plugin decides.
    assert "tok-secret" in _spawn_output(monkeypatch)


def test_session_scrubs_declared_patterns(monkeypatch):
    out = _spawn_output(monkeypatch, scrub_env=["*_TOKEN"])
    assert "tok-secret" not in out and "EMPTY" in out


def test_caller_env_is_a_grant_that_survives_the_scrub(monkeypatch):
    # Mirrors the MCP rule: a variable the caller hands over explicitly is
    # not the inherited leak the scrub exists to stop.
    out = _spawn_output(monkeypatch, scrub_env=["*_TOKEN"],
                        env={"MYAPP_TOKEN": "granted"})
    assert "granted" in out


def test_session_honours_exemptions(monkeypatch):
    out = _spawn_output(monkeypatch, scrub_env=["*_TOKEN", "!MYAPP_TOKEN"])
    assert "tok-secret" in out


# ---- the plugin owns the default -----------------------------------------

def _plugin(config):
    p = InteractiveShellPlugin()
    p._start_reaper = lambda: None
    p.initialize(config)
    return p


def test_plugin_absent_knob_is_the_framework_set():
    assert _plugin({})._scrub_secret_env == list(DEFAULT_SECRET_ENV_PATTERNS)
    assert _plugin(None)._scrub_secret_env == list(DEFAULT_SECRET_ENV_PATTERNS)


def test_plugin_none_is_announced(caplog):
    with caplog.at_level(logging.WARNING, logger="shared.secret_scrub"):
        p = _plugin({"scrub_secret_env": "none"})
    assert p._scrub_secret_env == []
    assert any("interactive_shell" in r.getMessage() for r in caplog.records)


def test_plugin_schema_declares_the_knob():
    props = InteractiveShellPlugin().get_config_schema()["properties"]
    assert props["scrub_secret_env"]["default"] == "default"


def test_plugin_spawn_scrubs_by_default(monkeypatch):
    monkeypatch.setenv("MYAPP_TOKEN", "tok-secret")
    p = _plugin({})
    result = p._exec_spawn({"command": _ECHO, "session_name": "scrub_default"})
    try:
        assert "tok-secret" not in result["output"] and "EMPTY" in result["output"]
    finally:
        for s in list(p._sessions.values()):
            s.close()
