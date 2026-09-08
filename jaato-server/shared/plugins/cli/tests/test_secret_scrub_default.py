"""The cli plugin scrubs secret env vars by DEFAULT (#863).

Before #863 the scrub was opt-in: a profile that declared no patterns handed
the daemon's full environment — provider keys included — to every command the
model ran.  These tests pin the flipped default at the plugin (the layer that
owns policy; ``run_command`` stays policy-free), the explicit opt-out, the
``!EXEMPT`` entry that keeps a developer CLI working, and fail-closed on a
malformed knob.
"""

import logging
import sys

import pytest

from shared.plugins.cli.plugin import CLIToolPlugin
from shared.secret_scrub import DEFAULT_SECRET_ENV_PATTERNS

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="sh -c")

_ECHO = 'sh -c \'echo "${MYAPP_TOKEN:-EMPTY}"\''


def _run(config, monkeypatch):
    monkeypatch.setenv("MYAPP_TOKEN", "tok-secret")
    plugin = CLIToolPlugin()
    plugin.initialize(config)
    result = plugin._execute({"command": _ECHO})
    assert "error" not in result, result
    return plugin, result["stdout"]


def test_absent_knob_scrubs_with_the_framework_set(monkeypatch):
    plugin, out = _run({}, monkeypatch)
    assert plugin._scrub_secret_env == list(DEFAULT_SECRET_ENV_PATTERNS)
    assert "tok-secret" not in out and "EMPTY" in out


def test_no_config_at_all_still_scrubs(monkeypatch):
    plugin, out = _run(None, monkeypatch)
    assert "tok-secret" not in out and "EMPTY" in out


def test_none_opts_out_and_is_announced(monkeypatch, caplog):
    with caplog.at_level(logging.WARNING, logger="shared.secret_scrub"):
        plugin, out = _run({"scrub_secret_env": "none"}, monkeypatch)
    assert plugin._scrub_secret_env == []
    assert "tok-secret" in out
    assert any("cli" in r.getMessage() and "DISABLED" in r.getMessage()
               for r in caplog.records)


def test_exemption_keeps_a_needed_token(monkeypatch):
    _, out = _run({"scrub_secret_env": ["default", "!MYAPP_TOKEN"]}, monkeypatch)
    assert "tok-secret" in out


def test_explicit_list_replaces_the_default(monkeypatch):
    # A narrower explicit set: *_SECRET only — MYAPP_TOKEN is not covered.
    _, out = _run({"scrub_secret_env": ["*_SECRET"]}, monkeypatch)
    assert "tok-secret" in out


def test_malformed_knob_fails_closed(monkeypatch, caplog):
    with caplog.at_level(logging.ERROR, logger="shared.secret_scrub"):
        plugin, out = _run({"scrub_secret_env": 42}, monkeypatch)
    assert plugin._scrub_secret_env == list(DEFAULT_SECRET_ENV_PATTERNS)
    assert "tok-secret" not in out
    assert any("fail closed" in r.getMessage() for r in caplog.records)


def test_config_schema_declares_the_grammar():
    schema = CLIToolPlugin().get_config_schema()["properties"]["scrub_secret_env"]
    assert schema["default"] == "default"
    assert "none" in schema["description"] and "!NAME" in schema["description"]
