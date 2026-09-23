"""``jaato-scaffold validate`` flags a GH_TOKEN the scrub makes inert (#1228).

Once ``GH_TOKEN=app://…`` reaches a session (#1226) it is only useful to
``gh`` / ``git`` if the model-driven subprocess surface does NOT scrub it.
``GH_TOKEN`` is in the default ``scrub_secret_env`` set (#863), so a profile
that declares the reference but keeps the default scrub on ``cli`` /
``interactive_shell`` is VALID and INERT — the token is stripped before ``gh``
runs.  The finding is ``gh_token_scrubbed_inert`` (warn); the fix is a
``!GH_TOKEN`` exemption.  MCP is deliberately never flagged (an MCP server
gets no GitHub token).
"""
from types import SimpleNamespace

import pytest

from jaato_server.shared.scaffold import introspect
from jaato_server.shared.scaffold.validate import validate_profile


def _validate(plugins, *, env=None, env_values=None, plugin_configs=None,
              scrub=None):
    prof = SimpleNamespace(provider=None, model="m", plugins=plugins,
                           plugin_configs=plugin_configs or {},
                           scrub_secret_env=scrub, env=env or {})
    return validate_profile(
        prof, providers=introspect.providers(), plugins=introspect.plugins(),
        gc_names=list(introspect.gc_strategies().keys()),
        env_values=env_values)


def _codes(diags, code):
    return [(d.severity, d.where) for d in diags if d.code == code]


APP = {"GH_TOKEN": "app://github"}


def test_fires_when_app_token_is_scrubbed_by_the_default_on_cli():
    diags = _validate(["cli"], env=APP)
    assert _codes(diags, "gh_token_scrubbed_inert") == [
        ("warn", "scrub_secret_env")]
    [msg] = [d.message for d in diags if d.code == "gh_token_scrubbed_inert"]
    assert "app://" in msg and "!GH_TOKEN" in msg and "'cli'" in msg


def test_fires_on_interactive_shell_too():
    diags = _validate(["interactive_shell"], env=APP)
    assert _codes(diags, "gh_token_scrubbed_inert") == [
        ("warn", "scrub_secret_env")]


def test_fires_from_the_workspace_env_not_only_the_profile():
    # The daemon reads GH_TOKEN=app://… from the workspace .env just as well.
    diags = _validate(["cli"], env_values=APP)
    found = _codes(diags, "gh_token_scrubbed_inert")
    assert found == [("warn", "scrub_secret_env")]


def test_cleared_by_the_exemption():
    diags = _validate(
        ["cli", "interactive_shell"], env=APP,
        plugin_configs={
            "cli": {"scrub_secret_env": ["default", "!GH_TOKEN"]},
            "interactive_shell": {"scrub_secret_env": ["default", "!GH_TOKEN"]},
        })
    assert _codes(diags, "gh_token_scrubbed_inert") == []


def test_not_fired_when_the_surface_is_not_enabled():
    # The token is declared, but no shell surface runs, so nothing scrubs it
    # away from a gh subprocess (there is none).
    diags = _validate(["todo"], env=APP)
    assert _codes(diags, "gh_token_scrubbed_inert") == []


def test_mcp_is_never_flagged():
    # MCP is in the scrub surfaces but keeps `default` by design — an MCP
    # server gets no GitHub token, so scrubbing GH_TOKEN there is correct.
    diags = _validate(["mcp"], env=APP)
    assert _codes(diags, "gh_token_scrubbed_inert") == []


def test_literal_token_is_out_of_scope():
    # The finding is about the app:// delivery (#1226/#1228); a plain literal
    # is a different case and is not reported here.
    diags = _validate(["cli"], env={"GH_TOKEN": "ghp_realtokenvalue"})
    assert _codes(diags, "gh_token_scrubbed_inert") == []


def test_disabled_scrub_does_not_double_report():
    # scrub_secret_env: none strips nothing, so GH_TOKEN already reaches gh —
    # that is secret_scrub_disabled's concern, not an inert-token one.
    diags = _validate(["cli"], env=APP, scrub="none")
    assert _codes(diags, "gh_token_scrubbed_inert") == []
    assert _codes(diags, "secret_scrub_disabled") == [("warn", "scrub_secret_env")]


def test_no_token_declared_is_silent():
    diags = _validate(["cli", "interactive_shell"])
    assert _codes(diags, "gh_token_scrubbed_inert") == []


def test_explain_gh_describes_the_setup():
    from jaato_server.shared.scaffold import explain
    _, text = explain.gh()
    assert "!GH_TOKEN" in text
    assert "GH_PROMPT_DISABLED" in text and "GIT_TERMINAL_PROMPT" in text
    assert "gh auth login" in text
