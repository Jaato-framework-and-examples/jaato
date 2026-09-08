"""``jaato-scaffold validate`` surfaces an unscrubbed subprocess surface (#863).

The scrub is ON by default, so a silent profile passes.  What the validator
flags is the deliberate ``none`` (profile-level or per-surface), a value the
grammar rejects (the plugin fails CLOSED on it, replacing the author's intent
with the default set), and a profile-level key with no surface to apply to.
"""
from types import SimpleNamespace

from shared.scaffold import introspect
from shared.scaffold.validate import validate_profile


def _validate(plugins, plugin_configs=None, scrub=None):
    prof = SimpleNamespace(provider=None, model="m", plugins=plugins,
                           plugin_configs=plugin_configs or {},
                           scrub_secret_env=scrub)
    return validate_profile(
        prof, providers=introspect.providers(), plugins=introspect.plugins(),
        gc_names=list(introspect.gc_strategies().keys()))


def _codes(diags, code):
    return [(d.severity, d.where) for d in diags if d.code == code]


def test_silent_profile_passes_because_the_default_scrubs():
    diags = _validate(["cli", "mcp", "interactive_shell"])
    assert _codes(diags, "secret_scrub_disabled") == []
    assert _codes(diags, "invalid_scrub_secret_env") == []


def test_profile_level_none_warns_per_enabled_surface():
    diags = _validate(["cli", "mcp"], scrub="none")
    found = _codes(diags, "secret_scrub_disabled")
    assert found == [("warn", "scrub_secret_env"), ("warn", "scrub_secret_env")]
    msgs = [d.message for d in diags if d.code == "secret_scrub_disabled"]
    assert any("'cli'" in m for m in msgs) and any("'mcp'" in m for m in msgs)
    assert all("scrub_secret_env: default" in m for m in msgs)


def test_per_surface_knob_decides_and_is_named():
    diags = _validate(["cli", "mcp"], {"cli": {"scrub_secret_env": []}},
                      scrub=["*_TOKEN"])
    assert _codes(diags, "secret_scrub_disabled") == [
        ("warn", "plugin_configs.cli.scrub_secret_env")]


def test_per_surface_knob_can_rescue_a_profile_level_none():
    diags = _validate(["cli"], {"cli": {"scrub_secret_env": "default"}}, scrub="none")
    assert _codes(diags, "secret_scrub_disabled") == []


def test_only_exemptions_is_disabled():
    diags = _validate(["cli"], scrub=["!GH_TOKEN"])
    assert len(_codes(diags, "secret_scrub_disabled")) == 1


def test_malformed_value_is_an_error_naming_fail_closed():
    diags = _validate(["mcp"], {"mcp": {"scrub_secret_env": {"x": 1}}})
    [(sev, where)] = _codes(diags, "invalid_scrub_secret_env")
    assert sev == "error" and where == "plugin_configs.mcp.scrub_secret_env"
    assert any("CLOSED" in d.message for d in diags
               if d.code == "invalid_scrub_secret_env")


def test_key_with_no_surface_is_inert_info():
    diags = _validate(["todo"], scrub="none")
    assert _codes(diags, "scrub_secret_env_inert") == [("info", "scrub_secret_env")]
    assert _codes(diags, "secret_scrub_disabled") == []


def test_explain_profile_lists_the_key_and_its_grammar():
    from shared.scaffold import explain
    _, text = explain.profile()
    assert "scrub_secret_env" in text
    assert "!NAME" in text and "announced at WARNING" in text
