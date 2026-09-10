"""``jaato-scaffold`` must SAY what the framework substitutes, and CHECK it.

Before this, ``validate.py`` contained the string "trace" zero times, and
``explain profile`` printed the block's one-line field description and nothing
about what a path may contain.  So the failure mode was the one the whole
scaffold verb family exists to prevent: a value that is quietly wrong, with
the tool that exists to catch it reporting nothing.

These are tests about the TOOL's output, deliberately — a rule the framework
enforces but the tool never mentions is a rule authors discover by incident.
"""

import pytest

from jaato_sdk.trace import TRACE_PATH_PLACEHOLDERS
from shared.plugins.subagent.config import EXPANSION_CONTEXT_VARS
from shared.scaffold import explain, introspect, validate


# ------------------------------------------------------------- introspect

def test_placeholders_are_computed_from_the_live_registries():
    """Neither vocabulary may be restated in the scaffold layer."""
    tokens = {ph.name for ph in introspect.placeholders()}
    for name in EXPANSION_CONTEXT_VARS:
        assert f"${{{name}}}" in tokens
    for token in TRACE_PATH_PLACEHOLDERS:
        assert token in tokens


def test_placeholders_are_grouped_by_when_they_resolve():
    """The distinction authors get wrong is TIME, so it must be the grouping."""
    resolvers = {ph.resolved_by for ph in introspect.placeholders()}
    assert len(resolvers) == 2, resolvers


# ---------------------------------------------------------------- explain

@pytest.mark.parametrize("topic", ["profile", "env"])
def test_every_placeholder_is_visible(topic):
    _, text = getattr(explain, topic)()
    for name in EXPANSION_CONTEXT_VARS:
        assert f"${{{name}}}" in text, f"{name} missing from `explain {topic}`"
    for token in TRACE_PATH_PLACEHOLDERS:
        assert token in text, f"{token} missing from `explain {topic}`"


def test_explain_profile_documents_the_trace_block_itself():
    _, text = explain.profile()
    assert "trace:" in text
    # the three things that are not on the one-line field description
    assert "provider_subagent_1" in text          # the implicit suffix
    assert "READER" in text                       # where a relative path lands
    assert "SUBSTITUTION" in text                 # what a value may contain


def test_explain_states_that_an_unknown_token_is_refused():
    """The rule and its enforcement must be described together."""
    _, text = explain.profile()
    assert "REFUSED" in text and "literal directory" in text


# --------------------------------------------------------------- validate

def _findings(profile):
    return {d.code for d in validate.validate_profile(
        profile,
        providers=introspect.providers(),
        plugins=introspect.plugins(),
        gc_names=list(introspect.gc_strategies().keys()),
    )}


class _Profile:
    """Minimal stand-in for a resolved profile (only the read fields)."""

    def __init__(self, trace=None, env=None):
        self.name = "t"
        self.trace = trace
        self.env = env or {}
        self.plugins = []
        self.provider = None
        self.model = None


class _Trace:
    def __init__(self, session_log=None, provider_log=None):
        self.session_log = session_log
        self.provider_log = provider_log


def test_a_sane_trace_block_produces_no_trace_findings():
    codes = _findings(_Profile(trace=_Trace(
        provider_log=".jaato/logs/provider{agent_suffix}.jsonl")))
    assert not {c for c in codes if c.startswith("trace_")}


def test_a_daemon_scoped_var_in_a_trace_path_is_reported():
    """Legal, and almost never what a per-session log path means."""
    assert "trace_path_daemon_scoped_var" in _findings(
        _Profile(trace=_Trace(provider_log="${workspaceRoot}/p.jsonl")))


def test_an_undefined_variable_is_reported():
    assert "trace_path_unexpanded_var" in _findings(
        _Profile(trace=_Trace(session_log="${NO_SUCH_VAR_XYZ}/s.jsonl")))


def test_a_variable_the_profile_itself_defines_is_not_reported():
    """The check must not punish an author for defining their own variable."""
    codes = _findings(_Profile(
        trace=_Trace(session_log="${MY_LOG_ROOT}/s.jsonl"),
        env={"MY_LOG_ROOT": "/var/log/acme"}))
    assert "trace_path_unexpanded_var" not in codes


def test_an_unknown_placeholder_reaching_via_the_env_map_is_reported():
    """The env route is not checked at load, so validate is the only reporter."""
    assert "trace_path_placeholder_unknown" in _findings(
        _Profile(env={"JAATO_PROVIDER_TRACE": "logs/{agent_id}/p.log"}))


def test_a_dead_env_value_shadowed_by_the_block_is_reported():
    """The typed block outranks the map, so the map's value never runs."""
    assert "trace_env_shadowed" in _findings(_Profile(
        trace=_Trace(provider_log="a.log"),
        env={"JAATO_PROVIDER_TRACE": "b.log"}))


def test_the_two_routes_agreeing_is_not_a_finding():
    codes = _findings(_Profile(env={"JAATO_PROVIDER_TRACE": "p.log"}))
    assert not {c for c in codes if c.startswith("trace_")}
