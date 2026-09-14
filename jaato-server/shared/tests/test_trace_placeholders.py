"""Substitution in trace paths: the two vocabularies, and what refuses what.

Companion to ``test_trace_profile_block.py`` (which owns the ``trace:`` block's
shape) and ``test_provider_trace_routing.py`` (which owns per-agent routing).
This module owns the seam BETWEEN them — that a value an author writes into a
log path is substituted by somebody, and that a token nobody substitutes is
refused rather than created as a directory.

The defect these are written against: ``trace:`` was applied VERBATIM while
the ``env:`` map beside it was expanded, so ``${HOME}/t.log`` in the validated
block produced a directory literally named ``${HOME}`` inside the workspace —
the #775 shape, in the block added to stop #775.
"""

import os

import pytest

from jaato_sdk.trace import (
    MAIN_AGENT_ID,
    TRACE_PATH_PLACEHOLDERS,
    _agent_trace_path,
    _substitute_agent_placeholders,
    clear_trace_agent_context,
    set_trace_agent_context,
    unknown_trace_placeholders,
)
from shared.plugins.subagent.config import (
    EXPANSION_CONTEXT_VARS,
    PATH_TYPED_ENV_VARS,
    TraceProfileConfig,
    expand_variables,
    parse_profile_env,
)


@pytest.fixture(autouse=True)
def _clean_agent_context():
    """No test may leak an agent context into the next one."""
    clear_trace_agent_context()
    yield
    clear_trace_agent_context()


# ------------------------------------------------- the per-agent vocabulary

def test_explicit_placeholder_is_honoured_where_the_author_put_it():
    set_trace_agent_context("subagent_1")
    assert _agent_trace_path("logs/{agent}/provider.log") == \
        "logs/subagent_1/provider.log"
    assert _agent_trace_path("provider{agent_suffix}.log") == \
        "provider_subagent_1.log"


def test_explicit_placeholder_suppresses_the_implicit_suffix():
    """An author who placed the id is not also given it a second time."""
    set_trace_agent_context("subagent_1")
    out = _agent_trace_path("logs/{agent}/provider.log")
    assert out.count("subagent_1") == 1


def test_implicit_suffix_is_unchanged_for_a_path_without_placeholders():
    """The historical behaviour every existing deployment depends on."""
    set_trace_agent_context("subagent_2")
    assert _agent_trace_path("/tmp/provider_trace.log") == \
        "/tmp/provider_trace_subagent_2.log"


def test_main_agent_differs_between_the_two_forms_on_purpose():
    """Implicit leaves the path alone; explicit names the main agent.

    A deployment that never asked for splitting keeps one file, while an
    author who asked for the id in the name gets it on every file rather than
    one anonymous file among named siblings.
    """
    for agent in (None, MAIN_AGENT_ID):
        set_trace_agent_context(agent)
        assert _agent_trace_path("provider.log") == "provider.log"
        assert _agent_trace_path("logs/{agent}/p.log") == "logs/main/p.log"
        assert _agent_trace_path("p{agent_suffix}.log") == "p.log"


def test_session_channel_substitutes_but_never_appends():
    """A session trace splits only when asked; a provider trace always does."""
    set_trace_agent_context("subagent_1")
    assert _substitute_agent_placeholders("session.log") == "session.log"
    assert _substitute_agent_placeholders("session{agent_suffix}.log") == \
        "session_subagent_1.log"


@pytest.mark.parametrize("agent_id, expect_segment", [
    ("../../etc/evil", ".._.._etc_evil"),   # separators neutralised
    ("..", MAIN_AGENT_ID),                  # a pure traversal segment
    (".", MAIN_AGENT_ID),                   # "this directory"
    ("", MAIN_AGENT_ID),                    # nothing left after filtering
])
def test_an_agent_id_contributes_exactly_one_path_segment(agent_id,
                                                          expect_segment):
    """An agent id can never redirect the write.

    The id is framework-generated today (``subagent_<n>``), but a placeholder
    may now sit MID-PATH rather than only in a suffix, so a `/` in it would
    silently write somewhere else entirely — and `.` / `..` survive a
    character filter while still meaning something to a path resolver.
    """
    set_trace_agent_context(agent_id)
    assert _agent_trace_path("logs/{agent}/p.log") == \
        f"logs/{expect_segment}/p.log"


# --------------------------------------------- telling the vocabularies apart

def test_a_dollar_variable_is_not_read_as_a_placeholder():
    """``${HOME}`` CONTAINS ``{HOME}``; the ``$`` is what separates them.

    Without this the first legitimate profile written against the feature —
    ``${HOME}/logs/p.log`` — is refused at load as naming an unknown token.
    """
    assert unknown_trace_placeholders("${HOME}/logs/p.log") == []
    assert unknown_trace_placeholders("${workspaceRoot}/{agent}/p.log") == []


def test_an_unknown_token_is_reported():
    assert unknown_trace_placeholders("logs/{agent_id}/p.log") == ["{agent_id}"]


def test_a_dollar_prefixed_known_token_is_left_for_the_expander():
    """``${agent}`` is an env var reference, not the per-agent placeholder."""
    set_trace_agent_context("subagent_1")
    assert _substitute_agent_placeholders("${agent}/p.log") == "${agent}/p.log"


# ------------------------------------------------------- the trace: block

def test_the_block_expands_variables_like_the_env_map_does(monkeypatch):
    """The validated route must not understand LESS than the untyped one."""
    monkeypatch.setenv("HOME", "/home/tester")
    cfg = TraceProfileConfig.from_dict({"provider_log": "${HOME}/p.log"})
    assert cfg.as_env()["JAATO_PROVIDER_TRACE"] == "/home/tester/p.log"


def test_the_block_leaves_per_agent_placeholders_for_the_reader(monkeypatch):
    """Two resolution TIMES: expansion must not consume the writer's tokens."""
    monkeypatch.setenv("HOME", "/home/tester")
    cfg = TraceProfileConfig.from_dict(
        {"provider_log": "${HOME}/p{agent_suffix}.log"})
    assert cfg.as_env()["JAATO_PROVIDER_TRACE"] == \
        "/home/tester/p{agent_suffix}.log"


def test_the_block_refuses_an_unknown_placeholder():
    with pytest.raises(ValueError, match=r"\{agent_id\}"):
        TraceProfileConfig.from_dict({"provider_log": "logs/{agent_id}/p.log"})


def test_the_refusal_names_what_is_allowed():
    """A refusal that does not say the vocabulary is a puzzle, not a message."""
    with pytest.raises(ValueError) as exc:
        TraceProfileConfig.from_dict({"session_log": "{nope}.log"})
    for token in TRACE_PATH_PLACEHOLDERS:
        assert token in str(exc.value)


# ------------------------------------------- the untyped env: map, #775's half

@pytest.mark.parametrize("switch", ["1", "0", "true", "False", "on", "off"])
@pytest.mark.parametrize("var", sorted(PATH_TYPED_ENV_VARS))
def test_a_switch_in_a_path_var_is_refused_in_profile_scope(var, switch):
    """The spelling that CAUSED #775, which the typed block alone did not close.

    Refusing ``trace: {provider_log: '1'}`` and accepting
    ``env: {JAATO_PROVIDER_TRACE: '1'}`` makes the typed block a suggestion:
    the author satisfies it by moving the same value one key over.
    """
    with pytest.raises(ValueError, match="switch, not a path"):
        parse_profile_env({"env": {var: switch}})


def test_the_refusal_points_at_the_typed_key_where_one_exists():
    with pytest.raises(ValueError, match=r"trace\.provider_log"):
        parse_profile_env({"env": {"JAATO_PROVIDER_TRACE": "1"}})


def test_a_real_path_and_an_unrelated_var_are_untouched():
    env = parse_profile_env(
        {"env": {"JAATO_PROVIDER_TRACE": "p.log", "SOME_FLAG": "1"}})
    assert env == {"JAATO_PROVIDER_TRACE": "p.log", "SOME_FLAG": "1"}


def test_every_path_var_names_itself_in_its_own_refusal():
    for var in PATH_TYPED_ENV_VARS:
        with pytest.raises(ValueError) as exc:
            parse_profile_env({"env": {var: "1"}})
        assert var in str(exc.value)


# --------------------------------------------------- the anti-drift guards

def test_expansion_context_vars_are_declared():
    """``EXPANSION_CONTEXT_VARS`` must BE the set ``expand_variables`` supplies.

    The declaration exists so ``jaato-scaffold explain`` renders the vocabulary
    instead of restating it.  A var added to the function body alone would be
    undocumented by construction — which is how the reference doc's table came
    to be missing ``jdtlsStateRoot``.
    """
    sentinel = "|".join(f"${{{name}}}" for name in EXPANSION_CONTEXT_VARS)
    expanded = expand_variables(sentinel)
    assert "${" not in expanded, (
        "a declared context var is not supplied by expand_variables")

    unknown = expand_variables("${definitely_not_a_context_var_xyz}")
    assert unknown == "${definitely_not_a_context_var_xyz}", (
        "undefined names must stay literal — the check above would otherwise "
        "pass for a var nothing supplies")


def test_placeholder_registry_is_the_single_source():
    """Everything that validates a token asks the SDK registry."""
    for token in TRACE_PATH_PLACEHOLDERS:
        assert unknown_trace_placeholders(f"logs/{token}/p.log") == []
        TraceProfileConfig.from_dict({"provider_log": f"p{token}.log"})
