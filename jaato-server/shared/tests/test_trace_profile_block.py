"""A trace path is a path, and the typed block is what can say so.

THE INCIDENT (issue #775).  ``JAATO_PROVIDER_TRACE`` is the FILE the
provider trace is written to.  Set through a profile's ``env:`` map as
``JAATO_PROVIDER_TRACE: "1"``, every session wrote its trace to a file
literally named ``1`` -- including eval-arm workspaces, contaminating the
very trees a comparative judge was diffing.  Nothing rejected it and
nothing could: ``env`` is ``Dict[str, str]`` and ``"1"`` is a valid string.

These tests pin the two halves of the fix:

  * the typed ``trace:`` block refuses a switch written into a path field,
    while still accepting the relative paths that are the supported
    per-session idiom (``jaato_sdk.trace._resolve_trace_file`` resolves
    them against the workspace, so each session gets its own file);
  * the block reaches the session env, above ``env:`` and below the
    post-auth overrides, so the promotion is wired end to end rather than
    parsed and dropped.

The second half is not paranoia: this repository has a named failure mode
for it -- FOUR ingresses build a ``SubagentProfile`` from a dict, and a
block field wired into three is silently inert in the fourth, which is
why ``parse_gc_block`` / ``parse_cache_block`` exist at all.
"""

from __future__ import annotations

import pytest

from shared.plugins.subagent.config import (
    TRACE_ENV_VARS,
    SubagentProfile,
    TraceProfileConfig,
    build_inline_profile,
    parse_trace_block,
)


# --------------------------------------------------------------------------
# What the block refuses
# --------------------------------------------------------------------------

@pytest.mark.parametrize("value", ["1", "0", "true", "False", "yes", "on", "none"])
def test_a_switch_is_not_a_path(value):
    """The incident value, and the rest of the vocabulary it belongs to.

    An author writing any of these means "turn the trace on".  There is no
    file they could plausibly have meant, so the value is refused instead
    of becoming a file with that name.
    """
    with pytest.raises(ValueError) as exc:
        TraceProfileConfig.from_dict({"provider_log": value})
    message = str(exc.value)
    assert "switch, not a path" in message
    # The message has to teach the fix, not just refuse: the author is
    # holding a boolean and needs to know both spellings of the answer.
    assert "relative" in message and "absolute" in message


def test_a_directory_is_not_a_file():
    with pytest.raises(ValueError, match="names a directory"):
        TraceProfileConfig.from_dict({"session_log": "/var/log/jaato/"})


@pytest.mark.parametrize("value", ["", "   ", 1, True, [], {}])
def test_a_non_path_value_is_refused(value):
    with pytest.raises(ValueError, match="non-empty string path"):
        TraceProfileConfig.from_dict({"provider_log": value})


def test_an_unknown_key_is_refused():
    """A typo in a typed block is a failure, not a silently ignored key."""
    with pytest.raises(ValueError, match="unknown key"):
        TraceProfileConfig.from_dict({"provider_trace": "/tmp/t.log"})


def test_a_non_mapping_block_is_refused():
    with pytest.raises(ValueError, match="must be a mapping"):
        TraceProfileConfig.from_dict("/tmp/t.log")


# --------------------------------------------------------------------------
# What the block accepts -- deliberately including relative paths
# --------------------------------------------------------------------------

def test_a_relative_path_is_the_per_session_idiom():
    """Relative is NOT the defect and must keep working.

    ``jaato_sdk.trace._resolve_trace_file`` joins a relative trace path
    onto ``JAATO_WORKSPACE_ROOT``, which the runner seeds per session --
    so a relative value gives each session its own file in its own
    workspace.  Rejecting relative paths here (the obvious-looking
    reading of the incident) would break the documented pattern.
    """
    cfg = TraceProfileConfig.from_dict(
        {"provider_log": ".jaato/logs/provider_trace.jsonl"})
    assert cfg.as_env() == {
        "JAATO_PROVIDER_TRACE": ".jaato/logs/provider_trace.jsonl"}


def test_an_absolute_path_is_shared_by_every_session_using_the_profile():
    cfg = TraceProfileConfig.from_dict({"session_log": "/tmp/jaato/session.jsonl"})
    assert cfg.as_env() == {"JAATO_TRACE_LOG": "/tmp/jaato/session.jsonl"}


def test_an_unset_key_seeds_nothing():
    """Half a block sets half the env -- not both keys with one empty."""
    cfg = TraceProfileConfig.from_dict({"provider_log": "/tmp/p.jsonl"})
    assert cfg.session_log is None
    assert set(cfg.as_env()) == {"JAATO_PROVIDER_TRACE"}


def test_an_absent_or_empty_block_is_none():
    assert parse_trace_block({}) is None
    assert parse_trace_block({"trace": None}) is None
    assert parse_trace_block({"trace": {}}) is None


def test_the_env_var_mapping_has_one_home():
    """``TRACE_ENV_VARS`` is what ``as_env`` and the catalog both read."""
    assert TRACE_ENV_VARS == {"session_log": "JAATO_TRACE_LOG",
                              "provider_log": "JAATO_PROVIDER_TRACE"}
    cfg = TraceProfileConfig(session_log="/a.log", provider_log="/b.log")
    assert set(cfg.as_env()) == set(TRACE_ENV_VARS.values())


# --------------------------------------------------------------------------
# The block reaches a profile, and the session env
# --------------------------------------------------------------------------

def test_an_inline_spec_carries_the_block():
    """One of the four ingresses, end to end."""
    profile = build_inline_profile(
        {"plugins": [], "trace": {"provider_log": "/tmp/p.jsonl"}})
    assert profile.trace is not None
    assert profile.trace.provider_log == "/tmp/p.jsonl"


def test_a_bad_block_fails_the_ingress_rather_than_being_dropped():
    with pytest.raises(ValueError, match="switch, not a path"):
        build_inline_profile({"plugins": [], "trace": {"provider_log": "1"}})


def test_the_block_outranks_the_untyped_env_map():
    """Typed beats stringly-typed, which is the point of promoting a knob.

    An author who sets both has one validated value and one unvalidated
    one; the validated one wins.  Checked through the real
    ``_resolve_session_env`` rather than a reimplementation of its
    precedence, because the precedence IS what is under test.
    """
    from server.core import JaatoServer

    server = JaatoServer.__new__(JaatoServer)
    server._session_env_resolved = False
    server._session_env = {}
    server.env_file = None
    server._env_overrides = {}
    server._workspace_path = None
    server._profile = SubagentProfile(
        name="t", description="t", plugins=[],
        env={"JAATO_PROVIDER_TRACE": "1"},
        trace=TraceProfileConfig(provider_log="/tmp/typed.jsonl"),
    )

    server._resolve_session_env()

    assert server._session_env["JAATO_PROVIDER_TRACE"] == "/tmp/typed.jsonl"


def test_the_env_map_still_works_on_its_own():
    """Promotion keeps the env var as the default -- nothing breaks."""
    from server.core import JaatoServer

    server = JaatoServer.__new__(JaatoServer)
    server._session_env_resolved = False
    server._session_env = {}
    server.env_file = None
    server._env_overrides = {}
    server._workspace_path = None
    server._profile = SubagentProfile(
        name="t", description="t", plugins=[],
        env={"JAATO_PROVIDER_TRACE": "legacy.log"},
    )

    server._resolve_session_env()

    assert server._session_env["JAATO_PROVIDER_TRACE"] == "legacy.log"


def test_inheritance_takes_the_whole_block_or_none_of_it():
    """Scalar-override: a child that redirects its trace redirects all of it.

    Merging ``session_log`` from a parent with ``provider_log`` from a
    child would produce a split diagnosis nobody asked for.
    """
    from shared.plugins.subagent.config import _merge_profiles

    parent = SubagentProfile(
        name="base", description="", plugins=[],
        trace=TraceProfileConfig(session_log="/parent.log",
                                 provider_log="/parent-p.log"))
    child = SubagentProfile(
        name="child", description="", plugins=[],
        trace=TraceProfileConfig(provider_log="/child-p.log"))

    errors: dict = {}
    merged = _merge_profiles("child", [parent], child, errors)
    assert not errors, errors

    assert merged.trace.provider_log == "/child-p.log"
    assert merged.trace.session_log is None


# --------------------------------------------------------------------------
# The block survives the runner->daemon wire
# --------------------------------------------------------------------------

def test_the_block_rides_the_isolated_runner_wire():
    """Producer shape -> boundary validator -> reconstructed profile.

    An isolated subagent is rebuilt from ``profile_payload`` by
    ``build_inline_profile``.  A key that is not on the allow-list is
    rejected outright, so "the block is honoured for main sessions and
    silently dropped for isolated subagents" is a shape this wire can
    actually take -- pinned here as a round trip rather than trusted.
    """
    from server.runner_rpc_handlers.profile_payload_schema import (
        PROFILE_PAYLOAD_ALLOWED_KEYS,
        validate_profile_payload,
    )

    assert "trace" in PROFILE_PAYLOAD_ALLOWED_KEYS

    payload = {"name": "sub", "plugins": [],
               "trace": {"provider_log": ".jaato/logs/p.jsonl"}}
    validate_profile_payload(payload)
    rebuilt = build_inline_profile(payload)

    assert rebuilt.trace.provider_log == ".jaato/logs/p.jsonl"


def test_the_wire_refuses_a_switch_too():
    """The boundary re-checks rather than deferring to reconstruction.

    ``runtime_limits`` is type-checked here and validated later; this
    block is validated here, because a bad value's whole problem is that
    every string-typed surface it crosses accepts it.
    """
    from server.runner_rpc_handlers.profile_payload_schema import (
        validate_profile_payload,
    )

    with pytest.raises(ValueError, match="switch, not a path"):
        validate_profile_payload({"name": "sub", "trace": {"provider_log": "1"}})
