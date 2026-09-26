"""Session-picker listing keys and ``session.new --model/--provider`` (1.27).

Covers:

* :func:`session_picker_fields` -- the one definition both client-facing
  listings (``session.list`` and the ``SessionInfoEvent`` snapshot) spread
  into each row;
* :func:`_apply_model_override` -- a profile is COPIED, never mutated; an
  inline spec is rewritten so its revive keeps the model; a profile-less
  session gets ``MODEL_NAME`` / ``JAATO_PROVIDER`` env overrides that are
  also returned for persistence;
* the argv parse in ``CommandRouter._handle_session_new`` and the
  ``--provider``-without-``--model`` refusal in ``create_session``.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

from jaato_server.server.command_router import CommandRouter
from jaato_server.server.session_manager import (
    RuntimeSessionInfo,
    SessionManager,
    _apply_model_override,
    _model_override_metadata,
    session_picker_fields,
)
from jaato_server.shared.plugins.subagent.config import (
    SubagentProfile,
    profile_from_snapshot,
    profile_to_snapshot,
)


# ------------------------------------------------------------- listing keys

def _row(**over):
    base = dict(
        session_id="s1", name="n", description=None,
        created_at="2026-09-01T00:00:00+00:00",
        last_activity="2026-09-02T00:00:00+00:00",
        model_provider="", model_name="", is_processing=True, is_loaded=True,
        client_count=0, turn_count=3,
    )
    base.update(over)
    return RuntimeSessionInfo(**base)


def test_picker_fields_carry_profile_activity_and_processing():
    fields = session_picker_fields(_row(profile_name="researcher"))
    assert fields == {
        "profile": "researcher",
        "last_activity": "2026-09-02T00:00:00+00:00",
        "is_processing": True,
        "created_at": "2026-09-01T00:00:00+00:00",
    }


def test_a_profile_less_row_says_empty_string_not_none():
    assert session_picker_fields(_row())["profile"] == ""


def test_a_duck_typed_row_gets_defaults_rather_than_raising():
    fields = session_picker_fields(SimpleNamespace(session_id="x"))
    assert fields == {"profile": "", "last_activity": "",
                      "is_processing": False, "created_at": ""}


def test_session_list_rows_carry_the_picker_keys():
    sent = []
    router = CommandRouter.__new__(CommandRouter)
    router._event_sink = SimpleNamespace(send_event=lambda cid, ev: sent.append(ev))
    router._sessions_visible_to = lambda cid: [_row(profile_name="p")]
    router._handle_session_list("c1", "s1")
    (row,) = sent[0].sessions
    assert row["profile"] == "p"
    assert row["is_processing"] is True
    assert row["created_at"] and row["last_activity"]


# ------------------------------------------------------------ model override

def _profile(**over):
    return SubagentProfile(name="researcher", description="d",
                           model="claude-x", provider="anthropic", **over)


def test_no_model_changes_nothing():
    prof = _profile()
    out = _apply_model_override(prof, None, {"A": "1"}, None, None)
    assert out == (prof, None, {"A": "1"}, None)


def test_a_profile_is_copied_not_mutated():
    prof = _profile()
    new, spec, env, persisted = _apply_model_override(
        prof, None, None, "gpt-5.1", "openai")
    assert (new.model, new.provider) == ("gpt-5.1", "openai")
    assert (prof.model, prof.provider) == ("claude-x", "anthropic")
    assert new is not prof
    assert env is None and persisted is None


def test_model_alone_keeps_the_profiles_provider():
    new, *_ = _apply_model_override(_profile(), None, None, "claude-y", None)
    assert (new.model, new.provider) == ("claude-y", "anthropic")


def test_the_override_survives_the_revive_snapshot():
    new, *_ = _apply_model_override(_profile(), None, None, "gpt-5.1", "openai")
    revived = profile_from_snapshot(profile_to_snapshot(new))
    assert (revived.model, revived.provider) == ("gpt-5.1", "openai")


def test_an_inline_spec_is_rewritten_so_its_revive_keeps_the_model():
    spec = {"model": "claude-x", "plugins": ["cli"]}
    _, new_spec, _, _ = _apply_model_override(
        _profile(), spec, None, "gpt-5.1", "openai")
    assert new_spec == {"model": "gpt-5.1", "provider": "openai", "plugins": ["cli"]}
    assert spec["model"] == "claude-x"


def test_no_profile_goes_through_env_overrides_and_is_persisted():
    prof, spec, env, persisted = _apply_model_override(
        None, None, {"MODEL_NAME": "old", "X": "1"}, "gpt-5.1", "openai")
    assert prof is None and spec is None
    assert env == {"MODEL_NAME": "gpt-5.1", "X": "1", "JAATO_PROVIDER": "openai"}
    assert persisted == {"MODEL_NAME": "gpt-5.1", "JAATO_PROVIDER": "openai"}
    session = SimpleNamespace(model_override_env=persisted)
    assert _model_override_metadata(session) == {"model_override_env": persisted}
    assert _model_override_metadata(SimpleNamespace()) == {}


# ------------------------------------------------------------ argv + refusal

class _CapturingManager:
    def __init__(self):
        self.kwargs = None

    def create_session(self, *args, **kwargs):
        self.kwargs = kwargs
        return ""


def _router(manager):
    router = CommandRouter.__new__(CommandRouter)
    router._session_manager = manager
    router._event_sink = SimpleNamespace(
        get_client_user=lambda cid: None, send_event=lambda *a: None)
    router._hint_available_auth_providers = lambda cid: None
    return router


def test_session_new_parses_model_and_provider():
    manager = _CapturingManager()
    _router(manager)._handle_session_new(
        "c1", ["mine", "--profile", "researcher", "--model", "gpt-5.1",
               "--provider", "openai", "k=v"], None)
    kw = manager.kwargs
    assert kw["profile_name"] == "researcher"
    assert kw["model_override"] == "gpt-5.1"
    assert kw["provider_override"] == "openai"
    assert kw["agent_params"] == {"k": "v"}


def test_session_new_without_flags_passes_no_overrides():
    manager = _CapturingManager()
    _router(manager)._handle_session_new("c1", ["--profile", "p"], None)
    assert "model_override" not in manager.kwargs
    assert manager.kwargs["profile_name"] == "p"


def test_provider_without_model_is_refused_as_an_invalid_spec():
    sm = SessionManager.__new__(SessionManager)
    sm._session_new_answer = threading.local()
    answered = []
    sm._answer_session_new = lambda cid, ev, **kw: answered.append((cid, ev))
    sm._answer_session_new_last_resort = lambda *a, **kw: None

    def _impl(*a, **kw):
        raise AssertionError("must be refused before creating anything")

    sm._create_session_impl = _impl
    assert sm.create_session("c1", provider_override="openai") == ""
    (cid, event), = answered
    assert cid == "c1"
    assert event.error_type == "InvalidSessionSpec"
    assert "--provider requires --model" in event.error
