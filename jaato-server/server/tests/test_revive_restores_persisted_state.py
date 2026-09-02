"""A revived session wakes from what it persisted, not from disk (#787).

THE FAILURE THIS EXISTS FOR.  A session whose profile declares a mandatory
``{{!py:...}}`` prefetch that reads ``context.agent_params`` — the
documented way to pass per-agent inputs to a prefetch — was created fine,
ran fine and persisted fine, and then could not be revived by anything:

    RunnerCallError: session.bootstrap failed: ToolError: session.bootstrap:
      dynamic-instructions abort: scripts/checkout_worktree.py: render raised
      RuntimeError: input.agent_params must carry both 'repo' and 'issue_id'

The params were not missing.  They were present at creation and absent on
the revive, because bootstrap REBUILT the system prompt — re-reading the
instruction layers, re-resolving the agent markdown and re-running the
prefetch — against ``agent_params`` that were never persisted.  The error
blamed the task definition, which was correct all along.

The fix is not to persist the params and re-run the script.  It is to stop
re-deriving: the session persists the RENDERED prompt and the RESOLVED
profile, and a revive restores them.  That also makes a prefetch run once
as documented (the reported one materialises a git worktree) and stops a
revived session's prompt from silently diverging from the one its own
history was produced under.

Two env knobs opt back into re-deriving, and the params are persisted for
the persona one.  The tests below pin both defaults, both opt-ins, and the
backward-compatible fallback for records written before any of this
existed.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from server.session_manager import Session, SessionManager
from server.revive_policy import (
    DISK,
    ENV_REVIVE_PERSONA,
    ENV_REVIVE_PROFILE,
    PERSISTED,
    persona_source,
    profile_source,
)
from shared.plugins.subagent.config import (
    build_inline_profile,
    profile_to_snapshot,
)


_RENDERED = "PERSONA AS RENDERED\n<worktree at /w/issue-787>"


def _profile(name="worker", model="m1"):
    return build_inline_profile(
        {"name": name, "model": model, "plugins": ["cli"]},
    )


def _state(**over):
    base = dict(
        session_id="20260901_215524",
        profile_name="worker",
        profile_spec=None,
        profile_snapshot=None,
        rendered_instructions=None,
        agent_params=None,
        agent_name="worker-persona",
        workspace_path="/w",
        config_root="/w/.jaato",
    )
    base.update(over)
    return SimpleNamespace(**base)


class _Manager:
    """A SessionManager stand-in recording what the revive path reached for.

    Only the two disk lookups matter here — whether the revive re-read the
    profile files and whether it re-rendered the persona (which is what
    re-runs the prefetch).  Everything else is irrelevant to the question.
    """

    #: ``None`` is a MEANINGFUL agent_result (agent not found on disk), so
    #: "not supplied" needs its own value.
    _DEFAULT = object()

    def __init__(self, agent_result=_DEFAULT, profile=None):
        self.resolved_agents = []
        self.resolved_profiles = []
        self._agent_result = (
            {"system_instructions": "RE-RENDERED FROM DISK"}
            if agent_result is _Manager._DEFAULT
            else agent_result
        )
        self._profile = profile if profile is not None else _profile("disk-worker")

    def _resolve_agent(self, name, params, workspace_path, config_root=None):
        self.resolved_agents.append((name, params, workspace_path, config_root))
        return self._agent_result

    def _resolve_profile(self, name, workspace_path, config_root=None,
                         env_file=None):
        self.resolved_profiles.append((name, workspace_path, config_root))
        return self._profile, None


def _persona(mgr, state, profile):
    return SessionManager._resolve_revive_persona(
        mgr, state, profile, session_id="s1",
        workspace_path="/w", config_root="/w/.jaato",
    )


def _recipe(mgr, state):
    return SessionManager._resolve_revive_profile(
        mgr, state, session_id="s1", workspace_path="/w",
        config_root="/w/.jaato", env_file=None,
    )


# ---------------------------------------------------------------- persona

def test_a_persisted_prompt_is_restored_and_no_prefetch_re_runs():
    """The regression, stated positively.

    ``_resolve_agent`` is the door to the persona markdown, and re-rendering
    it is what re-runs ``{{!py:...}}``.  Not knocking on that door is the
    whole fix: the returned override makes the runner skip assembly.
    """
    mgr = _Manager()
    profile = _profile()
    override = _persona(mgr, _state(rendered_instructions=_RENDERED), profile)

    assert override == _RENDERED
    assert mgr.resolved_agents == [], (
        "the revive re-resolved the persona from disk despite having the "
        "rendered prompt on hand — which re-runs the prefetch that made "
        "these sessions unwakeable"
    )
    assert profile.system_instructions != "RE-RENDERED FROM DISK"


def test_a_record_with_no_rendered_prompt_re_renders_as_before():
    """Backward compatibility: every session already on disk is pre-2.8.

    It carries no rendered prompt, so it must revive exactly as it did
    before this change rather than failing.
    """
    mgr = _Manager()
    profile = _profile()
    override = _persona(mgr, _state(), profile)

    assert override is None, "no prompt was persisted; nothing to restore"
    assert len(mgr.resolved_agents) == 1
    assert profile.system_instructions == "RE-RENDERED FROM DISK"


def test_the_re_render_path_passes_the_original_agent_params():
    """The proximate cause of #787, pinned.

    On the re-render path the prefetch DOES run, and it must run against
    the params the session was created with.  Handing it ``None`` is what
    produced "the task.yaml for this arm is missing one of them" for a
    task.yaml that was complete.
    """
    mgr = _Manager()
    params = {"repo": "jaato", "issue_id": "787"}
    _persona(mgr, _state(agent_params=params), _profile())

    assert mgr.resolved_agents, "the persona was never resolved"
    assert mgr.resolved_agents[0][1] == params, (
        "the persisted agent_params did not reach resolve_agent — a "
        "mandatory prefetch reading context.agent_params aborts "
        "session-prep, and the error blames the task definition"
    )


def test_the_persona_knob_forces_a_re_render(monkeypatch):
    """``JAATO_REVIVE_PERSONA=disk`` — testing an alternative persona.

    Deliberately re-runs the prefetch, against the ORIGINAL params.
    """
    monkeypatch.setenv(ENV_REVIVE_PERSONA, "disk")
    mgr = _Manager()
    profile = _profile()
    params = {"repo": "jaato"}
    override = _persona(
        mgr, _state(rendered_instructions=_RENDERED, agent_params=params),
        profile,
    )

    assert override is None, "the knob asked for a re-render"
    assert mgr.resolved_agents[0][1] == params
    assert profile.system_instructions == "RE-RENDERED FROM DISK"


def test_an_unresolvable_persona_does_not_fabricate_an_override():
    mgr = _Manager(agent_result=None)
    profile = _profile()
    assert _persona(mgr, _state(), profile) is None
    assert profile.system_instructions is None


# ----------------------------------------------------------------- recipe

def test_a_persisted_profile_snapshot_wins_over_the_file_on_disk():
    """The operator ruling: a profile edit must not reach a revived session.

    Before this, a named profile was re-resolved at revive time, so an edit
    between creation and revive silently changed what the session ran under
    — the same re-derivation that re-ran the prefetch.
    """
    mgr = _Manager(profile=_profile("worker", model="EDITED-ON-DISK"))
    snapshot = profile_to_snapshot(_profile("worker", model="AS-CREATED"))

    profile = _recipe(mgr, _state(profile_snapshot=snapshot))

    assert profile.model == "AS-CREATED"
    assert mgr.resolved_profiles == [], "the revive re-read the profile files"


def test_the_profile_knob_forces_a_disk_resolution(monkeypatch):
    """``JAATO_REVIVE_PROFILE=disk`` — interrogation's requirement.

    Interrogating a finished session selects a different contract via
    ``JAATO_PROFILE_SET``, and a profile set is resolved inside
    ``discover_profiles`` — so a frozen profile would make the set
    selection silently inert.  This is the axis where the DEFAULT is the
    wrong answer for that workflow, which is why the knob exists.
    """
    monkeypatch.setenv(ENV_REVIVE_PROFILE, "disk")
    mgr = _Manager(profile=_profile("worker", model="EDITED-ON-DISK"))
    snapshot = profile_to_snapshot(_profile("worker", model="AS-CREATED"))

    profile = _recipe(mgr, _state(profile_snapshot=snapshot))

    assert profile.model == "EDITED-ON-DISK"
    assert mgr.resolved_profiles, "the knob asked for a disk resolution"


def test_a_broken_snapshot_falls_back_to_the_disk_profile():
    """A snapshot that cannot rebuild must not make a session unloadable.

    The worst case of falling through is the pre-#787 behaviour.  The worst
    case of raising is the failure this whole change exists to remove.
    """
    mgr = _Manager(profile=_profile("worker", model="FROM-DISK"))
    profile = _recipe(mgr, _state(profile_snapshot={"gc": {"type": 1, "threshold_percent": "x"},
                                                    "budget_control": {"bogus": 1}}))
    assert profile.model == "FROM-DISK"
    assert mgr.resolved_profiles


def test_an_inline_session_still_restores_from_its_own_spec():
    """``profile_spec`` stays authoritative for inline sessions.

    They were already frozen — there is no name to re-resolve — so they
    must not be routed through the snapshot path even if one is present.
    """
    mgr = _Manager()
    spec = {"name": "nano-chat", "model": "INLINE", "plugins": []}
    profile = _recipe(mgr, _state(
        profile_spec=spec,
        profile_snapshot=profile_to_snapshot(_profile("worker", "SNAPSHOT")),
    ))
    assert profile.model == "INLINE"
    assert mgr.resolved_profiles == []


def test_a_pre_2_8_named_session_resolves_from_disk_exactly_as_before():
    mgr = _Manager(profile=_profile("worker", model="FROM-DISK"))
    profile = _recipe(mgr, _state())
    assert profile.model == "FROM-DISK"
    assert mgr.resolved_profiles == [("worker", "/w", "/w/.jaato")]


# ------------------------------------------------------------------ knobs

@pytest.mark.parametrize("raw,expected", [
    (None, PERSISTED),
    ("", PERSISTED),
    ("persisted", PERSISTED),
    ("disk", DISK),
    ("DISK", DISK),
    (" reload ", DISK),
    ("re-render", DISK),
    # An unrecognised value must NOT be read as the opt-in: silently doing
    # the opposite of what a typo asked for is worse than doing the default
    # and saying so.
    ("yes", PERSISTED),
])
def test_the_knobs_parse_defensively(monkeypatch, raw, expected):
    for var, fn in ((ENV_REVIVE_PROFILE, profile_source),
                    (ENV_REVIVE_PERSONA, persona_source)):
        monkeypatch.delenv(var, raising=False)
        if raw is not None:
            monkeypatch.setenv(var, raw)
        assert fn() == expected


def test_the_two_knobs_are_independent(monkeypatch):
    """The combination interrogation needs is neither knob's default.

    ``profile = disk`` (to pick up a different ``JAATO_PROFILE_SET``) with
    ``persona = persisted`` (to ask the session about the prompt it
    actually saw).  If one knob drove both, that combination would be
    unreachable and the knobs would be decoration.
    """
    monkeypatch.setenv(ENV_REVIVE_PROFILE, "disk")
    monkeypatch.delenv(ENV_REVIVE_PERSONA, raising=False)
    assert profile_source() == DISK
    assert persona_source() == PERSISTED


# ------------------------------------------------------- capture at save

class _RPCWithPrompt:
    def __init__(self, rendered=_RENDERED):
        self.calls = 0
        self._rendered = rendered

    def session_get_rendered_system_instruction_threadsafe(self, **kw):
        self.calls += 1
        return self._rendered


def _session(server=None, **over):
    fields = dict(
        session_id="s1", name="s", server=server, created_at="2026-09-01T00:00:00",
    )
    fields.update(over)
    return Session(**fields)


def test_the_first_save_captures_the_prompt_the_recipe_and_the_params():
    rpc = _RPCWithPrompt()
    params = {"repo": "jaato", "issue_id": "787"}
    session = _session(server=SimpleNamespace(_runner_rpc=rpc,
                                              _agent_params=params))
    profile = _profile("worker", "m1")

    SessionManager._capture_revive_snapshots(
        SimpleNamespace(), session, profile)

    assert session.rendered_instructions == _RENDERED
    assert session.profile_snapshot["model"] == "m1"
    assert session.agent_params == params


def test_capture_is_write_once_so_a_re_render_cannot_destroy_the_original():
    """Testing an alternative must not overwrite what it is compared against.

    A revive with ``JAATO_REVIVE_PERSONA=disk`` re-renders the prompt; its
    next save must still persist the ORIGINAL render, or interrogating the
    session later asks it about a prompt it never saw.
    """
    rpc = _RPCWithPrompt(rendered="THE RE-RENDER")
    session = _session(
        server=SimpleNamespace(_runner_rpc=rpc, _agent_params={"a": "2"}),
        rendered_instructions=_RENDERED,
        profile_snapshot={"name": "as-created", "plugins": []},
        agent_params={"a": "1"},
    )

    SessionManager._capture_revive_snapshots(
        SimpleNamespace(), session, _profile("re-resolved"))

    assert session.rendered_instructions == _RENDERED
    assert session.profile_snapshot["name"] == "as-created"
    assert session.agent_params == {"a": "1"}
    assert rpc.calls == 0, "the runner was queried for a value already held"


def test_an_inline_session_is_not_snapshotted_twice():
    """``profile_spec`` already freezes it; a second copy is two sources."""
    session = _session(
        server=SimpleNamespace(_runner_rpc=_RPCWithPrompt(), _agent_params={}),
        inline_profile_spec={"name": "nano", "plugins": []},
    )
    SessionManager._capture_revive_snapshots(
        SimpleNamespace(), session, _profile())
    assert session.profile_snapshot is None


def test_a_capture_failure_never_blocks_the_save():
    """A missing snapshot costs a re-derived revive.  A raise costs the session."""
    class _Boom:
        def session_get_rendered_system_instruction_threadsafe(self, **kw):
            raise RuntimeError("runner is gone")

    session = _session(server=SimpleNamespace(_runner_rpc=_Boom(),
                                              _agent_params={}))
    SessionManager._capture_revive_snapshots(
        SimpleNamespace(), session, object())  # not a profile → snapshot raises

    assert session.rendered_instructions is None
    assert session.profile_snapshot is None
