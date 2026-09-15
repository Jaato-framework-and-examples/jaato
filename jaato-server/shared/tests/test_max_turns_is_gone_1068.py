"""``max_turns`` is removed, and nothing may claim it bounds a session (#1068).

``SubagentProfile.max_turns`` was a complete field in every respect but the
one that mattered.  It was declared (``int = field(default=10)``), validated
(non-int and ``<= 0`` refused), inherited most-restrictive-wins, serialized
into snapshots and the runner RPC payload, exposed on the wire as
``ProfileSummary.max_turns``, rendered by ``explain profile``, and
**advertised to the model** by two tool descriptions promising that
"sessions auto-close after max_turns".

It was compared against a turn counter nowhere.  ``grep -c max_turns
jaato-server/shared/jaato_session.py`` returned 0.

The field is removed rather than implemented, because the bound it claimed
to be already exists and works: ``budget_control.limits.turns`` is fed
``turns=1`` per turn by ``_budget_observe_turn`` on every path, and a
``degrade`` rung whose ``action`` is ``abort`` reaches ``request_stop()``.
Two counters for one quantity is the "one check, one door" failure the
reversion meta-guard caught twice while #688 and #1069 were written.

What this module pins is the part a future change could silently undo:

A. the field is gone from the dataclass, the file-key set and the RPC
   allow-list -- and ``GCConfig.max_turns``, a different field on a
   different dataclass that happens to share the name, is untouched;
B. a profile file still carrying the key LOADS, and is reported by
   ``jaato-scaffold validate`` as ``removed_profile_key`` rather than
   erroring or passing silently;
C. no model-facing string claims a session auto-closes on a turn count,
   which is the half that made a parent agent leak its subagents.
"""

import dataclasses
import re
from pathlib import Path

import pytest

from shared.plugins.subagent import config as _cfg
from shared.plugins.subagent.config import (
    PROFILE_FILE_KEYS,
    PROFILE_REMOVED_FIELDS,
    SubagentProfile,
    profile_from_snapshot,
    profile_to_snapshot,
    validate_profile,
)
from shared.scaffold.validate import _profile_key_findings
from shared.tests.test_every_guard_detects_its_own_reversion import Reversion

_CONFIG = "jaato-server/shared/plugins/subagent/config.py"
_VALIDATE = "jaato-server/shared/scaffold/validate.py"
_PLUGIN = "jaato-server/shared/plugins/subagent/plugin.py"

REVERSIONS = [
    Reversion(
        target=_CONFIG,
        find="""PROFILE_REMOVED_FIELDS = {
    'max_turns': (""",
        replace="""PROFILE_REMOVED_FIELDS = {
    '_max_turns_unreported': (""",
        test="test_a_profile_still_carrying_the_key_is_reported_by_name",
        because=(
            "a workspace whose profiles declare max_turns is told nothing, "
            "so the key goes on being written and goes on doing nothing"
        ),
    ),
    Reversion(
        target=_VALIDATE,
        find="""        if key in PROFILE_REMOVED_FIELDS:""",
        replace="""        if False and key in PROFILE_REMOVED_FIELDS:""",
        test="test_a_removed_key_is_not_reported_as_an_unknown_one",
        because=(
            "a withdrawn field reads as a typo, and the message offers a "
            "near-miss instead of the successor"
        ),
    ),
    Reversion(
        target=_PLUGIN,
        find="""                    'A subagent session does NOT close itself after a fixed number '""",
        replace="""                    'While sessions auto-close after max_turns, explicit closure is preferred '""",
        test="test_no_model_facing_string_promises_a_turn_count_autoclose",
        because=(
            "the model is told again that a subagent closes itself, so a "
            "parent that declines to close_subagent leaks it"
        ),
    ),
]


_REPO = Path(__file__).resolve().parents[3]


# --------------------------------------------------------------- A. the field

def test_the_field_is_gone_from_the_dataclass():
    names = {f.name for f in dataclasses.fields(SubagentProfile)}
    assert "max_turns" not in names
    assert not hasattr(SubagentProfile(name="x", description="d"), "max_turns")


def test_it_is_gone_from_every_set_that_decides_what_a_profile_may_declare():
    assert "max_turns" not in PROFILE_FILE_KEYS
    from server.runner_rpc_handlers.profile_payload_schema import (
        PROFILE_PAYLOAD_ALLOWED_KEYS,
    )
    assert "max_turns" not in PROFILE_PAYLOAD_ALLOWED_KEYS


def test_the_gc_field_of_the_same_name_is_untouched():
    """``GCConfig.max_turns`` is a garbage-collection TRIGGER that happens to
    share the name.  Every ``turns >= config.max_turns`` comparison in the
    tree is that field, which is why the issue's first grep looked like the
    bound existed."""
    from shared.plugins.gc import GCConfig
    assert GCConfig(threshold_percent=90.0, max_turns=10).max_turns == 10
    gc_names = {f.name for f in dataclasses.fields(_cfg.GCProfileConfig)}
    assert "max_turns" in gc_names


def test_a_snapshot_does_not_carry_it_and_an_old_one_still_revives():
    """A session persisted before the removal must wake, not raise: the
    record reader gates on the MAJOR version, which a dropped key is
    indifferent to."""
    p = SubagentProfile(name="x", description="d", plugins=["cli"])
    snap = profile_to_snapshot(p)
    assert "max_turns" not in snap
    old = dict(snap, max_turns=7)
    assert profile_from_snapshot(old).name == "x"


# ------------------------------------------------- B. an existing profile file

@pytest.mark.parametrize("value", ["ten", 0, -1, True, 15])
def test_a_profile_file_still_carrying_the_key_loads(value):
    """Erroring here would fail every existing workspace over a line that
    has always been inert.  The loader is keyword-explicit, so the key is
    simply not read."""
    ok, errors, _warnings = validate_profile(
        {"name": "t", "description": "d", "max_turns": value})
    assert ok is True, errors
    assert not any("max_turns" in e for e in errors), errors


def test_a_profile_still_carrying_the_key_is_reported_by_name():
    found = _profile_key_findings(
        {"name": "p", "plugins": [], "max_turns": 5}, "p")
    codes = {f.code for f in found}
    assert "removed_profile_key" in codes, codes
    msg = next(f.message for f in found if f.code == "removed_profile_key")
    assert "budget_control" in msg, (
        "the message must name the successor; that is the whole reason this "
        "is not a bare unknown_profile_key")
    assert all(f.severity == "warn" for f in found)


def test_a_removed_key_is_not_reported_as_an_unknown_one():
    """``max_turns`` has no close match in the accepted set, so
    ``unknown_profile_key`` would have offered no fix at all."""
    found = _profile_key_findings({"name": "p", "max_turns": 5}, "p")
    assert [f.code for f in found] == ["removed_profile_key"]


def test_the_removed_set_is_not_empty_and_names_what_replaces_each_key():
    assert PROFILE_REMOVED_FIELDS
    for key, why in PROFILE_REMOVED_FIELDS.items():
        assert key not in PROFILE_FILE_KEYS, key
        assert why.strip(), key


# ------------------------------------------------- C. what the model is told

_AUTOCLOSE = re.compile(
    r"auto-?close[sd]?\s+after\s+max_turns|sessions?\s+auto-?close",
    re.IGNORECASE)


def test_no_model_facing_string_promises_a_turn_count_autoclose():
    """The two descriptions that said so are why a parent agent that
    declined to ``close_subagent`` -- reasoning correctly per its own
    instructions -- leaked the session."""
    text = (_REPO / _PLUGIN).read_text()
    hits = [ln.strip() for ln in text.splitlines() if _AUTOCLOSE.search(ln)]
    assert not hits, hits


def test_the_session_that_owns_the_turn_loop_still_does_not_mention_it():
    """The measurement the issue rests on, kept as a guard: if a future
    change reintroduces the name here, it is building the second counter."""
    text = (_REPO / "jaato-server/shared/jaato_session.py").read_text()
    assert "max_turns" not in text
