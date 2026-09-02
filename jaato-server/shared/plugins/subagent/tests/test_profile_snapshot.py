"""A resolved profile must survive a save/restore round trip unchanged.

Issue #787 ruled that a revived session keeps the recipe it was created
under: a profile edited between creation and revive must not reach the
revived session.  That ruling is only true if the snapshot the session
persists rebuilds into the SAME profile — a field the snapshot silently
drops is a recipe change with no edit behind it, which is worse than the
re-resolution it replaced, because nothing anywhere records that it
happened.

So the tests below compare whole dataclasses rather than spot-checking
fields, and assert the snapshot is a FIXED POINT (snapshot → profile →
snapshot is stable).  A field added to ``SubagentProfile`` and forgotten in
``profile_to_snapshot`` fails the first; a shape that drifts one level
deeper per cycle — which ``RuntimeLimits.extra`` did, because
``asdict``/``from_dict`` are not inverses for it — fails the second.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from shared.plugins.subagent.config import (
    PROFILE_SNAPSHOT_VERSION,
    SubagentProfile,
    build_inline_profile,
    parse_plugin_entry,
    profile_from_snapshot,
    profile_to_snapshot,
)


#: Deliberately exercises every structured sub-block and both plugin
#: modifiers, because the fields that get dropped are the ones nobody
#: writes a test for.
_FULL_SPEC = {
    "name": "worker",
    "model": "claude-sonnet-4-20250514",
    "provider": "anthropic",
    "plugins": ["cli", "todo(preload)", "file_edit([readFile,writeFile])"],
    "plugin_configs": {"anthropic": {"api_key": "pass://jaato/anthropic"}},
    "system_instructions": "you are a worker",
    "suppress_base_instructions": {"disk": True, "constants": True},
    "max_turns": 7,
    "gc": {"type": "budget", "threshold_percent": 77.0},
    "cache": {"enabled": True, "ttl": "1h"},
    "trace": {"provider_log": ".jaato/logs/provider.jsonl"},
    "env": {"FOO": "${BAR}", "TOKEN": "pass://jaato/token"},
    "runtime_limits": {"memory_max_mb": 512, "unknown_future_knob": 3},
    "budget_control": {"limits": {"usd": 1.5}},
    "model_tiers": {"vision": {"model": "some-vision-model"}},
    "quirks": {"prose_tool_calls": True},
    "completion_processors": [
        {"script": "scripts/p.py", "output": "out/p", "name": "acceptance"},
    ],
    "completion_payload_schema": {"type": "object"},
    "spawn_payload_schema": "schemas/spawn.json",
    "apparmor": True,
    "apparmor_fragments": ["net"],
}


def _full_profile() -> SubagentProfile:
    profile = build_inline_profile(_FULL_SPEC)
    # Set post-merge only fields the authored shape has no key for, so the
    # round trip is tested against a RESOLVED profile rather than an
    # authored one — which is what a session actually holds.
    profile.suppress_inherited_processors = ["inherited-one"]
    return profile


def test_every_field_survives_the_round_trip():
    profile = _full_profile()
    rebuilt = profile_from_snapshot(profile_to_snapshot(profile))
    assert dataclasses.asdict(rebuilt) == dataclasses.asdict(profile), (
        "a field was lost or altered by the snapshot round trip — a revived "
        "session would come back under a different recipe than it ran under, "
        "which is the thing #787 froze the profile to prevent"
    )


def test_the_snapshot_is_a_fixed_point():
    """profile -> snapshot -> profile -> snapshot must be stable.

    Not implied by the test above: a shape can rebuild into an equal
    profile while the snapshot itself grows a level each cycle, which is
    what ``RuntimeLimits.extra`` did (``from_dict`` parks unknown keys in
    ``extra``, and ``asdict`` emits ``extra`` as a key it then re-parks).
    A session is saved many times, so "stable once" is not enough.
    """
    profile = _full_profile()
    first = profile_to_snapshot(profile)
    second = profile_to_snapshot(profile_from_snapshot(first))
    assert first == second


def test_the_snapshot_is_json_serializable():
    """It is written into the session record, so it must survive json."""
    snapshot = profile_to_snapshot(_full_profile())
    assert json.loads(json.dumps(snapshot)) == snapshot
    assert snapshot["snapshot_version"] == PROFILE_SNAPSHOT_VERSION


def test_secret_uris_stay_unresolved():
    """A snapshot must not turn a URI into the credential behind it.

    A resolved profile holds ``pass://`` / ``vault://`` URIs verbatim —
    expansion happens later, daemon-side, at envelope-build time.  This
    test pins that, because the whole argument for persisting the recipe
    at all is that doing so lands nothing on disk that ``profile_spec``
    did not already land.
    """
    snapshot = profile_to_snapshot(_full_profile())
    assert snapshot["plugin_configs"]["anthropic"]["api_key"] == (
        "pass://jaato/anthropic")
    assert snapshot["env"]["TOKEN"] == "pass://jaato/token"


def test_plugin_modifiers_round_trip_through_their_string_form():
    """preload / tool-scope are re-emitted, not silently dropped.

    They live on ``SubagentProfile`` as two derived fields
    (``preloaded_plugins``, ``tool_scopes``) rather than on the ``plugins``
    strings, so a naive snapshot of ``profile.plugins`` loses both — and a
    revived session would quietly expose every tool of a scoped plugin.
    """
    snapshot = profile_to_snapshot(_full_profile())
    entries = {
        parse_plugin_entry(e)[0]: parse_plugin_entry(e)
        for e in snapshot["plugins"]
    }
    assert entries["todo"][1] is True, "preload modifier lost"
    assert entries["file_edit"][2] == ["readFile", "writeFile"], (
        "per-plugin tool allow-list lost")
    assert entries["cli"] == ("cli", False, None)


def test_inherits_is_dropped_because_a_snapshot_is_post_merge():
    profile = _full_profile()
    profile.inherits = ["base"]
    snapshot = profile_to_snapshot(profile)
    assert "inherits" not in snapshot
    assert profile_from_snapshot(snapshot).inherits is None, (
        "re-declaring the parents on a POST-merge snapshot would re-apply "
        "them on top of a profile that already carries their fields"
    )


def test_a_malformed_snapshot_raises_rather_than_half_building():
    with pytest.raises(ValueError):
        profile_from_snapshot("not a dict")
    with pytest.raises(ValueError):
        profile_from_snapshot({"plugins": [], "gc": {"threshold_percent": "x"},
                               "budget_control": {"nope": 1}})


def test_an_empty_snapshot_still_builds():
    """Forward compatibility: unknown keys are ignored, absent keys default.

    A snapshot written by a newer build must not make a session
    unloadable, and neither must a minimal one.
    """
    profile = profile_from_snapshot({"plugins": [], "some_future_key": 1})
    assert profile.plugins == []
    assert profile.max_turns == 10
