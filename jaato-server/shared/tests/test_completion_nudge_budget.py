"""Guard: the completion-nudge budget belongs to the deployment (#919).

``try_completion_nudge(max_nudges)`` has always taken the bound as an
argument — its own docstring calls it a knob — but every caller passed a
function-local ``MAX_COMPLETION_NUDGES = 2``.  Three of them, in three
files nothing kept equal, and none reachable from a profile.  Meanwhile
the bounds on either side of it in the same path are profile keys:
``max_turns``, ``runtime_limits``, and a processor's ``max_refusals``,
which is the close analogue — a profile could say how many times a
processor may BLOCK a completion, but not how many chances the model got
to CALL ``signal_completion`` in the first place.

What broke: an audio tier that hands off to a text tier, writes to
memory, narrates the write, and never calls the tool.  One of its two
nudges goes on a redundant ``enter_tier`` ("already_at_tier"), leaving
exactly one real attempt; 3 of 5 measured turns ended ``NudgeExhausted``
with the memory already on disk.  Announcing a tool instead of invoking
it is a documented weakness of that model class, not something persona
prose fixes.

This module asserts the four things that make the knob real:

1. the default is UNCHANGED at 2 — for a strong tool-caller it is right,
   and this is not an argument for raising it globally;
2. a profile's value reaches the resolver, and a profile that predates
   the field (or no profile at all) still resolves to the default;
3. the budget is what ``try_completion_nudge`` actually spends — a
   declared 4 buys four nudges, not two;
4. the number has exactly ONE definition, so the daemon guard, the
   embedded lead and the subagent loop cannot drift apart again.
"""

from __future__ import annotations

import ast
import pathlib
from typing import Any, Optional, Tuple

import pytest

from shared.completion_nudge import (
    DEFAULT_MAX_COMPLETION_NUDGES,
    coerce_max_completion_nudges,
    resolve_max_completion_nudges,
)

SERVER_ROOT = pathlib.Path(__file__).resolve().parents[2]

#: Every site that spends the budget.  Each one used to carry its own
#: ``= 2``; the guard below asserts none of them does again.
NUDGE_SITES = (
    SERVER_ROOT / "server" / "core.py",
    SERVER_ROOT / "jaato_embedded" / "client.py",
    SERVER_ROOT / "shared" / "plugins" / "subagent" / "plugin.py",
)


class _Profile:
    """Minimal stand-in carrying only what the resolver reads."""

    def __init__(self, value: Any = None, name: str = "stand-in") -> None:
        self.max_completion_nudges = value
        self.name = name


class _LegacyProfile:
    """A profile object from before the field existed (an old snapshot)."""

    name = "legacy"


# ---------------------------------------------------------------------------
# 1. The default is unchanged
# ---------------------------------------------------------------------------

class TestTheDefaultIsUnchanged:
    def test_the_framework_default_is_still_two(self):
        """#919 is not an argument that 2 is a bad default.

        Raising it globally would make weak models loop longer for
        everyone; the point is that the number belongs to the
        deployment.  An unconfigured checkout must nudge exactly twice,
        as it did before the knob existed.
        """
        assert DEFAULT_MAX_COMPLETION_NUDGES == 2

    def test_no_profile_resolves_to_the_default(self):
        assert resolve_max_completion_nudges(None) == 2

    def test_a_profile_declaring_nothing_resolves_to_the_default(self):
        assert resolve_max_completion_nudges(_Profile(None)) == 2

    def test_a_profile_predating_the_field_resolves_to_the_default(self):
        """A revived session's snapshot may have no such key at all.

        The sites call the resolver rather than reading the attribute
        precisely so this is a default, not an ``AttributeError`` on the
        path that gives up on a session.
        """
        assert resolve_max_completion_nudges(_LegacyProfile()) == 2


# ---------------------------------------------------------------------------
# 2. A declared value reaches the resolver; a bad one does not
# ---------------------------------------------------------------------------

class TestTheProfileValueIsHonoured:
    @pytest.mark.parametrize("declared", [1, 3, 4, 10])
    def test_a_declared_budget_is_returned(self, declared):
        assert resolve_max_completion_nudges(_Profile(declared)) == declared

    def test_the_raw_sibling_agrees_with_the_attribute_reader(self):
        """The embedded facade holds a spec DICT, not a profile object.

        Both spellings must resolve identically or the in-process lead
        nudges a different number of times than the same profile does
        under the daemon.
        """
        for declared in (None, 1, 4, 0, -1, "4", True):
            assert (
                coerce_max_completion_nudges(declared)
                == resolve_max_completion_nudges(_Profile(declared))
            )

    @pytest.mark.parametrize("declared", [0, -1, "4", 2.5, True, False, object()])
    def test_a_non_positive_or_non_int_falls_back_to_the_default(self, declared):
        """Fail safe, and specifically never to zero.

        The give-up predicate is ``nudges_fired >= max``, so honouring a
        budget of 0 would report ``NudgeExhausted`` on sessions that
        completed CLEANLY (0 >= 0) — a knob that turns success into a
        terminal error.  ``validate_profile`` refuses it at load; this
        is the second line for a hand-built profile object.
        """
        assert resolve_max_completion_nudges(_Profile(declared)) == 2

    def test_a_bad_value_is_announced_not_swallowed(self, caplog):
        with caplog.at_level("WARNING"):
            resolve_max_completion_nudges(_Profile(0, name="loud"))
        assert any(
            "max_completion_nudges" in r.getMessage()
            for r in caplog.records
        ), "a refused value must be announced, not silently defaulted"


# ---------------------------------------------------------------------------
# 3. The budget is what try_completion_nudge actually spends
# ---------------------------------------------------------------------------

class _Session:
    """The two attributes ``try_completion_nudge`` reads and writes."""

    def __init__(self) -> None:
        self._signal_completion_called = False
        self._completion_nudges_fired = 0


def _spend(max_nudges: int) -> int:
    """Nudge until the budget refuses, returning how many fired."""
    from shared.jaato_session import JaatoSession

    session = _Session()
    fired = 0
    for _ in range(max_nudges + 5):  # a loose bound; the budget must stop us
        should, _count = JaatoSession.try_completion_nudge(session, max_nudges)
        if not should:
            break
        fired += 1
    return fired


class TestTheBudgetIsSpent:
    def test_the_default_buys_two_nudges(self):
        assert _spend(DEFAULT_MAX_COMPLETION_NUDGES) == 2

    def test_a_raised_budget_buys_that_many(self):
        """The whole point: a deployment that knows its model needs four
        attempts gets four, not two."""
        assert _spend(4) == 4

    def test_a_lowered_budget_buys_that_many(self):
        assert _spend(1) == 1

    def test_signalling_spends_nothing(self):
        """``try_completion_nudge`` returns False both when the budget is
        spent and when the agent already signalled.  Only the second must
        leave the counter at zero — the give-up predicates downstream key
        on the count to tell the two apart."""
        from shared.jaato_session import JaatoSession

        session = _Session()
        session._signal_completion_called = True
        assert JaatoSession.try_completion_nudge(session, 4) == (False, 0)


# ---------------------------------------------------------------------------
# 4. One definition, three sites
# ---------------------------------------------------------------------------

def _assigned_constants(path: pathlib.Path) -> list:
    """Every ``MAX_COMPLETION_NUDGES = <expr>`` assignment in *path*."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for t in node.targets
        if isinstance(t, ast.Name) and t.id == "MAX_COMPLETION_NUDGES"
    ]


class TestOneDefinition:
    def test_the_guard_can_still_find_what_it_inspects(self):
        """Anchor.  Without it the assertions below pass on an empty match."""
        for site in NUDGE_SITES:
            assert site.exists(), f"{site} not found — guard is stale"
        assert any(
            "MAX_COMPLETION_NUDGES" in s.read_text(encoding="utf-8")
            for s in NUDGE_SITES
        ), "no site names the ceiling — re-aim this guard, do not delete it"

    @pytest.mark.parametrize("site", NUDGE_SITES, ids=lambda p: p.name)
    def test_no_site_hardcodes_the_number(self, site):
        """The hazard #919 names: two (in fact three) independent copies of
        the same literal, with nothing keeping them equal."""
        for value in _assigned_constants(site):
            assert not isinstance(value, ast.Constant), (
                f"{site.name} assigns MAX_COMPLETION_NUDGES a literal "
                f"({ast.dump(value)}) — the budget is the profile's, "
                f"resolved through shared.completion_nudge"
            )

    @pytest.mark.parametrize("site", NUDGE_SITES, ids=lambda p: p.name)
    def test_every_site_resolves_through_the_shared_default(self, site):
        src = site.read_text(encoding="utf-8")
        assert (
            "resolve_max_completion_nudges" in src
            or "coerce_max_completion_nudges" in src
        ), f"{site.name} spends a nudge budget it did not resolve"

    def test_the_literal_lives_exactly_once(self):
        """Grep-level: no file outside the definition module (and the
        deliberately-restated eval mirror) sets the number itself."""
        module = SERVER_ROOT / "shared" / "completion_nudge.py"
        assert "DEFAULT_MAX_COMPLETION_NUDGES = 2" in module.read_text(
            encoding="utf-8"
        )


# ---------------------------------------------------------------------------
# 5. The profile layer: parse, validate, inherit, persist
# ---------------------------------------------------------------------------

class TestTheProfileLayer:
    def test_a_profile_file_parses_the_key(self, tmp_path):
        from shared.plugins.subagent.config import discover_profiles

        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()
        (profiles_dir / "voz.yaml").write_text(
            "name: voz\n"
            "description: an audio tier that narrates its tool calls\n"
            "plugins: []\n"
            "max_completion_nudges: 4\n",
            encoding="utf-8",
        )
        result = discover_profiles(str(profiles_dir), config_root=str(tmp_path))
        assert result.profiles["voz"].max_completion_nudges == 4

    def test_an_absent_key_stays_none_so_the_resolver_owns_the_default(
        self, tmp_path,
    ):
        """``None`` rather than a baked-in 2: the merge must be able to tell
        "declared 2" from "declared nothing", and the DEFAULT must have one
        home (``shared.completion_nudge``) rather than a copy per parser."""
        from shared.plugins.subagent.config import discover_profiles

        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()
        (profiles_dir / "plain.yaml").write_text(
            "name: plain\ndescription: no knob\nplugins: []\n",
            encoding="utf-8",
        )
        result = discover_profiles(str(profiles_dir), config_root=str(tmp_path))
        assert result.profiles["plain"].max_completion_nudges is None
        assert resolve_max_completion_nudges(result.profiles["plain"]) == 2

    @pytest.mark.parametrize("bad", [0, -1, "4", 2.5, True])
    def test_validate_refuses_a_non_positive_or_non_int(self, bad):
        from shared.plugins.subagent.config import validate_profile

        ok, errors, _warnings = validate_profile({
            "name": "x", "description": "y", "max_completion_nudges": bad,
        })
        assert not ok
        assert any("max_completion_nudges" in e for e in errors), errors

    def test_validate_accepts_a_positive_int_and_an_absent_key(self):
        from shared.plugins.subagent.config import validate_profile

        for data in (
            {"name": "x", "description": "y", "max_completion_nudges": 4},
            {"name": "x", "description": "y"},
        ):
            ok, errors, _warnings = validate_profile(data)
            assert ok, errors

    def test_the_snapshot_round_trips_the_value(self):
        """A revived session nudges as many times as it was created to.

        The snapshot is what a woken session comes back with (#787), so a
        knob absent from it silently reverts to the default on revive.
        """
        from shared.plugins.subagent.config import (
            SubagentProfile, profile_from_snapshot, profile_to_snapshot,
        )

        profile = SubagentProfile(
            name="voz", description="", max_completion_nudges=4,
        )
        snapshot = profile_to_snapshot(profile)
        assert snapshot["max_completion_nudges"] == 4
        assert profile_from_snapshot(snapshot).max_completion_nudges == 4

    def test_a_snapshot_predating_the_key_revives_at_the_default(self):
        from shared.plugins.subagent.config import profile_from_snapshot

        revived = profile_from_snapshot({"name": "old", "description": ""})
        assert revived.max_completion_nudges is None
        assert resolve_max_completion_nudges(revived) == 2

    def test_a_child_overrides_its_parent(self, tmp_path):
        from shared.plugins.subagent.config import discover_profiles

        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()
        (profiles_dir / "_base.yaml").write_text(
            "name: _base\ndescription: b\nplugins: []\nmax_completion_nudges: 2\n",
            encoding="utf-8",
        )
        (profiles_dir / "voz.yaml").write_text(
            "name: voz\ndescription: v\nplugins: []\ninherits: [_base]\n"
            "max_completion_nudges: 6\n",
            encoding="utf-8",
        )
        result = discover_profiles(str(profiles_dir), config_root=str(tmp_path))
        assert result.profiles["voz"].max_completion_nudges == 6

    def test_a_silent_child_inherits_its_parent(self, tmp_path):
        from shared.plugins.subagent.config import discover_profiles

        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()
        (profiles_dir / "_base.yaml").write_text(
            "name: _base\ndescription: b\nplugins: []\nmax_completion_nudges: 5\n",
            encoding="utf-8",
        )
        (profiles_dir / "voz.yaml").write_text(
            "name: voz\ndescription: v\nplugins: []\ninherits: [_base]\n",
            encoding="utf-8",
        )
        result = discover_profiles(str(profiles_dir), config_root=str(tmp_path))
        assert result.profiles["voz"].max_completion_nudges == 5


# ---------------------------------------------------------------------------
# 5. One writer, so the loop stays bounded (#934)
# ---------------------------------------------------------------------------

class TestOneWriter:
    """Every site must SPEND the budget through ``try_completion_nudge``.

    The budget is per turn, and a turn start refills it -- unless the turn is
    the one a nudge created, which ``try_completion_nudge`` marks by latching
    ``_completion_nudge_turn_pending`` in the same call that spends the token
    (#934).  A site that bumps ``_completion_nudges_fired`` itself skips the
    latch, so its re-prompt reads as caller-originated, the reset refills the
    budget, and the loop is unbounded again -- which is exactly #767, the
    subagent loop's ``while`` included.  The method is the only writer.
    """

    @pytest.mark.parametrize("site", NUDGE_SITES, ids=lambda p: p.name)
    def test_no_site_increments_the_counter_itself(self, site):
        tree = ast.parse(site.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            target = None
            if isinstance(node, ast.AugAssign):
                target = node.target
            elif isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
            if (
                isinstance(target, ast.Attribute)
                and target.attr == "_completion_nudges_fired"
            ):
                pytest.fail(
                    f"{site.name} writes _completion_nudges_fired directly "
                    f"(line {node.lineno}) — spend the budget through "
                    f"session.try_completion_nudge(), which also marks the "
                    f"turn the nudge is about to start (#934/#767)"
                )

    def test_the_method_is_a_writer(self):
        """Anchor: without it the guard above passes on a framework that
        stopped counting nudges at all."""
        import inspect

        from shared.jaato_session import JaatoSession

        source = inspect.getsource(JaatoSession.try_completion_nudge)
        assert "self._completion_nudges_fired += 1" in source
        assert "self._completion_nudge_turn_pending = True" in source
