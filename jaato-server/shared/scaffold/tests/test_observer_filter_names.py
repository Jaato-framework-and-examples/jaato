"""The scaffolded observer must filter on event CLASS names.

A scaffolded observer connected, registered, printed its banner and then
received NOTHING for the entire life of the cascade — no error, no warning, no
timeout.  The template emitted ``EventType`` wire values
(``"session.terminated"``) while both filters between the script and the
daemon compare ``type(event).__name__``, which is ``"SessionTerminatedEvent"``.
No generated filter could ever match (jaato #821).

What makes it worth a guard rather than a fix-and-move-on is the failure mode:
it is indistinguishable from "the cascade produced no events".  Registration
succeeds, the daemon logs a healthy entry naming the filter, and a reader
debugging it reasonably suspects their cascade id, their timing, or the
daemon — the generated line is the LAST place they look, because scaffolded
output is documented as valid by construction.
"""

from __future__ import annotations

import argparse
import ast

import pytest

from jaato_sdk.events import check_event_type_names, known_event_class_names

from shared.scaffold import build, introspect
from shared.scaffold._client_templates import OBSERVER_TEMPLATE


@pytest.fixture(scope="module")
def observer_src(tmp_path_factory) -> str:
    ws = tmp_path_factory.mktemp("observer_ws")
    names = sorted(introspect.providers())
    assert names, "no providers installed — cannot scaffold a client"
    rc = build.run(argparse.Namespace(
        archetype="observer", workspace=str(ws), provider=names[0],
        model="test-model", set=None, agents=None, force=True,
        recoverable=False, json=False,
    ))
    assert rc == 0
    return (ws / "run_observer.py").read_text(encoding="utf-8")


def _event_types_literal(src: str) -> list:
    """The EVENT_TYPES list the generated observer actually passes."""
    tree = ast.parse(src)
    node = next(
        n.value for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and any(getattr(t, "id", None) == "EVENT_TYPES" for t in n.targets)
    )
    return [e.value for e in node.elts]


def test_the_filter_names_are_real_event_classes(observer_src):
    """Checked against the SDK's own registry, not a copy of four names.

    A second list of event names here would rot exactly the way the template
    did, and would go on passing while it did.
    """
    names = _event_types_literal(observer_src)
    assert names, "the observer passes no event-type filter at all"
    bad = check_event_type_names(names)
    assert not bad, (
        f"the generated observer filters on {sorted(bad)}, which match no "
        "event class — it would receive nothing, silently, for the life of "
        "every cascade it watched"
    )


def test_the_filter_is_reached_by_cascade_events(observer_src):
    """The names must be PASSED, not merely defined — a correct constant the
    call does not use is the same silence with extra steps."""
    assert "event_types=EVENT_TYPES" in observer_src


def test_no_wire_value_is_emitted(observer_src):
    """The specific mistake, stated as itself.

    ``check_event_type_names`` above would already catch it; this says which
    error it was, so a failure reads as "you wrote wire values again" rather
    than "one of these strings is unknown".
    """
    for name in _event_types_literal(observer_src):
        assert "." not in name, (
            f"{name!r} is an EventType wire value; filters compare against "
            "the class name (type(event).__name__)"
        )
        assert name in known_event_class_names()


def test_the_template_itself_carries_the_names():
    """Not a substitution the builder could stop making.

    The names are correct in the template, so every transport / flag
    combination emits the same working filter — there is no path through the
    generator that reintroduces wire values.
    """
    code = "\n".join(line.split("#", 1)[0]
                     for line in OBSERVER_TEMPLATE.splitlines())
    assert "SessionTerminatedEvent" in code
    # Comments stripped: the template WARNS about the wire form in prose, and
    # that warning is the reason the code below is right.
    assert "session.terminated" not in code
