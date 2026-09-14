"""An event-type filter that can never match is ANNOUNCED, not obeyed quietly.

THE FAILURE THIS EXISTS TO END.

``cascade_events(event_types=[...])`` and the daemon's cascade-client registry
both filter on ``type(event).__name__``.  The two vocabularies look
interchangeable — ``"SessionTerminatedEvent"`` versus ``"session.terminated"``
— and a filter written in the wrong one matches NOTHING while every other
signal says the subscription is healthy: registration succeeds, the daemon
logs a live entry naming the filter, and the iterator simply never yields.
That is indistinguishable from a cascade that produced no events, so a reader
debugging it suspects their cascade id, their timing, or the daemon.

``jaato-scaffold new observer`` shipped wire values, which meant every
generated observer was deaf for the entire life of every run it watched
(jaato #821).  The template is fixed; these tests cover the surface that
makes the SAME mistake visible when someone writes the filter by hand.
"""

from __future__ import annotations

import logging

import pytest

from jaato_sdk.events import (ClientType, EventType, check_event_type_names,
                              describe_event_type_problems,
                              known_event_class_names)


def test_class_names_are_the_vocabulary():
    known = known_event_class_names()
    assert "SessionTerminatedEvent" in known
    assert "AgentCompletedEvent" in known
    # ...and the wire values are NOT, which is the whole point.
    assert EventType.SESSION_TERMINATED.value not in known


def test_a_valid_filter_reports_nothing():
    assert check_event_type_names(
        ["ToolCallStartEvent", "TurnCompletedEvent",
         "AgentCompletedEvent", "SessionTerminatedEvent"]) == {}
    assert describe_event_type_problems(["SessionTerminatedEvent"]) is None


def test_none_and_empty_are_not_problems():
    """``None`` means "everything" and an empty list means the same; neither
    is a filter that fails to match."""
    assert check_event_type_names(None) == {}
    assert check_event_type_names([]) == {}


def test_a_wire_value_is_reported_WITH_the_class_to_use():
    """Naming the mistake is half the fix; naming the correction is the rest.

    The scaffolded observer's exact filter, which received nothing.
    """
    bad = check_event_type_names(
        ["tool.call_start", "turn.completed",
         "agent.completed", "session.terminated"])
    assert bad == {
        "tool.call_start": "ToolCallStartEvent",
        "turn.completed": "TurnCompletedEvent",
        "agent.completed": "AgentCompletedEvent",
        "session.terminated": "SessionTerminatedEvent",
    }
    msg = describe_event_type_problems(["session.terminated"])
    assert "SessionTerminatedEvent" in msg
    assert "wire value" in msg


def test_a_string_matching_nothing_is_reported_with_no_suggestion():
    """A typo is not a wire value, and pretending to correct it would send
    the reader after a class that does not exist either."""
    assert check_event_type_names(["NoSuchEvent"]) == {"NoSuchEvent": None}
    assert "matches no event class" in describe_event_type_problems(["NoSuchEvent"])


async def test_cascade_events_warns_on_a_filter_that_cannot_match(caplog):
    """The warning lands where the reader is — at the call site, on stderr via
    logging's last-resort handler even in a script that configures nothing."""
    from jaato_sdk.client.ipc import IPCClient

    client = IPCClient("/tmp/does-not-exist.sock", auto_start=False,
                       client_type=ClientType.API)

    async def _noop(_event):
        return False
    client._send_event = _noop            # nothing is really sent
    client._subscribe_events = lambda: _ClosedQueue()

    with caplog.at_level(logging.WARNING, logger="jaato_sdk.client.ipc"):
        async for _ in client.cascade_events(
                "cid-1", event_types=["session.terminated"]):
            pass

    assert any("SessionTerminatedEvent" in r.message for r in caplog.records), (
        "a filter that can never match was accepted silently; a deaf observer "
        "is indistinguishable from a cascade that produced no events"
    )


async def test_cascade_events_stays_quiet_on_a_correct_filter(caplog):
    """A warning nobody can act on trains readers to ignore warnings."""
    from jaato_sdk.client.ipc import IPCClient

    client = IPCClient("/tmp/does-not-exist.sock", auto_start=False,
                       client_type=ClientType.API)

    async def _noop(_event):
        return False
    client._send_event = _noop
    client._subscribe_events = lambda: _ClosedQueue()

    with caplog.at_level(logging.WARNING, logger="jaato_sdk.client.ipc"):
        async for _ in client.cascade_events(
                "cid-1", event_types=["SessionTerminatedEvent"]):
            pass

    assert not [r for r in caplog.records if "can never match" in r.message]


class _ClosedQueue:
    """A drain-loop queue that reports an ended connection immediately, so the
    iterator registers, warns (or does not), and exits without a daemon."""

    async def get(self):
        return None
