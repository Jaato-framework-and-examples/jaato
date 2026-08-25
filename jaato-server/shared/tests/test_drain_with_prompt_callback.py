"""The drain must survive a session that HAS a prompt-injected callback.

An orphaned fragment — the tail of the old child-drain block, left at the
wrong indent by a partial replacement in #612 — ran after each tier's loop
had exited:

    while True:
        msg = ...pop_first_matching(...)
        if msg is None:
            break
        ...
        if self._on_prompt_injected:
            self._on_prompt_injected(msg.text)

    if self._on_prompt_injected:          # <-- orphan, OUTSIDE the while
        self._on_prompt_injected(msg.text)     msg is None here, always

So every drain crashed with ``AttributeError: 'NoneType' object has no
attribute 'text'`` — killing the session, on both siblings of a cascade at
once — whenever a prompt-injected callback was installed.  The runner installs
one for the duration of every ``session.send_message`` RPC, so this was every
real turn.

**Every test written for that function passed**, because they all set
``_on_prompt_injected = None`` — and the crashing line sits BEHIND that guard.
The fixtures disabled the exact condition that triggers the bug.

That is the fourth instance in this arc of a fixture that cannot exhibit the
failure it covers, and the most direct: not a payload that couldn't
distinguish, but the guard condition itself set to the value that skips the
defect.  Found only by a real cascade run producing frames.
"""

import pytest

from shared.jaato_session import JaatoSession
from shared.message_queue import (
    HIGH_PRIORITY_SOURCES, IDLE_ONLY_SOURCES, MessageQueue, SourceType,
)


def _session(queue, callback):
    s = JaatoSession.__new__(JaatoSession)
    s._message_queue = queue
    s._agent_id = "bob"
    s._trace = lambda *a, **k: None
    s._activity_phase = None
    s._is_running = False
    s._on_continuation_needed = None
    s._on_prompt_injected = callback
    return s


def test_a_drain_with_a_callback_installed_does_not_crash():
    """The production configuration: the runner always installs one."""
    q = MessageQueue()
    q.put("hello", "alice", SourceType.SIBLING)
    _session(q, lambda _t: None)._drain_child_messages(None)


def test_an_empty_queue_with_a_callback_does_not_crash():
    """The orphan fired even with NOTHING to drain — the loop breaks on the
    first pop, and the fragment then reads ``.text`` off that None."""
    _session(MessageQueue(), lambda _t: None)._drain_child_messages(None)


def test_the_callback_fires_once_per_message_and_no_more():
    """The orphan also produced a SPURIOUS extra call per tier.

    Asserted on the exact sequence, not a count: an off-by-one that repeats
    the last message is a different defect from one that adds a blank, and a
    count cannot tell them apart.
    """
    q = MessageQueue()
    q.put("from the parent", "main", SourceType.PARENT)
    q.put("from a peer", "alice", SourceType.SIBLING)
    seen = []
    _session(q, seen.append)._drain_child_messages(None)
    assert seen == ["from the parent", "from a peer"]


@pytest.mark.parametrize(
    "source_type",
    sorted(HIGH_PRIORITY_SOURCES | IDLE_ONLY_SOURCES, key=lambda s: s.value),
    ids=lambda s: s.value,
)
def test_every_tier_drains_with_a_callback_installed(source_type):
    """Over the tier sets, so a new SourceType inherits the coverage."""
    q = MessageQueue()
    q.put(f"msg-{source_type.value}", "sender", source_type)
    seen = []
    collected = _session(q, seen.append)._drain_child_messages(None)
    assert seen == [f"msg-{source_type.value}"]
    assert f"msg-{source_type.value}" in collected


def test_no_callback_still_works():
    """The configuration the old tests used — must not regress either."""
    q = MessageQueue()
    q.put("hello", "alice", SourceType.SIBLING)
    assert "hello" in _session(q, None)._drain_child_messages(None)
