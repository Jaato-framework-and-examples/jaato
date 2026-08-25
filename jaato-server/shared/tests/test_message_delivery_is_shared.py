"""Every sender makes the queue-or-drive decision the SAME way.

Three senders reach a session that may be working or resting — a human, a
parent (``send_to_subagent``), a cascade peer (``send_to_sibling``) — and the
decision is identical: busy → queue on the sender's tier; idle → DRIVE, because
an injection into an idle session starts nothing.

It was written three times.  Two copies were right; the third queued into an
idle peer and reported ``accepted``.  Cloning a path this similar means each
copy rediscovers that ``inject_prompt`` only self-starts while
``_on_continuation_needed`` is installed — which is for the duration of a
``session.send_message`` RPC, not whenever the session happens to be idle.

These tests pin the CONSOLIDATION, not just the fixed behaviour: a fourth
sender that re-derives the branch would pass a behaviour test and fail here.
"""

import ast
import pathlib

import pytest

from shared.message_delivery import ACCEPTED, QUEUED, deliver


# ----------------------------------------------------------------------
# The decision itself
# ----------------------------------------------------------------------

def test_a_busy_target_is_queued():
    calls = []
    out = deliver(is_busy=lambda: True,
                  queue=lambda: calls.append("queue"),
                  drive=lambda: calls.append("drive"))
    assert out == QUEUED and calls == ["queue"]


def test_an_idle_target_is_driven():
    """The half every clone got wrong at least once."""
    calls = []
    out = deliver(is_busy=lambda: False,
                  queue=lambda: calls.append("queue"),
                  drive=lambda: calls.append("drive"))
    assert out == ACCEPTED and calls == ["drive"]


def test_exactly_one_mechanism_runs():
    """Delivering twice is as wrong as not delivering."""
    for busy in (True, False):
        calls = []
        deliver(is_busy=lambda: busy,
                queue=lambda: calls.append("q"), drive=lambda: calls.append("d"))
        assert len(calls) == 1


def test_a_failing_delivery_is_not_reported_as_success():
    """The mechanism's exception must reach the caller, not a status string."""
    def boom():
        raise RuntimeError("channel closed")
    with pytest.raises(RuntimeError):
        deliver(is_busy=lambda: False, queue=lambda: None, drive=boom)


# ----------------------------------------------------------------------
# The consolidation
# ----------------------------------------------------------------------

_SENDERS = [
    ("jaato-server/shared/plugins/subagent/plugin.py",
     "_execute_send_to_subagent"),
    ("jaato-server/server/session_manager.py",
     "deliver_sibling_message"),
]


def _fn_source(path, name):
    tree = ast.parse(pathlib.Path(path).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {path}")


@pytest.mark.parametrize("path,fn", _SENDERS, ids=[f for _, f in _SENDERS])
def test_each_sender_uses_the_shared_decision(path, fn):
    node = _fn_source(path, fn)
    calls = {c.func.id for c in ast.walk(node)
             if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}
    assert "deliver" in calls, (
        f"{fn} does not call shared.message_delivery.deliver — a re-derived "
        f"queue/drive branch is how this bug class returns"
    )


@pytest.mark.parametrize("path,fn", _SENDERS, ids=[f for _, f in _SENDERS])
def test_no_sender_re_derives_the_branch(path, fn):
    """No sender may test busy-ness and pick a mechanism itself.

    The tell is an ``if`` on a running/busy flag inside the sender.  That is
    the shape all three copies had, and the shape the shared decision exists
    to remove.
    """
    node = _fn_source(path, fn)
    for branch in [n for n in ast.walk(node) if isinstance(n, ast.If)]:
        rendered = ast.dump(branch.test)
        assert "is_running" not in rendered and "_model_running" not in rendered, (
            f"{fn} branches on busy-ness itself; pass it to deliver(is_busy=...)"
        )


def test_the_drain_reads_the_declared_tiers_not_named_accessors():
    """Adding a SourceType must not require editing the drain.

    ``message_queue`` declares membership once (HIGH_PRIORITY_SOURCES /
    IDLE_ONLY_SOURCES).  The drain enumerating tiers by named accessor is why
    SIBLING shipped with no drainer: the tier existed, nothing popped it, and
    a queued message was silently discarded on unload.
    """
    src = pathlib.Path("jaato-server/shared/jaato_session.py").read_text(encoding="utf-8")
    node = _fn_source("jaato-server/shared/jaato_session.py",
                      "_drain_child_messages")
    body = ast.get_source_segment(src, node) or ""
    assert "HIGH_PRIORITY_SOURCES" in body and "IDLE_ONLY_SOURCES" in body
    for named in ("pop_first_parent_message", "pop_first_child_message",
                  "pop_first_sibling_message"):
        assert named not in body, (
            f"_drain_child_messages still enumerates {named} — a new tier "
            f"would be silently undrained"
        )


def test_every_idle_only_tier_is_reachable_by_the_drain():
    """The property, over the tier set — not over the tiers I know today."""
    from shared.message_queue import (
        HIGH_PRIORITY_SOURCES, IDLE_ONLY_SOURCES, MessageQueue, SourceType,
    )
    from shared.jaato_session import JaatoSession

    q = MessageQueue()
    for src_type in sorted(HIGH_PRIORITY_SOURCES | IDLE_ONLY_SOURCES,
                           key=lambda s: s.value):
        q.put(f"msg-{src_type.value}", "sender", src_type)

    s = JaatoSession.__new__(JaatoSession)
    s._message_queue = q
    s._agent_id = "a"
    s._trace = lambda *a, **k: None
    s._on_prompt_injected = None
    s._activity_phase = None
    s._is_running = False
    s._on_continuation_needed = None

    collected = s._drain_child_messages(None)
    for src_type in HIGH_PRIORITY_SOURCES | IDLE_ONLY_SOURCES:
        assert f"msg-{src_type.value}" in collected, (
            f"{src_type.value} was queued and never drained")
    assert len(q) == 0
