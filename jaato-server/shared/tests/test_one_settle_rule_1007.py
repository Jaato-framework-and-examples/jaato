"""The facade settles a turn in ONE place (jaato #1007).

``complete`` settles on the daemon's confirmation (#767); ``ask`` and
``stream`` each carried a private, simpler rule -- first-of
``{TURN_COMPLETED, SESSION_TERMINATED}``.  Two rules, and the daemon emits
those two events from one thread 2-3 ms apart with the turn FIRST, so the
simpler rule settled on the turn and never saw the terminal.  A session its
budget ceiling had ended answered ``''`` like any other turn, and on a
cascade-stamped session -- where the terminal also unloads the session -- the
next call waited forever, because the daemon's immediate ``ErrorEvent`` reply
had no listener.  12+ minutes of a real run.

Three structural facts hold the fix up, and a later edit can undo any of them
without a behavioural test noticing until a session is over:

1. each verb settles through ``_TurnWatch`` -- a second private rule is how
   #1007 happened and the only thing that makes it unrepeatable is that there
   is nowhere to put one;
2. no verb subscribes to a settle-deciding event itself, which is what a
   private rule looks like in the source;
3. ``_TurnWatch.subscribe`` wires all four of those event types -- dropping
   ``ERROR`` restores the hang exactly, and it is the single line that does.

An AST guard, in the shape of ``test_budget_mid_turn_955`` and
``test_registry_iteration_snapshots``: the behaviour is covered by
``jaato-sdk/jaato_sdk/tests/test_a_turn_that_ended_the_session_says_so.py``;
this one refuses the SHAPE that produced the defect.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from shared.tests.test_every_guard_detects_its_own_reversion import Reversion

#: Repo root: this file is at <root>/jaato-server/shared/tests/<name>.py
ROOT = Path(__file__).resolve().parents[3]
FACADE = ROOT / "jaato-sdk" / "jaato_sdk" / "client" / "convenience.py"

#: The verbs a driver drives a turn with.  All three, because #1007's quiet
#: half ("returns the budget-cut turn as a normal reply") is not cascade-only
#: and the issue asks for one answer across them.
VERBS = ("ask", "complete", "stream")

#: Events whose arrival decides whether a call has reached its terminus.
#: Subscribing to one of these outside ``_TurnWatch`` IS a second settle rule,
#: whatever else the code around it says.
SETTLE_EVENTS = frozenset({
    "SESSION_TERMINATED",     # the unconditional terminus
    "TURN_COMPLETED",         # proposes one
    "AGENT_STATUS_CHANGED",   # confirms or withdraws the proposal
    "ERROR",                  # the session is not there at all
})


def _tree() -> ast.Module:
    return ast.parse(FACADE.read_text(), filename=str(FACADE))


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"{name} is gone from {FACADE.name}")


def _method(cls: ast.ClassDef, name: str):
    for node in cls.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and node.name == name:
            return node
    raise AssertionError(f"{cls.name}.{name} is gone from {FACADE.name}")


def _subscribed_events(node) -> set:
    """Event-type names this function subscribes to, by literal name."""
    found = set()
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        fn = sub.func
        if not (isinstance(fn, ast.Attribute)
                and fn.attr in ("subscribe", "subscribe_once")):
            continue
        if not sub.args:
            continue
        arg = sub.args[0]
        if isinstance(arg, ast.Attribute) and \
                isinstance(arg.value, ast.Name) and arg.value.id == "EventType":
            found.add(arg.attr)
    return found


def _calls_names(node) -> set:
    return {sub.func.id for sub in ast.walk(node)
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)}


@pytest.mark.parametrize("verb", VERBS)
def test_every_verb_settles_through_the_shared_watch(verb):
    """Fact 1: the rule is reached, not reimplemented."""
    fn = _method(_class(_tree(), "Session"), verb)
    assert "_TurnWatch" in _calls_names(fn), (
        f"Session.{verb} no longer builds a _TurnWatch, so it is settling by "
        f"some other rule. Two settle rules is what jaato #1007 was: the one "
        f"that did not read SESSION_TERMINATED returned a budget-ended turn "
        f"as an ordinary reply."
    )


@pytest.mark.parametrize("verb", VERBS)
def test_no_verb_subscribes_to_a_settle_event_itself(verb):
    """Fact 2: a private rule, in source form."""
    fn = _method(_class(_tree(), "Session"), verb)
    leaked = _subscribed_events(fn) & SETTLE_EVENTS
    assert not leaked, (
        f"Session.{verb} subscribes directly to {sorted(leaked)}. Deciding a "
        f"terminus outside _TurnWatch is a second settle rule, and the two "
        f"cannot be kept in agreement by review -- which is how jaato #1007 "
        f"shipped. Route it through the watch."
    )


def test_the_watch_wires_every_settle_event():
    """Fact 3: and ``ERROR`` is the one whose loss is a HANG, not a wrong
    answer.

    The daemon answers a request for an unloaded session immediately with
    ``ErrorEvent(error_type="SessionError")``.  Without this subscription the
    call waits on a terminal that is never coming -- #1007's headline symptom,
    and the only one that cost 12 minutes rather than an incorrect return
    value.
    """
    wired = _subscribed_events(_method(_class(_tree(), "_TurnWatch"),
                                       "subscribe"))
    missing = SETTLE_EVENTS - wired
    assert not missing, (
        f"_TurnWatch.subscribe no longer wires {sorted(missing)}. Every verb "
        f"settles through this method, so a type dropped here is dropped for "
        f"all three at once."
    )


REVERSIONS = [
    Reversion(
        target="jaato-sdk/jaato_sdk/client/convenience.py",
        find=(
            "        unsub_watch = watch.subscribe()\n"
            "        try:\n"
            "            await self._client.send_message(\n"
            "                prompt, parallel_tools=parallel_tools, "
            "attachments=attachments)\n"
            "            await self._wait_bounded(watch.wait(), timeout, "
            '"terminal event")\n'
            "        finally:\n"
            "            unsub_out()\n"
            "            unsub_media()\n"
            "            unsub_watch()\n"
            "        self._raise_if_needed(watch.box, end_is_error=True)\n"
            '        return "".join(chunks)'
        ),
        replace=(
            "        unsub_watch = self._client.subscribe_once(\n"
            "            EventType.SESSION_TERMINATED, watch.on_terminal)\n"
            "        try:\n"
            "            await self._client.send_message(\n"
            "                prompt, parallel_tools=parallel_tools, "
            "attachments=attachments)\n"
            "            await self._wait_bounded(watch.wait(), timeout, "
            '"terminal event")\n'
            "        finally:\n"
            "            unsub_out()\n"
            "            unsub_media()\n"
            "            unsub_watch()\n"
            "        self._raise_if_needed(watch.box, end_is_error=True)\n"
            '        return "".join(chunks)'
        ),
        test="test_no_verb_subscribes_to_a_settle_event_itself[ask]",
        because="a verb deciding its own terminus again -- the two-rule shape "
                "that let ask() settle on the turn event 2-3 ms before the "
                "terminal that said the session was over",
    ),
    Reversion(
        target="jaato-sdk/jaato_sdk/client/convenience.py",
        find=(
            "            self._client.subscribe(EventType.ERROR, self.on_error),\n"
            "        ]"
        ),
        replace="        ]",
        because="the facade going deaf to the daemon's 'Session not found' "
                "reply again, which is the unbounded hang itself",
        test="test_the_watch_wires_every_settle_event",
    ),
]
