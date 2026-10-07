"""Turn/idle suspension must cover a tool call that was ALREADY
waiting on the channel lock when the suspending answer landed.

Reported (the final rollout item from GitHub issue #1304, phase 3):
answering "t" (turn-scoped allow) for ``createPlan`` still left
``subscribeToEvents`` — called in the SAME parallel batch — prompting
separately, although CLAUDE.md's documented contract for turn
suspension is "every tool call for the rest of the turn".

``_turn_suspended`` / ``_idle_suspended`` ARE checked unconditionally
at the top of ``_check_permission_impl`` for every tool, so the
mechanism is turn-wide, not per-tool, by construction. The defect is
a race: with ``JAATO_PARALLEL_TOOLS`` on, two ASK-requiring tool calls
from one model turn are dispatched on separate threads and can both
read the (still-False) suspension flags before either is answered.
Both then queue on ``_policy_lock`` (held for the whole
recheck+channel-wait critical section — see
``test_policy_lock_ask_critical_section.py``). The first to acquire it
prompts, the user answers "t", ``_handle_channel_response`` sets
``_turn_suspended = True`` *while the lock is still held*, and the
call returns. The SECOND call then acquires the lock and re-runs
``policy.check()`` — which knows nothing about turn/idle suspension,
since that state lives outside ``PermissionPolicy`` — so
``policy_mutated`` is False and it fell through to
``channel.request_permission()`` a second time. Only ``_allow_all``
was re-checked in that fall-through section; ``_turn_suspended`` /
``_idle_suspended`` were not.

The fix re-checks idle/turn suspension (in the same priority order as
the top-of-function check) immediately after the channel lock is
acquired, so a call that queued behind an about-to-be-turn-suspended
sibling picks up the suspension instead of prompting a second time.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List

from ..channels import Channel, ChannelDecision, ChannelResponse, PermissionRequest
from ..plugin import PermissionPlugin


class _SingleShotBlockingChannel(Channel):
    """A channel that blocks its FIRST call on an external event and
    answers with a caller-supplied decision.  Any call beyond the
    first raises immediately — proving a second prompt never reached
    the channel, rather than merely asserting the return value."""

    def __init__(self, response: ChannelResponse) -> None:
        self._response = response
        self.entered = threading.Event()
        self.release = threading.Event()
        self.requests: List[PermissionRequest] = []
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return "single-shot-blocking"

    def request_permission(self, request: PermissionRequest) -> ChannelResponse:
        with self._lock:
            if self.requests:
                raise AssertionError(
                    f"channel prompted a second time for "
                    f"{request.tool_name!r} — turn/idle suspension "
                    f"granted by the first answer did not cover a "
                    f"concurrent tool call of the same turn (#1304)"
                )
            self.requests.append(request)
        self.entered.set()
        self.release.wait(timeout=5.0)
        return self._response

    def shutdown(self) -> None:
        pass


def _plugin_with_ask_policy() -> PermissionPlugin:
    plugin = PermissionPlugin()
    plugin.initialize({"policy": {"defaultPolicy": "ask"}})
    return plugin


def test_turn_suspend_answer_covers_a_concurrently_queued_ask() -> None:
    """createPlan (answered "t") and subscribeToEvents, dispatched in
    one parallel batch, must resolve to ONE prompt — the second call
    is auto-allowed under turn suspension, never shown its own card.
    """
    channel = _SingleShotBlockingChannel(
        response=ChannelResponse(
            request_id="r-1",
            decision=ChannelDecision.ALLOW_TURN,
            reason="user allowed for the turn",
        )
    )
    plugin = _plugin_with_ask_policy()
    plugin._channel = channel

    results: Dict[str, Any] = {}

    def _ask(name: str, tool: str) -> None:
        allowed, info = plugin.check_permission(tool, {})
        results[name] = (allowed, info)

    # createPlan reaches the channel first and blocks there — this is
    # the thread that will receive the "t" answer.
    first = threading.Thread(
        target=_ask, args=("first", "createPlan"), daemon=True
    )
    first.start()
    assert channel.entered.wait(timeout=5.0), "first call never reached the channel"

    # subscribeToEvents is issued concurrently (the same parallel
    # batch) and queues on the policy lock behind createPlan's
    # in-flight ASK.
    second = threading.Thread(
        target=_ask, args=("second", "subscribeToEvents"), daemon=True
    )
    second.start()
    # Give it a moment to actually attempt the lock acquisition —
    # this does not make the test flaky in the failing direction: if
    # the thread hasn't reached the lock yet, it will simply queue
    # a little later and the assertion below still holds, because
    # nothing releases the channel until after this sleep.
    time.sleep(0.05)

    # Answer "t" for createPlan.  Releasing the channel lets
    # ``_handle_channel_response`` set ``_turn_suspended`` while
    # still holding ``_policy_lock`` / ``_channel_lock``.
    channel.release.set()

    first.join(timeout=5.0)
    second.join(timeout=5.0)

    assert not first.is_alive() and not second.is_alive()

    allowed_first, info_first = results["first"]
    allowed_second, info_second = results["second"]

    assert allowed_first is True
    assert info_first["method"] == "turn_suspension"

    assert allowed_second is True
    assert info_second["method"] == "turn_suspension"

    # The load-bearing assertion: the channel was asked exactly once.
    assert len(channel.requests) == 1
    assert channel.requests[0].tool_name == "createPlan"

    # And the plugin-wide flag really is set for anything asked afterward.
    assert plugin._turn_suspended is True


def test_idle_suspend_answer_covers_a_concurrently_queued_ask() -> None:
    """Same race, for the "i" (idle) answer."""
    channel = _SingleShotBlockingChannel(
        response=ChannelResponse(
            request_id="r-1",
            decision=ChannelDecision.ALLOW_UNTIL_IDLE,
            reason="user allowed until idle",
        )
    )
    plugin = _plugin_with_ask_policy()
    plugin._channel = channel

    results: Dict[str, Any] = {}

    def _ask(name: str, tool: str) -> None:
        allowed, info = plugin.check_permission(tool, {})
        results[name] = (allowed, info)

    first = threading.Thread(
        target=_ask, args=("first", "createPlan"), daemon=True
    )
    first.start()
    assert channel.entered.wait(timeout=5.0)

    second = threading.Thread(
        target=_ask, args=("second", "subscribeToEvents"), daemon=True
    )
    second.start()
    time.sleep(0.05)

    channel.release.set()

    first.join(timeout=5.0)
    second.join(timeout=5.0)

    allowed_second, info_second = results["second"]
    assert allowed_second is True
    assert info_second["method"] == "idle_suspension"
    assert len(channel.requests) == 1


def test_without_a_suspending_answer_a_third_ask_still_prompts() -> None:
    """Control: a plain ALLOW ("y") does not suspend anything, so a
    concurrently queued call for a DIFFERENT tool still gets its own
    prompt.  Without this, the fix above could be satisfied by a
    version that skips every re-ask rather than only suspended ones.
    """
    channel = _SingleShotBlockingChannel(
        response=ChannelResponse(
            request_id="r-1",
            decision=ChannelDecision.ALLOW_ONCE,
            reason="user allowed once",
        )
    )
    plugin = _plugin_with_ask_policy()
    plugin._channel = channel

    def _ask() -> None:
        plugin.check_permission("createPlan", {})

    first = threading.Thread(target=_ask, daemon=True)
    first.start()
    assert channel.entered.wait(timeout=5.0)
    channel.release.set()
    first.join(timeout=5.0)

    assert plugin._turn_suspended is False
    assert plugin._idle_suspended is False
    assert plugin._allow_all is False

    # A second, later call for a different tool is NOT suspended —
    # it must reach the channel again.  Reuse the same single-shot
    # channel, which raises if asked twice; here it should raise
    # exactly because nothing suspended it, so drive it through a
    # fresh non-blocking scripted channel instead to assert the
    # positive case cleanly.
    class _Scripted(Channel):
        def __init__(self) -> None:
            self.requests: List[PermissionRequest] = []

        @property
        def name(self) -> str:
            return "scripted"

        def request_permission(self, request: PermissionRequest) -> ChannelResponse:
            self.requests.append(request)
            return ChannelResponse(
                request_id=request.request_id,
                decision=ChannelDecision.ALLOW_ONCE,
                reason="ok",
            )

        def shutdown(self) -> None:
            pass

    plugin._channel = _Scripted()
    allowed, info = plugin.check_permission("subscribeToEvents", {})
    assert allowed is True
    assert info["method"] == "user_approved"
    assert len(plugin._channel.requests) == 1
