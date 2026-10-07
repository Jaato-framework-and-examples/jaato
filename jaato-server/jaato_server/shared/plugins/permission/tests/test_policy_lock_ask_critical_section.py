"""Tests for the §3.7 + peer-review M3 ASK-side ``_policy_lock``
acquisition.

The race the fix protects against:

- Thread A enters ``check_permission`` for tool ``foo`` and the
  unlocked ``policy.check`` returns ASK_CHANNEL (rule miss).
- Thread B (a cross-session admin RPC) calls
  ``_policy.add_session_whitelist("foo")`` or ``add_blacklist`` —
  the policy's view of ``foo`` is now ALLOW or DENY.
- Thread A (without the fix) prompts the user about a tool that is
  no longer ambiguous.  The user's response is processed against
  the OLD policy snapshot, not the new one — and a sibling thread
  C arriving after the mutation would see ALLOW directly without a
  prompt.  The two threads disagree on whether to prompt for the
  same tool.

Fix: hold ``_policy_lock`` across the rule-recheck + channel-wait
critical section.  Inside the lock, re-run ``policy.check``; if it
no longer says ASK_CHANNEL, drop the lock and recurse so the
ALLOW/DENY branch handles the decision.

These tests pin the behaviour:

- Recursive resolution when a mutation lands BEFORE prompting.
- Mutations BLOCK on the policy lock until the prompt finishes.
- Concurrent mutations don't race the response-side side effects
  (``add_session_whitelist`` from "always" responses).
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

from ..channels import (
    Channel,
    ChannelDecision,
    ChannelResponse,
    PermissionRequest,
)
from ..plugin import PermissionPlugin


# ----------------------------------------------------------------------
# Test channels — programmable scripts that record + respond.
# ----------------------------------------------------------------------


class _ScriptedChannel(Channel):
    """Channel whose ``request_permission`` returns a sequence of
    canned responses.  Records each request received."""

    def __init__(self, responses: List[ChannelResponse]) -> None:
        self._responses = list(responses)
        self.requests: List[PermissionRequest] = []

    @property
    def name(self) -> str:
        return "scripted"

    def request_permission(
        self, request: PermissionRequest,
    ) -> ChannelResponse:
        self.requests.append(request)
        if not self._responses:
            raise AssertionError(
                "scripted channel exhausted; check is the ASK path "
                "being entered the right number of times?"
            )
        return self._responses.pop(0)

    def shutdown(self) -> None:
        pass


class _BlockingChannel(Channel):
    """Channel that blocks ``request_permission`` on an external
    event so the test can mutate the policy mid-prompt."""

    def __init__(self, response: ChannelResponse) -> None:
        self._response = response
        self.entered = threading.Event()
        self.release = threading.Event()
        self.requests: List[PermissionRequest] = []

    @property
    def name(self) -> str:
        return "blocking"

    def request_permission(
        self, request: PermissionRequest,
    ) -> ChannelResponse:
        self.requests.append(request)
        self.entered.set()
        # Block until the test signals release.
        self.release.wait(timeout=5.0)
        return self._response

    def shutdown(self) -> None:
        pass


# ----------------------------------------------------------------------
# Mid-ASK policy mutation: recheck flips to ALLOW → no prompt
# ----------------------------------------------------------------------


def test_mutation_to_allow_before_prompt_skips_channel() -> None:
    """If a session whitelist for the tool lands BEFORE the
    ASK-side recheck under the lock, the recheck observes ALLOW
    and the channel is never invoked."""
    channel = _ScriptedChannel(responses=[])  # asserts if invoked

    plugin = PermissionPlugin()
    plugin.initialize({
        "policy": {"defaultPolicy": "ask"},
    })
    plugin._channel = channel

    # Pre-mutate: add tool to session whitelist.  This simulates a
    # mutation that races the ASK in a way that actually completes
    # before the recheck runs.
    plugin._policy.add_session_whitelist("foo")

    allowed, info = plugin.check_permission("foo", {})
    assert allowed is True
    assert channel.requests == []  # never prompted


def test_mutation_to_deny_before_prompt_skips_channel() -> None:
    """A blacklist mutation that lands before the recheck causes
    DENY without prompting."""
    channel = _ScriptedChannel(responses=[])
    plugin = PermissionPlugin()
    plugin.initialize({
        "policy": {"defaultPolicy": "ask"},
    })
    plugin._channel = channel
    plugin._policy.add_session_blacklist("foo")

    allowed, info = plugin.check_permission("foo", {})
    assert allowed is False
    assert channel.requests == []


# ----------------------------------------------------------------------
# Default ASK path still prompts when no mutation lands
# ----------------------------------------------------------------------


def test_no_mutation_still_prompts() -> None:
    """Sanity check: the M3 fix does not break the normal ASK path —
    when no mutation races, the channel is still invoked once."""
    channel = _ScriptedChannel(responses=[
        ChannelResponse(
            request_id="r-1",
            decision=ChannelDecision.ALLOW,
            reason="ok",
        ),
    ])
    plugin = PermissionPlugin()
    plugin.initialize({
        "policy": {"defaultPolicy": "ask"},
    })
    plugin._channel = channel

    allowed, info = plugin.check_permission("foo", {})
    assert allowed is True
    assert len(channel.requests) == 1
    assert channel.requests[0].tool_name == "foo"


# ----------------------------------------------------------------------
# Concurrent mutation: blocks on policy lock until prompt resolves
# ----------------------------------------------------------------------


def test_concurrent_mutation_blocks_on_policy_lock() -> None:
    """A concurrent ``add_whitelist_tools`` call must block on the
    same ``_policy_lock`` the ASK path holds — so its mutation
    applies only AFTER the user has responded.  This pins the
    serialization invariant that motivates M3."""
    channel = _BlockingChannel(response=ChannelResponse(
        request_id="r-1",
        decision=ChannelDecision.DENY,
        reason="user denied",
    ))
    plugin = PermissionPlugin()
    plugin.initialize({
        "policy": {"defaultPolicy": "ask"},
    })
    plugin._channel = channel

    # Background: a long-poll thread starts the ASK and blocks
    # inside the channel.
    result_holder: Dict[str, Any] = {}

    def _ask() -> None:
        allowed, info = plugin.check_permission("bar", {})
        result_holder["allowed"] = allowed
        result_holder["info"] = info

    ask_thread = threading.Thread(target=_ask, daemon=True)
    ask_thread.start()

    # Wait for the channel to enter (proves we're inside
    # ``_policy_lock`` + ``request_permission``).
    assert channel.entered.wait(timeout=5.0)

    # Now try to add a whitelist for a different tool; this must
    # acquire the same ``_policy_lock`` and therefore block.
    mutation_done = threading.Event()

    def _mutate() -> None:
        plugin.add_whitelist_tools(["other"])
        mutation_done.set()

    mut_thread = threading.Thread(target=_mutate, daemon=True)
    mut_thread.start()

    # Give the mutation thread time to be scheduled and try to
    # acquire the lock; it should NOT complete because the ASK
    # holds the policy lock.
    time.sleep(0.1)
    assert not mutation_done.is_set(), (
        "add_whitelist_tools completed while ASK held _policy_lock; "
        "M3 fix is broken — mutations are NOT serialized against "
        "in-flight prompts."
    )

    # Release the channel — the prompt resolves, lock drops,
    # mutation can now complete.
    channel.release.set()
    ask_thread.join(timeout=5.0)
    assert mutation_done.wait(timeout=5.0)

    # Final state: ASK returned DENY, mutation applied after.
    assert result_holder["allowed"] is False
    assert "other" in plugin._policy.whitelist_tools


# ----------------------------------------------------------------------
# Recursion stability: recheck-flip path returns the new decision
# without re-prompting.
# ----------------------------------------------------------------------


def test_mutation_during_lock_acquisition_recurses_to_allow() -> None:
    """End-to-end M3 scenario:

    1. Thread enters ASK_CHANNEL (unlocked check returns ASK).
    2. Before the recheck under the lock runs, we mutate the
       policy to whitelist the tool.
    3. The recheck under the lock sees ALLOW, recurses, the
       recursive call's unlocked check sees ALLOW, returns true
       without ever invoking the channel.
    """
    # The channel asserts if invoked — proves the recursion lands
    # in ALLOW before any prompt.
    channel = _ScriptedChannel(responses=[])

    plugin = PermissionPlugin()
    plugin.initialize({
        "policy": {"defaultPolicy": "ask"},
    })
    plugin._channel = channel

    # We need to inject a mutation that lands AFTER the unlocked
    # check but BEFORE the locked recheck.  Patch
    # ``_policy.check`` so the second call sees the mutation.
    real_check = plugin._policy.check
    call_count = {"n": 0}

    def _wrapped_check(*args: Any, **kwargs: Any) -> Any:
        call_count["n"] += 1
        if call_count["n"] == 2:
            # Simulate the cross-thread mutation landing between
            # the two checks.
            plugin._policy.add_session_whitelist("baz")
        return real_check(*args, **kwargs)

    plugin._policy.check = _wrapped_check  # type: ignore[method-assign]

    allowed, info = plugin.check_permission("baz", {})

    assert allowed is True
    assert channel.requests == []  # recursion landed in ALLOW
    # Both unlocked + locked check happened, plus one for the
    # recursion's unlocked check.
    assert call_count["n"] >= 3
