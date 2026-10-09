"""`permissions default ask` stops approving tools without asking (#1403).

An ``all`` answer on a permission card sets the enforcer's ``_allow_all``
(``suspension_scope == "session"``).  ``check_permission`` asks the three
suspensions BEFORE the policy, so a later ``permissions default ask``
changed the default and decided nothing: the transcript said
``Session default policy: ask``, the status bar (correctly) still said
``allow (session)``, and the next ``writeNewFile`` ran with
``method=allow_all`` and no prompt.  Reproduced on a live daemon (echo
provider, runner-served path).

The fix: ``ask`` and ``deny`` tighten, so they clear every suspension and
say which one they cleared; per-tool session whitelist entries are kept and
named.  ``allow`` changes only the default.

The harness is #1412's: a real :class:`RunnerRPCClient` over a socketpair
to a real :class:`RunnerRPC`, the enforcer built by the bootstrap's own
:func:`build_session_permission_plugin`, and the command sent through the
runner's command path.  Only the operator's answer to the card is a
stand-in channel, because there is no operator.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Tuple

import pytest

from jaato_sdk.events import PermissionStatusEvent
from jaato_server.server.tests.test_permission_decisions_survive_a_revive_1412 import (
    _Runner,
)
from jaato_server.shared.plugins.permission.channels import (
    ChannelDecision,
    ChannelResponse,
)
from jaato_server.shared.tests.reversion import Reversion

_PLUGIN = "jaato-server/jaato_server/shared/plugins/permission/plugin.py"

_WRITE = ("writeNewFile", {"path": "README.md", "content": "x\n"})


class _Operator:
    """The person answering the permission card: records each prompt."""

    name = "operator"

    def __init__(self, answer: ChannelDecision) -> None:
        self.answer = answer
        self.asked: List[str] = []

    def request_permission(self, request: Any) -> ChannelResponse:
        self.asked.append(request.tool_name)
        return ChannelResponse(request_id=request.request_id,
                               decision=self.answer)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    ws = tmp_path / "ws"
    ws.mkdir()
    return ws


def _check(runner: _Runner) -> Tuple[bool, Dict[str, Any]]:
    return runner.plugin.check_permission(*_WRITE, call_id="c1")


def _statuses(server: Any) -> List[Tuple[str, Any]]:
    return [(e.effective_default, e.suspension_scope) for e in server.emitted
            if isinstance(e, PermissionStatusEvent)]


async def _answer_all(runner: _Runner) -> _Operator:
    """Step 1 of the report: answer the card with ALL."""
    operator = _Operator(ChannelDecision.ALLOW_ALL)
    runner.plugin._channel = operator
    allowed, info = await asyncio.to_thread(_check, runner)
    assert allowed and operator.asked == ["writeNewFile"]
    assert runner.plugin.suspension_scope == "session"
    return operator


async def test_default_ask_after_all_prompts_the_next_write(workspace) -> None:
    """The reported sequence: ALL, `permissions default ask`, a write."""
    r = _Runner(workspace, "s1")
    await r.start()
    try:
        operator = await _answer_all(r)
        server = r.server("s1")

        reply = await asyncio.to_thread(r.command, "default", "ask")
        assert "Cleared suspension (session)" in reply
        # What the status bar is told after the command.
        await asyncio.to_thread(server.emit_permission_status)
        assert _statuses(server) == [("ask", None)]

        # The next write reaches the operator instead of allow_all.
        operator.answer = ChannelDecision.ALLOW
        allowed, info = await asyncio.to_thread(_check, r)
        assert operator.asked == ["writeNewFile", "writeNewFile"]
        assert info.get("method") != "allow_all"
    finally:
        await r.stop()


async def test_default_deny_clears_an_idle_suspension(workspace) -> None:
    """`permissions suspend` then `default deny`: the write is refused."""
    r = _Runner(workspace, "s1")
    await r.start()
    try:
        r.plugin._channel = _Operator(ChannelDecision.ALLOW)
        await asyncio.to_thread(r.command, "suspend")
        assert r.plugin.suspension_scope == "idle"

        reply = await asyncio.to_thread(r.command, "default", "deny")
        assert "Cleared suspension (idle)" in reply
        allowed, info = await asyncio.to_thread(_check, r)
        assert not allowed, info
    finally:
        await r.stop()


async def test_a_kept_whitelist_entry_is_named(workspace) -> None:
    """An `always` answer is a per-tool decision: kept, and said so."""
    r = _Runner(workspace, "s1")
    await r.start()
    try:
        operator = _Operator(ChannelDecision.ALLOW_SESSION)
        r.plugin._channel = operator
        assert (await asyncio.to_thread(_check, r))[0]

        reply = await asyncio.to_thread(r.command, "default", "ask")
        assert "Still approved without asking" in reply
        assert "writeNewFile" in reply
        assert "permissions clear" in reply
    finally:
        await r.stop()


async def test_default_allow_leaves_a_suspension_alone(workspace) -> None:
    """Widening the default cannot be contradicted by a suspension."""
    r = _Runner(workspace, "s1")
    await r.start()
    try:
        await _answer_all(r)
        reply = await asyncio.to_thread(r.command, "default", "allow")
        assert "Cleared" not in reply
        assert r.plugin.suspension_scope == "session"
    finally:
        await r.stop()


REVERSIONS = [
    Reversion(
        target=_PLUGIN,
        find="""        if cleared:
            self.clear_all_suspensions()
            self._allow_all = False
""",
        replace="""        if cleared:
""",
        test="test_default_ask_after_all_prompts_the_next_write",
        because=(
            "the default changing while an 'all' answer keeps approving "
            "every tool with method=allow_all and no prompt"
        ),
    ),
    Reversion(
        target=_PLUGIN,
        find="""            self.clear_all_suspensions()
            self._allow_all = False
""",
        replace="""            self._allow_all = False
""",
        test="test_default_deny_clears_an_idle_suspension",
        because=(
            "only the 'all' flag cleared, so a `permissions suspend` still "
            "approves writes under a deny default"
        ),
    ),
    Reversion(
        target=_PLUGIN,
        find="""        kept = sorted(self._policy.session_whitelist)
        if kept:""",
        replace="""        kept = sorted(self._policy.session_whitelist)
        if False:""",
        test="test_a_kept_whitelist_entry_is_named",
        because=(
            "a tool still approved without asking after `default ask`, "
            "with nothing saying why"
        ),
    ),
]
