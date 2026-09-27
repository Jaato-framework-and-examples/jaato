"""A prompt still waiting for a person reaches a client that attaches later.

Reported from the web client: detach while the model waits on a permission
ASK or a clarification, reattach, and the transcript shows the call while
nothing can answer it.  On the runner path (the default) the two relays
emitted the prompt events ONCE, to whoever was attached then, and kept only
the futures, so ``emit_current_state`` had nothing to replay.  The relays
now keep the events for as long as the prompt is pending, and the attach
replays them last.
"""

from __future__ import annotations

import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List

from jaato_sdk.events import (
    ClarificationBatchEvent, PermissionInputModeEvent, PermissionRequestedEvent,
    PermissionResolvedEvent, ClarificationResolvedEvent,
)
from jaato_server.server.core import JaatoServer
from jaato_server.server.runner_rpc_handlers.clarification_relay import (
    ClarificationRelayHandler,
)
from jaato_server.server.runner_rpc_handlers.prompt_operator import (
    PromptOperatorHandler,
)
from jaato_server.shared.tests.reversion import Reversion

_HANDLERS = "jaato-server/jaato_server/server/runner_rpc_handlers"
_CORE = "jaato-server/jaato_server/server/core.py"

REVERSIONS = [
    Reversion(
        target=f"{_HANDLERS}/prompt_operator.py",
        find="        self._events[payload.request_id] = (event, input_mode_event)\n",
        replace="",
        test="test_a_pending_ask_is_kept_and_dropped_when_answered",
        because="a reattached client is never shown the permission ASK",
    ),
    Reversion(
        target=f"{_HANDLERS}/clarification_relay.py",
        find="        self._events[request_id] = event\n",
        replace="",
        test="test_a_pending_clarification_is_kept_and_dropped_when_answered",
        because="a reattached client is never shown the clarification",
    ),
    Reversion(
        target=_CORE,
        find="        self._replay_pending_prompts(emit)\n\n",
        replace="\n",
        test="test_the_attach_replays_the_prompts_after_the_stale_clear",
        because="the attach sends everything except the prompt",
    ),
    Reversion(
        target=_CORE,
        find="                and not (ask_relay is not None and ask_relay.has_pending_prompt())):\n",
        replace="):\n",
        test="test_the_stale_clear_spares_a_relayed_prompt",
        because="a recovered session clears a prompt that is really pending",
    ),
]


def _ask_args(request_id: str) -> dict:
    return {
        "request_id": request_id, "agent_id": "main", "tool_name": "cli",
        "tool_args": {"command": "ls"}, "response_options": [],
    }


def test_a_pending_ask_is_kept_and_dropped_when_answered():
    async def run() -> None:
        emitted: List[Any] = []
        h = PromptOperatorHandler(emitted.append)
        task = asyncio.ensure_future(h.handle(_ask_args("r1")))
        await asyncio.sleep(0)
        events = h.pending_events()
        assert [type(e) for e in events] == [
            PermissionRequestedEvent, PermissionInputModeEvent]
        assert events == emitted
        assert h.resolve_response("r1", "y")
        await task
        assert h.pending_events() == []
    asyncio.run(run())


def test_a_pending_clarification_is_kept_and_dropped_when_answered():
    async def run() -> None:
        h = ClarificationRelayHandler(lambda e: None)
        task = asyncio.ensure_future(h.handle({
            "request_id": "c1", "agent_id": "main",
            "questions": [{"text": "which?"}],
        }))
        await asyncio.sleep(0)
        [event] = h.pending_events()
        assert isinstance(event, ClarificationBatchEvent)
        assert event.request_id == "c1"
        assert h.resolve_response("c1", ["a"])
        await task
        assert h.pending_events() == []
    asyncio.run(run())


def _server_with(relays: List[Any]) -> JaatoServer:
    server = JaatoServer.__new__(JaatoServer)
    server._prompt_operator_handler = relays[0] if relays else None
    server._clarification_relay_handler = relays[1] if len(relays) > 1 else None
    server._main_agent_id = "main"
    server._pending_permission_request_id = None
    server._pending_clarification_request_id = None
    server._pending_reference_selection_request_id = None
    return server


def test_the_replay_sends_each_relays_pending_events():
    ask = SimpleNamespace(pending_events=lambda: ["ask", "mode"],
                          has_pending_prompt=lambda: True)
    clar = SimpleNamespace(pending_events=lambda: ["batch"],
                           has_pending_prompt=lambda: True)
    out: List[Any] = []
    JaatoServer._replay_pending_prompts(_server_with([ask, clar]), out.append)
    assert out == ["ask", "mode", "batch"]


def test_the_stale_clear_spares_a_relayed_prompt():
    ask = SimpleNamespace(pending_events=lambda: [], has_pending_prompt=lambda: True)
    clar = SimpleNamespace(pending_events=lambda: [], has_pending_prompt=lambda: False)
    out: List[Any] = []
    JaatoServer._emit_clear_stale_requests(_server_with([ask, clar]), out.append)
    kinds = [type(e) for e in out]
    assert PermissionResolvedEvent not in kinds
    assert ClarificationResolvedEvent in kinds


def test_the_attach_replays_the_prompts_after_the_stale_clear():
    """``emit_current_state`` reaches a dozen subsystems, so its call order
    is read off the source rather than driven."""
    # Beside this test, never via ``inspect``: the reversion meta-guard runs
    # this in a sandboxed copy of the tree.
    tree = ast.parse((Path(__file__).resolve().parents[1] / "core.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "emit_current_state":
            # Source order: ``ast.walk`` is breadth-first, so a call nested in
            # an ``if`` would otherwise sort after its later siblings.
            calls = [c.func.attr for c in sorted(
                (c for c in ast.walk(node)
                 if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)),
                key=lambda c: (c.lineno, c.col_offset))]
            assert "_replay_pending_prompts" in calls
            assert calls.index("_replay_pending_prompts") > calls.index(
                "_emit_clear_stale_requests")
            return
    raise AssertionError("no emit_current_state in core.py")
