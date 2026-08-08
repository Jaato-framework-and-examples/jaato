"""A scaffolded send-and-wait client must NOT hang on a plain turn.

Regression guard for the SESSION_TERMINATED-only completion wait (jaato
PR #316 class).  A completion-gated session emits ``SESSION_TERMINATED``;
a PLAIN turn that just answers emits only ``TURN_COMPLETED`` and the
session then goes IDLE — it never self-terminates.  A client that waits on
``SESSION_TERMINATED`` alone blocks forever on the plain path.  The
``client`` / ``host-tools`` archetypes therefore wait on first-of
``{TURN_COMPLETED, SESSION_TERMINATED}``.

This test generates a REAL client via ``build.run`` (so the whole template
+ build path is exercised), then drives the generated ``main()`` with a
fake client that emits ONLY ``TURN_COMPLETED`` — and asserts ``main()``
returns instead of blocking.  Wrapped in ``asyncio.wait_for`` so a
regression surfaces as a hard timeout, not a hung suite.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from jaato_sdk import EventType

from shared.scaffold import build, introspect
from shared.scaffold import _client_templates as tpl


def _a_provider() -> str:
    """Any installed provider — the client is never actually connected, so
    the choice is irrelevant; resolve from the environment, never hardcode."""
    names = sorted(introspect.providers())
    assert names, "no providers installed — cannot scaffold a client"
    return names[0]


def _generate(tmp_path: Path, archetype: str, *, recoverable: bool = False) -> Path:
    args = argparse.Namespace(
        archetype=archetype,
        workspace=str(tmp_path),
        provider=_a_provider(),
        model="test-model",
        set=None,
        agents=None,
        force=True,
        recoverable=recoverable,
        json=False,
    )
    rc = build.run(args)
    assert rc == 0, f"build.run({archetype}) returned {rc}"
    py = tmp_path / f"run_{archetype}.py"
    assert py.exists()
    return py


def _import(py: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeClient:
    """Minimal stand-in for IPCClient/IPCRecoveryClient that fires a
    scripted terminal event when ``send_message`` is called."""

    def __init__(self, fire):
        self._fire = fire                      # list[(EventType, event_obj)]
        self._handlers: dict = {}

    async def connect(self, timeout=0.0):
        return True

    def subscribe(self, event_type, handler):
        self._handlers.setdefault(event_type, []).append(handler)
        return lambda: None

    # subscribe_once shares storage — we fire each scripted event exactly once
    subscribe_once = subscribe

    async def create_session(self, *a, **k):
        return "sid-1"

    async def register_client_tools(self, tools):
        return None

    async def send_message(self, prompt):
        for event_type, ev in self._fire:
            for h in self._handlers.get(event_type, []):
                h(ev)

    async def disconnect(self):
        return None


def _run_main(mod, fire) -> int:
    mod._new_client = lambda: _FakeClient(fire)

    async def _go():
        return await asyncio.wait_for(mod.main(), timeout=5.0)

    return asyncio.run(_go())


@pytest.mark.parametrize("archetype", ["client", "host-tools"])
def test_plain_turn_does_not_hang(tmp_path, archetype):
    """Only TURN_COMPLETED fires (no SESSION_TERMINATED) — main() must return."""
    mod = _import(_generate(tmp_path, archetype), f"gen_{archetype}")
    rc = _run_main(mod, [(EventType.TURN_COMPLETED, SimpleNamespace())])
    assert rc == 0


def test_recoverable_client_plain_turn_does_not_hang(tmp_path):
    """The --recoverable variant shares the wait path — same guarantee."""
    mod = _import(_generate(tmp_path, "client", recoverable=True), "gen_client_rec")
    rc = _run_main(mod, [(EventType.TURN_COMPLETED, SimpleNamespace())])
    assert rc == 0


def test_error_terminal_surfaces_failure(tmp_path):
    """SESSION_TERMINATED(reason='error') must still drive a non-zero exit."""
    mod = _import(_generate(tmp_path, "client"), "gen_client_err")
    rc = _run_main(mod, [(
        EventType.SESSION_TERMINATED,
        SimpleNamespace(reason="error", error_type="APIError",
                        error_summary="boom"),
    )])
    assert rc == 1


def test_completion_gated_success_still_works(tmp_path):
    """A completion-gated turn emits TURN_COMPLETED then SESSION_TERMINATED
    (natural) — first-of resolves cleanly to a success exit."""
    mod = _import(_generate(tmp_path, "client"), "gen_client_gated")
    rc = _run_main(mod, [
        (EventType.TURN_COMPLETED, SimpleNamespace()),
        (EventType.SESSION_TERMINATED, SimpleNamespace(reason="natural")),
    ])
    assert rc == 0


def test_send_and_wait_templates_subscribe_to_both_terminals():
    """Static guard: the send-and-wait archetypes subscribe to BOTH
    TURN_COMPLETED and SESSION_TERMINATED (the anti-hang invariant)."""
    for src in (tpl.CLIENT_TEMPLATE, tpl.HOST_TOOLS_TEMPLATE):
        assert "EventType.SESSION_TERMINATED, on_done" in src
        assert "EventType.TURN_COMPLETED, on_done" in src


def test_fire_template_does_not_wait():
    """Fire-and-forget must not wait on any terminal."""
    assert "done.wait()" not in tpl.FIRE_TEMPLATE
    assert "subscribe_once(EventType.SESSION_TERMINATED" not in tpl.FIRE_TEMPLATE


def test_cascade_waits_on_session_terminated_only():
    """Cascade stages are completion-gated by design (signal_completion
    drives the slot handoff), so they intentionally wait on
    SESSION_TERMINATED only — NOT first-of."""
    assert "EventType.SESSION_TERMINATED, on_done" in tpl.CASCADE_TEMPLATE
    assert "EventType.TURN_COMPLETED, on_done" not in tpl.CASCADE_TEMPLATE
