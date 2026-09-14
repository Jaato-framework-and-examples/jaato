"""A scaffolded send-and-wait client must NOT hang, and must not hand-roll
the wait it hangs on.

TWO GUARANTEES, AND THE SECOND IS WHY THE FIRST STAYS TRUE
==========================================================

1. **No hang on a plain turn.**  A completion-gated session emits
   ``SESSION_TERMINATED``; a PLAIN turn that just answers emits only
   ``TURN_COMPLETED`` and the session then goes IDLE — it never
   self-terminates.  A client that waits on ``SESSION_TERMINATED`` alone
   blocks forever on the plain path (jaato PR #316 / #399).

2. **The wait is the SDK's, not the template's.**  That recipe lives in
   ``jaato_sdk.client.convenience``, whose docstring names it as the thing it
   exists to prevent.  The templates used to write it out by hand anyway —
   which is what made (1) something a generated script could regress on, and
   what left every scaffolded client behind the settle rule the facade
   learned in #767 (jaato #825 / #826 / #827).

So these tests generate a REAL client via ``build.run`` (the whole template +
build path), then drive the generated ``main()`` against a fake client that
emits ONLY ``TURN_COMPLETED`` — through the facade, exactly as a daemon
would — and assert ``main()`` returns instead of blocking.  Wrapped in
``asyncio.wait_for`` so a regression surfaces as a hard timeout, not a hung
suite.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from jaato_sdk import EventType
from jaato_sdk.client.convenience import Session

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
    """Minimal stand-in for the SDK clients that fires a scripted terminal
    event when ``send_message`` is called.

    Implements exactly the surface the facade's ``Session`` uses —
    ``subscribe`` / ``subscribe_once`` / ``send_message`` — plus the
    ``connect`` / ``create_session`` / ``disconnect`` the session context
    manager drives.  ``subscribe`` returns an unsubscribe callable because
    ``ask`` / ``stream`` / ``complete`` each clean up after themselves.
    """

    def __init__(self, fire):
        self._fire = fire                      # list[(EventType, event_obj)]
        self._handlers: dict = {}

    async def connect(self, timeout=0.0):
        return True

    def subscribe(self, event_type, handler):
        self._handlers.setdefault(event_type, []).append(handler)

        def _unsub():
            try:
                self._handlers[event_type].remove(handler)
            except (KeyError, ValueError):
                pass
        return _unsub

    # subscribe_once shares storage — we fire each scripted event exactly once
    subscribe_once = subscribe

    async def create_session(self, *a, **k):
        return "sid-1"

    async def register_client_tools(self, tools):
        return None

    async def send_message(self, prompt, **kwargs):
        for event_type, ev in self._fire:
            for h in list(self._handlers.get(event_type, [])):
                h(ev)

    async def disconnect(self):
        return None


class _FakeSessionContext:
    """Stands in for the facade's ``_SessionContext``: yields a REAL
    ``Session`` wrapping the fake client, so the generated script exercises
    the SDK's actual wait recipe rather than a mock of it."""

    def __init__(self, client):
        self._client = client

    async def __aenter__(self) -> Session:
        await self._client.connect()
        return Session(self._client, "sid-1", None)

    async def __aexit__(self, *exc) -> bool:
        await self._client.disconnect()
        return False


def _run_main(mod, fire) -> int:
    """Drive the generated ``main()`` with a scripted event sequence."""
    mod._open_session = lambda **spec: _FakeSessionContext(_FakeClient(fire))

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
    """SESSION_TERMINATED(reason='error') must still drive a non-zero exit.

    Through the facade it arrives as a typed ``AgentError`` rather than a
    reason string, which is why the template catches one.
    """
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


# ------------------------------------------------------------------ statics

#: Every archetype that takes a turn — i.e. everything but the read-only
#: observer, which attaches to someone else's cascade and sends nothing.
_TURN_TAKING = ("client", "host-tools", "cascade", "sweep")


@pytest.mark.parametrize("name", _TURN_TAKING)
def test_no_template_hand_rolls_the_wait(name):
    """The wait belongs to the SDK.

    The specific shapes banned here are the ones the templates actually
    shipped: an ``asyncio.Event`` woken by ``subscribe_once`` on the terminal
    events, then ``await done.wait()``.  That is ``convenience.py``'s recipe
    written out a second time — the second statement of a contract being the
    one that rots (jaato #825 / #826 / #827).
    """
    _, src, _ = tpl.TEMPLATES[name]
    assert "done.wait()" not in src, (
        f"{name} waits on its own asyncio.Event; the facade owns that recipe"
    )
    assert "subscribe_once(EventType.SESSION_TERMINATED" not in src, (
        f"{name} hand-rolls the terminal subscription the facade installs"
    )


@pytest.mark.parametrize("name", _TURN_TAKING)
def test_every_turn_taking_template_uses_the_facade(name):
    """...and it uses the facade INSTEAD, rather than merely not hand-rolling."""
    _, src, _ = tpl.TEMPLATES[name]
    assert "_open_session(" in src, f"{name} opens no facade session"
    assert any(f"s.{m}(" in src or f"stage.{m}(" in src
               for m in ("ask", "stream", "complete")), (
        f"{name} opens a session and never takes a turn through it"
    )


@pytest.mark.parametrize("name", ["client", "host-tools"])
def test_non_gated_archetypes_use_a_turn_method(name):
    """A NON-GATED session's turn IS its terminus, so ask/stream is right —
    they wait on first-of {TURN_COMPLETED, SESSION_TERMINATED}."""
    _, src, _ = tpl.TEMPLATES[name]
    assert "s.ask(" in src or "s.stream(" in src


@pytest.mark.parametrize("name", ["cascade", "sweep"])
def test_gated_archetypes_use_complete(name):
    """A COMPLETION-GATED session's terminus is signal_completion, not its
    first turn — and ``complete()`` is also the only method that RETURNS the
    payload the stage's own schema exists to produce (jaato #827)."""
    _, src, _ = tpl.TEMPLATES[name]
    assert ".complete(" in src, (
        f"{name} is completion-gated by construction and does not call "
        "Session.complete(); waiting on the terminal alone discards the payload"
    )


def _code(src: str) -> str:
    """Comments stripped — a template may WARN about a pattern in prose, and
    ``fire`` does exactly that ("NOT s.ask()/s.complete() — those WAIT")."""
    return "\n".join(line.split("#", 1)[0] for line in src.splitlines())


def test_fire_template_does_not_wait():
    """Fire-and-forget must not wait on any terminal — so NOT ask/complete
    either, which is the trap now that a turn method is one call away."""
    src = _code(tpl.FIRE_TEMPLATE)
    assert "done.wait()" not in src
    assert "subscribe_once(EventType.SESSION_TERMINATED" not in src
    assert "s.ask(" not in src and "s.complete(" not in src
    assert "s.client.send_message(" in tpl.FIRE_TEMPLATE
