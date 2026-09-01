"""Convenience facade (IPCClient.session / Session.ask / complete / ask()).

Drives the facade with a fake client so the contracts are exercised without a
daemon: the no-hang invariant (a plain turn emits only TURN_COMPLETED), source
filtering, typed completion, error raising, the permission fail-loud path, the
context-manager lifecycle + config pass-through (both config styles), and the
one-shot module function.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from jaato_sdk import (
    AgentError,
    IPCRecoveryClient,
    PermissionUnhandled,
    ReconnectingError,
    SessionCreateFailed,
    SessionRefused,
    Session,
    ask,
)
from jaato_sdk.events import EventType
from jaato_sdk.client.convenience import _SessionContext, open_session


class FakeClient:
    """Records calls; ``send_message`` fires a scripted event sequence into the
    synchronously-dispatched handlers (matching the real registry's inline
    sync-handler semantics)."""

    def __init__(self, fire=None, *, connect_ok=True, sid="sid-1", **ctor):
        self.ctor = ctor
        self._fire = fire or []
        self._connect_ok = connect_ok
        self._sid = sid
        self._handlers: dict = {}
        self.calls: list = []
        self.create_kwargs: dict | None = None

    async def connect(self, timeout=0.0):
        self.calls.append(("connect", timeout))
        return self._connect_ok

    async def create_session(self, **kwargs):
        """Mirror the real contract: return an id, or RAISE.

        ``sid=None`` used to mean "creation failed" and the facade checked for
        it.  Under the current contract ``None`` is not a value
        ``create_session`` can produce -- failures raise -- so the fake
        expresses failure the way the real client does.  A fake that kept
        returning ``None`` would be testing a state the production code can no
        longer be in, and the facade would build a Session around it.
        """
        self.create_kwargs = kwargs
        self.calls.append(("create_session", kwargs))
        if self._sid is None:
            raise SessionRefused(
                "the daemon refused session.new: no such profile",
                error_type="ProfileNotFoundError",
            )
        return self._sid

    async def register_client_tools(self, tools):
        self.calls.append(("register_client_tools", tools))

    def subscribe(self, event_type, handler):
        self._handlers.setdefault(event_type, []).append(handler)
        return lambda: self._handlers.get(event_type, []).remove(handler) \
            if handler in self._handlers.get(event_type, []) else None

    subscribe_once = subscribe

    async def send_message(self, prompt, parallel_tools=None, attachments=None):
        self.calls.append(("send_message", prompt))
        self.send_kwargs = {"parallel_tools": parallel_tools,
                            "attachments": attachments}
        for event_type, ev in self._fire:
            for h in list(self._handlers.get(event_type, [])):
                h(ev)

    async def respond_to_permission(self, request_id, response, edited_arguments=None):
        self.calls.append(("respond_to_permission", request_id, response))

    async def disconnect(self):
        self.calls.append(("disconnect",))


def _out(source, text):
    return (EventType.AGENT_OUTPUT, SimpleNamespace(source=source, text=text))


_TURN = (EventType.TURN_COMPLETED, SimpleNamespace())


# --------------------------------------------------------------- ask()

async def test_ask_plain_turn_does_not_hang_returns_model_text():
    c = FakeClient(fire=[_out("model", "Hello"), _TURN])
    s = Session(c, "sid-1")
    assert await s.ask("hi") == "Hello"      # returned via TURN_COMPLETED only


async def test_ask_filters_sources_by_default_to_model():
    c = FakeClient(fire=[_out("model", "A"), _out("tool", "[tool]"),
                         _out("system", "[sys]"), _TURN])
    s = Session(c, "sid-1")
    assert await s.ask("hi") == "A"


async def test_ask_widened_sources_collects_more():
    fire = [_out("model", "A"), _out("tool", "B"), _TURN]
    assert await Session(FakeClient(fire=fire), "s").ask("hi", sources=("model", "tool")) == "AB"
    assert await Session(FakeClient(fire=fire), "s").ask("hi", sources=None) == "AB"


async def test_ask_raises_agent_error_on_error_terminal():
    c = FakeClient(fire=[(EventType.SESSION_TERMINATED,
                          SimpleNamespace(reason="error", error_type="APIError",
                                          error_summary="boom"))])
    with pytest.raises(AgentError) as ei:
        await Session(c, "s").ask("hi")
    assert ei.value.error_type == "APIError"
    assert ei.value.error_summary == "boom"


# --------------------------------------------------------------- complete()

async def test_complete_returns_payload():
    c = FakeClient(fire=[
        (EventType.AGENT_COMPLETED, SimpleNamespace(payload={"name": "Alice", "age": 30})),
        (EventType.SESSION_TERMINATED, SimpleNamespace(reason="natural")),
    ])
    assert await Session(c, "s").complete("Alice is 30.") == {"name": "Alice", "age": 30}


async def test_complete_no_completion_returns_none_no_hang():
    # plain turn, model never called signal_completion -> only TURN_COMPLETED
    assert await Session(FakeClient(fire=[_TURN]), "s").complete("hi") is None


# ------------------------------- complete() settles on the SESSION's terminus

class ScriptedClient(FakeClient):
    """Delivers its script ONE EVENT PER LOOP TICK, from a background task.

    ``FakeClient`` fires everything inside ``send_message``, so by the time the
    call under test reaches its wait, the whole session has already happened --
    which makes "did it return too early?" unaskable: an early return still
    sees the last event.  That is precisely the question #767 turns on, so
    these tests need a client that lets the session arrive AFTER the wait
    begins, the way a daemon does.
    """

    async def send_message(self, prompt, parallel_tools=None, attachments=None):
        self.calls.append(("send_message", prompt))
        self.delivered = 0
        self._pump_task = asyncio.ensure_future(self._pump())

    async def _pump(self):
        for event_type, ev in self._fire:
            await asyncio.sleep(0)
            for h in list(self._handlers.get(event_type, [])):
                h(ev)
            self.delivered += 1


def _status(agent, status):
    return (EventType.AGENT_STATUS_CHANGED,
            SimpleNamespace(agent_id=agent, status=status))


def _turn(agent="main"):
    return (EventType.TURN_COMPLETED, SimpleNamespace(agent_id=agent))


#: The daemon's wire shape for a completion-gated session whose model ends its
#: loop in prose: the turn ends, the daemon RE-PROMPTS it
#: (``COMPLETION_NUDGE``) and announces the agent active again, twice, before
#: the agent finally signals.  The payload rides the LAST turn, so a call that
#: settles on the first one returns None for a session that completed -- and,
#: in the case that motivated this, hands a grader a half-written tree
#: (jaato #767, where an eval arm was graded 19s before the agent's first
#: commit and recorded FAIL on a tree that compiles).
_NUDGED_SESSION = [
    _status("main", "active"),           # send_message begins
    _turn(), _status("main", "active"),  # turn 1 -> nudge 1
    _turn(), _status("main", "active"),  # turn 2 -> nudge 2
    (EventType.AGENT_COMPLETED, SimpleNamespace(payload={"ok": True})),
    _turn(),
    (EventType.SESSION_TERMINATED, SimpleNamespace(reason="natural")),
]


async def test_complete_waits_out_a_nudged_session():
    c = ScriptedClient(fire=_NUDGED_SESSION)
    assert await Session(c, "s").complete("go") == {"ok": True}
    assert c.delivered == len(_NUDGED_SESSION), (
        "returned before the session reached its terminus")


async def test_complete_settles_when_the_agent_settles_not_when_a_turn_ends():
    """The status event after the turn is what makes it a terminus.

    Same opening as the nudged session, but the agent settles on its first
    turn -- so the call must return there and NOT wait for a
    SESSION_TERMINATED that a turn-per-message session never emits.
    """
    c = ScriptedClient(fire=[_status("main", "active"), _turn(),
                             _status("main", "done")])
    assert await Session(c, "s").complete("go") is None
    assert c.delivered == 3


async def test_complete_ignores_a_subagent_turn():
    """A subagent finishing a turn is not the parent's terminus.

    Before #767 any agent's TURN_COMPLETED ended the wait, so a parent that
    spawned one settled on the CHILD's first turn.
    """
    c = ScriptedClient(fire=[
        _status("main", "active"),
        _turn("sub-1"), _status("sub-1", "done"),   # not ours
        (EventType.AGENT_COMPLETED, SimpleNamespace(payload={"ok": True})),
        _turn("main"), _status("main", "done"),
    ])
    assert await Session(c, "s").complete("go") == {"ok": True}


async def test_complete_falls_back_to_the_turn_when_no_agent_is_announced():
    """Version skew costs accuracy, never a wait with no end.

    A daemon that never emits AGENT_STATUS_CHANGED(active) leaves nothing to
    latch, and the first turn settles the call exactly as it did before #767.
    """
    c = ScriptedClient(fire=[_turn()])
    assert await Session(c, "s").complete("go") is None
    assert c.delivered == 1


async def test_complete_raises_on_the_error_terminal_of_a_nudged_session():
    """A spent nudge budget is a terminal error, and it must reach the caller.

    This is the shape a completion-gated session takes when the model never
    signals: NudgeExhausted, after the daemon has asked twice.
    """
    c = ScriptedClient(fire=[
        _status("main", "active"),
        _turn(), _status("main", "active"),
        _turn(), _status("main", "active"),
        _turn(),
        (EventType.SESSION_TERMINATED,
         SimpleNamespace(reason="error", error_type="NudgeExhausted",
                         error_summary="asked twice, never signalled")),
    ])
    with pytest.raises(AgentError) as ei:
        await Session(c, "s").complete("go")
    assert ei.value.error_type == "NudgeExhausted"


# --------------------------------------------------------------- stream() (Phase 2)

async def test_stream_yields_chunks_then_stops():
    c = FakeClient(fire=[_out("model", "Hel"), _out("model", "lo"), _TURN])
    chunks = [x async for x in Session(c, "s").stream("hi")]
    assert chunks == ["Hel", "lo"]


async def test_stream_filters_sources_by_default():
    c = FakeClient(fire=[_out("model", "A"), _out("tool", "B"), _TURN])
    assert [x async for x in Session(c, "s").stream("hi")] == ["A"]
    c2 = FakeClient(fire=[_out("model", "A"), _out("tool", "B"), _TURN])
    assert [x async for x in Session(c2, "s").stream("hi", sources=None)] == ["A", "B"]


async def test_stream_raises_agent_error_on_error_terminal():
    c = FakeClient(fire=[(EventType.SESSION_TERMINATED,
                          SimpleNamespace(reason="error", error_type="APIError",
                                          error_summary="boom"))])
    with pytest.raises(AgentError):
        async for _ in Session(c, "s").stream("hi"):
            pass


# --------------------------------------------- send_message passthrough + ctor knobs

async def test_ask_forwards_parallel_tools_and_attachments():
    c = FakeClient(fire=[_out("model", "ok"), _TURN])
    att = [{"mime_type": "image/png", "data": "...", "display_name": "x.png"}]
    await Session(c, "s").ask("hi", parallel_tools=False, attachments=att)
    assert c.send_kwargs == {"parallel_tools": False, "attachments": att}


async def test_ask_defaults_send_kwargs_to_none():
    c = FakeClient(fire=[_out("model", "ok"), _TURN])
    await Session(c, "s").ask("hi")
    assert c.send_kwargs == {"parallel_tools": None, "attachments": None}


async def test_config_root_and_apparmor_forwarded_only_when_set():
    with_both = open_session(FakeClient, profile="x",
                             config_root="/cfg", apparmor=True)
    assert with_both._client.ctor.get("config_root") == "/cfg"
    assert with_both._client.ctor.get("apparmor") is True
    without = open_session(FakeClient, profile="x")
    assert "config_root" not in without._client.ctor
    assert "apparmor" not in without._client.ctor


async def test_session_exposes_underlying_client_for_mixing():
    """s.client gives low-level access on the SAME connection, so a builder can
    mix facade verbs with raw subscribe/events in one implementation."""
    captured = {}

    class _Cls(FakeClient):
        def __init__(self, **ctor):
            super().__init__(fire=[_out("model", "ok"), _TURN], **ctor)
            captured["instance"] = self

    async with open_session(_Cls, profile="x") as s:
        assert s.client is captured["instance"]          # same underlying client
        # a builder can attach their own low-level listener alongside facade calls
        s.client.subscribe(EventType.TOOL_CALL_END, lambda ev: None)
        assert await s.ask("hi") == "ok"


async def test_tool_level_error_in_completing_turn_does_not_raise():
    """A denied/failed tool that the turn recovers from completes normally
    (TURN_COMPLETED) — the facade raises AgentError ONLY on an error TERMINAL,
    not on in-turn tool failures (G5 contract)."""
    c = FakeClient(fire=[
        (EventType.AGENT_OUTPUT, SimpleNamespace(source="model", text="done")),
        # a tool-level failure event the facade does not treat as terminal:
        (EventType.TURN_COMPLETED, SimpleNamespace()),
    ])
    assert await Session(c, "s").ask("do the gated thing") == "done"  # no raise


# --------------------------------------------------- host tools via facade (Phase 2.1)

async def test_client_tools_registered_before_create_session():
    captured = {}
    tools = [{"name": "get_weather", "description": "d",
              "parameters": {"type": "object", "properties": {}},
              "handler": lambda a: {}}]

    class _Cls(FakeClient):
        def __init__(self, **ctor):
            super().__init__(fire=[_out("model", "ok"), _TURN], **ctor)
            captured["instance"] = self

    async with open_session(_Cls, profile={"model": "m"}, client_tools=tools) as s:
        assert await s.ask("Use the tool") == "ok"
    calls = [c[0] for c in captured["instance"].calls]
    # ordering is load-bearing: register must precede create_session
    assert calls.index("register_client_tools") < calls.index("create_session")
    assert ("register_client_tools", tools) in captured["instance"].calls


async def test_no_client_tools_means_no_registration():
    captured = {}

    class _Cls(FakeClient):
        def __init__(self, **ctor):
            super().__init__(fire=[_TURN], **ctor)
            captured["instance"] = self

    async with open_session(_Cls, profile={"model": "m"}) as s:
        await s.ask("hi")
    assert not any(c[0] == "register_client_tools"
                   for c in captured["instance"].calls)


# --------------------------------------------------------- recovery parity (Phase 2)

async def test_recovery_session_classmethod_delegates():
    """IPCRecoveryClient.session returns a session context manager backed by an
    IPCRecoveryClient — same facade, auto-reconnect client."""
    cm = IPCRecoveryClient.session(profile={"model": "m", "provider": "p"},
                                   env_file=".env")
    assert isinstance(cm, _SessionContext)
    assert isinstance(cm._client, IPCRecoveryClient)


async def test_on_status_change_forwarded_only_when_set():
    cb = lambda s: None
    with_cb = open_session(FakeClient, profile="x", on_status_change=cb)
    assert with_cb._client.ctor.get("on_status_change") is cb
    without = open_session(FakeClient, profile="x")
    assert "on_status_change" not in without._client.ctor


# --------------------------------------------------------------- permissions

async def test_unhandled_permission_denies_and_then_raises():
    c = FakeClient(fire=[_TURN])
    s = Session(c, "s")                       # no on_permission
    await s._on_perm(SimpleNamespace(request_id="r1", tool_name="delete_file"))
    assert ("respond_to_permission", "r1", "n") in c.calls   # denied to unstick
    with pytest.raises(PermissionUnhandled) as ei:
        await s.ask("hi")                     # in-flight turn surfaces it
    assert ei.value.tool_name == "delete_file"


async def test_permission_callback_is_used():
    c = FakeClient()
    s = Session(c, "s", on_permission=lambda ev: "y")
    await s._on_perm(SimpleNamespace(request_id="r1", tool_name="delete_file"))
    assert ("respond_to_permission", "r1", "y") in c.calls


async def test_async_permission_callback_is_awaited():
    c = FakeClient()

    async def approve(ev):
        return "a"

    s = Session(c, "s", on_permission=approve)
    await s._on_perm(SimpleNamespace(request_id="r1", tool_name="x"))
    assert ("respond_to_permission", "r1", "a") in c.calls


# ------------------------------------------------ context manager + pass-through

async def test_session_lifecycle_and_programmatic_passthrough():
    captured = {}

    class _Cls(FakeClient):
        def __init__(self, **ctor):
            super().__init__(fire=[_out("model", "ok"), _TURN], **ctor)
            captured["instance"] = self

    cm = open_session(_Cls, profile={"model": "m", "provider": "p"},
                      agent="pirate", connect_timeout=1.0)
    async with cm as s:
        assert await s.ask("hi") == "ok"
    c = captured["instance"]
    # forwarded create_session config (programmatic inline spec + persona)
    assert c.create_kwargs["profile"] == {"model": "m", "provider": "p"}
    assert c.create_kwargs["agent"] == "pirate"
    # lifecycle: connected then disconnected
    assert ("connect", 1.0) in c.calls
    assert ("disconnect",) in c.calls


async def test_declarative_passthrough():
    captured = {}

    class _Cls(FakeClient):
        def __init__(self, **ctor):
            super().__init__(fire=[_TURN], **ctor)
            captured["instance"] = self

    async with open_session(_Cls, profile="researcher", agent="pirate",
                            agent_params={"tone": "terse"}) as s:
        await s.ask("hi")
    ck = captured["instance"].create_kwargs
    assert ck["profile"] == "researcher"       # named = declarative
    assert ck["agent"] == "pirate"
    assert ck["agent_params"] == {"tone": "terse"}


async def test_connect_failure_raises():
    class _Cls(FakeClient):
        def __init__(self, **ctor):
            super().__init__(connect_ok=False, **ctor)

    with pytest.raises(ConnectionError):
        async with open_session(_Cls, profile="x"):
            pass


async def test_session_new_failure_raises_and_disconnects():
    captured = {}

    class _Cls(FakeClient):
        def __init__(self, **ctor):
            super().__init__(sid=None, **ctor)
            captured["instance"] = self

    # RuntimeError, still: ``SessionCreateFailed`` subclasses it precisely so
    # handlers written against the old "caller raises RuntimeError" contract
    # keep catching.  Asserting the base here (not the subclass) is the point
    # -- it is the compatibility guarantee out-of-tree consumers rely on.
    with pytest.raises(RuntimeError) as excinfo:
        async with open_session(_Cls, profile="x"):
            pass
    assert isinstance(excinfo.value, SessionCreateFailed)
    # The daemon's own reason reaches the caller.  The facade used to replace
    # it with a guess -- "check provider auth" -- which was wrong for every
    # refusal that was not an auth failure.
    assert "no such profile" in str(excinfo.value)
    assert excinfo.value.may_exist is False
    assert ("disconnect",) in captured["instance"].calls


async def test_the_facade_disconnects_even_when_create_raises():
    """The connection must not leak on the exception path.

    The old ``if not sid: disconnect(); raise`` ran the disconnect only on the
    falsy-return path.  An exception out of ``create_session`` -- already
    possible before this change, since ``IPCRecoveryClient`` raises
    ``ReconnectingError`` / ``ConnectionClosedError`` from ``_check_can_send``
    -- skipped it and leaked the client.
    """
    captured = {}

    class _Boom(FakeClient):
        def __init__(self, **ctor):
            super().__init__(**ctor)
            captured["instance"] = self

        async def create_session(self, **kwargs):
            self.calls.append(("create_session", kwargs))
            raise ReconnectingError()

    with pytest.raises(ReconnectingError):
        async with open_session(_Boom, profile="x"):
            pass
    assert ("disconnect",) in captured["instance"].calls, (
        "an exception out of create_session leaked the connection"
    )


# --------------------------------------------------------------- module ask()

async def test_module_ask_delegates_to_session(monkeypatch):
    import jaato_sdk.client.ipc as ipc_mod
    seen = {}

    class _FakeSession:
        async def ask(self, prompt, *, sources):
            seen["prompt"] = prompt
            seen["sources"] = sources
            return f"answer:{prompt}"

    class _FakeCM:
        def __init__(self, **kw):
            seen["session_kwargs"] = kw

        async def __aenter__(self):
            return _FakeSession()

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(ipc_mod.IPCClient, "session",
                        staticmethod(lambda **kw: _FakeCM(**kw)))

    out = await ask("Who are you?", profile={"model": "m", "provider": "p"})
    assert out == "answer:Who are you?"
    assert seen["sources"] == ("model",)
    assert seen["session_kwargs"]["profile"] == {"model": "m", "provider": "p"}


async def test_raising_on_permission_responds_and_surfaces_error():
    """A raising ``on_permission`` must NOT deadlock the turn — ``_on_perm``
    ALWAYS responds (deny), and the callback's error surfaces on the in-flight
    ask/complete via ``_raise_if_needed`` (the in-process channel blocks a
    worker thread on the response with no transport timeout, so a response is
    mandatory)."""
    c = FakeClient()
    boom = RuntimeError("callback blew up")

    def raising(ev):
        raise boom

    s = Session(c, "s", on_permission=raising)
    await s._on_perm(SimpleNamespace(request_id="r1", tool_name="x"))

    # Guaranteed response (deny) — no deadlock.
    assert ("respond_to_permission", "r1", "n") in c.calls
    # The callback's real error is surfaced on the in-flight turn.
    with pytest.raises(RuntimeError) as ei:
        s._raise_if_needed({})
    assert ei.value is boom


async def test_async_raising_on_permission_also_guarded():
    """The guard covers a coroutine on_permission that raises too."""
    c = FakeClient()

    async def raising(ev):
        raise ValueError("async boom")

    s = Session(c, "s", on_permission=raising)
    await s._on_perm(SimpleNamespace(request_id="r2", tool_name="y"))
    assert ("respond_to_permission", "r2", "n") in c.calls
    with pytest.raises(ValueError):
        s._raise_if_needed({})
