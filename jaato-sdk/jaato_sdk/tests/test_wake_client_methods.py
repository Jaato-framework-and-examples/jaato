"""Typed wake-primitive client methods: bind_wake / unbind_wake / cascade_register.

Replace the string-RPC (`execute_command("session.bind_wake", ...)`) +
`subscribe_once`+future boilerplate with typed `await` methods.  The result
handler is armed via a SYNCHRONOUS `subscribe` BEFORE the command is sent (no
subscribe-after-send race — matters on the recovery client, no replay buffer),
and results are filtered by `wake_ref` so concurrent binds don't cross.
"""

import asyncio
import pytest

from jaato_sdk import IPCClient
from jaato_sdk.client.recovery import IPCRecoveryClient
from jaato_sdk.client.ws import WSRecoveryClient
from jaato_sdk.events import ClientType, EventType, SessionWokenEvent, WakeBindResultEvent


def _make_client() -> IPCClient:
    c = IPCClient(
        socket_path="/tmp/_unused_wake_client.sock",
        client_type=ClientType.API,
        auto_start=False,
    )
    c._connected = True
    return c


@pytest.mark.asyncio
async def test_bind_wake_sends_command_and_returns_typed_result():
    c = _make_client()
    sent = []

    async def fake_exec(cmd, args=None):
        sent.append((cmd, args))
        # daemon reply: the handler must ALREADY be armed (subscribe-before-send)
        assert c._registry._handlers.get(EventType.WAKE_BIND_RESULT.value), \
            "result handler must be subscribed before the command is sent"
        c._registry.dispatch(WakeBindResultEvent(
            wake_ref=args[0], outcome="ok", detail="bind_wake: ok",
            expires_at=123.0, endpoint="http://x/wake"))

    c.execute_command = fake_exec
    result = await c.bind_wake("github-pr:o/r#1", ["KEYA", "KEYB"])
    assert sent == [("session.bind_wake", ["github-pr:o/r#1", "KEYA", "KEYB"])]
    assert result.outcome == "ok" and result.wake_ref == "github-pr:o/r#1"
    assert result.expires_at == 123.0 and result.endpoint == "http://x/wake"
    # handler cleaned up
    assert not c._registry._handlers.get(EventType.WAKE_BIND_RESULT.value)


@pytest.mark.asyncio
async def test_bind_wake_filters_by_wake_ref():
    c = _make_client()

    async def fake_exec(cmd, args=None):
        # a result for a DIFFERENT wake_ref must NOT resolve this call
        c._registry.dispatch(WakeBindResultEvent(wake_ref="other#9", outcome="ok"))
        c._registry.dispatch(WakeBindResultEvent(wake_ref=args[0], outcome="unauthorized"))

    c.execute_command = fake_exec
    result = await c.bind_wake("mine#1", ["K"])
    assert result.wake_ref == "mine#1" and result.outcome == "unauthorized"


@pytest.mark.asyncio
async def test_bind_wake_times_out_without_result():
    c = _make_client()

    async def fake_exec(cmd, args=None):
        pass  # daemon never replies

    c.execute_command = fake_exec
    with pytest.raises(asyncio.TimeoutError):
        await c.bind_wake("x#1", ["K"], timeout=0.05)
    assert not c._registry._handlers.get(EventType.WAKE_BIND_RESULT.value)  # cleaned up


@pytest.mark.asyncio
async def test_unbind_wake_sends_and_returns():
    c = _make_client()
    sent = []

    async def fake_exec(cmd, args=None):
        sent.append((cmd, args))
        c._registry.dispatch(WakeBindResultEvent(wake_ref=args[0], outcome="ok"))

    c.execute_command = fake_exec
    result = await c.unbind_wake("pr#2")
    assert sent == [("session.unbind_wake", ["pr#2"])]
    assert result.outcome == "ok"


@pytest.mark.asyncio
async def test_cascade_register_derives_class_name_and_is_fire_and_forget():
    c = _make_client()
    sent = []

    async def fake_exec(cmd, args=None):
        sent.append((cmd, args))

    c.execute_command = fake_exec
    # pass the CLASS (footgun-proof) — the right "SessionWokenEvent" name is derived
    await c.cascade_register("bot-cid", "observer", [SessionWokenEvent])
    assert sent == [("cascade.register", ["bot-cid", "observer", "SessionWokenEvent"])]


@pytest.mark.asyncio
async def test_cascade_register_accepts_string_names_too():
    c = _make_client()
    sent = []
    c.execute_command = lambda cmd, args=None: sent.append((cmd, args)) or _noop()
    await c.cascade_register("cid", "observer", ["SessionWokenEvent"])
    assert sent == [("cascade.register", ["cid", "observer", "SessionWokenEvent"])]


async def _noop():
    return None


def test_recovery_clients_expose_the_typed_methods():
    # WSRecoveryClient (what the bot uses) inherits them from IPCRecoveryClient.
    for name in ("bind_wake", "unbind_wake", "cascade_register"):
        assert hasattr(IPCRecoveryClient, name)
        assert getattr(WSRecoveryClient, name) is getattr(IPCRecoveryClient, name)
