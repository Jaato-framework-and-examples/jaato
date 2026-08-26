"""Tests for ``IPCClient.create_session`` polymorphic ``profile`` parameter.

Verifies that:
- ``profile=str`` produces ``args=["--profile", str]`` with no payload
- ``profile=dict`` produces ``payload={"spec": dict}`` and no ``--profile``
  in args
- ``agent`` and ``agent_params`` work in both modes (orthogonal to profile)
- ``profile`` of any other type raises ``TypeError``

These tests don't connect — they patch ``_send_event`` to capture the
``CommandRequest`` produced by ``create_session``.
"""

import asyncio

import pytest

from jaato_sdk import IPCClient
from jaato_sdk.events import ClientType
from jaato_sdk.events import CommandRequest


def _make_client_capture():
    """Return (client, captured_events) tuple.

    ``captured_events`` accumulates every event ``_send_event`` is called
    with, letting tests assert on the exact ``CommandRequest`` shape.
    """
    client = IPCClient(socket_path="/tmp/_unused_create_session.sock", client_type=ClientType.API, auto_start=False)
    captured = []

    async def fake_send_event(event):
        captured.append(event)
        # MUST report success.  ``_send_event`` returns whether the write
        # actually went, and ``create_session`` now fails fast on a falsy
        # answer rather than waiting out the timeout for a reply to a command
        # it knows it never sent.  A fake that returns ``None`` is claiming
        # the socket was gone, and every test here would assert on the
        # not-sent path instead of the argv shape it is named for.
        return True

    async def fake_await_session_info(request_id=None):
        # Short-circuit the post-send wait so the test asserts only on
        # the CommandRequest shape.  Returns an id because create_session
        # returns ``str`` now -- failures raise, so ``None`` is not a value
        # this can produce.
        return "sess-fake"

    client._send_event = fake_send_event
    client._await_session_info = fake_await_session_info
    return client, captured


def _spec_payload(cmd):
    """The payload MINUS the correlation token.

    ``create_session`` stamps ``payload["request_id"]`` on every send so the
    answering event can be matched to this call.  These tests are about
    profile routing, so they assert on what is left after removing it --
    rather than on a bare ``payload is None``, which stopped being true the
    day correlation landed and left all nine tests in this file red.
    """
    if cmd.payload is None:
        return None
    rest = {k: v for k, v in cmd.payload.items() if k != "request_id"}
    assert "request_id" in cmd.payload, (
        "every session.new must carry a request_id — without it a stale "
        "SessionInfoEvent from an earlier create can satisfy this wait"
    )
    return rest or None


@pytest.mark.asyncio
async def test_profile_string_routes_to_argv():
    """``profile=str`` produces ``--profile`` flag, no payload."""
    client, captured = _make_client_capture()

    await client.create_session(profile="researcher")

    assert len(captured) == 1
    cmd = captured[0]
    assert isinstance(cmd, CommandRequest)
    assert cmd.command == "session.new"
    assert "--profile" in cmd.args
    assert "researcher" in cmd.args
    assert _spec_payload(cmd) is None


@pytest.mark.asyncio
async def test_profile_dict_routes_to_payload():
    """``profile=dict`` produces ``payload={'spec': dict}``, no --profile arg."""
    client, captured = _make_client_capture()

    spec = {
        "model": "claude-sonnet-4-5",
        "plugins": ["cli", "web_search"],
        "system_instructions": "You are a researcher.",
    }
    await client.create_session(profile=spec)

    cmd = captured[0]
    assert "--profile" not in cmd.args
    assert _spec_payload(cmd) == {"spec": spec}


@pytest.mark.asyncio
async def test_profile_dict_with_name_and_agent():
    """Inline spec composes with name and agent independently."""
    client, captured = _make_client_capture()

    await client.create_session(
        name="ops-task",
        profile={"model": "claude-sonnet-4-5"},
        agent="reviewer",
        agent_params={"focus": "security"},
    )

    cmd = captured[0]
    # Name is the first bare arg
    assert cmd.args[0] == "ops-task"
    # Inline spec rides on payload, not args
    assert _spec_payload(cmd) == {"spec": {"model": "claude-sonnet-4-5"}}
    # Agent and params still work
    assert "--agent" in cmd.args
    assert "reviewer" in cmd.args
    assert "focus=security" in cmd.args


@pytest.mark.asyncio
async def test_profile_string_with_agent_orthogonal():
    """Profile name + agent compose normally — no interaction."""
    client, captured = _make_client_capture()

    await client.create_session(
        profile="researcher",
        agent="reviewer",
    )

    cmd = captured[0]
    assert "--profile" in cmd.args
    assert "researcher" in cmd.args
    assert "--agent" in cmd.args
    assert "reviewer" in cmd.args
    assert _spec_payload(cmd) is None


@pytest.mark.asyncio
async def test_profile_none_omits_both_paths():
    """No profile, no spec — args empty (or just name), payload None."""
    client, captured = _make_client_capture()

    await client.create_session()

    cmd = captured[0]
    assert "--profile" not in cmd.args
    assert _spec_payload(cmd) is None


@pytest.mark.asyncio
async def test_profile_invalid_type_raises():
    """Non-str / non-dict / non-None profile is a programmer error."""
    client, _ = _make_client_capture()

    with pytest.raises(TypeError, match="must be str.*or dict"):
        await client.create_session(profile=42)

    with pytest.raises(TypeError):
        await client.create_session(profile=["not", "a", "spec"])


# ----------------------------------------------------------------------
# Phase 2 cascade-sharing (server 0.6.144+, SDK 0.13.X+)
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cascade_driver_id_emits_argv_flag():
    """``cascade_driver_id=<id>`` appends ``--cascade-driver-id <id>`` to
    the session.new argv.  Server-side command_router parses it +
    threads through to PoolManager.acquire_slot for affinity routing."""
    client, captured = _make_client_capture()

    await client.create_session(cascade_driver_id="cascade-abc123")

    cmd = captured[0]
    assert "--cascade-driver-id" in cmd.args
    idx = cmd.args.index("--cascade-driver-id")
    assert cmd.args[idx + 1] == "cascade-abc123"


@pytest.mark.asyncio
async def test_cascade_driver_id_none_omits_flag():
    """Default (cascade_driver_id=None) — no flag emitted.  Preserves
    Phase 1 semantics for non-cascade sessions."""
    client, captured = _make_client_capture()

    await client.create_session()

    cmd = captured[0]
    assert "--cascade-driver-id" not in cmd.args


@pytest.mark.asyncio
async def test_cascade_driver_id_composes_with_profile_agent_params():
    """All four kwargs (profile, agent, agent_params, cascade_driver_id)
    coexist on one create_session call without colliding."""
    client, captured = _make_client_capture()

    await client.create_session(
        name="my-session",
        profile="researcher",
        agent="discovery",
        agent_params={"target": "kb"},
        cascade_driver_id="cascade-xyz789",
    )

    cmd = captured[0]
    # Name first; flags + agent_params follow.
    assert "my-session" in cmd.args
    assert "--profile" in cmd.args
    assert "researcher" in cmd.args
    assert "--agent" in cmd.args
    assert "discovery" in cmd.args
    assert "target=kb" in cmd.args
    assert "--cascade-driver-id" in cmd.args
    assert "cascade-xyz789" in cmd.args


@pytest.mark.asyncio
async def test_cascade_driver_id_empty_string_omits_flag():
    """Falsy values (empty string) also omit the flag — defensive
    treatment, matches the ``if cascade_driver_id:`` gate."""
    client, captured = _make_client_capture()

    await client.create_session(cascade_driver_id="")

    cmd = captured[0]
    assert "--cascade-driver-id" not in cmd.args
