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

    async def fake_await_session_info():
        # Short-circuit the post-send wait so the test asserts only on
        # the CommandRequest shape.  In SDK 0.13.0+ create_session
        # always calls _await_session_info; mocking it returns control
        # to the test immediately.
        return None

    client._send_event = fake_send_event
    client._await_session_info = fake_await_session_info
    return client, captured


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
    assert cmd.payload is None


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
    assert cmd.payload == {"spec": spec}


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
    assert cmd.payload == {"spec": {"model": "claude-sonnet-4-5"}}
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
    assert cmd.payload is None


@pytest.mark.asyncio
async def test_profile_none_omits_both_paths():
    """No profile, no spec — args empty (or just name), payload None."""
    client, captured = _make_client_capture()

    await client.create_session()

    cmd = captured[0]
    assert "--profile" not in cmd.args
    assert cmd.payload is None


@pytest.mark.asyncio
async def test_profile_invalid_type_raises():
    """Non-str / non-dict / non-None profile is a programmer error."""
    client, _ = _make_client_capture()

    with pytest.raises(TypeError, match="must be str.*or dict"):
        await client.create_session(profile=42)

    with pytest.raises(TypeError):
        await client.create_session(profile=["not", "a", "spec"])
