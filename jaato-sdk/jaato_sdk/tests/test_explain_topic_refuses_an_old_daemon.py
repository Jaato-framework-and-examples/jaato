"""``explain_topic`` refuses a daemon below 1.18 rather than waiting on it.

The 1.7 rule, applied to a verb rather than a payload: an additive FIELD
degrades harmlessly, and a missing VERB does not.  An older daemon ignores
``scaffold.explain`` in silence, and silence on this verb is exactly the
answer it exists to stop being read as *there is no such topic* — the caller
would wait out its deadline and report the topic missing, which is the
original defect arriving through the fix.

So the refusal is raised client-side, before anything reaches the socket, and
it names the daemon's spoken version: that is the actionable half, because the
remedy is upgrading a process the reader may not own.
"""

import asyncio

import pytest

from jaato_sdk.client.ipc import IPCClient
from jaato_sdk.events import PROTOCOL_VERSION, ClientType


def _client(spoken):
    client = IPCClient("/tmp/does-not-exist.sock", client_type=ClientType.API,
                       auto_start=False)
    client._server_protocol_version = spoken
    return client


def _sent(client):
    """Whatever the client tried to put on the wire."""
    return getattr(client, "_guard_sent", [])


@pytest.fixture(autouse=True)
def _capture(monkeypatch):
    async def fake_send(self, event):
        self._guard_sent = getattr(self, "_guard_sent", []) + [event]
    monkeypatch.setattr(IPCClient, "_send_event", fake_send, raising=True)


def test_the_floor_is_the_version_that_introduced_the_verb():
    assert IPCClient.MIN_SCAFFOLD_EXPLAIN_PROTOCOL == "1.18"


def test_a_daemon_at_the_floor_is_asked():
    client = _client(IPCClient.MIN_SCAFFOLD_EXPLAIN_PROTOCOL)
    asyncio.run(client.explain_topic("reactors"))
    assert len(_sent(client)) == 1
    assert _sent(client)[0].command == "scaffold.explain"


def test_todays_protocol_is_at_or_above_the_floor():
    """A floor above the version this SDK speaks would refuse every daemon."""
    client = _client(PROTOCOL_VERSION)
    asyncio.run(client.explain_topic("reactors"))
    assert len(_sent(client)) == 1


def test_an_older_daemon_is_refused_with_nothing_sent():
    client = _client("1.17")
    with pytest.raises(ValueError) as exc:
        asyncio.run(client.explain_topic("reactors"))
    assert "1.17" in str(exc.value), (
        "the refusal must name what the daemon SPEAKS — the remedy is "
        "upgrading a process the reader may not own")
    assert "1.18" in str(exc.value)
    assert _sent(client) == [], (
        "nothing may reach a daemon that would ignore it: the caller would "
        "then wait out a deadline and report the topic missing")


def test_a_client_that_never_connected_is_refused_too():
    """No version is not a version at or above the floor."""
    client = _client(None)
    with pytest.raises(ValueError):
        asyncio.run(client.explain_topic("reactors"))


def test_both_argument_positions_are_always_sent():
    """Dropping an absent one shifts the rest.

    ``explain_topic(name=X)`` must not put an argument where the handler
    reads a topic name — which is what a ``[a for a in ... if a]`` filter
    would do, and which fails as a *wrong answer* rather than an error.
    """
    client = _client(PROTOCOL_VERSION)
    asyncio.run(client.explain_topic("plugin", "memory"))
    asyncio.run(client.explain_topic())
    asyncio.run(client.explain_topic(None, "memory"))
    sent = _sent(client)
    assert sent[0].args == ["plugin", "memory"]
    assert sent[1].args == ["", ""]
    assert sent[2].args == ["", "memory"], (
        "an absent topic must hold its position, or the name is read as one")


def test_there_is_no_workspace_parameter():
    """A read-only report must not become a second path ingress.

    A workspace-reading topic reads the caller's own workspace, which the
    daemon resolves from the session it is attached to or the workspace it
    declared — both entitlement-checked at the handshake.  A parameter here
    would let this verb name a directory that neither check ever saw.
    """
    import inspect
    params = inspect.signature(IPCClient.explain_topic).parameters
    assert "workspace" not in params
    assert set(params) == {"self", "topic", "name"}
