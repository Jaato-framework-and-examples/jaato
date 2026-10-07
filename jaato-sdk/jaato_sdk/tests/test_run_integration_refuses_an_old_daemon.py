"""``run_integration`` refuses a daemon below 1.21 rather than waiting on it.

The sibling of ``explain_topic``'s guard (#1263).  The 1.7 rule applied to a
verb: an older daemon ignores ``scaffold.integration`` in silence, and silence
here is indistinguishable from *the skill was installed* — a caller that
reported an install would report one that never happened.  So the refusal is
raised client-side, before anything reaches the socket, naming the daemon's
spoken version, because the remedy is upgrading a process the reader may not
own.
"""

import asyncio
import inspect

import pytest

from jaato_sdk.client.ipc import IPCClient
from jaato_sdk.client.recovery import IPCRecoveryClient
from jaato_sdk.events import PROTOCOL_VERSION, ClientType


def _client(spoken):
    client = IPCClient("/tmp/does-not-exist.sock", client_type=ClientType.API,
                       auto_start=False)
    client._server_protocol_version = spoken
    return client


def _sent(client):
    return getattr(client, "_guard_sent", [])


@pytest.fixture(autouse=True)
def _capture(monkeypatch):
    async def fake_send(self, event):
        self._guard_sent = getattr(self, "_guard_sent", []) + [event]
    monkeypatch.setattr(IPCClient, "_send_event", fake_send, raising=True)


def test_the_floor_is_the_version_that_introduced_the_verb():
    assert IPCClient.MIN_SCAFFOLD_INTEGRATION_PROTOCOL == "1.21"


def test_a_daemon_at_the_floor_is_asked():
    client = _client(IPCClient.MIN_SCAFFOLD_INTEGRATION_PROTOCOL)
    asyncio.run(client.run_integration("claude-code"))
    assert len(_sent(client)) == 1
    assert _sent(client)[0].command == "scaffold.integration"
    assert _sent(client)[0].args == ["claude-code"]


def test_todays_protocol_is_at_or_above_the_floor():
    """A floor above the version this SDK speaks would refuse every daemon."""
    client = _client(PROTOCOL_VERSION)
    asyncio.run(client.run_integration("claude-code"))
    assert len(_sent(client)) == 1


def test_an_older_daemon_is_refused_with_nothing_sent():
    client = _client("1.20")
    with pytest.raises(ValueError) as exc:
        asyncio.run(client.run_integration("claude-code"))
    assert "1.20" in str(exc.value), (
        "the refusal must name what the daemon SPEAKS — the remedy is "
        "upgrading a process the reader may not own")
    assert "1.21" in str(exc.value)
    assert _sent(client) == [], (
        "nothing may reach a daemon that would ignore it: the caller would "
        "then report a skill installed that never was")


def test_a_client_that_never_connected_is_refused_too():
    client = _client(None)
    with pytest.raises(ValueError):
        asyncio.run(client.run_integration("claude-code"))


def test_the_recovery_client_carries_the_same_verb():
    """Parity: the recovery client forwards to the inner client, which is
    what raises against an old daemon."""
    assert hasattr(IPCRecoveryClient, "run_integration")
    params = inspect.signature(IPCRecoveryClient.run_integration).parameters
    assert set(params) == {"self", "name"}


def test_there_is_no_workspace_parameter():
    """A verb that writes into the workspace must not name a directory the
    daemon's entitlement checks never saw — it resolves the caller's own."""
    params = inspect.signature(IPCClient.run_integration).parameters
    assert set(params) == {"self", "name"}
