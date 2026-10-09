"""The SDK's half of the plugin-tool verbs (#1606).

Two client-side decisions, each checked without a daemon:

* **The 1.7 rule.**  A daemon below 1.37 ignores the requests in silence, and
  the caller would wait out its timeout for a call nobody ran, so the SDK
  refuses before anything reaches the socket.
* **Whose ASK is it.**  The call runs in a session the caller is not attached
  to, so its permission ASK arrives tagged with the call's ``request_id``
  (``origin_request_id``).  ``invoke_plugin_tool`` answers exactly its own --
  through ``on_permission``, or ``"n"`` when there is none, so the call is
  refused rather than left waiting on nobody -- and leaves anyone else's alone.
"""

import asyncio

import pytest

from jaato_sdk.client.ipc import IPCClient
from jaato_sdk.client.recovery import IPCRecoveryClient
from jaato_sdk.events import (
    ClientType,
    PermissionRequestedEvent,
    PermissionResponseRequest,
    PluginToolInvokeRequest,
    PluginToolInvokeResultEvent,
    PROTOCOL_VERSION,
)


def _client(spoken):
    client = IPCClient("/tmp/does-not-exist.sock", client_type=ClientType.API,
                       auto_start=False)
    client._server_protocol_version = spoken
    return client


@pytest.fixture
def sent(monkeypatch):
    """Capture what the client sends; answer an invoke like the daemon does."""
    log = []

    async def fake_send(self, event):
        log.append(event)
        if isinstance(event, PluginToolInvokeRequest):
            rid = event.request_id
            replies = [
                PermissionRequestedEvent(request_id="perm-other",
                                         origin_request_id="pti_someone_else"),
                PermissionRequestedEvent(request_id="perm-mine",
                                         origin_request_id=rid),
                PluginToolInvokeResultEvent(request_id=rid, ok=True, success=True),
            ]
            for q in list(self._event_subscribers):
                for reply in replies:
                    q.put_nowait(reply)
        return True

    monkeypatch.setattr(IPCClient, "_send_event", fake_send, raising=True)
    return log


def _answers(log):
    return [(e.request_id, e.response) for e in log
            if isinstance(e, PermissionResponseRequest)]


def test_the_floor_is_the_version_that_introduced_the_verbs():
    assert IPCClient.MIN_PLUGIN_TOOL_PROTOCOL == "1.37"
    _client(PROTOCOL_VERSION)._require_plugin_tool_protocol("x")


@pytest.mark.parametrize("method", ["describe_plugin_tool", "invoke_plugin_tool"])
def test_an_older_daemon_is_refused_with_nothing_sent(sent, method):
    client = _client("1.36")
    with pytest.raises(ValueError) as exc:
        asyncio.run(getattr(client, method)("template", "listAvailableTemplates"))
    assert "1.36" in str(exc.value) and "1.37" in str(exc.value)
    assert sent == []


def test_the_call_answers_its_own_ask_and_no_one_elses(sent):
    client = _client(PROTOCOL_VERSION)
    seen = []
    answer = asyncio.run(client.invoke_plugin_tool(
        "template", "renderTemplateToFile", {"x": 1},
        on_permission=lambda ev: (seen.append(ev.request_id), "y")[1]))
    assert answer.success is True
    assert seen == ["perm-mine"]
    assert _answers(sent) == [("perm-mine", "y")]


def test_an_ask_with_no_answerer_is_refused(sent):
    client = _client(PROTOCOL_VERSION)
    asyncio.run(client.invoke_plugin_tool("template", "renderTemplateToFile"))
    assert _answers(sent) == [("perm-mine", "n")]


def test_the_recovery_client_carries_both_verbs():
    for name in ("describe_plugin_tool", "invoke_plugin_tool"):
        assert callable(getattr(IPCRecoveryClient, name))
