"""The Python SDK's paged history verb (protocol 1.28).

* a daemon below 1.28 is refused with nothing sent (the 1.7 missing-verb
  rule: it would answer "Unknown request type", never the page);
* the answer is matched by ``request_id``, so another event arriving first
  is not handed back;
* the request carries the cursor and budget it was given.
"""

import asyncio

import pytest

from jaato_sdk.client.ipc import IPCClient
from jaato_sdk.events import (
    PROTOCOL_VERSION,
    AgentOutputEvent,
    ClientType,
    HistoryPageEvent,
    HistoryPageRequest,
)


def _client(spoken=PROTOCOL_VERSION):
    client = IPCClient("/tmp/does-not-exist.sock", client_type=ClientType.API,
                       auto_start=False)
    client._server_protocol_version = spoken
    return client


def test_the_floor_is_the_version_that_introduced_the_verb():
    assert IPCClient.MIN_HISTORY_PAGE_PROTOCOL == "1.28"
    assert tuple(int(p) for p in PROTOCOL_VERSION.split(".")) >= (1, 28)


def test_an_older_daemon_is_refused_with_nothing_sent():
    client = _client("1.27")
    sent = []

    async def fake_send(event):
        sent.append(event)
        return True
    client._send_event = fake_send  # type: ignore[assignment]
    with pytest.raises(ValueError, match="1.28"):
        asyncio.run(client.request_history_page())
    assert sent == []


def test_the_answer_is_correlated_and_the_request_carries_cursor_and_budget():
    client = _client()
    sent = []

    async def fake_send(event):
        sent.append(event)
        for q in list(client._event_subscribers):
            q.put_nowait(AgentOutputEvent(agent_id="main", source="model",
                                          text="noise", mode="write"))
            q.put_nowait(HistoryPageEvent(request_id="someone-else"))
            q.put_nowait(HistoryPageEvent(request_id=event.request_id,
                                          before="3:abc", has_more=True))
        return True
    client._send_event = fake_send  # type: ignore[assignment]

    page = asyncio.run(client.request_history_page(
        "main", before="9:def", max_lines=50))
    assert isinstance(sent[0], HistoryPageRequest)
    assert sent[0].before == "9:def" and sent[0].max_lines == 50
    assert page.request_id == sent[0].request_id and page.before == "3:abc"
