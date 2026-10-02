"""The Python SDK's runner-pool verbs (protocol 1.35).

* a daemon below 1.35 is REFUSED with nothing sent: it would never answer,
  and "resized" would describe a pool nobody changed;
* ``resize_pool`` sends the sizes it was given and returns the answer that
  carries its ``request_id``;
* ``pool_status`` is the same request with no sizes, so it only reads.
"""

import asyncio

import pytest

from jaato_sdk.client.ipc import IPCClient
from jaato_sdk.events import (
    PROTOCOL_VERSION,
    ClientType,
    PoolStatusEvent,
    PoolStatusRequest,
)


def _client(spoken=PROTOCOL_VERSION):
    client = IPCClient("/tmp/does-not-exist.sock", client_type=ClientType.API,
                       auto_start=False)
    client._server_protocol_version = spoken
    return client


def _answering(client):
    sent = []

    async def fake_send(event):
        sent.append(event)
        answer = PoolStatusEvent(request_id=event.request_id,
                                 target_size=event.target_size or 2)
        for q in list(client._event_subscribers):
            q.put_nowait(PoolStatusEvent(request_id="someone_else"))
            q.put_nowait(answer)
        return True

    client._send_event = fake_send  # type: ignore[assignment]
    return sent


def test_an_older_daemon_is_refused_with_nothing_sent():
    client = _client("1.34")
    sent = _answering(client)
    with pytest.raises(ValueError, match="1.35"):
        asyncio.run(client.resize_pool(4))
    assert sent == []


def test_resize_sends_the_sizes_and_returns_its_own_answer():
    client = _client()
    sent = _answering(client)
    answer = asyncio.run(client.resize_pool(4, 9))
    (request,) = sent
    assert isinstance(request, PoolStatusRequest)
    assert (request.target_size, request.max_size) == (4, 9)
    assert answer.request_id == request.request_id
    assert answer.target_size == 4


def test_status_sends_no_sizes():
    client = _client()
    sent = _answering(client)
    asyncio.run(client.pool_status())
    assert (sent[0].target_size, sent[0].max_size) == (None, None)
