"""The Python SDK's memory verbs (#1232, protocol 1.22).

Four properties, each a way a caller could be misled:

* a daemon below 1.22 is REFUSED with nothing sent -- it would answer
  "Unknown request type" with no request_id and never the result the call
  waits on;
* the answer is matched by ``request_id``, so an unrelated event (or another
  caller's answer) arriving first is not handed back;
* a connection that closes before the answer raises, rather than returning an
  empty list that reads as "nothing remembered";
* approve / dismiss are ``update_memory`` with the maturity filled in.
"""

import asyncio

import pytest

from jaato_sdk.client.ipc import IPCClient
from jaato_sdk.client.recovery import IPCRecoveryClient
from jaato_sdk.events import (
    PROTOCOL_VERSION,
    AgentOutputEvent,
    ClientType,
    MemoryDeleteRequest,
    MemoryDeleteResultEvent,
    MemoryGetRequest,
    MemoryGetResultEvent,
    MemoryListEvent,
    MemoryListRequest,
    MemoryUpdateRequest,
    MemoryUpdateResultEvent,
)


def _client(spoken=PROTOCOL_VERSION):
    client = IPCClient("/tmp/does-not-exist.sock", client_type=ClientType.API,
                       auto_start=False)
    client._server_protocol_version = spoken
    return client


def _answering(client, make_answer, *, before=()):
    """Make ``client`` answer each sent request through the drain fan-out."""
    sent = []

    async def fake_send(event):
        sent.append(event)
        for ev in before:
            for q in list(client._event_subscribers):
                q.put_nowait(ev)
        answer = make_answer(event)
        if answer is not None:
            for q in list(client._event_subscribers):
                q.put_nowait(answer)
        return True

    client._send_event = fake_send  # type: ignore[assignment]
    return sent


def test_the_floor_is_the_version_that_introduced_the_verbs():
    assert IPCClient.MIN_MEMORY_VERBS_PROTOCOL == "1.22"
    assert PROTOCOL_VERSION.split(".")[0] == "1"
    assert tuple(int(p) for p in PROTOCOL_VERSION.split(".")) >= (1, 22)


@pytest.mark.parametrize("call", [
    lambda c: c.list_memories(),
    lambda c: c.get_memory("m1"),
    lambda c: c.update_memory("m1", description="x"),
    lambda c: c.approve_memory("m1"),
    lambda c: c.dismiss_memory("m1"),
    lambda c: c.delete_memory("m1"),
])
def test_an_older_daemon_is_refused_with_nothing_sent(call):
    client = _client("1.21")
    sent = _answering(client, lambda e: None)
    with pytest.raises(ValueError) as exc:
        asyncio.run(call(client))
    assert "1.21" in str(exc.value) and "1.22" in str(exc.value)
    assert sent == []


def test_the_list_answer_is_matched_by_request_id():
    client = _client()
    decoy = MemoryListEvent(request_id="someone-else", memories=[{"id": "x"}])

    def answer(event):
        assert isinstance(event, MemoryListRequest)
        assert event.request_id
        return MemoryListEvent(
            request_id=event.request_id, memories=[{"id": "mine"}],
            may_curate=True, source="runner",
        )

    _answering(client, answer, before=(
        AgentOutputEvent(source="model", text="hi", mode="write"), decoy))
    got = asyncio.run(client.list_memories(timeout=2))
    assert got.memories == [{"id": "mine"}]
    assert got.may_curate is True


def test_the_request_itself_is_never_mistaken_for_its_answer():
    """A daemon echo of the request (same request_id, same type) is not the
    answer -- only a result type is."""
    client = _client()

    def answer(event):
        return MemoryGetResultEvent(
            request_id=event.request_id, memory_id="m1",
            memory={"id": "m1", "content": "c"})

    _answering(client, answer, before=())
    got = asyncio.run(client.get_memory("m1", timeout=2))
    assert isinstance(got, MemoryGetResultEvent)
    assert got.memory["content"] == "c"


def test_a_closed_connection_raises_rather_than_answering_empty():
    client = _client()

    def closes(event):
        for q in list(client._event_subscribers):
            q.put_nowait(None)
        return None

    _answering(client, closes)
    with pytest.raises(ConnectionError):
        asyncio.run(client.list_memories(timeout=2))


def test_a_failed_send_raises_at_once():
    client = _client()

    async def fake_send(event):
        return False

    client._send_event = fake_send  # type: ignore[assignment]
    with pytest.raises(ConnectionError):
        asyncio.run(client.list_memories(timeout=30))


def test_no_answer_is_a_timeout_not_an_empty_list():
    client = _client()
    _answering(client, lambda e: None)
    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(client.list_memories(timeout=0.05))


@pytest.mark.parametrize("method,maturity", [
    ("approve_memory", "validated"),
    ("dismiss_memory", "dismissed"),
])
def test_approve_and_dismiss_are_maturity_updates(method, maturity):
    client = _client()
    sent = _answering(client, lambda e: MemoryUpdateResultEvent(
        request_id=e.request_id, memory_id=e.memory_id))
    asyncio.run(getattr(client, method)("m1", timeout=2))
    assert isinstance(sent[0], MemoryUpdateRequest)
    assert sent[0].maturity == maturity
    assert sent[0].description is None and sent[0].content is None


def test_delete_sends_the_delete_request():
    client = _client()
    sent = _answering(client, lambda e: MemoryDeleteResultEvent(
        request_id=e.request_id, memory_id=e.memory_id))
    got = asyncio.run(client.delete_memory("m1", timeout=2))
    assert isinstance(sent[0], MemoryDeleteRequest)
    assert got.memory_id == "m1"


def test_the_recovery_client_carries_every_verb():
    for name in ("list_memories", "get_memory", "update_memory",
                 "approve_memory", "dismiss_memory", "delete_memory"):
        assert callable(getattr(IPCRecoveryClient, name, None)), name
