"""The Python SDK's reference-claim verbs (protocol 1.32).

* a daemon below 1.32 is refused with nothing sent (the 1.7 missing-verb
  rule: it would drop the request, and "promoted" would describe a catalog
  nobody changed);
* each answer is matched by ``request_id``, so another event arriving first
  is not handed back;
* a promotion sends the typed request carrying the action and claim.
"""

import asyncio

import pytest

from jaato_sdk.client.ipc import IPCClient
from jaato_sdk.events import (
    PROTOCOL_VERSION,
    ClientType,
    ReferenceClaimsEvent,
    ReferenceClaimsRequest,
    ReferenceCurationRequest,
    ReferenceCurationResultEvent,
)


def _client(spoken=PROTOCOL_VERSION):
    client = IPCClient("/tmp/does-not-exist.sock", client_type=ClientType.API,
                       auto_start=False)
    client._server_protocol_version = spoken
    return client


def _answering(client, answer_for):
    sent = []

    async def fake_send(event):
        sent.append(event)
        for q in list(client._event_subscribers):
            q.put_nowait(answer_for(event, "someone-else"))
            q.put_nowait(answer_for(event, event.request_id))
        return True
    client._send_event = fake_send  # type: ignore[assignment]
    return sent


def test_the_floor_is_the_version_that_introduced_the_verbs():
    assert IPCClient.MIN_REFERENCE_CURATION_PROTOCOL == "1.32"
    assert tuple(int(p) for p in PROTOCOL_VERSION.split(".")) >= (1, 32)


@pytest.mark.parametrize("call", [
    lambda c: c.list_reference_claims(),
    lambda c: c.promote_reference_claim("c1"),
    lambda c: c.dismiss_reference_claim("c1"),
])
def test_an_older_daemon_is_refused_with_nothing_sent(call):
    client = _client("1.31")
    sent = _answering(client, lambda e, rid: ReferenceClaimsEvent(request_id=rid))
    with pytest.raises(ValueError, match="1.32"):
        asyncio.run(call(client))
    assert sent == []


def test_the_listing_is_correlated():
    client = _client()
    sent = _answering(client, lambda e, rid: ReferenceClaimsEvent(
        request_id=rid, claims=[{"claim_id": rid}], may_curate=True))
    answer = asyncio.run(client.list_reference_claims())
    assert isinstance(sent[0], ReferenceClaimsRequest)
    assert answer.request_id == sent[0].request_id
    assert answer.claims == [{"claim_id": sent[0].request_id}]


@pytest.mark.parametrize("method,action", [
    ("promote_reference_claim", "promote"),
    ("dismiss_reference_claim", "dismiss"),
])
def test_curation_sends_the_typed_request_and_returns_its_answer(method, action):
    client = _client()
    sent = _answering(client, lambda e, rid: ReferenceCurationResultEvent(
        request_id=rid, action=e.action, claim_id=e.claim_id))
    answer = asyncio.run(getattr(client, method)("c7"))
    assert isinstance(sent[0], ReferenceCurationRequest)
    assert (sent[0].action, sent[0].claim_id) == (action, "c7")
    assert answer.request_id == sent[0].request_id and answer.action == action
