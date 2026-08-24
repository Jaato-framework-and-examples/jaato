"""A create must be answered by ITS OWN event, not by whichever arrives.

``_await_session_info`` matched on SHAPE — any ``SessionInfoEvent`` carrying a
session_id — and ``_subscribe_events`` drains the buffered-event list into every
new subscription.  So a stale event from an EARLIER create satisfied a LATER
wait and returned an id the caller never created.  Reproduced on demand by the
cascade-coordination example via a refused ``sibling_name``: same id returned
twice, one refusal in the daemon log between them.  Latent for any two rapid
``session.new`` calls.

The fix correlates: the client mints a ``request_id`` into the existing generic
``payload``, the daemon echoes it on whichever event answers, and the wait
accepts only events bearing it.  ``ErrorEvent`` echoes it too — otherwise, with
two creates in flight and one refused, the SUCCEEDING caller could observe the
other's failure.
"""
import asyncio
import inspect

import pytest

from jaato_sdk.client.ipc import IPCClient, _protocol_compatible
from jaato_sdk.events import (
    ErrorEvent, PROTOCOL_VERSION, SessionInfoEvent,
)

RID = "req_deadbeefcafe"


def _client(protocol="1.1"):
    c = IPCClient.__new__(IPCClient)
    c._server_protocol_version = protocol
    return c


# ------------------------------------------------------------- the wire

def test_create_session_puts_a_request_id_on_the_wire():
    cap = {}

    async def _send(ev):
        cap["payload"] = ev.payload
        raise _Stop()

    class _Stop(Exception):
        pass

    c = IPCClient.__new__(IPCClient)
    c._send_event = _send
    try:
        asyncio.run(IPCClient.create_session(c, "n"))
    except _Stop:
        pass
    rid = (cap["payload"] or {}).get("request_id")
    assert rid and rid.startswith("req_"), (
        "no correlation id was sent, so the daemon has nothing to echo and the "
        "wait falls back to matching on shape")


def test_the_id_rides_the_existing_payload_not_a_new_field():
    """``payload`` is the documented generic escape hatch; no schema change."""
    src = inspect.getsource(IPCClient.create_session)
    assert 'payload["request_id"] = req_id' in src


# ------------------------------------------------------- the correlation

def test_my_own_answer_is_accepted():
    assert _client()._correlates(
        SessionInfoEvent(session_id="mine", request_id=RID), RID)


def test_another_calls_answer_is_rejected():
    assert not _client()._correlates(
        SessionInfoEvent(session_id="theirs", request_id="req_other"), RID), (
        "a concurrent create's answer satisfied this wait")


def test_a_stale_unstamped_event_is_rejected_on_a_current_daemon():
    """The exact bug: a buffered event from an earlier create."""
    assert not _client()._correlates(
        SessionInfoEvent(session_id="stale"), RID)


def test_another_calls_ERROR_is_rejected():
    """Two creates in flight, one refused — the other must not see it."""
    assert not _client()._correlates(
        ErrorEvent(error="refused", request_id="req_other"), RID)


def test_my_own_error_is_accepted():
    assert _client()._correlates(ErrorEvent(error="refused", request_id=RID), RID)


def test_no_request_id_means_no_correlation_asked_for():
    assert _client()._correlates(SessionInfoEvent(session_id="any"), None)


# --------------------------------------------------------- version skew

def test_an_older_protocol_falls_back_rather_than_hanging():
    """Requiring the echo unconditionally would hang every create."""
    assert _client(protocol="1.0")._correlates(
        SessionInfoEvent(session_id="stale"), RID)


def test_the_fallback_warns_once(caplog):
    c = _client(protocol="1.0")
    with caplog.at_level("WARNING"):
        c._correlates(SessionInfoEvent(session_id="a"), RID)
        c._correlates(SessionInfoEvent(session_id="b"), RID)
    hits = [r for r in caplog.records if "predates session.new request" in r.getMessage()]
    assert len(hits) == 1, (
        "silent fallback leaves the bug in place with nothing to notice; "
        "warning per event trains the reader to skim")


def test_the_gate_is_the_PROTOCOL_version_not_the_package_version():
    """``server_version`` is diagnostics-only and says nothing about wire shape."""
    src = inspect.getsource(IPCClient._correlates)
    assert "server_protocol_version" in src
    assert "self.server_version" not in src


def test_the_protocol_minor_was_bumped_for_the_additive_field():
    assert PROTOCOL_VERSION == "1.1"
    assert _protocol_compatible(PROTOCOL_VERSION, "1.0"), (
        "clients declaring 1.0 must still connect — the field is additive")
