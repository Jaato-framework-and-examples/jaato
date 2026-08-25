"""A refused spawn must reach the cascade stream, not only the requester.

``_emit_cascade_refusal`` exists because, without it, "the cascade is out of
budget" and "the daemon hung" are indistinguishable from outside — and they
want opposite responses (finish gracefully vs escalate).

It emitted to the REQUESTING CLIENT only.  Two audiences were missed:

- a cascade **observer** watching the cid — the design's own observation
  surface, and the audience a *cascade-budget* refusal is most obviously for.
  It saw a session that never appeared, and no reason.
- a **headless** spawn (reactor- or cascade-driven) carries the synthetic
  ``_HEADLESS_CLIENT_ID``, so the emit reached nobody and the refusal existed
  only as a log line.

Reported by the cascade-coordination probe, whose driver hung fifteen minutes
waiting on a completion from a session that had never been allowed to exist.
"""

import threading

import pytest

from jaato_sdk.events import ErrorEvent
from server.session_manager import SessionManager


CID = "cid-1"


class _Refusal(Exception):
    """Shaped like ``CascadeExhaustedError`` — carries the cid itself."""

    def __init__(self, cid=CID):
        super().__init__("cascade has no headroom left on tokens")
        self.cascade_driver_id = cid

    def as_payload(self):
        return {
            "cascade_driver_id": self.cascade_driver_id,
            "reason": "cascade_budget_exhausted",
            "exhausted_dimensions": ["tokens"],
            "cascade_remaining": {"tokens": 0.0},
        }


class _NoWakeBindings:
    def has_live_binding_for_cid(self, cid):
        return False


def _sm():
    sm = SessionManager.__new__(SessionManager)
    sm._sessions = {}
    sm._lock = threading.RLock()
    sm._client_to_session = {}
    sm._cascade_clients = {}
    sm._cascade_clients_lock = threading.Lock()
    sm._cascade_client_idle_timeout = 300.0
    sm._cascade_client_sweep_stop = None
    sm._cascade_client_sweep_thread = None
    sm._cid_last_session_ts = {}
    sm._HEADLESS_CLIENT_ID = "_headless"
    sm._pending_wakes = {}
    sm._wake_binding_registry = _NoWakeBindings()
    sm.to_client = []
    sm._event_callback = lambda cid_, ev: sm.to_client.append((cid_, ev))
    return sm


def _observer(sm, client_id="obs"):
    seen = []
    sm.register_in_process_client(
        client_id=client_id, callback=seen.append,
        cascade_driver_id=CID, role="observer",
    )
    return seen


# ----------------------------------------------------------------------

def test_the_observer_is_told_a_spawn_was_refused():
    """The gap: a cascade-budget refusal never reached the cascade stream."""
    sm = _sm()
    seen = _observer(sm)
    sm._emit_cascade_refusal("client-1", "sess-refused", _Refusal())

    assert len(seen) == 1, "the cascade observer saw nothing"
    assert seen[0].error_type == "CascadeExhaustedError"
    assert seen[0].recoverable is False


def test_the_observer_learns_WHICH_session_was_refused():
    """On a shared stream, "a spawn was refused" is not actionable alone."""
    sm = _sm()
    seen = _observer(sm)
    sm._emit_cascade_refusal("client-1", "sess-refused", _Refusal())
    assert seen[0].session_id == "sess-refused"


def test_the_structured_evidence_survives_to_the_observer():
    """The point of ``details`` is branching without parsing prose."""
    sm = _sm()
    seen = _observer(sm)
    sm._emit_cascade_refusal("client-1", "sess-refused", _Refusal())
    d = seen[0].details
    assert d["reason"] == "cascade_budget_exhausted"
    assert d["exhausted_dimensions"] == ["tokens"]
    assert d["cascade_remaining"] == {"tokens": 0.0}


def test_the_requesting_client_is_still_told():
    """The pre-existing audience must not be traded for the new one."""
    sm = _sm()
    _observer(sm)
    sm._emit_cascade_refusal("client-1", "sess-refused", _Refusal())
    assert [c for c, _ in sm.to_client] == ["client-1"]


def test_a_headless_spawn_still_reaches_the_cascade():
    """A reactor-driven child has no real requester.

    Its refusal used to exist only as a log line; the cascade stream is the
    only audience it can have.
    """
    sm = _sm()
    seen = _observer(sm)
    sm._emit_cascade_refusal(sm._HEADLESS_CLIENT_ID, "sess-refused", _Refusal())
    assert len(seen) == 1


def test_the_requester_is_not_told_twice_when_it_is_also_an_observer():
    """A driver that watches its own cascade must not see it doubled."""
    sm = _sm()
    seen = []
    sm.register_in_process_client(
        client_id="_cascade:obs", callback=seen.append,
        cascade_driver_id=CID, role="observer",
        delivery_target_id="client-1",
    )
    sm._emit_cascade_refusal("client-1", "sess-refused", _Refusal())
    assert [c for c, _ in sm.to_client] == ["client-1"]
    assert seen == [], "delivered twice on the same connection"


def test_an_observer_of_a_DIFFERENT_cascade_is_not_told():
    """The cid is the blast radius — refusals do not leak across cascades."""
    sm = _sm()
    seen = []
    sm.register_in_process_client(
        client_id="other", callback=seen.append,
        cascade_driver_id="cid-OTHER", role="observer",
    )
    sm._emit_cascade_refusal("client-1", "sess-refused", _Refusal())
    assert seen == []


def test_a_refusal_without_a_cid_does_not_crash():
    """Not every refusal shape carries one; delivery is best-effort."""
    sm = _sm()
    seen = _observer(sm)
    sm._emit_cascade_refusal("client-1", "sess-refused", _Refusal(cid=None))
    assert seen == []
    assert [c for c, _ in sm.to_client] == ["client-1"]
