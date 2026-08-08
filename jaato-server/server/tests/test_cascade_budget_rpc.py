"""IPC verbs for cascade budgets: cascade.budget.set / .get / .clear.

The declaration surface a plain IPCClient needs — set_cascade_budget lives
on SessionManager and is unreachable in-process from an external driver.
"""

import json
from types import SimpleNamespace

import pytest

from server.command_router import CommandRouter


class _Sink:
    def __init__(self):
        self.events = []

    def send_event(self, client_id, event):
        self.events.append((client_id, event))

    def last(self):
        return self.events[-1][1]


class _SM:
    def __init__(self):
        import threading
        self._cascade_budgets = {}
        self._cascade_budgets_lock = threading.Lock()

    def __getattr__(self, name):
        from server.session_manager import SessionManager
        fn = getattr(SessionManager, name)
        return lambda *a, **k: fn(self, *a, **k)


def _router():
    sink, sm = _Sink(), _SM()
    r = SimpleNamespace(_event_sink=sink, _session_manager=sm)
    for m in ("_handle_cascade_budget_set", "_handle_cascade_budget_get",
              "_handle_cascade_budget_clear"):
        setattr(r, m, (lambda n: (lambda *a, **k:
                getattr(CommandRouter, n)(r, *a, **k)))(m))
    return r, sink, sm


def test_set_declares_the_pool():
    r, sink, sm = _router()
    r._handle_cascade_budget_set("c1", ["cid1"], {"limits": {"tokens": 12000}})
    assert sm.get_cascade_budget("cid1").remaining()["tokens"] == 12000.0
    assert "cascade.budget.set" in sink.last().message


def test_set_rejects_a_malformed_budget_without_crashing():
    r, sink, _ = _router()
    r._handle_cascade_budget_set("c1", ["cid1"], {"limits": {"bogus": 1}})
    assert sink.last().error_type == "UsageError"
    assert "unknown dimension" in sink.last().error


def test_set_rejects_an_empty_payload():
    r, sink, sm = _router()
    r._handle_cascade_budget_set("c1", ["cid1"], {})
    assert sink.last().error_type == "UsageError"
    assert sm.get_cascade_budget("cid1") is None


def test_set_requires_a_cid():
    r, sink, _ = _router()
    r._handle_cascade_budget_set("c1", [], {"limits": {"tokens": 1}})
    assert sink.last().error_type == "UsageError"


def test_get_reports_remaining_as_json():
    r, sink, _ = _router()
    r._handle_cascade_budget_set("c1", ["cid1"], {"limits": {"tokens": 12000}})
    r._handle_cascade_budget_get("c1", ["cid1"])
    body = json.loads(sink.last().message)
    assert body["declared"] is True
    assert body["remaining"]["tokens"] == 12000.0
    assert body["limits"]["tokens"] == 12000


def test_get_shows_depletion_between_stages():
    """The client-side witness: the pool visibly draining across the chain."""
    r, sink, sm = _router()
    r._handle_cascade_budget_set("c1", ["cid1"], {"limits": {"tokens": 12000}})
    sm.get_cascade_budget("cid1").spend(tokens=9300)
    r._handle_cascade_budget_get("c1", ["cid1"])
    body = json.loads(sink.last().message)
    assert body["remaining"]["tokens"] == 2700.0
    assert "tokens" in body["pressure"]


def test_get_on_an_uncapped_cid_says_so():
    r, sink, _ = _router()
    r._handle_cascade_budget_get("c1", ["nope"])
    assert json.loads(sink.last().message)["declared"] is False


def test_clear_drops_it():
    r, sink, sm = _router()
    r._handle_cascade_budget_set("c1", ["cid1"], {"limits": {"tokens": 1}})
    r._handle_cascade_budget_clear("c1", ["cid1"])
    assert sm.get_cascade_budget("cid1") is None


def test_set_accepts_a_degrade_ladder_too():
    r, sink, sm = _router()
    r._handle_cascade_budget_set("c1", ["cid1"], {
        "limits": {"tokens": 100},
        "degrade": [{"at": 100, "action": "abort"}]})
    assert len(sm.get_cascade_budget("cid1").config.degrade) == 1
