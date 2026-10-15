"""Tests for ``SessionManager.is_cid_known`` (#1701).

The predicate lets a reactor tell a cascade it is creating (new cid →
register as owner) from one it is joining (known cid → register as
observer, since a driver created it and may own it).

Pins each of the four sources the daemon knows a cid from, the empty
cid, and the cid being forgotten once the cascade-client sweep has
reaped it.
"""

from __future__ import annotations

import threading
import time
from typing import Optional
from unittest.mock import MagicMock

import pytest

from jaato_server.server.session_manager import SessionManager


class _NoWakeBindings:
    def has_live_binding_for_cid(self, cid):
        return False


def _make_sm() -> SessionManager:
    """SessionManager skeleton with the attributes ``is_cid_known``,
    the cascade-client registry and the sweep touch.

    Mirrors ``test_cascade_as_client_phase1.py:_make_sm``.
    """
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
    sm._cancelled_cids = set()
    sm._cancelled_cids_lock = threading.Lock()
    sm._wake_binding_registry = _NoWakeBindings()
    return sm


def _make_session(session_id: str, cid: Optional[str] = None):
    session = MagicMock()
    session.session_id = session_id
    session.cascade_driver_id = cid
    return session


class TestUnknown:
    def test_fresh_daemon_knows_no_cid(self):
        sm = _make_sm()
        assert sm.is_cid_known("abc") is False

    @pytest.mark.parametrize("cid", ["", None])
    def test_empty_cid_is_unknown(self, cid):
        sm = _make_sm()
        # Even with a session lacking a cid loaded, an empty cid is
        # never "known".
        sm._sessions["s1"] = _make_session("s1", cid=None)
        assert sm.is_cid_known(cid) is False

    def test_other_cids_do_not_leak(self):
        sm = _make_sm()
        sm.register_in_process_client(
            client_id="obs", callback=MagicMock(),
            cascade_driver_id="abc", role="observer",
        )
        sm._sessions["s1"] = _make_session("s1", cid="abc")
        sm._record_cid_session_activity("abc")
        sm._cancelled_cids.add("abc")
        assert sm.is_cid_known("xyz") is False


class TestSources:
    @pytest.mark.parametrize("role", ["owner", "observer"])
    def test_registered_cascade_client(self, role):
        sm = _make_sm()
        sm.register_in_process_client(
            client_id=f"_cascade:abc:{role}", callback=MagicMock(),
            cascade_driver_id="abc", role=role,
        )
        assert sm.is_cid_known("abc") is True

    def test_unregistered_client_no_longer_counts(self):
        sm = _make_sm()
        sm.register_in_process_client(
            client_id="obs", callback=MagicMock(),
            cascade_driver_id="abc", role="observer",
        )
        sm.unregister_cascade_client("abc", "obs")
        assert sm.is_cid_known("abc") is False

    def test_loaded_session_stamped_with_cid(self):
        sm = _make_sm()
        sm._sessions["s1"] = _make_session("s1", cid="abc")
        assert sm.is_cid_known("abc") is True

    def test_session_created_then_unloaded(self):
        # The reactor's case: the triggering stage may already be
        # unloaded by the default policy when the action runs.
        sm = _make_sm()
        sm._sessions["s1"] = _make_session("s1", cid="abc")
        sm._record_cid_session_activity("abc")
        del sm._sessions["s1"]
        assert sm.is_cid_known("abc") is True

    def test_cancelled_cascade(self):
        sm = _make_sm()
        sm._cancelled_cids.add("abc")
        assert sm.is_cid_known("abc") is True


class TestForgotten:
    def test_forgotten_after_sweep_reaps_registration(self):
        sm = _make_sm()
        sm._cascade_client_idle_timeout = 0.05
        sm.register_in_process_client(
            client_id="obs", callback=MagicMock(),
            cascade_driver_id="abc", role="observer",
        )
        sm._record_cid_session_activity("abc")
        # Registration and last session both idle past the timeout,
        # and no session carries the cid.
        sm._cascade_clients["abc"][0].registered_at = time.monotonic() - 1.0
        sm._cid_last_session_ts["abc"] = time.monotonic() - 1.0
        assert sm.is_cid_known("abc") is True

        sm._cascade_client_sweep_once()

        assert "abc" not in sm._cascade_clients
        assert "abc" not in sm._cid_last_session_ts
        assert sm.is_cid_known("abc") is False

    def test_not_forgotten_while_a_session_carries_it(self):
        sm = _make_sm()
        sm._cascade_client_idle_timeout = 0.05
        sm.register_in_process_client(
            client_id="obs", callback=MagicMock(),
            cascade_driver_id="abc", role="observer",
        )
        sm._cascade_clients["abc"][0].registered_at = time.monotonic() - 1.0
        sm._sessions["s1"] = _make_session("s1", cid="abc")

        sm._cascade_client_sweep_once()

        assert sm.is_cid_known("abc") is True


class TestReactorRoleChoice:
    """The pattern jaato-premium's reactor follows: owner when the cid
    is new, observer when it is known."""

    @staticmethod
    def _register_as_reactor(sm: SessionManager, cid: str) -> str:
        role = "observer" if sm.is_cid_known(cid) else "owner"
        sm.register_in_process_client(
            client_id=f"_reactor:{cid}", callback=MagicMock(),
            cascade_driver_id=cid, role=role,
        )
        return role

    def test_creating_a_cascade_takes_owner(self):
        sm = _make_sm()
        assert self._register_as_reactor(sm, "new-cid") == "owner"

    def test_joining_a_driver_owned_cascade_takes_observer(self):
        sm = _make_sm()
        sm.register_in_process_client(
            client_id="_cascade:drv", callback=MagicMock(),
            cascade_driver_id="drv", role="owner",
        )
        sm._record_cid_session_activity("drv")
        # Without the check, a second owner registration raises.
        assert self._register_as_reactor(sm, "drv") == "observer"
        roles = sorted(e.role for e in sm._cascade_clients["drv"])
        assert roles == ["observer", "owner"]
