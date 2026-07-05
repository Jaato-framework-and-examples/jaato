"""CommandRouter handlers for session.bind_wake / session.unbind_wake.

Pins: the bound session is always the CALLER'S own session (resolved from
client_id — hijack-proof), the outcome is forwarded via WakeBindResultEvent,
and a caller with no active session is refused without touching the registry.
"""

from unittest.mock import MagicMock

from server.command_router import CommandRouter
from server.wake_binding_registry import BindOutcome


def _router():
    r = CommandRouter.__new__(CommandRouter)
    r._session_manager = MagicMock()
    r._event_sink = MagicMock()
    return r


def _session(sid="s1", ws="/ws/bot"):
    s = MagicMock()
    s.session_id = sid
    s.workspace_path = ws
    return s


def test_bind_wake_binds_callers_own_session(tmp_path):
    r = _router()
    r._session_manager.get_client_session.return_value = _session("s1", "/ws/bot")
    r._session_manager.bind_wake.return_value = BindOutcome.OK
    r._session_manager.resolve_wake_binding.return_value = MagicMock(expires_at=1234.0)

    r._handle_session_bind_wake(
        "c1", ["github-pr:o/r#1"],
        {"wake_ref": "github-pr:o/r#1", "trust_keys": ["PEMKEY"]})

    # bound the CALLER's session_id + workspace (not anything caller-supplied)
    r._session_manager.bind_wake.assert_called_once_with(
        "github-pr:o/r#1", "s1", "/ws/bot", ["PEMKEY"], None)
    evt = r._event_sink.send_event.call_args[0][1]
    assert evt.wake_ref == "github-pr:o/r#1"
    assert evt.outcome == "ok"
    assert evt.expires_at == 1234.0


def test_bind_wake_normalizes_single_string_key(tmp_path):
    # A lone PEM string must NOT be split into single-char "keys".
    r = _router()
    r._session_manager.get_client_session.return_value = _session("s1", "/ws")
    r._session_manager.bind_wake.return_value = BindOutcome.OK
    r._session_manager.resolve_wake_binding.return_value = MagicMock(expires_at=1.0)

    r._handle_session_bind_wake("c1", [], {"wake_ref": "w", "trust_keys": "PEMKEY"})

    r._session_manager.bind_wake.assert_called_once_with("w", "s1", "/ws", ["PEMKEY"], None)


def test_bind_wake_non_iterable_keys_no_crash(tmp_path):
    r = _router()
    r._session_manager.get_client_session.return_value = _session("s1", "/ws")
    r._session_manager.bind_wake.return_value = BindOutcome.NO_KEYS

    # trust_keys as an int must not crash the handler — normalized to [].
    r._handle_session_bind_wake("c1", [], {"wake_ref": "w", "trust_keys": 123})

    r._session_manager.bind_wake.assert_called_once_with("w", "s1", "/ws", [], None)


def test_bind_wake_bad_ttl_coerced_to_none(tmp_path):
    r = _router()
    r._session_manager.get_client_session.return_value = _session("s1", "/ws")
    r._session_manager.bind_wake.return_value = BindOutcome.OK
    r._session_manager.resolve_wake_binding.return_value = MagicMock(expires_at=1.0)

    r._handle_session_bind_wake(
        "c1", [], {"wake_ref": "w", "trust_keys": ["k"], "ttl_seconds": "not-an-int"})

    # bad ttl → None (registry falls back to its default), never propagated raw
    assert r._session_manager.bind_wake.call_args[0][4] is None


def test_bind_wake_no_session_refused(tmp_path):
    r = _router()
    r._session_manager.get_client_session.return_value = None

    r._handle_session_bind_wake("c1", [], {"wake_ref": "w", "trust_keys": ["k"]})

    r._session_manager.bind_wake.assert_not_called()
    evt = r._event_sink.send_event.call_args[0][1]
    assert evt.outcome == "no_session"


def test_bind_wake_forwards_failure_outcome(tmp_path):
    r = _router()
    r._session_manager.get_client_session.return_value = _session()
    r._session_manager.bind_wake.return_value = BindOutcome.UNAUTHORIZED

    r._handle_session_bind_wake("c1", [], {"wake_ref": "w", "trust_keys": ["k"]})

    evt = r._event_sink.send_event.call_args[0][1]
    assert evt.outcome == "unauthorized"
    # no expiry echoed on failure
    assert evt.expires_at == 0.0


def test_unbind_wake_owner(tmp_path):
    r = _router()
    r._session_manager.get_client_session.return_value = _session("s1")
    r._session_manager.unbind_wake.return_value = BindOutcome.OK

    r._handle_session_unbind_wake("c1", ["w"], {"wake_ref": "w"})

    r._session_manager.unbind_wake.assert_called_once_with("w", "s1")
    evt = r._event_sink.send_event.call_args[0][1]
    assert evt.outcome == "ok"


def test_unbind_wake_no_session_refused(tmp_path):
    r = _router()
    r._session_manager.get_client_session.return_value = None

    r._handle_session_unbind_wake("c1", ["w"], {"wake_ref": "w"})

    r._session_manager.unbind_wake.assert_not_called()
    evt = r._event_sink.send_event.call_args[0][1]
    assert evt.outcome == "no_session"
