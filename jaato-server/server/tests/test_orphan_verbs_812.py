"""The IPC verbs that make an orphaned session actionable (#812).

``session.orphans`` lists the LOADED sessions nothing is consuming;
``session.stop <id>`` stops any one of them.  Both are shaped after the
existing session-management verbs (``session.list``, ``cascade.cancel``)
rather than as a parallel surface: the listing answers with the same
``SessionListEvent``, and the stop confirms with the same
``SystemMessageEvent`` / ``ErrorEvent`` pair ``cascade.cancel`` uses.

The confirmation wording is asserted because #812's operator had to choose
between killing a circumstantially-identified process and waiting for a
budget to burn: "cancelled mid-turn", "was idle" and "not loaded" are three
different answers and an operator acts differently on each.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from jaato_sdk.events import CommandRequest

from server.command_router import CommandRouter


def _make_router():
    """A CommandRouter with mocked collaborators.

    Mirrors ``test_cascade_cancel.py::TestCommandRouterCascadeCancel``: the
    router's only job here is argument validation, delegation and the
    operator-facing confirmation.
    """
    session_manager = MagicMock()
    event_sink = MagicMock()
    event_sink.get_client_workspace.return_value = "/ws"
    event_sink.get_client_user.return_value = None
    router = CommandRouter.__new__(CommandRouter)
    router._session_manager = session_manager
    router._event_sink = event_sink
    return router, session_manager, event_sink


def _sent(event_sink):
    """Every event the router sent, in order."""
    return [call[0][1] for call in event_sink.send_event.call_args_list]


class TestSessionOrphansVerb:
    def test_answers_with_the_orphan_rows(self):
        router, sm, sink = _make_router()
        sm.list_orphan_sessions.return_value = [
            {"session_id": "20260903_084517", "orphaned_seconds": 431.0,
             "is_processing": True, "runner": {"runner_pid": 4242}},
        ]
        router._handle_session_orphans("client_ipc_14")

        event = _sent(sink)[0]
        assert event.sessions[0]["session_id"] == "20260903_084517"
        assert event.sessions[0]["runner"]["runner_pid"] == 4242

    def test_an_empty_listing_is_an_answer_not_silence(self):
        router, sm, sink = _make_router()
        sm.list_orphan_sessions.return_value = []
        router._handle_session_orphans("c1")
        assert _sent(sink)[0].sessions == []

    def test_dispatch_routes_the_command(self):
        router, sm, sink = _make_router()
        sm.list_orphan_sessions.return_value = []
        router.handle_request("c1", "", CommandRequest(
            command="session.orphans", args=[]))
        sm.list_orphan_sessions.assert_called_once()


class TestSessionStopVerb:
    def test_stops_the_named_session(self):
        router, sm, sink = _make_router()
        sm.stop_session.return_value = {
            "session_id": "s1", "found": True, "stopped": True,
            "was_processing": True, "reason": "operator_request",
        }
        router._handle_session_stop("c1", ["s1"])

        sm.stop_session.assert_called_once_with(
            "s1", reason="operator_request")
        assert "mid-turn" in _sent(sink)[0].message

    def test_an_idle_session_is_reported_differently(self):
        router, sm, sink = _make_router()
        sm.stop_session.return_value = {
            "session_id": "s1", "found": True, "stopped": False,
            "was_processing": False, "reason": "operator_request",
        }
        router._handle_session_stop("c1", ["s1"])
        assert "idle" in _sent(sink)[0].message

    def test_an_unknown_session_says_so_rather_than_claiming_a_stop(self):
        router, sm, sink = _make_router()
        sm.stop_session.return_value = {
            "session_id": "nope", "found": False, "stopped": False,
            "was_processing": False, "reason": "operator_request",
        }
        router._handle_session_stop("c1", ["nope"])
        assert "not loaded" in _sent(sink)[0].message

    def test_a_missing_id_is_a_usage_error_naming_the_discovery_verb(self):
        router, sm, sink = _make_router()
        router._handle_session_stop("c1", [])

        event = _sent(sink)[0]
        assert event.error_type == "UsageError"
        assert "session.orphans" in event.error
        sm.stop_session.assert_not_called()

    def test_dispatch_routes_the_command(self):
        router, sm, sink = _make_router()
        sm.stop_session.return_value = {
            "session_id": "s1", "found": True, "stopped": False,
            "was_processing": False, "reason": "operator_request",
        }
        router.handle_request("c1", "", CommandRequest(
            command="session.stop", args=["s1"]))
        sm.stop_session.assert_called_once()


class TestSessionListCarriesTheNewFacts:
    def test_a_row_says_whether_anything_is_consuming_the_session(self):
        """A listing that shows a session and not whether anyone is
        watching it, nor which process runs it, is the listing #812's
        reporter had."""
        router, sm, sink = _make_router()
        row = MagicMock()
        row.session_id = "s1"
        row.name = "arm"
        row.description = ""
        row.model_provider = "openrouter"
        row.model_name = "m"
        row.is_loaded = True
        row.client_count = 0
        row.turn_count = 3
        row.workspace_path = "/ws"
        row.orphaned = True
        row.runner = {"runner_pid": 4242}
        sm.list_sessions.return_value = [row]

        router._handle_session_list("c1", "other")
        payload = _sent(sink)[0].sessions[0]
        assert payload["orphaned"] is True
        assert payload["runner"] == {"runner_pid": 4242}


class TestTheSdkRefusesAgainstAnOldDaemon:
    """An additive FIELD degrades harmlessly; a missing VERB does not.

    An older daemon does not recognise ``session.stop``, so the call is a
    silent no-op — and its caller concludes a runaway session has been
    stopped.  Same verdict as #845: refused, not degraded.
    """

    def _client(self, spoken):
        from jaato_sdk.client.ipc import IPCClient
        client = IPCClient.__new__(IPCClient)
        # ``server_protocol_version`` is a read-only property over this
        # attribute -- set the backing field, which is what ``connect()``
        # populates from the ConnectedEvent handshake.
        client._server_protocol_version = spoken
        return client

    @pytest.mark.asyncio
    async def test_an_old_daemon_is_refused(self):
        client = self._client("1.6")
        with pytest.raises(ValueError, match="does not serve"):
            await client.stop_session("s1")
        with pytest.raises(ValueError, match="does not serve"):
            await client.list_orphan_sessions()

    @pytest.mark.asyncio
    async def test_an_unknown_version_is_refused_too(self):
        """"I have not been told what this daemon can do" is not a licence
        to report a stop that may not have happened."""
        client = self._client(None)
        with pytest.raises(ValueError, match="unknown"):
            await client.stop_session("s1")

    @pytest.mark.asyncio
    async def test_an_empty_id_is_refused_before_the_protocol_check(self):
        client = self._client("1.7")
        with pytest.raises(ValueError, match="session_id is required"):
            await client.stop_session("")

    @pytest.mark.asyncio
    async def test_a_current_daemon_is_allowed_and_sends_the_verb(self):
        sent = []
        client = self._client("1.7")

        async def _capture(event):
            sent.append(event)
        client._send_event = _capture

        await client.stop_session("s1")
        await client.list_orphan_sessions()
        assert [e.command for e in sent] == ["session.stop", "session.orphans"]
        assert sent[0].args == ["s1"]

    def test_a_newer_daemon_satisfies_the_floor(self):
        from jaato_sdk.client.ipc import _protocol_compatible, IPCClient
        assert _protocol_compatible("1.8", IPCClient.MIN_SESSION_STOP_PROTOCOL)
        assert not _protocol_compatible(
            "1.6", IPCClient.MIN_SESSION_STOP_PROTOCOL)
