"""Phase 4 cascade-sharing — LSP plugin lifecycle pin tests.

Pins the contract that the cascade-sharing arc depends on:

  - ``reset_for_next_session()`` is a NO-OP — every piece of LSP
    plugin state survives so subsequent sessions of the same cascade
    don't pay the jdtls cold-start tax (~30-60s per stage on Maven
    workspaces) and the enrichment chain's ``_connected_servers``
    lookup keeps returning the right answer.

  - ``shutdown()`` clears EVERYTHING — used at slot-teardown
    (cascade-end) when the slot finally exits.  Without this, the
    runner subprocess would leak its LSP server child processes
    (jdtls, pyright, etc.) at cascade-end.

Phase 1a (#160) added the docstring source-pin test.  This file is
Phase 4's behaviour pin: explicit assertions on every state attribute
the LSP plugin holds, verified against a real-but-mocked plugin
instance.

Design reference: ``docs/design/runner-cascade-sharing.md`` §6 Phase 4.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from shared.plugins.lsp.plugin import LSPToolPlugin


# ======================================================================
# State preservation: reset_for_next_session is NO-OP
# ======================================================================


class TestResetForNextSessionPreservesAllState:
    """Per the docstring contract: every attribute survives reset.

    Each test seeds ONE attribute with a sentinel value, fires
    ``reset_for_next_session()``, asserts the attribute is unchanged.
    Distributing across multiple tests gives the failure surface a
    chance to localize ('which field broke?') rather than a single
    big assertion sweep.
    """

    def _plugin(self) -> LSPToolPlugin:
        return LSPToolPlugin()

    def test_clients_dict_preserved(self) -> None:
        """_clients carries already-connected LSPClient instances.
        Tearing them down between sessions re-triggers the jdtls
        cold-start tax (~30-60s per stage on Maven)."""
        p = self._plugin()
        sentinel_clients = {"java": MagicMock(name="jdtls_client"),
                            "python": MagicMock(name="pyright_client")}
        p._clients = dict(sentinel_clients)
        p.reset_for_next_session()
        assert p._clients == sentinel_clients
        # Identity check: same dict object — no rebuild.
        assert p._clients["java"] is sentinel_clients["java"]

    def test_connected_servers_set_preserved(self) -> None:
        """_connected_servers is the membership set the enrichment
        chain consults.  Clearing recreates the v151 multi-instance
        state-isolation bug."""
        p = self._plugin()
        p._connected_servers = {"java", "python"}
        p.reset_for_next_session()
        assert p._connected_servers == {"java", "python"}

    def test_config_cache_preserved(self) -> None:
        """_config_cache is parsed .lsp.json.  The workspace config
        doesn't change within a cascade (workspace_path is constant
        per design §3 decision 4)."""
        p = self._plugin()
        cached = {"servers": {"java": {"command": "jdtls"}}}
        p._config_cache = dict(cached)
        p.reset_for_next_session()
        assert p._config_cache == cached

    def test_background_thread_preserved(self) -> None:
        """The background thread owns LSP client lifecycles.  Joining
        + restarting it discards the in-flight request/response
        queues."""
        p = self._plugin()
        fake_thread = MagicMock(spec_set=["is_alive", "join", "start"])
        fake_thread.is_alive.return_value = True
        p._thread = fake_thread
        p.reset_for_next_session()
        assert p._thread is fake_thread
        fake_thread.join.assert_not_called()  # not torn down

    def test_request_response_queues_preserved(self) -> None:
        """Request/response queues are the runtime channels between
        the background thread and tool handlers."""
        p = self._plugin()
        rq = MagicMock(name="request_queue")
        respq = MagicMock(name="response_queue")
        p._request_queue = rq
        p._response_queue = respq
        p.reset_for_next_session()
        assert p._request_queue is rq
        assert p._response_queue is respq

    def test_initialized_flag_preserved(self) -> None:
        """_initialized=True means initialize() has run.  Re-running
        would re-spawn LSP server subprocesses and reset
        _connected_servers."""
        p = self._plugin()
        p._initialized = True
        p.reset_for_next_session()
        assert p._initialized is True

    def test_loop_reference_preserved(self) -> None:
        """The asyncio loop powering LSPClient I/O survives so
        existing in-flight tasks don't lose their loop reference."""
        p = self._plugin()
        fake_loop = MagicMock(name="asyncio_loop")
        p._loop = fake_loop
        p.reset_for_next_session()
        assert p._loop is fake_loop

    def test_errlog_preserved(self) -> None:
        """Stderr capture handle stays open across cascade sessions
        so the operator's diagnostic log doesn't truncate at every
        session boundary."""
        p = self._plugin()
        fake_errlog = MagicMock(name="errlog")
        p._errlog = fake_errlog
        p.reset_for_next_session()
        assert p._errlog is fake_errlog
        fake_errlog.close.assert_not_called()

    def test_failed_servers_preserved(self) -> None:
        """_failed_servers carries error reasons for LSP servers that
        couldn't connect.  Useful diagnostic context across cascade
        sessions; re-attempting connect within a cascade is governed
        by the retry-autoconnect signal, not by clearing this dict."""
        p = self._plugin()
        failures = {"rust-analyzer": "binary not found at /usr/bin/rust-analyzer"}
        p._failed_servers = dict(failures)
        p.reset_for_next_session()
        assert p._failed_servers == failures


# ======================================================================
# State teardown: shutdown clears EVERYTHING
# ======================================================================


class TestShutdownClearsAllState:
    """Symmetric to the reset NO-OP: shutdown() is the cascade-end
    bookend that fires when the slot finally exits.  All resources
    held across cascade sessions get released here."""

    def test_shutdown_clears_clients(self) -> None:
        p = LSPToolPlugin()
        p._clients = {"java": MagicMock()}
        p.shutdown()
        assert p._clients == {}

    def test_shutdown_clears_connected_servers(self) -> None:
        p = LSPToolPlugin()
        p._connected_servers = {"java", "python"}
        p.shutdown()
        assert p._connected_servers == set()

    def test_shutdown_clears_loop_thread_queues(self) -> None:
        p = LSPToolPlugin()
        fake_thread = MagicMock(spec_set=["is_alive", "join"])
        fake_thread.is_alive.return_value = False  # avoid join() blocking
        p._thread = fake_thread
        p._loop = MagicMock(name="loop")
        p._request_queue = MagicMock(name="rq")
        p._response_queue = MagicMock(name="respq")
        p.shutdown()
        assert p._thread is None
        assert p._loop is None
        assert p._request_queue is None
        assert p._response_queue is None

    def test_shutdown_resets_initialized_flag(self) -> None:
        p = LSPToolPlugin()
        p._initialized = True
        p.shutdown()
        assert p._initialized is False

    def test_shutdown_closes_errlog(self) -> None:
        p = LSPToolPlugin()
        fake_errlog = MagicMock(name="errlog")
        p._errlog = fake_errlog
        p.shutdown()
        assert p._errlog is None
        fake_errlog.close.assert_called_once()

    def test_shutdown_clears_failed_servers(self) -> None:
        p = LSPToolPlugin()
        p._failed_servers = {"rust-analyzer": "boom"}
        p.shutdown()
        assert p._failed_servers == {}


# ======================================================================
# Asymmetry pin: reset != shutdown
# ======================================================================


class TestResetShutdownAsymmetry:
    """The whole point of cascade sharing: reset and shutdown DIFFER.
    reset preserves state (NO-OP).  shutdown clears state.  Pin the
    asymmetry so a future refactor doesn't accidentally merge them."""

    def test_reset_does_not_clear_what_shutdown_clears(self) -> None:
        """Seed state, fire reset, verify state survives, then fire
        shutdown, verify state cleared.  Same plugin instance walks
        both methods to demonstrate the contract."""
        p = LSPToolPlugin()
        # Seed.
        client = MagicMock(name="jdtls_client")
        p._clients = {"java": client}
        p._connected_servers = {"java"}
        p._config_cache = {"servers": {"java": {"command": "jdtls"}}}
        p._initialized = True
        fake_thread = MagicMock(spec_set=["is_alive", "join"])
        fake_thread.is_alive.return_value = False
        p._thread = fake_thread

        # Reset — should be no-op.
        p.reset_for_next_session()
        assert p._clients == {"java": client}
        assert p._connected_servers == {"java"}
        assert p._config_cache != {}
        assert p._initialized is True
        assert p._thread is fake_thread

        # Shutdown — should clear.
        p.shutdown()
        assert p._clients == {}
        assert p._connected_servers == set()
        assert p._initialized is False
        assert p._thread is None
