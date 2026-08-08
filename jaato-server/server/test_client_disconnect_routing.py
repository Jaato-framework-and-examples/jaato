"""Regression tests for the transport-agnostic client-disconnect helper.

Background: long-running daemons hit ``Errno 24 inotify instance limit``
because IPC client disconnects never propagated to
``SessionManager.detach_client``.  WS already had the call but IPC
did not — feature-parity gap.  Server 0.6.45 centralises the
detach into ``CommandRouter.handle_client_disconnect`` and wires it
from both transports.

These tests pin the wiring so the gap can't reopen silently.
"""

from unittest.mock import MagicMock

from server.command_router import CommandRouter
from server.ipc import JaatoIPCServer


def test_handle_client_disconnect_calls_session_manager_detach():
    """CommandRouter.handle_client_disconnect → SessionManager.detach_client."""
    session_manager = MagicMock()
    event_sink = MagicMock()
    router = CommandRouter(
        session_manager=session_manager,
        event_sink=event_sink,
        daemon_plugins={},
    )

    router.handle_client_disconnect("ipc_42")

    session_manager.detach_client.assert_called_once_with("ipc_42")


def test_ipc_server_accepts_on_client_disconnect_callback():
    """IPC server constructor exposes on_client_disconnect param.

    Pins the API so __main__.py's wiring stays valid.  The 0.6.45 fix
    required adding this constructor parameter; if it's renamed or
    removed without updating __main__.py, the daemon silently regresses
    to the leaky pre-0.6.45 behaviour.
    """
    cb = MagicMock()
    ipc = JaatoIPCServer(
        socket_path="/tmp/jaato-test-disconnect.sock",
        on_client_disconnect=cb,
    )
    assert ipc._on_client_disconnect is cb
