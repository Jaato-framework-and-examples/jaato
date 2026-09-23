"""The IPC transport carries its peer, and the two client-path
chokepoints refuse what that peer could not reach.

The unit half is ``shared/tests/test_peer_identity.py``; this is the
wiring — the three places a correct mechanism could still be inert:

1. ``JaatoIPCServer.get_client_user`` was a hardcoded ``None``, so the
   attribution plumbing above it (``created_by``, the ledger ``user_id``,
   the ``user.id`` span) could never fire on this transport however well
   #859 wired it.
2. ``CommandRouter._handle_set_workspace`` accepts ``args[0]`` and stores
   it; the daemon then opens it with its own credential.
3. ``SessionManager._apply_client_config`` accepts ``working_dir`` /
   ``config_root`` / ``env_file`` from the handshake, and a half-applied
   handshake is its own bug — so a refusal must apply NOTHING.

Every deny case here is paired with the same call under a peer that IS
entitled, because a refusal that would have happened anyway proves
nothing about the check.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from jaato_server.shared.peer_identity import PeerCredentials

OTHER_UID = 65534


def _other_peer() -> PeerCredentials:
    return PeerCredentials(uid=OTHER_UID, gid=OTHER_UID, pid=4321, username=None)


def _self_peer() -> PeerCredentials:
    return PeerCredentials(uid=os.getuid(), gid=os.getgid(), pid=os.getpid())


@pytest.fixture
def open_dir():
    """A directory rooted at ``/tmp`` so only its own mode decides.

    See the same fixture in ``shared/tests/test_peer_identity.py``:
    pytest's ``tmp_path`` sits under a ``0700`` parent, which would make
    every refusal here pass for a reason that has nothing to do with the
    code under test.
    """
    path = Path(tempfile.mkdtemp(prefix="jaato-peer-wire-"))
    os.chmod(path, 0o755)
    try:
        yield path
    finally:
        os.chmod(path, 0o755)
        shutil.rmtree(path, ignore_errors=True)


# ----------------------------------------------------------------------
# 1. The transport reports its peer, and renders it for attribution
# ----------------------------------------------------------------------


def _ipc_server_with_client(peer: Optional[PeerCredentials]) -> Any:
    """A ``JaatoIPCServer`` holding one fabricated connection.

    Constructed without starting a socket: everything under test reads
    ``self._clients``, and binding a real socket would make the test about
    the filesystem instead.
    """
    from jaato_server.server.ipc import IPCClientConnection, JaatoIPCServer

    server = JaatoIPCServer(socket_path="/tmp/does-not-need-to-exist.sock")
    server._clients["ipc_1"] = IPCClientConnection(
        reader=None, writer=None, client_id="ipc_1",
        session_id=None, connected_at="now", peer=peer,
    )
    return server


def test_ipc_reports_the_peer_it_captured() -> None:
    peer = _other_peer()
    server = _ipc_server_with_client(peer)
    assert server.get_client_peer("ipc_1") is peer


def test_ipc_renders_the_peer_as_the_client_user() -> None:
    """This is what makes ``Session.created_by`` and the ledger's
    ``user_id`` non-empty on IPC — it used to be a hardcoded ``None``."""
    server = _ipc_server_with_client(
        PeerCredentials(uid=1000, gid=1000, username="ana"),
    )
    assert server.get_client_user("ipc_1") == "ana"


def test_a_transport_with_no_peer_reports_none_for_both() -> None:
    """Windows pipes and non-Linux sockets: unchanged from before, and
    #859's consumers omit the key rather than writing a ``None``."""
    server = _ipc_server_with_client(None)
    assert server.get_client_peer("ipc_1") is None
    assert server.get_client_user("ipc_1") is None


def test_an_unknown_client_is_not_an_error() -> None:
    server = _ipc_server_with_client(_other_peer())
    assert server.get_client_peer("ipc_99") is None
    assert server.get_client_user("ipc_99") is None


def test_set_client_user_cannot_override_the_kernel() -> None:
    """The one transport where identity is checkable must not accept a
    client's claim to be somebody else."""
    server = _ipc_server_with_client(
        PeerCredentials(uid=1000, gid=1000, username="ana"),
    )
    server.set_client_user("ipc_1", "root")
    assert server.get_client_user("ipc_1") == "ana"


def test_the_websocket_adapter_reports_no_peer() -> None:
    """A WS client may be on another machine, so there is no local account
    for the kernel to vouch for — and the guards must read that as "not
    applicable", never as a denial."""
    from jaato_server.server.websocket import WSEventSinkAdapter

    adapter = WSEventSinkAdapter.__new__(WSEventSinkAdapter)
    assert adapter.get_client_peer("ws_1") is None


def test_the_composite_sink_asks_every_transport() -> None:
    """And tolerates a sink predating the method — an out-of-tree
    transport contributes ``None`` rather than raising, because an absent
    peer is a valid answer here."""
    from jaato_server.server.event_sink import CompositeEventSink

    class _Old:
        """A sink with no ``get_client_peer`` at all."""

    class _New:
        def get_client_peer(self, client_id: str) -> Optional[PeerCredentials]:
            return _other_peer() if client_id == "ipc_1" else None

    composite = CompositeEventSink()
    composite.add_sink(_Old())       # type: ignore[arg-type]
    composite.add_sink(_New())       # type: ignore[arg-type]
    assert composite.get_client_peer("ipc_1") is not None
    assert composite.get_client_peer("ws_1") is None


# ----------------------------------------------------------------------
# 2. set_workspace
# ----------------------------------------------------------------------


class _RecordingSink:
    """An ``EventSink`` that records instead of delivering."""

    def __init__(self, peer: Optional[PeerCredentials]) -> None:
        self._peer = peer
        self.events: List[Any] = []
        self.workspaces: Dict[str, str] = {}

    def send_event(self, client_id: str, event: Any) -> None:
        self.events.append(event)

    def set_client_workspace(self, client_id: str, workspace_path: str) -> None:
        self.workspaces[client_id] = workspace_path

    def get_client_workspace(self, client_id: str) -> Optional[str]:
        return self.workspaces.get(client_id)

    def get_client_user(self, client_id: str) -> Optional[str]:
        return self._peer.identity if self._peer else None

    def get_client_peer(self, client_id: str) -> Optional[PeerCredentials]:
        return self._peer

    def set_client_user(self, client_id: str, user_id: str) -> None:
        pass

    def set_client_session(self, client_id: str, session_id: str) -> None:
        pass

    def broadcast_event(self, event: Any) -> None:
        pass


def _router(sink: _RecordingSink) -> Any:
    from jaato_server.server.command_router import CommandRouter

    router = CommandRouter.__new__(CommandRouter)
    router._event_sink = sink
    return router


def test_set_workspace_refuses_a_path_the_peer_cannot_reach(
    open_dir: Path,
) -> None:
    os.chmod(open_dir, 0o700)
    sink = _RecordingSink(_other_peer())
    _router(sink)._handle_set_workspace("ipc_1", [str(open_dir)])

    assert sink.workspaces == {}, "the workspace must not be stored"
    assert len(sink.events) == 1
    assert sink.events[0].error_type == "PeerPathNotReachable"
    assert str(open_dir) in sink.events[0].error


def test_set_workspace_accepts_the_same_path_when_the_peer_can_reach_it(
    open_dir: Path,
) -> None:
    """The control: identical call, identical peer, one chmod apart."""
    os.chmod(open_dir, 0o755)
    sink = _RecordingSink(_other_peer())
    _router(sink)._handle_set_workspace("ipc_1", [str(open_dir)])

    assert sink.workspaces == {"ipc_1": str(open_dir)}
    assert sink.events == []


def test_set_workspace_is_unchanged_with_no_peer(open_dir: Path) -> None:
    """A transport that reports no peer behaves exactly as it did before
    this check existed."""
    os.chmod(open_dir, 0o700)
    sink = _RecordingSink(None)
    _router(sink)._handle_set_workspace("ipc_1", [str(open_dir)])

    assert sink.workspaces == {"ipc_1": str(open_dir)}
    assert sink.events == []


def test_set_workspace_is_unchanged_for_the_daemons_own_account(
    open_dir: Path,
) -> None:
    """A connection from the daemon's own uid is skipped rather than
    satisfied — it could reach this path by simpler means anyway.  Note
    this is decided per CONNECTION, so it says nothing about how the
    daemon treats anybody else."""
    os.chmod(open_dir, 0o700)
    sink = _RecordingSink(_self_peer())
    _router(sink)._handle_set_workspace("ipc_1", [str(open_dir)])

    assert sink.workspaces == {"ipc_1": str(open_dir)}
    assert sink.events == []


# ----------------------------------------------------------------------
# 3. the client-config handshake
# ----------------------------------------------------------------------


def _manager() -> Any:
    from jaato_server.server.session_manager import SessionManager

    manager = SessionManager.__new__(SessionManager)
    manager._emitted = []                                   # type: ignore[attr-defined]
    manager._emit_to_client = (                             # type: ignore[assignment]
        lambda client_id, event: manager._emitted.append(event)
    )
    return manager


def _config(**fields: Any) -> Any:
    from jaato_sdk.events import ClientConfigRequest

    return ClientConfigRequest(**fields)


def test_the_handshake_is_refused_and_nothing_is_applied(
    open_dir: Path,
) -> None:
    """A half-applied handshake — a good ``working_dir`` beside a dropped
    ``config_root`` — is its own silent-wrong-directory bug, so the
    refusal is all-or-nothing."""
    os.chmod(open_dir, 0o700)
    manager = _manager()
    refused = manager._reject_unentitled_client_paths(
        "ipc_1", _config(working_dir=str(open_dir)), _other_peer(),
    )

    assert refused is True
    assert len(manager._emitted) == 1
    assert manager._emitted[0].error_type == "PeerPathNotReachable"


def test_the_handshake_passes_when_the_peer_can_reach_it(
    open_dir: Path,
) -> None:
    os.chmod(open_dir, 0o755)
    manager = _manager()
    refused = manager._reject_unentitled_client_paths(
        "ipc_1", _config(working_dir=str(open_dir)), _other_peer(),
    )

    assert refused is False
    assert manager._emitted == []


def test_every_path_field_is_checked_not_just_working_dir(
    open_dir: Path,
) -> None:
    """``config_root`` is the sharper one: it selects which
    ``<provider>_auth.json`` the session resolves."""
    private = open_dir / "private"
    private.mkdir()
    os.chmod(private, 0o700)
    os.chmod(open_dir, 0o755)

    manager = _manager()
    refused = manager._reject_unentitled_client_paths(
        "ipc_1",
        _config(working_dir=str(open_dir), config_root=str(private)),
        _other_peer(),
    )

    assert refused is True
    assert "config_root" in manager._emitted[0].error


def test_no_peer_leaves_the_handshake_alone(open_dir: Path) -> None:
    os.chmod(open_dir, 0o700)
    manager = _manager()
    assert manager._reject_unentitled_client_paths(
        "ipc_1", _config(working_dir=str(open_dir)), None,
    ) is False
    assert manager._emitted == []
