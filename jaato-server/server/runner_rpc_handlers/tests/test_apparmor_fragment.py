"""Tests for the daemon-side ``apparmor.add_reference_fragment`` handler.

Phase 3 §3.2.2.

Three test surfaces:
1. Handler-only (with a stub AppArmorManager).
2. End-to-end through the bidirectional RPC channel from §3.2.
3. Validation-rejection cases (relative path, glob metacharacters,
   missing args, session-id mismatch).
"""

from __future__ import annotations

import asyncio
import socket
import threading
from typing import Any, Dict, List, Optional, Tuple

import pytest

from server.runner.rpc import RunnerRPC
from server.runner.rpc_client import (
    RunnerRPCClient as RunnerSideClient,
    RunnerRPCError,
)
from server.runner_rpc_client import RunnerRPCClient as DaemonRPCClient
from server.runner_rpc_handlers.apparmor_fragment import (
    AppArmorFragmentHandler,
    register as register_apparmor_fragment,
)
from server.runner_rpc_server import RunnerRPCServer


# ----------------------------------------------------------------------
# Stub AppArmor manager
# ----------------------------------------------------------------------


class _StubAppArmor:
    """Minimal stand-in mirroring the methods the handler touches.

    Records calls + supports validator-failure / load-failure paths.
    """

    # AppArmor rule globs that the real validator rejects.
    _APPARMOR_GLOB_CHARS = frozenset("[]{}*?\\")

    def __init__(
        self,
        *,
        validation_error: Optional[str] = None,
        load_succeeds: bool = True,
        load_raises: Optional[BaseException] = None,
    ) -> None:
        self._validation_error = validation_error
        self._load_succeeds = load_succeeds
        self._load_raises = load_raises
        self.calls: List[Tuple[str, str, str]] = []

    def _validate_path_for_fragment(self, path: str) -> Optional[str]:
        if self._validation_error is not None:
            return self._validation_error
        # Mirror the real validator's checks.
        if not path:
            return "path is empty"
        if not path.startswith("/"):
            return f"path must be absolute: {path!r}"
        if "\n" in path or "\r" in path:
            return f"path contains newline/CR: {path!r}"
        bad = sorted(set(path) & self._APPARMOR_GLOB_CHARS)
        if bad:
            return (
                f"path contains AppArmor glob metacharacter(s) "
                f"{bad!r}: {path!r}"
            )
        return None

    def add_reference_fragment(
        self, session_id: str, ref_id: str, path: str,
    ) -> bool:
        self.calls.append((session_id, ref_id, path))
        if self._load_raises is not None:
            raise self._load_raises
        return self._load_succeeds


# ----------------------------------------------------------------------
# Handler in isolation
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handler_happy_path_loads_fragment() -> None:
    aa = _StubAppArmor()
    handler = AppArmorFragmentHandler(aa, session_id="sess-A")
    result = await handler.handle({
        "ref_id": "ref-1", "path": "/srv/data/x.txt",
    })
    assert result == {"ok": True}
    assert aa.calls == [("sess-A", "ref-1", "/srv/data/x.txt")]


@pytest.mark.asyncio
async def test_handler_rejects_relative_path() -> None:
    aa = _StubAppArmor()
    handler = AppArmorFragmentHandler(aa, session_id="sess-A")
    result = await handler.handle({
        "ref_id": "ref-1", "path": "relative/x.txt",
    })
    assert result["ok"] is False
    assert "absolute" in result["error"]
    # Should NOT have called the manager.
    assert aa.calls == []


@pytest.mark.asyncio
async def test_handler_rejects_glob_metacharacters() -> None:
    aa = _StubAppArmor()
    handler = AppArmorFragmentHandler(aa, session_id="sess-A")
    result = await handler.handle({
        "ref_id": "ref-1", "path": "/srv/data/file*.txt",
    })
    assert result["ok"] is False
    assert "glob metacharacter" in result["error"]
    assert aa.calls == []


@pytest.mark.asyncio
async def test_handler_load_returning_false_propagates() -> None:
    aa = _StubAppArmor(load_succeeds=False)
    handler = AppArmorFragmentHandler(aa, session_id="sess-A")
    result = await handler.handle({
        "ref_id": "r", "path": "/p/x",
    })
    assert result["ok"] is False
    assert "returned False" in result["error"]


@pytest.mark.asyncio
async def test_handler_load_exception_returns_error_dict() -> None:
    aa = _StubAppArmor(load_raises=PermissionError("no parser"))
    handler = AppArmorFragmentHandler(aa, session_id="sess-A")
    result = await handler.handle({
        "ref_id": "r", "path": "/p/x",
    })
    assert result["ok"] is False
    assert "PermissionError" in result["error"]


@pytest.mark.asyncio
async def test_handler_missing_ref_id_raises() -> None:
    handler = AppArmorFragmentHandler(_StubAppArmor(), session_id="s")
    with pytest.raises(ValueError, match="missing ref_id"):
        await handler.handle({"path": "/p/x"})


@pytest.mark.asyncio
async def test_handler_missing_path_raises() -> None:
    handler = AppArmorFragmentHandler(_StubAppArmor(), session_id="s")
    with pytest.raises(ValueError, match="missing path"):
        await handler.handle({"ref_id": "r"})


@pytest.mark.asyncio
async def test_handler_session_id_mismatch_raises() -> None:
    """If the runner echoes a session_id in the payload, it must
    match the handler's bound id.  Sanity check against confused-
    deputy bugs."""
    handler = AppArmorFragmentHandler(_StubAppArmor(), session_id="sess-A")
    with pytest.raises(ValueError, match="payload session_id"):
        await handler.handle({
            "ref_id": "r", "path": "/p/x", "session_id": "sess-B",
        })


@pytest.mark.asyncio
async def test_handler_session_id_echo_match_succeeds() -> None:
    aa = _StubAppArmor()
    handler = AppArmorFragmentHandler(aa, session_id="sess-A")
    result = await handler.handle({
        "ref_id": "r", "path": "/p/x", "session_id": "sess-A",
    })
    assert result == {"ok": True}


@pytest.mark.asyncio
async def test_handler_no_session_id_echo_succeeds() -> None:
    """Payload without session_id (the common case) skips the
    echo check and uses the handler-bound id."""
    aa = _StubAppArmor()
    handler = AppArmorFragmentHandler(aa, session_id="sess-A")
    result = await handler.handle({"ref_id": "r", "path": "/p/x"})
    assert result == {"ok": True}
    assert aa.calls == [("sess-A", "r", "/p/x")]


@pytest.mark.asyncio
async def test_handler_unavailable_apparmor_path() -> None:
    """When the AppArmorManager has no validator (test-injectable
    null), the handler skips validation and trusts the manager."""

    class _NoValidator:
        def __init__(self):
            self.calls = []

        def add_reference_fragment(self, sid, rid, p):
            self.calls.append((sid, rid, p))
            return True

    aa = _NoValidator()
    handler = AppArmorFragmentHandler(aa, session_id="s")
    result = await handler.handle({"ref_id": "r", "path": "/p/x"})
    assert result == {"ok": True}
    assert aa.calls == [("s", "r", "/p/x")]


# ----------------------------------------------------------------------
# End-to-end through the bidirectional RPC channel
# ----------------------------------------------------------------------


class _RPCPair:
    def __init__(self, runner_rpc, runner_thread, daemon_client):
        self.runner_rpc = runner_rpc
        self.runner_thread = runner_thread
        self.daemon_client = daemon_client


async def _make_pair(rpc_server: RunnerRPCServer) -> _RPCPair:
    daemon_sock, runner_sock = socket.socketpair(
        socket.AF_UNIX, socket.SOCK_STREAM,
    )

    def _no_executor(name, args):
        return False, {"error": "no executor"}

    runner_rpc = RunnerRPC(runner_sock, _no_executor)
    thread = threading.Thread(
        target=runner_rpc.serve, name="rpc-serve-test", daemon=True,
    )
    thread.start()

    daemon_client = DaemonRPCClient(
        daemon_sock, runner_pid=0, rpc_server=rpc_server,
    )
    await daemon_client.start()
    return _RPCPair(runner_rpc, thread, daemon_client)


async def _teardown(pair: _RPCPair) -> None:
    pair.runner_rpc.shutdown()
    pair.runner_thread.join(timeout=2)
    pair.daemon_client._closed = True
    if pair.daemon_client._writer is not None:
        try:
            pair.daemon_client._writer.close()
            await pair.daemon_client._writer.wait_closed()
        except Exception:
            pass
    if pair.daemon_client._read_task is not None:
        pair.daemon_client._read_task.cancel()
        try:
            await pair.daemon_client._read_task
        except (asyncio.CancelledError, Exception):
            pass


@pytest.mark.asyncio
async def test_e2e_add_fragment_round_trip() -> None:
    aa = _StubAppArmor()
    handler = AppArmorFragmentHandler(aa, session_id="sess-1")
    rpc_server = RunnerRPCServer()
    register_apparmor_fragment(rpc_server, handler)

    pair = await _make_pair(rpc_server)
    try:
        wrapper = RunnerSideClient(pair.runner_rpc)
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: wrapper.add_reference_fragment(
                ref_id="ref-1",
                path="/srv/data/x.txt",
                session_id="sess-1",
                timeout=2,
            ),
        )
        assert result == {"ok": True}
        assert aa.calls == [("sess-1", "ref-1", "/srv/data/x.txt")]
    finally:
        await _teardown(pair)


@pytest.mark.asyncio
async def test_e2e_validation_failure_returns_error_dict() -> None:
    aa = _StubAppArmor()
    handler = AppArmorFragmentHandler(aa, session_id="sess-1")
    rpc_server = RunnerRPCServer()
    register_apparmor_fragment(rpc_server, handler)

    pair = await _make_pair(rpc_server)
    try:
        wrapper = RunnerSideClient(pair.runner_rpc)
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: wrapper.add_reference_fragment(
                ref_id="ref-1",
                path="relative/path.txt",
                timeout=2,
            ),
        )
        assert result["ok"] is False
        assert "absolute" in result["error"]
    finally:
        await _teardown(pair)


@pytest.mark.asyncio
async def test_e2e_session_id_mismatch_raises_runner_rpc_error() -> None:
    """Session-id mismatch raises a ValueError on the daemon side
    which the handler dispatcher serializes into a typed error
    envelope; the runner-side wrapper translates that to
    RunnerRPCError."""
    handler = AppArmorFragmentHandler(_StubAppArmor(), session_id="sess-A")
    rpc_server = RunnerRPCServer()
    register_apparmor_fragment(rpc_server, handler)

    pair = await _make_pair(rpc_server)
    try:
        wrapper = RunnerSideClient(pair.runner_rpc)
        loop = asyncio.get_running_loop()
        with pytest.raises(RunnerRPCError, match="ValueError"):
            await loop.run_in_executor(
                None,
                lambda: wrapper.add_reference_fragment(
                    ref_id="r", path="/p/x", session_id="sess-B", timeout=2,
                ),
            )
    finally:
        await _teardown(pair)
