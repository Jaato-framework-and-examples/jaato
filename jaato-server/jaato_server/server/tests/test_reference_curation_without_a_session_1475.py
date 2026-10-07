"""Reference curation from a session-less IPC client is answered (#1475).

A client that connected over IPC and declared a workspace, but attached to
no session, sent ``list_reference_claims`` / ``promote_reference_claim``
and waited out its whole timeout (180 s for promote).  Two defects, one
on each side of the socket:

1. ``JaatoIPCServer`` routes a request only when the client has a session
   or the request is in ``_SESSIONLESS_REQUEST_TYPES``.  The four
   reference-curation requests were not in it, although their handlers
   resolve the workspace from the CONNECTION and answer ``no_workspace``
   themselves -- the design the memory verbs got in #1232.
2. The gate's refusal carried no ``request_id``, and
   ``IPCClient._correlated_request`` discards anything that does not echo
   its id, so ANY correlated request refused there hung until its timeout.
   The refusal now echoes the id, and the SDK turns a correlated
   ``ErrorEvent`` into :class:`~jaato_sdk.client.errors.RequestRefused`.

Driven end to end: a real ``JaatoIPCServer`` on a temporary socket, wired
to a real ``CommandRouter`` (as ``__main__`` wires them), and a real
``IPCClient``.  Only the session manager is a stand-in -- no session exists,
which is the state under test.  Every wait is short, so a regression fails
in seconds rather than hanging.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

import pytest

from jaato_server.shared.tests.reversion import Reversion

_IPC = "jaato-server/jaato_server/server/ipc.py"
_SDK_IPC = "jaato-sdk/jaato_sdk/client/ipc.py"

#: Seconds a correlated call may wait.  A working path answers in
#: milliseconds; the old defect is a wait of exactly this long.
_WAIT = 3.0


class _NoSessionManager:
    """A session manager with no sessions: the state #1475 is about."""

    def get_client_session(self, client_id: str) -> Any:
        return None

    def get_session(self, session_id: str) -> Any:
        return None

    def creator_in_workspace(self, *args: Any, **kwargs: Any) -> Optional[str]:
        return None

    def _workspace_owner_of(self, workspace: str) -> Optional[str]:
        return None

    def handle_request(self, *args: Any, **kwargs: Any) -> None:
        """``ClientConfigRequest`` lands here; nothing to apply."""

    def detach_client(self, client_id: str) -> None:
        pass

    def unregister_all_cascade_clients_for_connection(self, *a: Any, **k: Any) -> None:
        pass

    def __getattr__(self, name: str) -> Callable[..., None]:
        # Any other daemon-side bookkeeping the router touches is a no-op.
        return lambda *a, **k: None


def _drive(body: Callable[[Any, Path], Awaitable[None]]) -> None:
    """Run ``body(client, workspace)`` against a live IPC server."""
    from jaato_sdk.client.ipc import IPCClient
    from jaato_sdk.events import ClientType

    from jaato_server.server.command_router import CommandRouter
    from jaato_server.server.ipc import JaatoIPCServer

    tmp = tempfile.mkdtemp(prefix="jaato1475-", dir="/tmp")
    workspace = Path(tmp) / "ws"
    (workspace / ".jaato").mkdir(parents=True)
    sock = os.path.join(tmp, "d.sock")

    async def main() -> None:
        server = JaatoIPCServer(socket_path=sock)
        router = CommandRouter(
            session_manager=_NoSessionManager(),  # type: ignore[arg-type]
            event_sink=server, daemon_plugins={})
        server._on_session_request = router.handle_request
        task = asyncio.create_task(server.start())
        try:
            deadline = time.monotonic() + 5
            while not os.path.exists(sock):
                assert time.monotonic() < deadline, "IPC server did not bind"
                await asyncio.sleep(0.02)
            client = IPCClient(sock, auto_start=False, client_type=ClientType.API,
                               workspace_path=str(workspace))
            assert await client.connect(timeout=5)
            # set_workspace is fire-and-forget; wait until it is applied.
            deadline = time.monotonic() + 5
            while not any(server.get_client_workspace(cid)
                          for cid in list(server._clients)):
                assert time.monotonic() < deadline, "workspace not declared"
                await asyncio.sleep(0.02)
            try:
                await body(client, workspace)
            finally:
                await client.disconnect()
        finally:
            await server.stop()
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    try:
        asyncio.run(main())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_reference_listings_are_answered_without_a_session() -> None:
    """(a) The two listings get the daemon's answer, not a timeout."""

    async def body(client: Any, workspace: Path) -> None:
        claims = await client.list_reference_claims(timeout=_WAIT)
        assert claims.ok is True, claims
        assert claims.claims == []
        assert claims.may_curate is True  # unowned workspace
        catalog = await client.list_reference_catalog(timeout=_WAIT)
        assert catalog.ok is True, catalog

    _drive(body)


def test_promote_is_answered_without_a_session() -> None:
    """(a) A promotion reaches the curation handler and gets ITS refusal.

    The claim does not exist, so the answer is a correlated ``not_found``
    from the handler -- proof it was routed, where the defect answered
    nothing at all and the call waited out 180 s.
    """

    async def body(client: Any, workspace: Path) -> None:
        result = await client.promote_reference_claim(
            "20260929T100000Z-abcd1234", timeout=_WAIT)
        assert result.ok is False
        assert result.category == "not_found", result
        assert result.claim_id == "20260929T100000Z-abcd1234"

    _drive(body)


def test_a_request_still_refused_at_the_gate_fails_at_once() -> None:
    """(b) A correlated request that needs a session fails now, with why.

    ``HistoryPageRequest`` still needs a session.  The refusal echoes its
    ``request_id``, and the SDK raises ``RequestRefused`` carrying the
    ``no_session`` category -- well inside the wait it used to exhaust.
    """
    from jaato_sdk.client.errors import RequestRefused

    async def body(client: Any, workspace: Path) -> None:
        started = time.monotonic()
        with pytest.raises(RequestRefused) as caught:
            await client.request_history_page(timeout=_WAIT)
        assert time.monotonic() - started < _WAIT / 2
        err = caught.value
        assert err.category == "no_session"
        assert err.details.get("request_type") == "HistoryPageRequest"
        assert err.request_id
        assert "no session" in str(err)

    _drive(body)


REVERSIONS = [
    Reversion(
        target=_IPC,
        find=") + MEMORY_REQUEST_TYPES + REFERENCE_CURATION_REQUEST_TYPES",
        replace=") + MEMORY_REQUEST_TYPES",
        test="test_reference_listings_are_answered_without_a_session",
        because=(
            "the reference-curation requests are dropped at the IPC session "
            "gate, so a session-less curator waits out its timeout"),
    ),
    Reversion(
        target=_IPC,
        find="            request_id=request_id,\n"
             "            details={\"category\": \"no_session\"",
        replace="            request_id=None,\n"
                "            details={\"category\": \"no_session\"",
        test="test_a_request_still_refused_at_the_gate_fails_at_once",
        because=(
            "a refusal that does not echo the request_id is discarded by "
            "the correlated wait, which then times out"),
    ),
    Reversion(
        target=_SDK_IPC,
        find="        if isinstance(answer, ErrorEvent):\n"
             "            # A correlated refusal (#1475)",
        replace="        if False:\n"
                "            # A correlated refusal (#1475)",
        test="test_a_request_still_refused_at_the_gate_fails_at_once",
        because=(
            "the SDK hands a correlated ErrorEvent back as if it were the "
            "request's own answer instead of raising with the reason"),
    ),
]
