"""``app://`` references resolved over IPC, by the workspace owner's account (#1609).

A session the daemon starts on its own (a reactor stage, a ``session.wake``
from something other than the application, a revive) re-resolves its
environment without asking the client that created it.  On WebSocket the
daemon asks the owning application over its bind channel (#1226); over IPC
nobody was wired, so every ``app://`` reference was dropped.

An IPC connection now sends ``secret.resolver.register`` and answers
``secret.resolve`` for the workspaces its OS account owns.  These guards
drive a real ``JaatoIPCServer``, a real ``IPCClient`` and a real
``JaatoServer._resolve_session_env`` and pin:

* a registered connection of the owning account resolves the reference into
  the session env, and is asked with the workspace and the name;
* a workspace owned by another uid is not answered by it;
* an answer is taken only from the connection the request was sent to;
* WS ownership wins over IPC in the chained resolver;
* a disconnect unregisters, so the next spawn drops the reference again;
* the daemon wiring hands the IPC resolver to the session manager, which
  hands both to every session it constructs;
* the SDK refuses the verb below protocol 1.40, and on WebSocket.
"""

from __future__ import annotations

import ast
import asyncio
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import pytest

from jaato_server.server.app_secret import (
    AppSecretAnswer,
    AppSecretResolver,
    chain_app_secret_resolvers,
)
from jaato_server.server.core import JaatoServer
from jaato_server.server.ipc_app_secret import (
    IPC_OWNER_PREFIX,
    IPCAppSecretResolver,
    PendingSecretResolves,
)
from jaato_server.shared.plugins.subagent.config import (
    AppSecretReference,
    SecretResolveContext,
)
from jaato_server.shared.tests.reversion import Reversion

_IPC = "jaato-server/jaato_server/server/ipc.py"
_IPC_SECRET = "jaato-server/jaato_server/server/ipc_app_secret.py"
_APP_SECRET = "jaato-server/jaato_server/server/app_secret.py"
_SM = "jaato-server/jaato_server/server/session_manager.py"
_MAIN = "jaato-server/jaato_server/server/__main__.py"

_REPO = Path(__file__).resolve().parents[4]


# --------------------------------------------------------------- harness


def _drive(body: Callable[[Any, Any], Any]) -> None:
    """Run ``body(ipc_server, client)`` against a real IPC server and client."""
    from jaato_sdk.client.ipc import IPCClient
    from jaato_sdk.events import ClientType
    from jaato_server.server.ipc import JaatoIPCServer

    tmp = tempfile.mkdtemp(prefix="jaato1609-", dir="/tmp")
    sock = os.path.join(tmp, "d.sock")

    async def main() -> None:
        server = JaatoIPCServer(socket_path=sock)
        task = asyncio.create_task(server.start())
        try:
            deadline = time.monotonic() + 5
            while not os.path.exists(sock):
                assert time.monotonic() < deadline, "IPC server did not bind"
                await asyncio.sleep(0.02)
            client = IPCClient(sock, auto_start=False, client_type=ClientType.API)
            assert await client.connect(timeout=5)
            try:
                await body(server, client)
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


def _session(workspace: str, resolver: Any, env: str = "GH_TOKEN=app://github\n") -> JaatoServer:
    envf = os.path.join(workspace, ".env")
    with open(envf, "w") as fh:
        fh.write(env)
    srv = JaatoServer(env_file=envf, provider=None, workspace_path=workspace,
                      session_id="s1609")
    srv.set_app_secret_resolver(resolver)
    return srv


def _resolve_off_loop(srv: JaatoServer) -> Any:
    """Session-env resolution runs off the loop in the daemon; so here."""
    return asyncio.to_thread(srv._resolve_session_env)


# --------------------------------------------------- end to end over IPC


def test_a_registered_owner_resolves_the_reference(tmp_path: Path) -> None:
    calls: list = []
    got: dict = {}

    def resolver(*, user: str, workspace: str, name: str) -> Any:
        calls.append((user, workspace, name))
        return {"value": "tok-1609", "expires_at": "2099-01-01T00:00:00Z"}

    async def body(server: Any, client: Any) -> None:
        answer = await client.register_secret_resolver(resolver, timeout=5)
        got["answer"] = answer
        srv = _session(str(tmp_path), chain_app_secret_resolvers(
            None, server.app_secret_resolver))
        await _resolve_off_loop(srv)
        got["env"] = dict(srv._session_env)
        got["expiries"] = dict(srv._app_secret_expiries)
        got["granted"] = srv.granted_env_names()

    _drive(body)
    assert got["answer"].status == "ok"
    assert got["answer"].uid == os.getuid()
    assert got["env"]["GH_TOKEN"] == "tok-1609"
    assert got["expiries"] == {"GH_TOKEN": "2099-01-01T00:00:00Z"}
    assert got["granted"] == ["GH_TOKEN"]
    assert len(calls) == 1
    _, workspace, name = calls[0]
    assert (workspace, name) == (str(tmp_path), "github")


def test_a_workspace_of_another_account_is_not_answered(tmp_path: Path) -> None:
    calls: list = []
    got: dict = {}

    async def body(server: Any, client: Any) -> None:
        await client.register_secret_resolver(
            lambda **kw: calls.append(kw) or "stolen", timeout=5)
        ipc = server.app_secret_resolver
        # The workspace belongs to a different uid than this connection's.
        ipc._owner_uid_of = lambda path: os.getuid() + 1
        srv = _session(str(tmp_path), chain_app_secret_resolvers(None, ipc))
        await _resolve_off_loop(srv)
        got["env"] = dict(srv._session_env)

    _drive(body)
    assert "GH_TOKEN" not in got["env"]
    assert calls == []


def test_a_disconnect_unregisters_the_resolver(tmp_path: Path) -> None:
    got: dict = {}

    async def body(server: Any, client: Any) -> None:
        await client.register_secret_resolver(lambda **kw: "tok", timeout=5)
        await client.disconnect()
        deadline = time.monotonic() + 5
        ipc = server.app_secret_resolver
        while ipc._registration_for(os.getuid()) is not None:
            assert time.monotonic() < deadline, "registration outlived its connection"
            await asyncio.sleep(0.02)
        srv = _session(str(tmp_path), chain_app_secret_resolvers(None, ipc))
        await _resolve_off_loop(srv)
        got["env"] = dict(srv._session_env)

    _drive(body)
    assert "GH_TOKEN" not in got["env"]


# ----------------------------------------------------- unit-level rules


def test_an_answer_is_taken_only_from_the_connection_asked() -> None:
    pending = PendingSecretResolves()
    fut = pending.open("r1", "ipc_1")
    assert pending.deliver("ipc_2", "r1", "forged") is False
    assert not fut.done()
    assert pending.deliver("ipc_1", "r1", "real") is True
    assert fut.result(timeout=1) == "real"


def test_ws_ownership_wins_over_ipc(tmp_path: Path) -> None:
    ws_calls: list = []
    ipc_calls: list = []

    def ws_transport(app_id: str, user: str, workspace: str, name: str,
                     timeout: float) -> AppSecretAnswer:
        ws_calls.append(app_id)
        return AppSecretAnswer(status="ok", value="from-ws")

    def ipc_transport(*a: Any) -> AppSecretAnswer:
        ipc_calls.append(a)
        return AppSecretAnswer(status="ok", value="from-ipc")

    ws = AppSecretResolver(owner_of=lambda p: "acme:alice", transport=ws_transport)
    ipc = IPCAppSecretResolver(ipc_transport, owner_uid_of=lambda p: os.getuid())
    ipc.register("ipc_1", os.getuid(), "me")
    srv = _session(str(tmp_path), chain_app_secret_resolvers(ws, ipc))
    srv._resolve_session_env()
    assert srv._session_env["GH_TOKEN"] == "from-ws"
    assert ws_calls == ["acme"] and ipc_calls == []

    # An unowned WS workspace falls through to the owning account.
    ws_unowned = AppSecretResolver(owner_of=lambda p: None, transport=ws_transport)
    srv = _session(str(tmp_path), chain_app_secret_resolvers(ws_unowned, ipc))
    srv._resolve_session_env()
    assert srv._session_env["GH_TOKEN"] == "from-ipc"


def test_no_registration_names_the_uid_in_the_reason() -> None:
    ipc = IPCAppSecretResolver(lambda *a: AppSecretAnswer(status="ok", value="x"))
    ctx = SecretResolveContext(workspace_path="/w",
                               workspace_owner=f"{IPC_OWNER_PREFIX}4242",
                               session_id="s")
    answer = ipc.resolve_reference(AppSecretReference(name="github", required=False), ctx)
    assert answer.status == "unreachable"
    assert "uid 4242" in (answer.detail or "")


# ------------------------------------------------------------- wiring


def _calls_in(rel: str) -> set:
    tree = ast.parse((_REPO / rel).read_text())
    return {getattr(n.func, "attr", getattr(n.func, "id", ""))
            for n in ast.walk(tree) if isinstance(n, ast.Call)}


def test_the_daemon_hands_the_ipc_resolver_to_the_session_manager() -> None:
    assert "set_ipc_app_secret_resolver" in _calls_in(_MAIN)


def test_every_session_gets_both_resolvers_chained() -> None:
    import inspect

    from jaato_server.server.session_manager import SessionManager

    src = inspect.getsource(SessionManager._construct_and_initialize_server)
    i = src.index("set_app_secret_resolver(")
    call = src[i:src.index("_resolve_session_env()", i)]
    assert "chain_app_secret_resolvers" in call
    assert "_ipc_app_secret_resolver" in call


# ---------------------------------------------------------------- the SDK


def test_the_sdk_refuses_below_1_40() -> None:
    from jaato_sdk.client.ipc import IPCClient

    from jaato_sdk.events import ClientType

    client = IPCClient("/nonexistent.sock", auto_start=False,
                       client_type=ClientType.API)
    client._server_protocol_version = "1.39"
    with pytest.raises(ValueError, match="1.40"):
        asyncio.run(client.register_secret_resolver(lambda **kw: "x"))


def test_the_sdk_normalises_resolver_returns() -> None:
    from jaato_sdk.client.ipc import _secret_outcome

    assert _secret_outcome("t") == {"status": "ok", "value": "t"}
    assert _secret_outcome(None) == {"status": "not_found"}
    assert _secret_outcome({"status": "denied", "detail": "no"}) == {
        "status": "denied", "detail": "no"}
    assert _secret_outcome({"status": "ok"})["status"] == "error"
    assert _secret_outcome(42)["status"] == "error"


def test_the_ws_client_refuses_the_ipc_verb() -> None:
    from jaato_sdk.client.ws import WSClient

    client = WSClient("ws://127.0.0.1:1")
    with pytest.raises(NotImplementedError):
        asyncio.run(client.register_secret_resolver(lambda **kw: "x"))


REVERSIONS = [
    Reversion(
        target=_IPC,
        find="        self._app_secret_resolver.register(client_id, peer.uid, peer.identity)\n",
        replace="",
        test="test_a_registered_owner_resolves_the_reference",
        because="the daemon answers ok and never records the registration",
    ),
    Reversion(
        target=_IPC_SECRET,
        find="                if reg.uid == uid:\n",
        replace="                if True:\n",
        test="test_a_workspace_of_another_account_is_not_answered",
        because="any registered account answers for every workspace",
    ),
    Reversion(
        target=_IPC_SECRET,
        find="            if entry is None or entry[0] != client_id:\n",
        replace="            if entry is None:\n",
        test="test_an_answer_is_taken_only_from_the_connection_asked",
        because="a second connection of the account can answer for the first",
    ),
    Reversion(
        target=_IPC,
        find="            self._app_secret_resolver.unregister(client_id)\n",
        replace="",
        test="test_a_disconnect_unregisters_the_resolver",
        because="a closed connection keeps being asked until the deadline",
    ),
    Reversion(
        target=_APP_SECRET,
        find="    return ChainedAppSecretResolver(ws, ipc)\n",
        replace="    return ChainedAppSecretResolver(ipc, ws)\n",
        test="test_ws_ownership_wins_over_ipc",
        because="a local account answers for a workspace an application owns",
    ),
    Reversion(
        target=_SM,
        find='            getattr(self, "_ipc_app_secret_resolver", None),\n',
        replace="            None,\n",
        test="test_every_session_gets_both_resolvers_chained",
        because="sessions never see the IPC resolver",
    ),
]
