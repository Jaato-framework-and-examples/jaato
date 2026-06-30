"""WebSocket transport for the jaato SDK — :class:`WSClient`.

``WSClient`` is :class:`~jaato_sdk.client.ipc.IPCClient` with the transport
swapped from a Unix domain socket to a WebSocket. The wire protocol above the
transport is **identical** — the same command/event JSON frames — so the
convenience facade (``Session.ask`` / ``.complete`` / ``.stream``) rides on it
**unchanged**. Use it to drive a *remote* jaato daemon (started with
``--web-socket``) over ``ws://`` or ``wss://``.

Only the transport seam is overridden:

* ``connect`` — open the WebSocket (with optional bearer token) instead of a
  Unix socket / Windows pipe, then run the shared :meth:`IPCClient._handshake`.
* ``_read_message`` / ``_write_message`` — WS ``recv`` / ``send``. WebSocket
  frames self-delimit, so there is **no** 4-byte length prefix (unlike IPC).
* ``disconnect`` — close the WebSocket (plus the inherited drain-task teardown).

Everything else (the drain loop, event dispatch, ``create_session`` /
``send_message`` / ``respond_to_permission`` / ``subscribe`` / the
``_handshake``) is inherited verbatim.

Auth: when a ``token`` is supplied it is sent as a ``?token=<token>`` query
parameter on the Upgrade URL (the browser-compatible form the server accepts;
the server also accepts an ``Authorization: Bearer`` header). A bad token is
closed by the server with WS code 1008 before any session work — surfaced here
as a ``ConnectionError`` from the handshake.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from jaato_sdk.events import ClientType

from .config import RecoveryConfig
from .ipc import IPCClient
from .recovery import IPCRecoveryClient, StatusCallback


def _ws_module() -> Any:
    """Import the ``websockets`` client library, with a clear install hint.

    Imported lazily so the SDK's IPC/in-process paths don't require
    ``websockets`` — only the WebSocket transport does.
    """
    try:
        import websockets  # noqa: F401

        return websockets
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise ImportError(
            "The jaato WebSocket transport requires the 'websockets' package. "
            "Install it with:  pip install 'websockets>=12.0'"
        ) from exc


class WSClient(IPCClient):
    """:class:`IPCClient` over a WebSocket transport (a remote daemon).

    Construct via :meth:`session` (the facade entry point) rather than directly.
    See the module docstring for the transport-seam overrides; the protocol
    layer and the full facade-client contract are inherited from
    :class:`IPCClient`.
    """

    def __init__(
        self,
        url: str,
        *,
        token: Optional[str] = None,
        client_type: ClientType = ClientType.API,
        env_file: str = ".env",
        workspace_path: Optional[str] = None,
        config_root: Optional[str] = None,
        min_protocol_version: Optional[str] = None,
    ) -> None:
        super().__init__(
            socket_path="",  # unused — a WebSocket has no socket file / pipe
            client_type=client_type,
            auto_start=False,  # a remote WS daemon is never auto-started
            env_file=env_file,
            workspace_path=workspace_path,
            config_root=config_root,
            min_protocol_version=min_protocol_version,
        )
        self._url = url
        self._token = token
        self._ws: Any = None

    @classmethod
    def session(
        cls,
        url: str,
        *,
        token: Optional[str] = None,
        client_type: ClientType = ClientType.API,
        profile: Any = None,
        agent: Optional[str] = None,
        agent_params: Any = None,
        cascade_driver_id: Optional[str] = None,
        on_permission: Any = None,
        client_tools: Any = None,
        workspace_path: Optional[str] = None,
        config_root: Optional[str] = None,
        env_file: str = ".env",
        connect_timeout: float = 120.0,
        min_protocol_version: Optional[str] = None,
        **_ignored: Any,
    ) -> Any:
        """Async context manager that connects + creates a session over a
        WebSocket — the WS analog of :meth:`IPCClient.session`.

        ``IPCClient.session`` routes through ``open_session`` (hardwired for the
        IPC ctor: ``socket_path`` / ``auto_start`` / ``apparmor``), so the WS
        transport supplies its own entry that wires the WS ctor knobs (``url`` /
        ``token``) and reuses the transport-agnostic ``_SessionContext``::

            async with WSClient.session("wss://host:8080", token="...",
                                        profile="researcher") as s:
                print(await s.ask("Research X"))
        """
        from .convenience import _SessionContext

        client = cls(
            url,
            token=token,
            client_type=client_type,
            env_file=env_file,
            workspace_path=workspace_path,
            config_root=config_root,
            min_protocol_version=min_protocol_version,
        )
        create_kwargs = dict(
            profile=profile,
            agent=agent,
            agent_params=agent_params,
            cascade_driver_id=cascade_driver_id,
        )
        return _SessionContext(
            client,
            create_kwargs,
            on_permission,
            connect_timeout,
            client_tools=client_tools,
        )

    # ---- transport overrides (the only IPC-vs-WS difference) ----

    async def connect(self, timeout: float = 5.0) -> bool:
        """Open the WebSocket and run the shared post-transport handshake."""
        websockets = _ws_module()
        url = self._url
        if self._token:
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}token={self._token}"
        try:
            self._ws = await asyncio.wait_for(websockets.connect(url), timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise ConnectionError(
                f"WebSocket connect timed out after {timeout}s ({self._url})"
            ) from exc
        except Exception as exc:
            raise ConnectionError(
                f"WebSocket connect failed ({self._url}): {exc}"
            ) from exc
        self._connected = True
        # The shared handshake reads the ConnectedEvent, gates protocol compat,
        # sends workspace + client config, and starts the drain reader.
        return await self._handshake()

    async def _read_message(self) -> Optional[str]:
        """Read one WebSocket frame (already delimited — no length prefix).

        Returns ``None`` on a closed connection so the drain loop exits cleanly,
        matching the IPC transport's end-of-stream contract.
        """
        if self._ws is None:
            return None
        try:
            msg = await self._ws.recv()
        except Exception:
            # ConnectionClosed (normal/abnormal) or any recv error -> the stream
            # has ended; signal end-of-stream like the IPC transport does.
            return None
        if isinstance(msg, (bytes, bytearray)):
            return msg.decode("utf-8")
        return msg

    async def _write_message(self, message: str) -> None:
        """Send one WebSocket text frame (no length prefix)."""
        if self._ws is None:
            raise ConnectionError("Not connected")
        try:
            await self._ws.send(message)
        except Exception as exc:
            raise ConnectionError(f"WebSocket send failed: {exc}") from exc

    async def disconnect(self) -> None:
        """Tear down the drain task + state (inherited), then close the WS."""
        await super().disconnect()  # cancels drain (interrupts recv) + clears state
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None


class WSRecoveryClient(IPCRecoveryClient):
    """:class:`IPCRecoveryClient` over a WebSocket transport — auto-reconnect
    for a *remote* daemon.

    The recovery state machine, exponential-backoff reconnect loop, session
    reattachment, and event pump are all inherited from
    :class:`IPCRecoveryClient`; only the inner-client factory
    (:meth:`_make_client`) is overridden to build a :class:`WSClient`
    (``url`` / ``token``) instead of an ``IPCClient`` (``socket_path`` /
    ``auto_start``).  Reattachment works identically because the server's
    replay-on-attach is transport-agnostic (``emit_current_state`` /
    ``_emit_conversation_replay`` live on the server core, not the IPC sink).

    Construct via :meth:`session` (the facade entry point) rather than directly.
    """

    def __init__(
        self,
        url: str,
        *,
        token: Optional[str] = None,
        client_type: ClientType = ClientType.API,
        config: Optional[RecoveryConfig] = None,
        env_file: str = ".env",
        workspace_path: Optional[Any] = None,
        config_root: Optional[str] = None,
        on_status_change: Optional[StatusCallback] = None,
        min_protocol_version: Optional[str] = None,
    ) -> None:
        super().__init__(
            socket_path="",  # unused — a WebSocket has no socket file / pipe
            client_type=client_type,
            config=config,
            auto_start=False,  # a remote WS daemon is never auto-started
            env_file=env_file,
            workspace_path=workspace_path,
            on_status_change=on_status_change,
            min_protocol_version=min_protocol_version,
        )
        self._url = url
        self._token = token
        self._config_root = config_root

    def _make_client(self, *, auto_start: bool) -> IPCClient:
        """Build the inner :class:`WSClient`. ``auto_start`` is meaningless for
        WS (a remote daemon is never auto-started) and is ignored; everything
        else mirrors the inherited IPC factory."""
        return WSClient(
            self._url,
            token=self._token,
            client_type=self._client_type,
            env_file=self._env_file,
            workspace_path=str(self._workspace_path) if self._workspace_path else None,
            config_root=self._config_root,
            min_protocol_version=self._min_protocol_version,
        )

    @classmethod
    def session(
        cls,
        url: str,
        *,
        token: Optional[str] = None,
        client_type: ClientType = ClientType.API,
        profile: Any = None,
        agent: Optional[str] = None,
        agent_params: Any = None,
        cascade_driver_id: Optional[str] = None,
        on_permission: Any = None,
        client_tools: Any = None,
        workspace_path: Optional[str] = None,
        config_root: Optional[str] = None,
        env_file: str = ".env",
        connect_timeout: float = 120.0,
        on_status_change: Optional[StatusCallback] = None,
        min_protocol_version: Optional[str] = None,
        **_ignored: Any,
    ) -> Any:
        """Auto-reconnect WS analog of :meth:`IPCRecoveryClient.session`: the
        convenience facade (``Session.ask`` / ``.complete`` / ``.stream``)
        backed by a ``WSRecoveryClient`` so the session survives daemon
        restarts over a WebSocket.  Adds ``on_status_change=`` (the
        reconnection-status callback) on top of :meth:`WSClient.session`'s
        knobs::

            async with WSRecoveryClient.session("wss://host:8080", token="...",
                                                profile="researcher",
                                                on_status_change=print) as s:
                print(await s.ask("Long task…"))
        """
        from .convenience import _SessionContext

        client = cls(
            url,
            token=token,
            client_type=client_type,
            env_file=env_file,
            workspace_path=workspace_path,
            config_root=config_root,
            on_status_change=on_status_change,
            min_protocol_version=min_protocol_version,
        )
        create_kwargs = dict(
            profile=profile,
            agent=agent,
            agent_params=agent_params,
            cascade_driver_id=cascade_driver_id,
        )
        return _SessionContext(
            client,
            create_kwargs,
            on_permission,
            connect_timeout,
            client_tools=client_tools,
        )
