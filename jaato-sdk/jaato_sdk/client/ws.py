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

Message size (#1279, #1586): the client RECEIVES up to ``max_message_size``
bytes per message, default :data:`~jaato_sdk.ws_limits.DEFAULT_WS_MAX_MESSAGE_SIZE`
(16 MiB, the daemon's own default).  It used to pass no ``max_size`` and so
kept ``websockets``' 1 MiB: a larger daemon message closed the connection with
1009 and the waiting call timed out.  A connection that ends on a 1009 now
records a :class:`~jaato_sdk.client.errors.MessageTooLarge`, raised to the
waiting call and announced on the event stream; a message larger than the
daemon advertises it accepts is refused before it is sent.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from jaato_sdk.events import ClientType

from jaato_sdk.ws_limits import (
    DEFAULT_WS_MAX_MESSAGE_SIZE,
    MESSAGE_TOO_BIG_CLOSE_CODE,
    close_frame_of,
)

from .config import RecoveryConfig
from .errors import MessageTooLarge
from .ipc import IPCClient
from .recovery import IPCRecoveryClient, StatusCallback

logger = logging.getLogger(__name__)


def _resolve_max_message_size(value: Optional[int]) -> int:
    """Validate ``max_message_size=``: ``None`` = the shared default.

    A non-positive or non-integer value is refused at construction rather
    than handed to ``websockets`` (where ``None`` means "no limit at all" and
    ``0`` refuses every message) -- a typo must not silently change what the
    client can receive.
    """
    if value is None:
        return DEFAULT_WS_MAX_MESSAGE_SIZE
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            f"max_message_size must be a positive number of bytes, got {value!r}")
    return value


def _too_large_from_close(exc: BaseException, limit: int,
                          advertised: Optional[int]) -> Optional[MessageTooLarge]:
    """The :class:`MessageTooLarge` a 1009 close means, else ``None``.

    Who STARTED the close decides the direction: this client refusing an
    inbound message, or the daemon refusing one this client sent.  The
    other end echoes the code, so both frames read 1009 either way.
    """
    frame, initiator = close_frame_of(exc)
    if frame is None or getattr(frame, "code", None) != MESSAGE_TOO_BIG_CLOSE_CODE:
        return None
    reason = getattr(frame, "reason", "") or ""
    if initiator == "self":
        return MessageTooLarge(
            f"the daemon sent a message larger than this client's "
            f"{limit}-byte limit (1009: {reason}); raise max_message_size= "
            f"on the client to match the daemon's --ws-max-message-size",
            direction="inbound", limit=limit,
            close_code=MESSAGE_TOO_BIG_CLOSE_CODE, reason=reason)
    return MessageTooLarge(
        f"the daemon refused a message this client sent as too large "
        f"(1009: {reason}); its limit is "
        f"{advertised if advertised else 'not advertised'} bytes "
        f"(--ws-max-message-size)",
        direction="outbound", limit=advertised,
        close_code=MESSAGE_TOO_BIG_CLOSE_CODE, reason=reason)


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
        ssl: Any = None,
        ca: Optional[str] = None,
        presentation: Optional[Any] = None,
        max_message_size: Optional[int] = None,
    ) -> None:
        super().__init__(
            socket_path="",  # unused — a WebSocket has no socket file / pipe
            client_type=client_type,
            auto_start=False,  # a remote WS daemon is never auto-started
            env_file=env_file,
            workspace_path=workspace_path,
            config_root=config_root,
            min_protocol_version=min_protocol_version,
            presentation=presentation,
        )
        self._url = url
        self._token = token
        # The largest message this client RECEIVES, in bytes (#1279).  Fixed
        # at ``websockets.connect`` time -- the daemon's advertised limit
        # arrives only in the ConnectedEvent that connection carries, too
        # late to configure it -- so the default is the daemon's own default
        # rather than ``websockets``' 1 MiB.  A deployment that raised
        # ``--ws-max-message-size`` raises this to match.
        self._max_message_size = _resolve_max_message_size(max_message_size)
        # TLS for wss://. ``ssl`` is passed straight to ``websockets.connect``
        # (an ``ssl.SSLContext``, or ``True``/``False``); ``ca`` is a
        # convenience — a CA-bundle path we load into a default context (the
        # Python analog of Node's ``NODE_EXTRA_CA_CERTS`` for self-signed / dev
        # certs). ``ssl`` wins if both are given. Both unset → ``websockets``
        # uses its default verifying context for ``wss://`` (and no TLS for
        # ``ws://``).
        self._ssl = ssl
        self._ca = ca
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
        session_timeout: Optional[float] = None,
        min_protocol_version: Optional[str] = None,
        ssl: Any = None,
        ca: Optional[str] = None,
        presentation: Optional[Any] = None,
        max_message_size: Optional[int] = None,
        **_ignored: Any,
    ) -> Any:
        """Async context manager that connects + creates a session over a
        WebSocket — the WS analog of :meth:`IPCClient.session`.

        ``IPCClient.session`` routes through ``open_session`` (hardwired for the
        IPC ctor: ``socket_path`` / ``auto_start`` / ``apparmor``), so the WS
        transport supplies its own entry that wires the WS ctor knobs (``url`` /
        ``token`` / ``ssl`` / ``ca``) and reuses the transport-agnostic
        ``_SessionContext``::

            async with WSClient.session("wss://host:8080", token="...",
                                        ca="~/.jaato/certs/ca.crt",
                                        profile="researcher") as s:
                print(await s.ask("Research X"))

        ``session_timeout`` bounds the wait for the ``session.new``
        confirmation (unset = ``create_session``'s own 60s); it is distinct
        from ``connect_timeout``.  See :func:`~.convenience.open_session`.
        ``max_message_size`` is the client's receive ceiling (see the
        constructor).
        """
        from .convenience import _SessionContext, facade_create_kwargs

        client = cls(
            url,
            token=token,
            client_type=client_type,
            env_file=env_file,
            workspace_path=workspace_path,
            config_root=config_root,
            min_protocol_version=min_protocol_version,
            ssl=ssl,
            ca=ca,
            presentation=presentation,
            max_message_size=max_message_size,
        )
        # The shared builder, so a knob forwarded over IPC is forwarded here
        # too -- ``**_ignored`` above would otherwise swallow it silently.
        create_kwargs = facade_create_kwargs(
            profile=profile,
            agent=agent,
            agent_params=agent_params,
            cascade_driver_id=cascade_driver_id,
            session_timeout=session_timeout,
        )
        return _SessionContext(
            client,
            create_kwargs,
            on_permission,
            connect_timeout,
            client_tools=client_tools,
        )

    @property
    def is_connected(self) -> bool:
        """Whether the WebSocket is open.

        The inherited property tests the IPC ``_writer``, which a WebSocket
        client never has, so it read ``False`` on every live WS connection.
        """
        return self._connected and self._ws is not None

    @property
    def max_message_size(self) -> int:
        """The largest message this client receives, in bytes (#1279)."""
        return self._max_message_size

    # ---- transport overrides (the only IPC-vs-WS difference) ----

    def _resolve_ssl(self) -> Any:
        """Resolve the TLS arg for ``websockets.connect``.

        ``ssl`` passes through verbatim (an ``ssl.SSLContext`` or ``True`` /
        ``False``); else ``ca`` is loaded into a default verifying context;
        else ``None`` (websockets' default — verifying for ``wss://``, no TLS
        for ``ws://``)."""
        if self._ssl is not None:
            return self._ssl
        if self._ca:
            import ssl as _ssl
            ctx = _ssl.create_default_context()
            ctx.load_verify_locations(self._ca)
            return ctx
        return None

    async def connect(self, timeout: float = 5.0) -> bool:
        """Open the WebSocket and run the shared post-transport handshake."""
        websockets = _ws_module()
        url = self._url
        if self._token:
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}token={self._token}"
        connect_kwargs: dict = {"max_size": self._max_message_size}
        ssl_arg = self._resolve_ssl()
        if ssl_arg is not None:
            connect_kwargs["ssl"] = ssl_arg
        try:
            self._ws = await asyncio.wait_for(
                websockets.connect(url, **connect_kwargs), timeout=timeout)
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
        ok = await self._handshake()
        self._warn_if_daemon_sends_more()
        return ok

    def _warn_if_daemon_sends_more(self) -> None:
        """Say so when the daemon accepts more than this client receives.

        The daemon's ceiling governs both directions on its side, so a
        daemon that accepts 64 MiB may also SEND that much, and this client
        would refuse it with 1009.  The limit cannot be raised on a live
        connection, so this is a WARNING naming the knob, once per connect.
        """
        advertised = self._server_max_message_size
        if advertised and advertised > self._max_message_size:
            logger.warning(
                "WSClient: the daemon accepts messages up to %d bytes but this "
                "client receives at most %d; a larger daemon message will close "
                "the connection with 1009.  Pass max_message_size=%d to match.",
                advertised, self._max_message_size, advertised)

    async def _read_message(self) -> Optional[str]:
        """Read one WebSocket frame (already delimited — no length prefix).

        Returns ``None`` on a closed connection so the drain loop exits cleanly,
        matching the IPC transport's end-of-stream contract.
        """
        if self._ws is None:
            return None
        try:
            msg = await self._ws.recv()
        except Exception as exc:
            # ConnectionClosed (normal/abnormal) or any recv error -> the stream
            # has ended; signal end-of-stream like the IPC transport does.  A
            # 1009 is recorded first, so the waiter is told WHY (#1279).
            cause = _too_large_from_close(
                exc, self._max_message_size, self._server_max_message_size)
            if cause is not None:
                logger.warning("WSClient: connection closed: %s", cause)
                self._close_cause = cause
            return None
        if isinstance(msg, (bytes, bytearray)):
            return msg.decode("utf-8")
        return msg

    async def _write_message(self, message: str) -> None:
        """Send one WebSocket text frame (no length prefix).

        Raises :class:`MessageTooLarge` BEFORE sending a message larger than
        the daemon advertised it accepts: sent, it would be answered with a
        1009 that ends the connection and loses everything in flight on it.
        """
        if self._ws is None:
            raise ConnectionError("Not connected")
        limit = self._server_max_message_size
        if limit:
            size = len(message.encode("utf-8"))
            if size > limit:
                raise MessageTooLarge(
                    f"this message is {size} bytes and the daemon accepts at "
                    f"most {limit} (--ws-max-message-size); it was not sent",
                    direction="outbound", limit=limit, size=size)
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
        ssl: Any = None,
        ca: Optional[str] = None,
        presentation: Optional[Any] = None,
        max_message_size: Optional[int] = None,
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
            presentation=presentation,
        )
        self._url = url
        self._token = token
        self._config_root = config_root
        self._ssl = ssl
        self._ca = ca
        # Validated now, so a bad value fails at construction rather than on
        # the first (re)connect; every inner client is built with it.
        self._max_message_size = _resolve_max_message_size(max_message_size)

    def _make_client(self, *, auto_start: bool) -> IPCClient:
        """Build the inner :class:`WSClient`. ``auto_start`` is meaningless for
        WS (a remote daemon is never auto-started) and is ignored; everything
        else (incl. the TLS ``ssl`` / ``ca`` knobs) mirrors the inherited IPC
        factory — so every reconnect uses the same CA / context."""
        return WSClient(
            self._url,
            token=self._token,
            client_type=self._client_type,
            env_file=self._env_file,
            workspace_path=str(self._workspace_path) if self._workspace_path else None,
            config_root=self._config_root,
            min_protocol_version=self._min_protocol_version,
            ssl=self._ssl,
            ca=self._ca,
            presentation=self._presentation,
            max_message_size=self._max_message_size,
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
        session_timeout: Optional[float] = None,
        on_status_change: Optional[StatusCallback] = None,
        min_protocol_version: Optional[str] = None,
        ssl: Any = None,
        ca: Optional[str] = None,
        presentation: Optional[Any] = None,
        max_message_size: Optional[int] = None,
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

        ``session_timeout`` is forwarded exactly as in :meth:`WSClient.session`.
        """
        from .convenience import _SessionContext, facade_create_kwargs

        client = cls(
            url,
            token=token,
            client_type=client_type,
            env_file=env_file,
            workspace_path=workspace_path,
            config_root=config_root,
            on_status_change=on_status_change,
            min_protocol_version=min_protocol_version,
            ssl=ssl,
            ca=ca,
            presentation=presentation,
            max_message_size=max_message_size,
        )
        # Same shared builder as WSClient.session -- ``**_ignored`` would
        # otherwise swallow a forwarded knob here too.
        create_kwargs = facade_create_kwargs(
            profile=profile,
            agent=agent,
            agent_params=agent_params,
            cascade_driver_id=cascade_driver_id,
            session_timeout=session_timeout,
        )
        return _SessionContext(
            client,
            create_kwargs,
            on_permission,
            connect_timeout,
            client_tools=client_tools,
        )
