"""Custom stdlib CONNECT-allowlist proxy (Phase 5 §5.11b).

A minimal HTTP proxy that implements ONLY the ``CONNECT`` verb (which HTTPS
clients use to open a tunnel through an HTTP proxy) plus a hostname allowlist.
No TLS termination, no content inspection — the proxy parses the destination
``host:port`` from the CONNECT line, checks the :class:`AllowlistConfig`, and
either replies ``200`` and splices bytes, or replies ``403``.

Security posture (see design doc §2.2 cons):
- Narrow surface: only CONNECT; GET/POST/etc. are rejected ``405``.
- Strict input validation: bounded request read, allow-listed hostname chars,
  port range check — defends against CONNECT-line CRLF/header smuggling.
- DNS is resolved *proxy-side* at ``create_connection`` time, so the runner
  does not do its own DNS for tunneled HTTPS (a DNS-exfil mitigation).

The proxy binds ``127.0.0.1:0`` (an OS-assigned free port) and runs an accept
loop in a daemon thread; each accepted connection is handled on its own thread.
"""

from __future__ import annotations

import logging
import select
import socket
import threading
from typing import Optional, Set

from .config import AllowlistConfig, _HOST_CHARS

logger = logging.getLogger(__name__)

# Bounded read for the CONNECT request line + headers (a well-formed CONNECT
# preamble is tiny; anything larger is malformed/hostile).
_MAX_REQUEST_BYTES = 8192
_ACCEPT_TIMEOUT = 0.5      # seconds — lets the accept loop poll the stop flag
_TUNNEL_CHUNK = 65536
_CONNECT_TIMEOUT = 10      # seconds to establish the upstream connection


class ProxyStats:
    """Thread-safe counters for telemetry."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.connections_total = 0
        self.allowed_total = 0
        self.denied_total = 0
        self.malformed_total = 0

    def _inc(self, attr: str) -> None:
        with self._lock:
            setattr(self, attr, getattr(self, attr) + 1)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "connections_total": self.connections_total,
                "allowed_total": self.allowed_total,
                "denied_total": self.denied_total,
                "malformed_total": self.malformed_total,
            }


class ConnectAllowlistProxy:
    """A single-session CONNECT-allowlist proxy bound to loopback.

    Lifecycle: ``start()`` binds + spawns the accept thread and returns; the
    bound port is available as :attr:`port` / :meth:`proxy_url`.  ``stop()``
    is idempotent — it closes the listen socket, wakes the accept loop, and
    closes any in-flight client sockets so tunnels drain.
    """

    def __init__(self, allowlist: AllowlistConfig, host: str = "127.0.0.1"):
        self._allowlist = allowlist
        self._bind_host = host
        self._listen_sock: Optional[socket.socket] = None
        self._accept_thread: Optional[threading.Thread] = None
        self._stopping = threading.Event()
        self._active: Set[socket.socket] = set()
        self._active_lock = threading.Lock()
        self.port: Optional[int] = None
        self.stats = ProxyStats()

    # -- lifecycle --------------------------------------------------------

    def start(self) -> int:
        """Bind to a free loopback port and start accepting.  Returns the port."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self._bind_host, 0))
        sock.listen(64)
        sock.settimeout(_ACCEPT_TIMEOUT)
        self._listen_sock = sock
        self.port = sock.getsockname()[1]
        self._accept_thread = threading.Thread(
            target=self._accept_loop, name=f"egress-proxy-{self.port}", daemon=True,
        )
        self._accept_thread.start()
        logger.info("egress proxy listening on %s:%d", self._bind_host, self.port)
        return self.port

    def proxy_url(self) -> str:
        if self.port is None:
            raise RuntimeError("proxy not started")
        return f"http://{self._bind_host}:{self.port}"

    def stop(self) -> None:
        """Idempotent shutdown: stop accepting, close active tunnels, join."""
        self._stopping.set()
        if self._listen_sock is not None:
            try:
                self._listen_sock.close()
            except OSError:
                pass
        with self._active_lock:
            active = list(self._active)
        for s in active:
            try:
                s.close()
            except OSError:
                pass
        if self._accept_thread is not None:
            self._accept_thread.join(timeout=2.0)

    # -- accept loop ------------------------------------------------------

    def _accept_loop(self) -> None:
        while not self._stopping.is_set():
            try:
                client, _addr = self._listen_sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break  # listen socket closed during stop()
            t = threading.Thread(
                target=self._handle_client, args=(client,), daemon=True,
            )
            t.start()

    # -- per-connection ---------------------------------------------------

    def _handle_client(self, client: socket.socket) -> None:
        self.stats._inc("connections_total")
        with self._active_lock:
            self._active.add(client)
        upstream: Optional[socket.socket] = None
        try:
            client.settimeout(_CONNECT_TIMEOUT)
            preamble = self._read_preamble(client)
            if preamble is None:
                self.stats._inc("malformed_total")
                self._reply(client, 400, "Bad Request")
                return

            parsed = self._parse_connect(preamble)
            if parsed is None:
                self.stats._inc("malformed_total")
                # Distinguish "not CONNECT" from "malformed CONNECT".
                method = preamble.split(b" ", 1)[0][:16].decode("latin-1", "replace")
                if method.upper() != "CONNECT":
                    self._reply(client, 405, "Method Not Allowed")
                else:
                    self._reply(client, 400, "Bad Request")
                return

            host, port = parsed
            if not self._allowlist.allows(host, port):
                self.stats._inc("denied_total")
                if self._allowlist.log_denied:
                    logger.warning(
                        "egress DENY %s:%d (not in allowlist)", host, port)
                self._reply(client, 403, "Forbidden")
                return

            try:
                upstream = socket.create_connection((host, port), _CONNECT_TIMEOUT)
            except OSError as exc:
                logger.info("egress upstream connect failed %s:%d: %s",
                            host, port, exc)
                self._reply(client, 502, "Bad Gateway")
                return

            self.stats._inc("allowed_total")
            self._reply(client, 200, "Connection Established")
            with self._active_lock:
                self._active.add(upstream)
            client.settimeout(None)
            self._splice(client, upstream)
        except OSError:
            pass
        finally:
            with self._active_lock:
                self._active.discard(client)
                if upstream is not None:
                    self._active.discard(upstream)
            for s in (client, upstream):
                if s is not None:
                    try:
                        s.close()
                    except OSError:
                        pass

    def _read_preamble(self, client: socket.socket) -> Optional[bytes]:
        """Read up to the header terminator (CRLFCRLF), bounded."""
        buf = b""
        while b"\r\n\r\n" not in buf:
            if len(buf) >= _MAX_REQUEST_BYTES:
                return None
            try:
                chunk = client.recv(_MAX_REQUEST_BYTES - len(buf))
            except (socket.timeout, OSError):
                return None
            if not chunk:
                return None
            buf += chunk
        return buf

    def _parse_connect(self, preamble: bytes):
        """Parse + strictly validate a CONNECT request line.

        Returns ``(host, port)`` or ``None`` if not a valid CONNECT.
        """
        try:
            request_line = preamble.split(b"\r\n", 1)[0].decode("latin-1")
        except Exception:
            return None
        parts = request_line.split(" ")
        if len(parts) != 3 or parts[0].upper() != "CONNECT":
            return None
        target = parts[1]
        # Reject stray control chars that survived latin-1 decode.
        if any(ord(c) < 0x20 for c in target):
            return None
        # target is host:port. IPv6 literals ([::1]:443) are not supported in v1.
        if target.startswith("["):
            return None
        if ":" not in target:
            return None
        host, _, port_s = target.rpartition(":")
        if not host or not _HOST_CHARS.match(host):
            return None
        if "*" in host:  # a real destination is never a wildcard
            return None
        try:
            port = int(port_s)
        except ValueError:
            return None
        if not (1 <= port <= 65535):
            return None
        return host.lower(), port

    def _reply(self, client: socket.socket, code: int, reason: str) -> None:
        try:
            client.sendall(
                f"HTTP/1.1 {code} {reason}\r\n\r\n".encode("latin-1"))
        except OSError:
            pass

    def _splice(self, a: socket.socket, b: socket.socket) -> None:
        """Bidirectionally pipe bytes between two sockets until either closes."""
        socks = [a, b]
        while not self._stopping.is_set():
            try:
                readable, _, errored = select.select(socks, [], socks, 1.0)
            except (OSError, ValueError):
                break
            if errored:
                break
            for src in readable:
                dst = b if src is a else a
                try:
                    data = src.recv(_TUNNEL_CHUNK)
                except OSError:
                    return
                if not data:
                    return
                try:
                    dst.sendall(data)
                except OSError:
                    return
