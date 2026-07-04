"""End-to-end proxy behavior over real loopback sockets (§5.11b).

Each test stands up a tiny echo "origin" server, points a
``ConnectAllowlistProxy`` at an allowlist, and drives a raw client socket
through the CONNECT handshake — asserting allow/deny/malformed outcomes and
that an allowed tunnel actually passes bytes end to end.
"""

import socket
import threading

import pytest

from server.egress_proxy.config import AllowlistConfig
from server.egress_proxy.proxy import ConnectAllowlistProxy


class _EchoOrigin:
    """A loopback TCP server that echoes whatever it receives."""

    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(4)
        self.port = self.sock.getsockname()[1]
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def _loop(self):
        self.sock.settimeout(0.3)
        while not self._stop.is_set():
            try:
                conn, _ = self.sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(target=self._echo, args=(conn,), daemon=True).start()

    def _echo(self, conn):
        try:
            while True:
                data = conn.recv(4096)
                if not data:
                    break
                conn.sendall(data)
        except OSError:
            pass
        finally:
            conn.close()

    def close(self):
        self._stop.set()
        try:
            self.sock.close()
        except OSError:
            pass


def _connect_request(host: str, port: int) -> bytes:
    return f"CONNECT {host}:{port} HTTP/1.1\r\nHost: {host}:{port}\r\n\r\n".encode()


def _status_line(resp: bytes) -> bytes:
    return resp.split(b"\r\n", 1)[0]


@pytest.fixture
def origin():
    o = _EchoOrigin()
    yield o
    o.close()


def _proxy(allowlist):
    p = ConnectAllowlistProxy(allowlist)
    p.start()
    return p


def test_allowed_host_tunnels_bytes(origin):
    proxy = _proxy(AllowlistConfig(allowed_hosts=["127.0.0.1"],
                                   allowed_ports=[origin.port]))
    try:
        c = socket.create_connection(("127.0.0.1", proxy.port), 2)
        c.sendall(_connect_request("127.0.0.1", origin.port))
        resp = c.recv(4096)
        assert b"200" in _status_line(resp), resp
        # tunnel is live — echo round-trips through it
        c.sendall(b"ping-through-tunnel")
        assert c.recv(4096) == b"ping-through-tunnel"
        c.close()
        assert proxy.stats.snapshot()["allowed_total"] == 1
    finally:
        proxy.stop()


def test_denied_host_gets_403(origin):
    # allowlist does NOT include 127.0.0.1
    proxy = _proxy(AllowlistConfig(allowed_hosts=["api.anthropic.com"],
                                   allowed_ports=[443]))
    try:
        c = socket.create_connection(("127.0.0.1", proxy.port), 2)
        c.sendall(_connect_request("127.0.0.1", origin.port))
        resp = c.recv(4096)
        assert b"403" in _status_line(resp), resp
        c.close()
        assert proxy.stats.snapshot()["denied_total"] == 1
    finally:
        proxy.stop()


def test_denied_port_gets_403(origin):
    # host allowed, but only port 443 — origin is on a random high port
    proxy = _proxy(AllowlistConfig(allowed_hosts=["127.0.0.1"],
                                   allowed_ports=[443]))
    try:
        c = socket.create_connection(("127.0.0.1", proxy.port), 2)
        c.sendall(_connect_request("127.0.0.1", origin.port))
        assert b"403" in _status_line(c.recv(4096))
        c.close()
    finally:
        proxy.stop()


def test_non_connect_method_gets_405():
    proxy = _proxy(AllowlistConfig(allowed_hosts=["x.com"]))
    try:
        c = socket.create_connection(("127.0.0.1", proxy.port), 2)
        c.sendall(b"GET http://x.com/ HTTP/1.1\r\nHost: x.com\r\n\r\n")
        assert b"405" in _status_line(c.recv(4096))
        c.close()
    finally:
        proxy.stop()


def test_malformed_connect_gets_400():
    proxy = _proxy(AllowlistConfig(allowed_hosts=["x.com"]))
    try:
        c = socket.create_connection(("127.0.0.1", proxy.port), 2)
        c.sendall(b"CONNECT no-port-here HTTP/1.1\r\n\r\n")
        assert b"400" in _status_line(c.recv(4096))
        c.close()
        assert proxy.stats.snapshot()["malformed_total"] == 1
    finally:
        proxy.stop()


def test_ipv6_literal_target_rejected():
    proxy = _proxy(AllowlistConfig(allowed_hosts=["x.com"]))
    try:
        c = socket.create_connection(("127.0.0.1", proxy.port), 2)
        c.sendall(b"CONNECT [::1]:443 HTTP/1.1\r\n\r\n")
        # v1 does not support IPv6 literals -> malformed CONNECT -> 400
        assert b"400" in _status_line(c.recv(4096))
        c.close()
    finally:
        proxy.stop()


def test_crlf_injection_in_host_rejected():
    proxy = _proxy(AllowlistConfig(allowed_hosts=["x.com"]))
    try:
        c = socket.create_connection(("127.0.0.1", proxy.port), 2)
        # a smuggled control char in the target must not be accepted
        c.sendall(b"CONNECT x.com\x00evil:443 HTTP/1.1\r\n\r\n")
        assert b"400" in _status_line(c.recv(4096))
        c.close()
    finally:
        proxy.stop()


def test_stop_is_idempotent():
    proxy = _proxy(AllowlistConfig(allowed_hosts=["x.com"]))
    proxy.stop()
    proxy.stop()  # must not raise
