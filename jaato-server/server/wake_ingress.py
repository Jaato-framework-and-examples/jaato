"""Daemon-tier HTTP wake ingress (mode-B Stage-B verify).

This is the always-up ingress that turns an authenticated external wake into a
``session.wake``.  It lives at the DAEMON tier — NOT the runner-tier webhook
plugin — because it must survive session unload (a wake can arrive days after
the session went idle-detached; the runner-bound webhook listener would be gone).

Two stages:

- **Stage A — transport hygiene** (:class:`WakeIngressServer`): IP-allowlist
  (CIDR), per-IP token-bucket rate limit, optional TLS.  DoS/abuse bounding,
  NOT authorization.  Derived from ``shared/plugins/webhook/http_server.py``.
- **Stage B — the authoritative auth** (:func:`process_wake`): resolve
  ``wake_ref`` → binding → OR-verify the Ed25519/RSA signature over the RAW
  request body against ``binding.trust_keys`` → replay-window check on the
  signed ``ts`` → drive ``wake_session``.  Session-scoped: the key is the target
  session's own declared trust, never a daemon-global config.

Canonical signed body (sealed with the relay author 2026-07-05):
``B = {"wake_ref", "text", "source", "event_id", "ts"}`` JSON; the relay signs
the RAW bytes of ``B`` and sends ``X-Jaato-Wake-Signature: base64(sig)`` (plus an
untrusted ``X-Jaato-Wake-Key-Id`` hint).  We verify over the raw bytes we
received — never a re-serialization — and only ACT after verification passes.
"""
from __future__ import annotations

import ipaddress
import json
import logging
import ssl
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from .session_manager import WakeOutcome
from .wake_binding_registry import WakeBinding
from .wake_verify import verify_wake_signature

logger = logging.getLogger(__name__)

SIGNATURE_HEADER = "x-jaato-wake-signature"
KEY_ID_HEADER = "x-jaato-wake-key-id"  # untrusted hint; we OR-verify all keys

# Type of the injected wake driver: (session_id, text, source, event_id) ->
# (WakeOutcome, detail).  In production this is SessionManager.wake_session.
WakeFn = Callable[[str, str, str, Optional[str]], Tuple["WakeOutcome", str]]
ResolveFn = Callable[[str], Optional[WakeBinding]]


@dataclass
class WakeIngressConfig:
    """Operator config for the wake ingress (``.jaato/wake.json``).

    ``replay_window_seconds`` is a knob (not a hardcoded constant) so an
    operator with unusual relay clock skew can tune it.  ``host`` defaults to
    loopback — expose via a reverse proxy or set ``0.0.0.0`` deliberately.
    """
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 9110
    path: str = "/wake"
    # Operator-declared PUBLIC URL where this daemon's ingress is reachable
    # (through the reverse proxy / tunnel the operator runs).  ``host``/``port``
    # are the local bind; ``public_url`` is what an external relay must POST to.
    # Surfaced to a binding session via WakeBindResultEvent.endpoint so the bot
    # can embed it as the relay's routing marker WITHOUT any bot-side config.
    public_url: str = ""
    allowed_ips: List[str] = field(default_factory=list)
    rate_limit_per_second: int = 0
    replay_window_seconds: int = 300
    max_body_size: int = 1 << 20  # 1 MiB
    tls_certfile: Optional[str] = None
    tls_keyfile: Optional[str] = None

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "WakeIngressConfig":
        """Build from a dict, coercing defensively — a bad field type falls
        back to that field's default rather than raising (mirrors
        :meth:`from_file`'s "malformed config never crashes" contract)."""
        c = cls()
        if not isinstance(d, Mapping):
            return c

        def _int(key: str, default: int) -> int:
            try:
                return int(d.get(key, default))
            except (TypeError, ValueError):
                logger.warning("wake ingress: bad %s %r — using default %d",
                               key, d.get(key), default)
                return default

        c.enabled = bool(d.get("enabled", c.enabled))
        c.host = str(d.get("host", c.host))
        c.port = _int("port", c.port)
        c.path = str(d.get("path", c.path))
        pub = d.get("public_url", c.public_url)
        c.public_url = str(pub).strip() if isinstance(pub, str) else ""
        ips = d.get("allowed_ips", [])
        c.allowed_ips = [str(x) for x in ips] if isinstance(ips, list) else []
        c.rate_limit_per_second = _int("rate_limit_per_second",
                                       c.rate_limit_per_second)
        c.replay_window_seconds = _int("replay_window_seconds",
                                       c.replay_window_seconds)
        c.max_body_size = _int("max_body_size", c.max_body_size)
        tls = d.get("tls", {})
        if isinstance(tls, Mapping) and tls.get("enabled"):
            cert, key = tls.get("certfile"), tls.get("keyfile")
            if cert and key:
                c.tls_certfile, c.tls_keyfile = str(cert), str(key)
            else:
                # Fail closed: TLS was requested but is incomplete — refuse to
                # serve the ingress in plaintext.  Disable rather than downgrade.
                logger.error(
                    "wake ingress: tls.enabled but certfile/keyfile missing — "
                    "disabling ingress (refusing to serve plaintext)")
                c.enabled = False
        return c

    @classmethod
    def from_file(cls, path) -> "WakeIngressConfig":
        """Load from a daemon-level JSON file (``~/.jaato/wake.json``).  A
        missing/unreadable file yields the default (disabled) config — the
        ingress is opt-in."""
        import pathlib
        p = pathlib.Path(path)
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return cls()
        except (OSError, ValueError) as exc:
            logger.warning("wake ingress: config unreadable at %s: %s — disabled",
                           p, exc)
            return cls()
        return cls.from_dict(data)


def _lower_headers(headers: Mapping[str, str]) -> Dict[str, str]:
    """Case-insensitive header lookup (HTTP header names are case-insensitive)."""
    return {str(k).lower(): v for k, v in headers.items()}


def process_wake(
    raw_body: bytes,
    headers: Mapping[str, str],
    resolve_binding: ResolveFn,
    wake_fn: WakeFn,
    replay_window_seconds: int,
    now: float,
) -> Tuple[int, str]:
    """Stage B: authenticate and route a wake request that has already passed
    transport hygiene.  Returns ``(http_status, detail)``.

    Pure + injectable (``resolve_binding`` / ``wake_fn``) so the security-
    critical flow is unit-testable without a socket or a live SessionManager.

    Discipline: parse only to READ the routing ref, verify the signature over
    the RAW body, and ACT (drive the wake) only AFTER verification + replay
    checks pass.  Status mapping follows the sealed contract — 2xx success
    (incl. idempotent duplicate), 4xx permanent (don't retry), 5xx transient.
    """
    h = _lower_headers(headers)
    sig_b64 = h.get(SIGNATURE_HEADER)
    if not sig_b64:
        return (401, "missing wake signature")

    # Parse the body ONCE — used pre-verify only for routing (wake_ref); the
    # rest of its fields are consumed only after the signature validates.
    try:
        obj = json.loads(raw_body)
    except (ValueError, UnicodeDecodeError):
        return (400, "malformed body (not JSON)")
    if not isinstance(obj, dict):
        return (400, "malformed body (not an object)")

    wake_ref = obj.get("wake_ref")
    if not isinstance(wake_ref, str) or not wake_ref:
        return (400, "missing wake_ref")

    binding = resolve_binding(wake_ref)
    if binding is None:
        return (404, "unknown or expired wake_ref")

    # AUTHORITATIVE gate: signature over the RAW bytes against the binding's keys.
    if not verify_wake_signature(binding.trust_keys, raw_body, sig_b64):
        return (401, "signature verification failed")

    # From here the body is authentic — enforce replay + read the payload.
    ts = obj.get("ts")
    if not isinstance(ts, (int, float)) or isinstance(ts, bool):
        return (400, "missing or non-numeric ts")
    if abs(now - float(ts)) > replay_window_seconds:
        return (401, "stale timestamp (outside replay window)")

    text = obj.get("text")
    if not isinstance(text, str) or not text:
        return (400, "missing text")
    source = obj.get("source")
    source = source if isinstance(source, str) and source else "wake"
    event_id = obj.get("event_id")
    event_id = event_id if isinstance(event_id, str) and event_id else None

    outcome, detail = wake_fn(binding.session_id, text, source, event_id)
    if outcome in (WakeOutcome.OK, WakeOutcome.DUPLICATE):
        return (200, detail)
    if outcome in (WakeOutcome.REVIVE_FAILED, WakeOutcome.NOT_DRIVABLE):
        return (503, detail)  # transient — safe to retry
    return (400, detail)  # INVALID / UNRESOLVED — permanent


class WakeIngressServer:
    """Daemon-tier HTTP listener for the wake ingress.

    Stage A (transport hygiene) lives here — IP-allowlist + token-bucket rate
    limit + optional TLS, derived from the webhook HTTP server — and each POST
    to ``config.path`` is handed to :func:`process_wake` for Stage B.  Runs in a
    daemon thread; started once at daemon boot when the config is enabled.
    """

    def __init__(
        self,
        config: WakeIngressConfig,
        resolve_binding: ResolveFn,
        wake_fn: WakeFn,
    ) -> None:
        self.config = config
        self._resolve_binding = resolve_binding
        self._wake_fn = wake_fn
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._rate_lock = threading.Lock()
        self._rate_buckets: Dict[str, List[float]] = {}
        self._allowed_networks = []
        for cidr in config.allowed_ips:
            try:
                self._allowed_networks.append(ipaddress.ip_network(cidr, strict=False))
            except ValueError:
                logger.warning("wake ingress: ignoring invalid allowed_ips entry %r",
                               cidr)

    # ---- Stage A — transport hygiene (derived from webhook http_server) ------

    def check_ip_allowed(self, client_ip: str) -> bool:
        if not self._allowed_networks:
            return True
        try:
            addr = ipaddress.ip_address(client_ip)
            if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped:
                addr = addr.ipv4_mapped
            return any(addr in net for net in self._allowed_networks)
        except ValueError:
            logger.warning("wake ingress: unparseable client IP %s", client_ip)
            return False

    def check_rate_limit(self, client_ip: str) -> bool:
        limit = self.config.rate_limit_per_second
        if limit <= 0:
            return True
        now = time.monotonic()
        with self._rate_lock:
            bucket = self._rate_buckets.get(client_ip)
            if bucket is None:
                self._rate_buckets[client_ip] = [limit - 1, now]
                return True
            elapsed = now - bucket[1]
            bucket[0] = min(limit, bucket[0] + elapsed * limit)
            bucket[1] = now
            if bucket[0] >= 1:
                bucket[0] -= 1
                return True
            return False

    # ---- lifecycle -----------------------------------------------------------

    def start(self) -> bool:
        """Bind + serve in a daemon thread.  Returns True if listening."""
        if not self.config.enabled:
            return False
        try:
            handler = _make_handler(self)
            httpd = ThreadingHTTPServer((self.config.host, self.config.port), handler)
            if self.config.tls_certfile and self.config.tls_keyfile:
                ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                ctx.load_cert_chain(self.config.tls_certfile, self.config.tls_keyfile)
                httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
        except (OSError, ssl.SSLError) as exc:
            logger.error("wake ingress: failed to start on %s:%s — %s",
                         self.config.host, self.config.port, exc)
            return False
        self._httpd = httpd
        self._thread = threading.Thread(
            target=httpd.serve_forever, name="wake-ingress", daemon=True)
        self._thread.start()
        logger.info("wake ingress listening on %s:%s%s (tls=%s)",
                    self.config.host, self.config.port, self.config.path,
                    bool(self.config.tls_certfile))
        return True

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None


def _make_handler(server: WakeIngressServer):
    class WakeHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # silence default stderr logging
            logger.debug("wake ingress: " + fmt, *args)

        def _respond(self, status: int, detail: str) -> None:
            body = json.dumps({"status": status, "detail": detail}).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):  # noqa: N802
            cfg = server.config
            if self.path.split("?", 1)[0] != cfg.path:
                self._respond(404, "no such path")
                return
            client_ip = self.client_address[0]
            if not server.check_ip_allowed(client_ip):
                self._respond(403, "forbidden")
                return
            if not server.check_rate_limit(client_ip):
                self._respond(429, "rate limit exceeded")
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
            except (TypeError, ValueError):
                self._respond(400, "bad Content-Length")
                return
            if length <= 0 or length > cfg.max_body_size:
                self._respond(413, "missing or oversized body")
                return
            try:
                raw = self.rfile.read(length)
            except OSError as exc:
                self._respond(400, f"failed to read body: {exc}")
                return
            headers = {k: v for k, v in self.headers.items()}
            try:
                status, detail = process_wake(
                    raw, headers, server._resolve_binding, server._wake_fn,
                    cfg.replay_window_seconds, time.time())
            except Exception:  # noqa: BLE001 — never leak a stack to the caller
                logger.exception("wake ingress: unhandled error processing request")
                self._respond(500, "internal error")
                return
            self._respond(status, detail)

    return WakeHandler
