"""Lightweight HTTP server for receiving webhooks.

Runs Python's stdlib ``http.server.HTTPServer`` in a dedicated daemon thread.
Routes incoming POSTs to a callback after body-size checks and route matching.
The callback is called from the server thread — it must be thread-safe
(``TaskEventBus.publish()`` is).

Corporate hardening features (all stdlib, no external deps):
- TLS/SSL with optional mutual TLS (client certificate verification)
- IP allowlisting with CIDR support via ``ipaddress`` module
- Per-IP token-bucket rate limiting
"""

import ipaddress
import json
import logging
import ssl
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Callable, Dict, List, Optional, Union

from .config import SECRET_ALGO_TOKEN, RouteConfig, WebhookConfig
from .replay import ReplayCache
from .signature_schemes import DEFAULT_MAX_AGE_SECONDS
from .routes import match_route, parse_webhook_request

logger = logging.getLogger(__name__)

# Type alias for parsed IP networks/addresses used in allowlisting
_IPNetwork = Union[ipaddress.IPv4Network, ipaddress.IPv6Network]
_IPAddress = Union[ipaddress.IPv4Address, ipaddress.IPv6Address]


class WebhookHTTPServer:
    """HTTP server that receives webhooks in a background thread.

    Lifecycle:
        1. ``__init__()`` — stores config, no socket bound yet.
        2. ``start()`` — binds socket, starts server thread.
        3. Requests handled → ``on_webhook`` called for each valid POST.
        4. ``stop()`` — shuts down server, joins thread.

    Thread safety: ``on_webhook`` is called from the server thread.
    The caller must ensure the callback is safe to invoke from any thread.

    Attributes:
        config: The WebhookConfig controlling host, port, routes, etc.
        on_webhook: Callback invoked for each valid webhook POST.
        is_running: Whether the server thread is alive.
        events_received: Per-route counter of received events.
    """

    def __init__(
        self,
        config: WebhookConfig,
        on_webhook: Callable[[str, str, Dict[str, str], Any], None],
    ):
        """Initialize the webhook HTTP server.

        Args:
            config: Webhook configuration with host, port, routes, etc.
            on_webhook: Callback invoked for each valid POST with args:
                (route_name, event_type, headers_dict, payload_dict).
        """
        self.config = config
        self.on_webhook = on_webhook
        self.is_running = False
        self.events_received: Dict[str, int] = {
            name: 0 for name in config.routes
        }
        self.requests_blocked_ip: int = 0
        self.requests_blocked_rate: int = 0

        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

        # Pre-parse allowed_ips into network objects for fast matching
        self._allowed_networks: List[_IPNetwork] = []
        for entry in config.allowed_ips:
            try:
                if '/' in entry:
                    self._allowed_networks.append(
                        ipaddress.ip_network(entry, strict=False)
                    )
                else:
                    # Single IP → /32 or /128 network for uniform matching
                    addr = ipaddress.ip_address(entry)
                    prefix = 32 if isinstance(addr, ipaddress.IPv4Address) else 128
                    self._allowed_networks.append(
                        ipaddress.ip_network(f"{entry}/{prefix}", strict=False)
                    )
            except ValueError:
                logger.warning("Ignoring invalid allowed_ip entry: %s", entry)

        # Rate limiter state: ip_str → (token_count, last_refill_time)
        self._rate_buckets: Dict[str, List[float]] = {}
        self._rate_lock = threading.Lock()

        # One replay cache per listener, shared by every route; keys are
        # namespaced by route name so two routes cannot collide on a shared
        # delivery-id space.  Constructed unconditionally: which routes it
        # actually protects is a per-route property (``replay_protected``),
        # decided per request, and an unused cache costs one empty dict.
        self._replay_cache = ReplayCache(max_entries=config.replay_cache_size)
        self.requests_refused_replay: int = 0

    def start(self) -> None:
        """Start the HTTP server in a background daemon thread.

        When TLS is enabled in config, wraps the server socket with an
        ``ssl.SSLContext``. Supports server-only TLS and mutual TLS
        (when ``ca_certfile`` is set, clients must present a valid cert).

        Raises:
            OSError: If the port is already in use or bind fails.
            ssl.SSLError: If TLS cert/key files are invalid.
            FileNotFoundError: If TLS cert/key files don't exist.
        """
        if self.is_running:
            logger.warning("Webhook HTTP server already running")
            return

        self._warn_on_weak_posture()

        handler = _create_handler(self)
        self._server = HTTPServer((self.config.host, self.config.port), handler)
        self._server.timeout = 1.0  # Allow periodic shutdown checks

        # Wrap socket with TLS if configured
        tls = self.config.tls
        if tls.enabled:
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.load_cert_chain(certfile=tls.certfile, keyfile=tls.keyfile)
            if tls.ca_certfile:
                ctx.load_verify_locations(cafile=tls.ca_certfile)
                ctx.verify_mode = ssl.CERT_REQUIRED
            else:
                ctx.verify_mode = ssl.CERT_NONE
            self._server.socket = ctx.wrap_socket(
                self._server.socket, server_side=True
            )
            logger.info(
                "TLS enabled (mutual=%s)", bool(tls.ca_certfile)
            )

        self._thread = threading.Thread(
            target=self._serve,
            name=f"webhook-http-{self.config.port}",
            daemon=True,
        )
        self._thread.start()
        self.is_running = True

        scheme = "https" if tls.enabled else "http"
        logger.info(
            "Webhook HTTP server listening on %s://%s:%d",
            scheme,
            self.config.host,
            self.config.port,
        )

    def _serve(self) -> None:
        """Server loop running in the background thread."""
        try:
            self._server.serve_forever()
        except Exception:
            logger.exception("Webhook HTTP server crashed")
        finally:
            self.is_running = False

    def _warn_on_weak_posture(self) -> None:
        """Announce weak authentication postures at listener startup.

        Called once from :meth:`start`, before the socket is bound, so an
        operator sees the posture in the same place they see the listener come
        up.  Three cases, each the framework's standing "announce, never
        silently accept" rule (cf. ``--ws-unsafe-no-auth`` and
        ``scrub_secret_env: none``):

        * **No routes at all** — the listener 404s everything.
        * **An unsigned route** (``allow_unauthenticated`` with no transport
          auth) — anyone who can reach the port can drive agent sessions.
        * **A plain-token route** (``secret_algo='token'``) — weaker than an
          HMAC over the body: the secret is in every request, so it is readable
          by anything that terminates TLS, and requests replay against any
          payload.  Louder when TLS is off, since the secret is then sent in
          the clear.  Announcing it is what stops ``token`` from becoming the
          quiet path of least resistance for a producer that DOES sign bodies.

        A fourth, orthogonal to all three, is delegated to
        :meth:`_announce_replay_posture`: how long a valid request STAYS
        valid.  It is asked of every route, because it is a property of the
        signature scheme rather than of the authentication mode.
        """
        if not self.config.routes:
            logger.warning(
                "Webhook listener on %s:%d has NO routes configured — every "
                "request will 404. Declare routes in the webhook config.",
                self.config.host, self.config.port,
            )
            return

        transport_auth = self.transport_authenticated()
        for name, route in self.config.routes.items():
            has_secret = bool(route.secret_header and route.secret_algo)
            if not has_secret and not transport_auth and route.allow_unauthenticated:
                logger.warning(
                    "Webhook route '%s' (%s) accepts UNSIGNED requests "
                    "(allow_unauthenticated=true, no mutual-TLS / IP-allowlist) "
                    "— anyone who can reach %s:%d can inject events into agent "
                    "sessions.",
                    name, route.path, self.config.host, self.config.port,
                )
            elif has_secret and route.secret_algo == SECRET_ALGO_TOKEN:
                self._warn_plain_token_route(name, route)
            # Independent of HOW the route authenticates: announce for how
            # LONG that authentication stays valid.  An unconditional call so
            # this loop gains no branch (its score is near the ratchet).
            self._announce_replay_posture(name, route, has_secret)

    def _warn_plain_token_route(self, name: str, route: RouteConfig) -> None:
        """Warn that one route authenticates with a plain shared secret.

        Split from :meth:`_warn_on_weak_posture` so the TLS-on / TLS-off
        wording can differ without nesting another branch in the route loop.

        Args:
            name: The route's config key, so the operator can find it.
            route: The route declaring ``secret_algo='token'``.
        """
        if self.config.tls.enabled:
            logger.warning(
                "Webhook route '%s' (%s) authenticates with a PLAIN shared "
                "secret (secret_algo='token', header %s). The secret travels in "
                "every request and is readable by anything that terminates TLS; "
                "requests are replayable against any payload. Use "
                "secret_algo='hmac-sha256' if the producer signs request bodies.",
                name, route.path, route.secret_header,
            )
        else:
            logger.warning(
                "Webhook route '%s' (%s) authenticates with a PLAIN shared "
                "secret (secret_algo='token', header %s) over PLAINTEXT HTTP on "
                "%s:%d — the secret is sent in the clear and anyone who observes "
                "one request can replay it forever. Enable tls.enabled, or use "
                "secret_algo='hmac-sha256' if the producer signs request bodies.",
                name, route.path, route.secret_header,
                self.config.host, self.config.port,
            )

    def _announce_replay_posture(
        self, name: str, route: RouteConfig, has_secret: bool,
    ) -> None:
        """Announce how long an accepted request on ``route`` stays valid (#713).

        The three postures, and why each is at the level it is:

        * **No timestamp binding and no delivery id** — the WEAK default, and
          the state every route configured before #713 is in.  A signature over
          the body alone (or a plain token) authenticates the same bytes
          forever, so whoever observes one delivery replays it indefinitely.
          WARNING, naming both remedies.  This is the cost of the
          backward-compatibility choice made deliberately in #713: the
          protection could not be switched on for these routes, because the
          sender signs no timestamp and there is nothing to check — so what is
          switched on instead is saying so.
        * **A delivery id but no timestamp binding** — bounded protection:
          a verbatim replay and a sender retry are refused within the TTL, and
          an attacker who rewrites the unsigned id is not.  INFO, because the
          operator chose it knowingly and the bound is stated.
        * **A timestamp-bound scheme with the window disabled**
          (``max_age_seconds: 0``) — the explicit opt-out, so it announces
          itself, the posture ``scrub_secret_env: none`` and
          ``--ws-unsafe-no-auth`` already take.

        A timestamp-bound scheme with a live window says nothing: that is the
        strong posture, and a log line per listener start for the correct
        configuration is noise.

        Args:
            name: The route's config key, so the operator can find it.
            route: The route being announced.
            has_secret: Whether the route authenticates with a shared secret at
                all.  An unsigned route is already announced above, and adding
                "its signature never expires" to a route that has no signature
                would be false.
        """
        if not has_secret:
            return

        if route.binds_timestamp():
            if route.max_age_seconds <= 0:
                logger.warning(
                    "Webhook route '%s' (%s) uses signature scheme '%s' but has "
                    "max_age_seconds=0, so the freshness window is DISABLED — a "
                    "captured request stays valid indefinitely. The replay cache "
                    "still refuses repeats for %ds.",
                    name, route.path, route.signature_scheme,
                    DEFAULT_MAX_AGE_SECONDS,
                )
            return

        if route.replay_key_header:
            logger.info(
                "Webhook route '%s' (%s) binds no timestamp (signature_scheme "
                "'%s'), so a captured request does not expire; replays are "
                "refused for %ds by delivery id (%s). An attacker who rewrites "
                "that header is not caught — no sender here signs its headers.",
                name, route.path, route.signature_scheme,
                route.max_age_seconds or DEFAULT_MAX_AGE_SECONDS,
                route.replay_key_header,
            )
            return

        logger.warning(
            "Webhook route '%s' (%s) has NO replay protection: signature_scheme "
            "'%s' binds no timestamp and no replay_key_header is set, so a "
            "request captured once authenticates forever and re-drives agent "
            "sessions on every replay. Set signature_scheme 'slack-v0' or "
            "'stripe-v1' if the sender signs a timestamp, or replay_key_header "
            "(GitHub: X-GitHub-Delivery, GitLab: X-Gitlab-Event-UUID) for "
            "delivery-id deduplication.",
            name, route.path, route.signature_scheme,
        )

    def stop(self) -> None:
        """Stop the HTTP server and join the thread."""
        if self._server:
            self._server.shutdown()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        self.is_running = False
        self._server = None
        self._thread = None
        logger.info("Webhook HTTP server stopped")

    def transport_authenticated(self) -> bool:
        """True when the deployment authenticates callers below the secret layer.

        Either mutual TLS (``tls.enabled`` with a ``ca_certfile``, so clients
        must present a valid client certificate) or a non-empty IP allowlist.
        A route without a shared secret may accept requests only when this is
        True or the route sets ``allow_unauthenticated`` — otherwise the request
        is refused (fail-closed).
        """
        tls = self.config.tls
        mutual_tls = bool(tls.enabled and tls.ca_certfile)
        return mutual_tls or bool(self._allowed_networks)

    def check_ip_allowed(self, client_ip: str) -> bool:
        """Check if a client IP is in the allowed list.

        When ``allowed_ips`` is empty (default), all IPs are allowed.
        IPv4-mapped IPv6 addresses (``::ffff:1.2.3.4``) are normalized
        to their IPv4 equivalents before matching.

        Args:
            client_ip: Client IP address string from the socket.

        Returns:
            True if allowed (or no allowlist configured).
        """
        if not self._allowed_networks:
            return True
        try:
            addr = ipaddress.ip_address(client_ip)
            # Normalize IPv4-mapped IPv6 → IPv4
            if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped:
                addr = addr.ipv4_mapped
            return any(addr in net for net in self._allowed_networks)
        except ValueError:
            logger.warning("Cannot parse client IP for allowlist check: %s", client_ip)
            return False

    def check_rate_limit(self, client_ip: str) -> bool:
        """Check if a client IP is within the rate limit.

        Uses a simple token-bucket algorithm: each IP gets
        ``rate_limit_per_second`` tokens per second, with a burst capacity
        of ``rate_limit_per_second`` (i.e., 1 second of burst).

        When ``rate_limit_per_second`` is 0 (default), rate limiting is
        disabled and all requests pass.

        Args:
            client_ip: Client IP address string.

        Returns:
            True if within rate limit (or rate limiting disabled).
        """
        limit = self.config.rate_limit_per_second
        if limit <= 0:
            return True

        now = time.monotonic()
        with self._rate_lock:
            if client_ip not in self._rate_buckets:
                # [tokens_remaining, last_refill_time]
                self._rate_buckets[client_ip] = [limit - 1, now]
                return True

            bucket = self._rate_buckets[client_ip]
            elapsed = now - bucket[1]
            # Refill tokens based on elapsed time
            bucket[0] = min(limit, bucket[0] + elapsed * limit)
            bucket[1] = now

            if bucket[0] >= 1:
                bucket[0] -= 1
                return True
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Return server statistics.

        Returns:
            Dict with listening status, host, port, route stats, totals,
            and security counters.
        """
        route_stats = []
        for name, route in self.config.routes.items():
            route_stats.append({
                "name": name,
                "path": route.path,
                "events_received": self.events_received.get(name, 0),
            })
        stats = {
            "listening": self.is_running,
            "host": self.config.host,
            "port": self.config.port,
            "tls_enabled": self.config.tls.enabled,
            "routes": route_stats,
            "total_events_received": sum(self.events_received.values()),
        }
        if self._allowed_networks:
            stats["ip_allowlist_size"] = len(self._allowed_networks)
            stats["requests_blocked_ip"] = self.requests_blocked_ip
        if self.config.rate_limit_per_second > 0:
            stats["rate_limit_per_second"] = self.config.rate_limit_per_second
            stats["requests_blocked_rate"] = self.requests_blocked_rate
        stats["requests_refused_replay"] = self.requests_refused_replay
        stats["replay_cache"] = self._replay_cache.get_stats()
        return stats


def _create_handler(server_instance: WebhookHTTPServer):
    """Create a request handler class bound to a server instance.

    Uses a closure to pass the ``WebhookHTTPServer`` to the handler
    without subclassing or global state.

    Args:
        server_instance: The WebhookHTTPServer that owns this handler.

    Returns:
        A BaseHTTPRequestHandler subclass.
    """

    class WebhookHandler(BaseHTTPRequestHandler):
        """HTTP request handler for webhook POST requests.

        Only POST is accepted. All other methods return 405.
        Responses are JSON with appropriate status codes.
        """

        def do_POST(self):
            """Handle POST request — the main webhook entry point.

            Security checks are applied in order:
            1. IP allowlist (if configured)
            2. Rate limit (if configured)
            3. Route matching
            4. Body size limit
            5. Shared-secret verification (per-route: HMAC over the body, or a
               constant-time compare of a plain token header)
            """
            config = server_instance.config

            # IP allowlist check
            client_ip = self.client_address[0]
            if not server_instance.check_ip_allowed(client_ip):
                server_instance.requests_blocked_ip += 1
                logger.warning("Blocked request from non-allowed IP: %s", client_ip)
                self._respond(403, {"error": "Forbidden"})
                return

            # Rate limit check
            if not server_instance.check_rate_limit(client_ip):
                server_instance.requests_blocked_rate += 1
                self._respond(429, {"error": "Rate limit exceeded"})
                return

            # Route matching
            result = match_route(self.path, config.routes)
            if result is None:
                self._respond(404, {"error": "No route matches this path"})
                return
            route_name, route = result

            # Body size check
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length > config.max_body_size:
                self._respond(413, {"error": "Request body too large"})
                return

            # Read body
            try:
                body = self.rfile.read(content_length)
            except Exception as e:
                self._respond(400, {"error": f"Failed to read body: {e}"})
                return

            # Parse and validate
            headers = {k: v for k, v in self.headers.items()}
            event, err_status, err_msg = parse_webhook_request(
                body, headers, route_name, route, config.secret,
                transport_authenticated=server_instance.transport_authenticated(),
                replay_cache=server_instance._replay_cache,
            )

            if event is None:
                if err_status == 409:
                    server_instance.requests_refused_replay += 1
                self._respond(err_status, {"error": err_msg})
                return

            # Publish via callback
            try:
                server_instance.on_webhook(
                    route_name,
                    event["event_type"],
                    event["headers"],
                    event["payload"],
                )
                server_instance.events_received[route_name] = (
                    server_instance.events_received.get(route_name, 0) + 1
                )
            except Exception:
                logger.exception("Error in webhook callback for route '%s'", route_name)
                self._respond(500, {"error": "Internal processing error"})
                return

            self._respond(200, {"status": "accepted"})

        def do_GET(self):
            """Reject GET requests with 405."""
            self._respond(405, {"error": "Only POST is accepted"})

        def do_PUT(self):
            """Reject PUT requests with 405."""
            self._respond(405, {"error": "Only POST is accepted"})

        def do_DELETE(self):
            """Reject DELETE requests with 405."""
            self._respond(405, {"error": "Only POST is accepted"})

        def _respond(self, status: int, body: dict) -> None:
            """Send a JSON response.

            Args:
                status: HTTP status code.
                body: Response body dict (JSON-serialized).
            """
            payload = json.dumps(body).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args):
            """Route access logs through the plugin logger."""
            logger.debug("webhook-http: %s", format % args)

    return WebhookHandler
