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

from .config import RouteConfig, WebhookConfig
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

        # Fail-loud on posture: no routes at all, or routes that accept unsigned
        # requests, are surfaced at startup so the operator can't miss them.
        if not self.config.routes:
            logger.warning(
                "Webhook listener on %s:%d has NO routes configured — every "
                "request will 404. Declare routes in the webhook config.",
                self.config.host, self.config.port,
            )
        transport_auth = self.transport_authenticated()
        for name, route in self.config.routes.items():
            has_hmac = bool(route.secret_header and route.secret_algo)
            if not has_hmac and not transport_auth and route.allow_unauthenticated:
                logger.warning(
                    "Webhook route '%s' (%s) accepts UNSIGNED requests "
                    "(allow_unauthenticated=true, no mutual-TLS / IP-allowlist) "
                    "— anyone who can reach %s:%d can inject events into agent "
                    "sessions.",
                    name, route.path, self.config.host, self.config.port,
                )

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
        """True when the deployment authenticates callers below the HMAC layer.

        Either mutual TLS (``tls.enabled`` with a ``ca_certfile``, so clients
        must present a valid client certificate) or a non-empty IP allowlist.
        A route without an HMAC secret may accept requests only when this is
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
            5. HMAC signature verification (per-route)
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
            )

            if event is None:
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
