"""Lightweight HTTP server for receiving webhooks.

Runs Python's stdlib ``http.server.HTTPServer`` in a dedicated daemon thread.
Routes incoming POSTs to a callback after body-size checks and route matching.
The callback is called from the server thread — it must be thread-safe
(``TaskEventBus.publish()`` is).

No external dependencies — uses only the standard library.
"""

import json
import logging
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Callable, Dict, Optional

from .config import RouteConfig, WebhookConfig
from .routes import match_route, parse_webhook_request

logger = logging.getLogger(__name__)


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

        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the HTTP server in a background daemon thread.

        Raises:
            OSError: If the port is already in use or bind fails.
        """
        if self.is_running:
            logger.warning("Webhook HTTP server already running")
            return

        handler = _create_handler(self)
        self._server = HTTPServer((self.config.host, self.config.port), handler)
        self._server.timeout = 1.0  # Allow periodic shutdown checks

        self._thread = threading.Thread(
            target=self._serve,
            name=f"webhook-http-{self.config.port}",
            daemon=True,
        )
        self._thread.start()
        self.is_running = True
        logger.info(
            "Webhook HTTP server listening on %s:%d",
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

    def get_stats(self) -> Dict[str, Any]:
        """Return server statistics.

        Returns:
            Dict with listening status, host, port, route stats, and totals.
        """
        route_stats = []
        for name, route in self.config.routes.items():
            route_stats.append({
                "name": name,
                "path": route.path,
                "events_received": self.events_received.get(name, 0),
            })
        return {
            "listening": self.is_running,
            "host": self.config.host,
            "port": self.config.port,
            "routes": route_stats,
            "total_events_received": sum(self.events_received.values()),
        }


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
            """Handle POST request — the main webhook entry point."""
            config = server_instance.config

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
