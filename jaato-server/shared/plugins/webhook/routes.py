"""Route matching and HMAC secret verification for webhook requests.

Routes map URL paths to named event sources. Each route can optionally
verify request signatures using HMAC-SHA256 and extract event types
from headers.
"""

import hashlib
import hmac
import json
import logging
from typing import Any, Dict, Optional, Tuple

from .config import RouteConfig

logger = logging.getLogger(__name__)


def match_route(
    path: str,
    routes: Dict[str, RouteConfig],
) -> Optional[Tuple[str, RouteConfig]]:
    """Find the route that matches a request path.

    Routes are matched as exact path strings — no prefix matching or
    glob patterns. First match wins when multiple routes share a path
    (shouldn't happen with valid config).

    Args:
        path: Request URL path (e.g., '/webhook/github').
        routes: Named routes from WebhookConfig.

    Returns:
        Tuple of (route_name, RouteConfig) or None if no match.
    """
    for name, route in routes.items():
        if route.path == path:
            return name, route
    return None


def verify_signature(
    body: bytes,
    secret: str,
    signature_header: str,
    algo: str,
) -> bool:
    """Verify HMAC signature of a webhook request body.

    Only 'hmac-sha256' is supported. The signature header value may
    optionally be prefixed with 'sha256=' (GitHub style).

    Args:
        body: Raw request body bytes.
        secret: Shared secret for HMAC computation.
        signature_header: Value of the signature header from the request.
        algo: Algorithm string (must be 'hmac-sha256').

    Returns:
        True if signature is valid.
    """
    if algo != 'hmac-sha256':
        logger.warning("Unsupported signature algorithm: %s", algo)
        return False

    if not signature_header:
        return False

    # Strip optional 'sha256=' prefix (GitHub convention)
    expected_sig = signature_header
    if expected_sig.startswith('sha256='):
        expected_sig = expected_sig[7:]

    computed = hmac.new(
        secret.encode('utf-8'),
        body,
        hashlib.sha256,
    ).hexdigest()

    return hmac.compare_digest(computed, expected_sig)


def parse_webhook_request(
    body: bytes,
    headers: Dict[str, str],
    route_name: str,
    route: RouteConfig,
    global_secret: Optional[str] = None,
    transport_authenticated: bool = False,
) -> Tuple[Optional[Dict[str, Any]], Optional[int], Optional[str]]:
    """Parse and validate a webhook request against a matched route.

    Performs content-type check, secret verification, JSON parsing,
    and event type extraction.

    Args:
        body: Raw request body bytes.
        headers: Request headers (case-insensitive keys recommended).
        route_name: Name of the matched route.
        route: RouteConfig for the matched route.
        global_secret: Global secret from WebhookConfig (used if route
            has secret_header but no per-route secret in metadata).
        transport_authenticated: True when the deployment authenticates callers
            below the HMAC layer (mutual TLS or a configured IP allowlist). The
            caller (``WebhookHTTPServer``) decides this. It lets a route without
            an HMAC secret still accept requests without an explicit
            ``allow_unauthenticated`` opt-in.

    Returns:
        Tuple of (parsed_event_dict, None, None) on success, or
        (None, http_status_code, error_message) on failure.
    """
    # Case-insensitive header lookup
    lower_headers = {k.lower(): v for k, v in headers.items()}

    # Content-type check
    content_type = lower_headers.get('content-type', '')
    if 'application/json' not in content_type:
        return None, 415, "Content-Type must be application/json"

    # Secret verification.  A route is authenticated by ONE of: an HMAC secret
    # (verified here), mutual TLS / an IP allowlist (transport_authenticated,
    # decided by the caller), or an explicit allow_unauthenticated opt-in.  A
    # route with none of these is REFUSED (fail-closed) so an operator cannot
    # accidentally expose an open endpoint that dispatches into agent sessions.
    #
    # A route that declares EITHER secret_header or secret_algo is treated as
    # "intends HMAC": an incomplete pair is a misconfiguration and is refused
    # (500), never silently downgraded to unsigned — otherwise a typo would
    # skip verification while transport auth quietly let the request through.
    intends_hmac = bool(route.secret_header or route.secret_algo)
    if intends_hmac:
        if not (route.secret_header and route.secret_algo):
            logger.warning(
                "Route '%s' has an incomplete signature config — it sets "
                "secret_header or secret_algo but not both; refusing (fail-closed).",
                route_name,
            )
            return None, 500, (
                "Server misconfigured: route sets secret_header or secret_algo but "
                "not both, so signature verification cannot run."
            )
        secret = route.metadata.get('secret') or global_secret
        if not secret:
            logger.warning(
                "Route '%s' requires signature verification but no secret configured",
                route_name,
            )
            return None, 500, "Server misconfigured: no secret for signature verification"

        sig_value = lower_headers.get(route.secret_header.lower(), '')
        if not verify_signature(body, secret, sig_value, route.secret_algo):
            logger.warning("HMAC verification failed for route '%s'", route_name)
            return None, 403, "Signature verification failed"
    elif not (transport_authenticated or route.allow_unauthenticated):
        logger.warning(
            "Route '%s' has no signature verification and the deployment "
            "provides no mutual-TLS / IP-allowlist auth — refusing the request.",
            route_name,
        )
        return None, 401, (
            "Route has no signature verification configured. Set secret_header + "
            "secret_algo, front the listener with mutual TLS or an allowed_ips "
            "allowlist, or set allow_unauthenticated: true to deliberately accept "
            "unsigned requests."
        )

    # Parse JSON body
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        return None, 400, f"Invalid JSON body: {e}"

    # Extract event type from header
    event_type = "unknown"
    if route.event_type_header:
        event_type = lower_headers.get(
            route.event_type_header.lower(), "unknown"
        )

    # Build event dict
    event = {
        "source": route_name,
        "event_type": event_type,
        "headers": headers,
        "payload": payload,
        "metadata": route.metadata,
    }

    return event, None, None
