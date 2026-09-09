"""Route matching and shared-secret verification for webhook requests.

Routes map URL paths to named event sources. Each route can optionally
authenticate the caller with the route's shared secret and extract event
types from headers.

Two verification modes, selected by ``RouteConfig.secret_algo``:

* ``hmac-sha256`` — the header carries an HMAC digest over the request BODY.
  Not replayable and not readable by a TLS-terminating hop.
* ``token`` — the header carries the shared secret VERBATIM, compared for
  equality in constant time.  Strictly weaker (replayable; exposed to anything
  that terminates TLS), and the only mode available for producers that do not
  sign bodies — GitLab's ``X-Gitlab-Token`` is the canonical case.  Pair it
  with TLS.

Both are one predicate to the caller: a route declaring ``secret_header`` +
``secret_algo`` is authenticated, and an incomplete or unrecognised pair is
REFUSED rather than downgraded to unsigned.
"""

import hashlib
import hmac
import json
import logging
from typing import Any, Dict, Optional, Tuple

from .config import SECRET_ALGO_TOKEN, SECRET_ALGOS, RouteConfig

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


def _constant_time_equals(a: str, b: str) -> bool:
    """Constant-time string comparison that never raises on the input.

    ``hmac.compare_digest`` rejects a non-ASCII ``str`` with ``TypeError``, and
    both operands here come off the wire — a header value an attacker chooses.
    Encoding to UTF-8 first makes the comparison total (identical results for
    the ASCII inputs both modes actually produce) so a crafted header can never
    turn a verification failure into a 500.
    """
    return hmac.compare_digest(a.encode('utf-8'), b.encode('utf-8'))


def verify_signature(
    body: bytes,
    secret: str,
    signature_header: str,
    algo: str,
) -> bool:
    """Verify a webhook request against the route's shared secret.

    Dispatches on ``algo`` (one of ``SECRET_ALGOS``):

    * ``'hmac-sha256'`` — recomputes an HMAC-SHA256 over ``body`` keyed by
      ``secret`` and compares it to the header value, which may optionally
      carry a ``'sha256='`` prefix (GitHub convention).
    * ``'token'`` — compares the header value to ``secret`` itself, in
      constant time.  No prefix is stripped and the body is not read: the
      header IS the credential, so any transformation of it would either
      accept a secret nobody configured or reject one that matches.

    An algorithm outside ``SECRET_ALGOS`` returns False (fail-closed) — it is
    already a hard error at config-validation time, and reaching here means a
    route was constructed programmatically around that check.

    Args:
        body: Raw request body bytes (read only by ``'hmac-sha256'``).
        secret: The route's shared secret.
        signature_header: Value of ``route.secret_header`` from the request.
        algo: Verification mode; see ``SECRET_ALGOS``.

    Returns:
        True if the request is authenticated.
    """
    if algo not in SECRET_ALGOS:
        logger.warning("Unsupported signature algorithm: %s", algo)
        return False

    if not signature_header:
        return False

    if algo == SECRET_ALGO_TOKEN:
        # An empty configured secret must never authenticate an empty header.
        # The caller already 500s on a missing secret; this is the second lock.
        if not secret:
            return False
        return _constant_time_equals(signature_header, secret)

    # Strip optional 'sha256=' prefix (GitHub convention)
    expected_sig = signature_header
    if expected_sig.startswith('sha256='):
        expected_sig = expected_sig[7:]

    computed = hmac.new(
        secret.encode('utf-8'),
        body,
        hashlib.sha256,
    ).hexdigest()

    return _constant_time_equals(computed, expected_sig)


def _authenticate_route(
    body: bytes,
    lower_headers: Dict[str, str],
    route_name: str,
    route: RouteConfig,
    global_secret: Optional[str],
    transport_authenticated: bool,
) -> Optional[Tuple[int, str]]:
    """Decide whether a request to ``route`` is authenticated.

    A route is authenticated by ONE of: the route's shared secret (verified
    here, in whichever mode ``secret_algo`` names), mutual TLS / an IP allowlist
    (``transport_authenticated``, decided by ``WebhookHTTPServer``), or an
    explicit ``allow_unauthenticated`` opt-in.  A route with none of these is
    REFUSED (fail-closed) so an operator cannot accidentally expose an open
    endpoint that dispatches into agent sessions.

    A route declaring EITHER ``secret_header`` or ``secret_algo`` is treated as
    "intends a secret check": an incomplete pair is a misconfiguration and is
    refused (500), never silently downgraded to unsigned — otherwise a typo
    would skip verification while transport auth quietly let the request
    through.  That predicate is mode-agnostic on purpose: adding ``'token'`` to
    the vocabulary widens what ``secret_algo`` may SAY, never what it may omit.

    Args:
        body: Raw request body bytes.
        lower_headers: Request headers with lowercased keys.
        route_name: Name of the matched route (for logs).
        route: The matched route.
        global_secret: Fallback secret from ``WebhookConfig``.
        transport_authenticated: Whether the deployment authenticates below
            the shared-secret layer.

    Returns:
        None when the request is authenticated, else ``(status, message)``
        describing the refusal.
    """
    intends_secret_check = bool(route.secret_header or route.secret_algo)

    if not intends_secret_check:
        if transport_authenticated or route.allow_unauthenticated:
            return None
        logger.warning(
            "Route '%s' has no secret verification and the deployment "
            "provides no mutual-TLS / IP-allowlist auth — refusing the request.",
            route_name,
        )
        return 401, (
            "Route has no secret verification configured. Set secret_header + "
            "secret_algo ('hmac-sha256' where the producer signs bodies, 'token' "
            "where it sends a plain shared secret such as X-Gitlab-Token), front "
            "the listener with mutual TLS or an allowed_ips allowlist, or set "
            "allow_unauthenticated: true to deliberately accept unsigned requests."
        )

    if not (route.secret_header and route.secret_algo):
        logger.warning(
            "Route '%s' has an incomplete secret config — it sets "
            "secret_header or secret_algo but not both; refusing (fail-closed).",
            route_name,
        )
        return 500, (
            "Server misconfigured: route sets secret_header or secret_algo but "
            "not both, so secret verification cannot run."
        )

    # An algo the tree does not implement never degrades to unverified.
    # validate_config() rejects it, but a programmatically built route bypasses
    # that, and verify_signature() alone returning False would read to the
    # caller as a bad signature rather than as a broken server.
    if route.secret_algo not in SECRET_ALGOS:
        logger.warning(
            "Route '%s' declares unknown secret_algo '%s'; refusing (fail-closed).",
            route_name, route.secret_algo,
        )
        return 500, (
            "Server misconfigured: unknown secret_algo, so secret "
            "verification cannot run."
        )

    secret = route.metadata.get('secret') or global_secret
    if not secret:
        logger.warning(
            "Route '%s' requires secret verification but no secret configured",
            route_name,
        )
        return 500, "Server misconfigured: no secret for signature verification"

    sig_value = lower_headers.get(route.secret_header.lower(), '')
    if not verify_signature(body, secret, sig_value, route.secret_algo):
        logger.warning(
            "Secret verification (%s) failed for route '%s'",
            route.secret_algo, route_name,
        )
        return 403, "Signature verification failed"

    return None


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
            below the shared-secret layer (mutual TLS or a configured IP
            allowlist). The caller (``WebhookHTTPServer``) decides this. It lets
            a route without a secret still accept requests without an explicit
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

    auth_failure = _authenticate_route(
        body, lower_headers, route_name, route,
        global_secret=global_secret,
        transport_authenticated=transport_authenticated,
    )
    if auth_failure is not None:
        status, message = auth_failure
        return None, status, message

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
