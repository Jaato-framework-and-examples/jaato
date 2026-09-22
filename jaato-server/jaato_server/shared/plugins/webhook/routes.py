"""Route matching and shared-secret verification for webhook requests.

Routes map URL paths to named event sources. Each route can optionally
authenticate the caller with the route's shared secret and extract event
types from headers.

Two verification modes, selected by ``RouteConfig.secret_algo``:

* ``hmac-sha256`` — the header carries an HMAC digest.  Not readable by a
  TLS-terminating hop.
* ``token`` — the header carries the shared secret VERBATIM, compared for
  equality in constant time.  Strictly weaker (exposed to anything that
  terminates TLS, and replayable against any payload), and the only mode
  available for producers that do not sign bodies — GitLab's
  ``X-Gitlab-Token`` is the canonical case.  Pair it with TLS.

Both are one predicate to the caller: a route declaring ``secret_header`` +
``secret_algo`` is authenticated, and an incomplete or unrecognised pair is
REFUSED rather than downgraded to unsigned.

**What ``secret_algo`` cannot say is what was SIGNED** (#713).  An HMAC over
the body alone binds no time, so the digest authenticates the same request
forever: whoever observes one delivery — a proxy log, a mirrored port, a
misrouted retry — replays it verbatim, indefinitely, and it authenticates every
time.  On a listener whose whole purpose is to drive agent sessions that is not
a duplicate row, it is a re-triggered turn: tool calls, ledger spend, and
whatever side effects the persona authorises.

Two mechanisms close it, and they are complementary rather than alternatives:

1. **A freshness window**, from ``RouteConfig.signature_scheme`` +
   ``max_age_seconds``.  Only a scheme that binds a timestamp INTO the
   signature can carry one — see :mod:`.signature_schemes`.  It bounds how long
   a captured request stays useful, and catches nothing inside the window.
2. **A replay cache**, :class:`.replay.ReplayCache`, keyed per delivery and
   TTL'd to that window.  This is the part that catches a replay inside it.

Order of operations is load-bearing.  The signature is verified FIRST, because
in both timestamped schemes the timestamp is part of the signed payload and is
therefore only evidence once the signature holds.  The freshness window is
second.  The replay cache is recorded LAST, only for a delivery that has passed
both — recording earlier would let anyone who can reach the port poison the
cache with a guessed delivery id and have the genuine delivery refused as a
replay, building a denial of service out of the anti-replay control.
"""

import hashlib
import hmac
import json
import logging
import time
from typing import Any, Dict, Optional, Tuple

from .config import (
    SECRET_ALGO_HMAC_SHA256,
    SECRET_ALGO_TOKEN,
    SECRET_ALGOS,
    RouteConfig,
)
from .replay import ReplayCache, replay_key_for
from . import signature_schemes as schemes

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
    now: float,
    replay_cache: Optional[ReplayCache],
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
        now: Current Unix time, injected by the caller so the freshness window
            and the replay TTL are stated rather than slept through.
        replay_cache: The listener's replay cache, or None when it has none
            (an embedded caller, or a deployment that disabled it).

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

    scheme_failure = _scheme_config_error(route_name, route)
    if scheme_failure is not None:
        return scheme_failure

    secret = route.metadata.get('secret') or global_secret
    if not secret:
        logger.warning(
            "Route '%s' requires secret verification but no secret configured",
            route_name,
        )
        return 500, "Server misconfigured: no secret for signature verification"

    return _verify_credential(
        body, lower_headers, route_name, route, secret, now, replay_cache,
    )


def _scheme_config_error(
    route_name: str,
    route: RouteConfig,
) -> Optional[Tuple[int, str]]:
    """Refuse a route whose replay-protection keys cannot be honoured (#713).

    ``validate_config`` reports every one of these, but a route can be built
    programmatically around that check — which is exactly the reasoning the
    ``secret_algo`` guard above already records.  The refusal is a **500**, not
    a 403: it describes a broken server, not a bad request, and the two must
    not read the same to whoever is looking at the logs.

    Nothing here falls through to a weaker check.  A mistyped
    ``signature_scheme`` does not degrade to ``'body'``; a Slack route with no
    ``timestamp_header`` does not verify the body alone.  Widening the scheme
    vocabulary widens what a route may SAY, never what it may omit.

    Args:
        route_name: Name of the matched route (for logs).
        route: The matched route.

    Returns:
        None when the scheme keys are coherent, else ``(500, message)``.
    """
    scheme = route.signature_scheme
    if scheme not in schemes.SIGNATURE_SCHEMES:
        logger.warning(
            "Route '%s' declares unknown signature_scheme '%s'; refusing "
            "(fail-closed).", route_name, scheme,
        )
        return 500, (
            "Server misconfigured: unknown signature_scheme, so the signed "
            "payload cannot be reconstructed."
        )

    if not route.binds_timestamp():
        if route.timestamp_header:
            logger.warning(
                "Route '%s' sets timestamp_header on signature_scheme '%s', "
                "which does not sign the timestamp — refusing (fail-closed), "
                "because checking a value the replayer can rewrite is not "
                "replay protection.", route_name, scheme,
            )
            return 500, (
                "Server misconfigured: timestamp_header is set on a "
                "signature_scheme that does not cover the timestamp with the "
                "signature."
            )
        return None

    # From here the scheme binds a timestamp, so both halves must be present
    # and the MAC must be the one the scheme is defined over.
    if route.secret_algo != SECRET_ALGO_HMAC_SHA256:
        logger.warning(
            "Route '%s' pairs signature_scheme '%s' with secret_algo '%s'; "
            "refusing (fail-closed).", route_name, scheme, route.secret_algo,
        )
        return 500, (
            "Server misconfigured: this signature_scheme is defined over an "
            f"HMAC-SHA256 digest, so secret_algo must be "
            f"'{SECRET_ALGO_HMAC_SHA256}'."
        )

    wants_header = (
        schemes.SCHEME_TIMESTAMP_SOURCE[scheme] == schemes.TIMESTAMP_FROM_HEADER
    )
    if wants_header and not route.timestamp_header:
        logger.warning(
            "Route '%s' declares signature_scheme '%s' but names no "
            "timestamp_header; refusing (fail-closed).", route_name, scheme,
        )
        return 500, (
            "Server misconfigured: this signature_scheme signs a timestamp "
            "carried in its own header, but the route names no "
            "timestamp_header."
        )
    if not wants_header and route.timestamp_header:
        logger.warning(
            "Route '%s' declares signature_scheme '%s', which carries its "
            "timestamp inside the signature header, but also names a "
            "timestamp_header; refusing (fail-closed).", route_name, scheme,
        )
        return 500, (
            "Server misconfigured: this signature_scheme reads its timestamp "
            "from the signature header, so timestamp_header must not be set."
        )
    return None


def _run_scheme(
    body: bytes,
    secret: str,
    signature_value: str,
    lower_headers: Dict[str, str],
    route: RouteConfig,
) -> schemes.SignatureVerdict:
    """Verify the request against the route's signature scheme.

    The ``'body'`` branch delegates to :func:`verify_signature`, so every
    pre-#713 route is verified by byte-identical code on a byte-identical
    input.  Its verdict carries **no** replay material: with no timestamp in
    the signed payload the digest is a pure function of the body, so two
    genuine deliveries of the same payload would be indistinguishable from a
    replay and the second one wrongly refused.  Such a route keys its cache on
    a delivery id or gets none — see :func:`.replay.replay_key_for`.

    Args:
        body: Raw request body bytes.
        secret: The route's shared secret.
        signature_value: The ``secret_header`` value from the request.
        lower_headers: Request headers with lowercased keys.
        route: The matched route.  Its scheme keys have already been checked
            by :func:`_scheme_config_error`, so ``timestamp_header`` is present
            wherever the scheme needs one.

    Returns:
        The scheme's verdict.  Freshness is NOT applied here — one policy, one
        place, so it cannot drift per sender.
    """
    scheme = route.signature_scheme

    if scheme == schemes.SCHEME_SLACK_V0:
        timestamp_raw = lower_headers.get(route.timestamp_header.lower(), '')
        return schemes.verify_slack_v0(body, secret, signature_value, timestamp_raw)

    if scheme == schemes.SCHEME_STRIPE_V1:
        return schemes.verify_stripe_v1(body, secret, signature_value)

    ok = verify_signature(body, secret, signature_value, route.secret_algo)
    return schemes.SignatureVerdict(
        ok, schemes.REASON_OK if ok else schemes.REASON_BAD_SIGNATURE,
    )


def _verify_credential(
    body: bytes,
    lower_headers: Dict[str, str],
    route_name: str,
    route: RouteConfig,
    secret: str,
    now: float,
    replay_cache: Optional[ReplayCache],
) -> Optional[Tuple[int, str]]:
    """Run signature, then freshness, then replay — in that order.

    The order is the security argument, not a style choice:

    1. **Signature.**  In both timestamped schemes the timestamp is part of
       the signed payload, so it is evidence of nothing until the signature
       holds.  Checking freshness first would also let an unauthenticated
       caller probe the window.
    2. **Freshness.**  Bounds how long a captured delivery stays useful.
    3. **Replay.**  Catches the copy that arrives INSIDE the window, which
       freshness by construction cannot.  Recorded last, so a request that
       failed either earlier check never writes to the cache — otherwise
       anyone who can reach the port could poison it with a guessed delivery
       id and have the genuine delivery refused.

    Args:
        body: Raw request body bytes.
        lower_headers: Request headers with lowercased keys.
        route_name: Name of the matched route.
        route: The matched route.
        secret: The resolved shared secret.
        now: Current Unix time, injected by the caller.
        replay_cache: The listener's cache, or None when it has none.

    Returns:
        None when the request is authenticated and fresh and new, else
        ``(status, message)``.
    """
    signature_value = lower_headers.get(route.secret_header.lower(), '')
    verdict = _run_scheme(body, secret, signature_value, lower_headers, route)

    if not verdict.ok:
        logger.warning(
            "Secret verification failed for route '%s' (algo=%s scheme=%s): %s",
            route_name, route.secret_algo, route.signature_scheme, verdict.reason,
        )
        return 403, "Signature verification failed"

    if route.binds_timestamp() and not schemes.is_fresh(
        verdict.timestamp, now, route.max_age_seconds
    ):
        logger.warning(
            "Route '%s': signature verified but the signed timestamp (%s) is "
            "outside the %ds freshness window — refusing as stale/skewed.",
            route_name, verdict.timestamp, route.max_age_seconds,
        )
        return 403, "Request timestamp is outside the freshness window"

    return _check_replay(
        route_name, route, lower_headers, verdict, now, replay_cache,
    )


def _check_replay(
    route_name: str,
    route: RouteConfig,
    lower_headers: Dict[str, str],
    verdict: schemes.SignatureVerdict,
    now: float,
    replay_cache: Optional[ReplayCache],
) -> Optional[Tuple[int, str]]:
    """Record an authenticated delivery, refusing it if it is a repeat.

    A route with nothing unique-per-delivery to key on gets no cache and no
    refusal — the honest outcome, since keying on a constant credential would
    reject the second legitimate request rather than a replay.  That case is
    announced once at listener startup rather than silently accepted.

    Disabling the freshness window (``max_age_seconds: 0``) does NOT disable
    the cache: the TTL falls back to the default window, so an operator who
    turns the window off still keeps short-horizon replay refusal rather than
    losing both controls with one key.

    Args:
        route_name: Name of the matched route, used to namespace the key.
        route: The matched route.
        lower_headers: Request headers with lowercased keys.
        verdict: The verified signature verdict.
        now: Current Unix time.
        replay_cache: The listener's cache, or None.

    Returns:
        None when the delivery is new, else ``(409, message)``.
    """
    if replay_cache is None:
        return None

    delivery_id = None
    if route.replay_key_header:
        delivery_id = lower_headers.get(route.replay_key_header.lower()) or None

    key = replay_key_for(
        route_name, delivery_id, verdict.replay_material, route.binds_timestamp(),
    )
    if key is None:
        return None

    ttl = (
        route.max_age_seconds if route.max_age_seconds > 0
        else schemes.DEFAULT_MAX_AGE_SECONDS
    )
    if replay_cache.check_and_record(key, now, ttl):
        logger.warning(
            "Route '%s': refusing a REPLAYED delivery — this request was "
            "already accepted within the last %ds.", route_name, ttl,
        )
        return 409, "Duplicate delivery: this request has already been accepted"

    return None


def parse_webhook_request(
    body: bytes,
    headers: Dict[str, str],
    route_name: str,
    route: RouteConfig,
    global_secret: Optional[str] = None,
    transport_authenticated: bool = False,
    now: Optional[float] = None,
    replay_cache: Optional[ReplayCache] = None,
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
        now: Current Unix time.  Defaults to ``time.time()``; a caller passes
            it so the freshness window (#713) is a value the test states
            rather than a clock it waits on.
        replay_cache: The listener's :class:`.replay.ReplayCache`.  None means
            no replay refusal — every other check is unchanged, so a caller
            that does not supply one is exactly as safe as it was before.

    Returns:
        Tuple of (parsed_event_dict, None, None) on success, or
        (None, http_status_code, error_message) on failure.  A replayed
        delivery is ``409`` — deliberately distinct from ``403``, so an
        operator can tell a repeat from a forgery in the access log.
    """
    if now is None:
        now = time.time()

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
        now=now,
        replay_cache=replay_cache,
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
