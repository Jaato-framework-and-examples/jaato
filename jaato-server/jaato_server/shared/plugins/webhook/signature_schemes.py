"""How a webhook route's signed payload is CONSTRUCTED, as distinct from the MAC.

``RouteConfig.secret_algo`` says how the route's shared secret is checked —
an HMAC digest, or a verbatim token.  It cannot say *what was signed*, and for
the two senders that bind a timestamp into the signature that is the whole
question:

* **Slack** signs ``v0:{timestamp}:{body}`` and sends the digest as
  ``v0=<hex>`` in ``X-Slack-Signature``, with the timestamp beside it in
  ``X-Slack-Request-Timestamp``.
* **Stripe** signs ``{timestamp}.{body}`` and sends *both* in one header —
  ``Stripe-Signature: t=<unix>,v1=<hex>`` (possibly several ``v1`` values
  during a secret rotation).

Verifying either as "HMAC over the body" fails outright, which is why routes
for those senders could previously only be configured as ``body`` and so
ignored the timestamp the sender went to the trouble of signing (#713).

``RouteConfig.signature_scheme`` names the construction:

===============  ==========================================================
``body``         the request body alone (default; GitHub, GitLab, and every
                 pre-#713 route).  Binds no timestamp, so a signature over
                 it never expires.
``slack-v0``     ``v0:{ts}:{body}``, timestamp from ``timestamp_header``.
``stripe-v1``    ``{ts}.{body}``, timestamp from ``t=`` in the signature
                 header itself.
===============  ==========================================================

Three rules this module exists to hold:

* **No cross-mode leniency.**  A scheme verifies its own construction and
  nothing else.  ``slack-v0`` does not accept a bare hex digest without the
  ``v0=`` prefix, and does not fall back to an HMAC over the body if the
  timestamped basestring fails to match.  Either transformation would accept
  a signature built a way nobody configured.
* **The timestamp is only trustworthy once the signature verifies**, because
  in both schemes the timestamp is part of the signed payload.  So callers
  verify FIRST and test freshness SECOND — never the reverse, which would let
  an unauthenticated caller probe the window and, worse, would treat an
  attacker-chosen timestamp as evidence of anything.
* **Unparseable is a refusal, never a skip.**  A route that asked for a
  timestamp and did not get a usable one is refused; falling through would
  turn a malformed header into a bypass of the freshness window.

stdlib only, like the rest of this plugin.
"""

import hashlib
import hmac
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# The vocabulary of ``RouteConfig.signature_scheme``.  Anything outside it is a
# hard error at config-validation time and a 500 at request time — the same
# fail-closed posture ``secret_algo`` takes, for the same reason: a typo must
# never leave a route verified a way the operator did not choose.
SCHEME_BODY = 'body'
SCHEME_SLACK_V0 = 'slack-v0'
SCHEME_STRIPE_V1 = 'stripe-v1'
SIGNATURE_SCHEMES = (SCHEME_BODY, SCHEME_SLACK_V0, SCHEME_STRIPE_V1)

# Schemes that bind a timestamp into the signed payload.  Only these can carry
# a freshness window: a timestamp outside the signature is attacker-rewritable,
# so checking it would be theatre.
TIMESTAMP_BOUND_SCHEMES = (SCHEME_SLACK_V0, SCHEME_STRIPE_V1)

# Where each timestamp-bound scheme reads its timestamp from.  ``'header'``
# means the route must name a ``timestamp_header``; ``'signature'`` means the
# scheme parses it out of the signature header itself and a separate
# ``timestamp_header`` would be a second, unused source (refused).
TIMESTAMP_FROM_HEADER = 'header'
TIMESTAMP_FROM_SIGNATURE = 'signature'
SCHEME_TIMESTAMP_SOURCE: Dict[str, str] = {
    SCHEME_SLACK_V0: TIMESTAMP_FROM_HEADER,
    SCHEME_STRIPE_V1: TIMESTAMP_FROM_SIGNATURE,
}

# Slack's documented freshness window, and a sane default for every scheme.
DEFAULT_MAX_AGE_SECONDS = 300

# Refusal reasons.  Machine tokens rather than prose so a log line is greppable
# and a test can assert the verdict rather than the sentence.
REASON_OK = 'ok'
REASON_NO_SIGNATURE = 'no_signature'
REASON_BAD_SIGNATURE = 'bad_signature'
REASON_NO_TIMESTAMP = 'no_timestamp'
REASON_BAD_TIMESTAMP = 'bad_timestamp'
REASON_STALE_TIMESTAMP = 'stale_timestamp'


@dataclass(frozen=True)
class SignatureVerdict:
    """The outcome of verifying one request against one scheme.

    Attributes:
        ok: True when the signature verified.  Freshness is NOT included —
            the caller applies the window, because the policy is one policy
            across every scheme and must not drift per-sender.
        reason: One of the ``REASON_*`` tokens, for logs and tests.
        timestamp: The signed Unix timestamp, when the scheme carries one.
            Trustworthy only when ``ok`` is True, since it is the signature
            that proves it.
        replay_material: A value unique to THIS delivery that can key a replay
            cache — the signature digest, which for every timestamp-bound
            scheme varies with both the body and the timestamp.  ``None``
            whenever the credential does not distinguish one delivery from
            another: ``token`` mode, where it is constant, and the ``body``
            scheme, where it is a pure function of the payload, so two genuine
            deliveries of the same body would be indistinguishable from a
            replay.  Those routes key on a delivery id instead, or get no
            replay cache at all.
    """
    ok: bool
    reason: str
    timestamp: Optional[int] = None
    replay_material: Optional[str] = None


def constant_time_equals(a: str, b: str) -> bool:
    """Constant-time string comparison that never raises on the input.

    ``hmac.compare_digest`` rejects a non-ASCII ``str`` with ``TypeError``, and
    both operands here come off the wire — a header value an attacker chooses.
    Encoding to UTF-8 first makes the comparison total (identical results for
    the ASCII inputs every scheme actually produces) so a crafted header can
    never turn a verification failure into a 500.

    Args:
        a: One side of the comparison.
        b: The other side.

    Returns:
        True when the two strings are byte-identical.
    """
    return hmac.compare_digest(a.encode('utf-8'), b.encode('utf-8'))


def hmac_sha256_hex(secret: str, payload: bytes) -> str:
    """HMAC-SHA256 of ``payload`` keyed by ``secret``, as lowercase hex.

    Args:
        secret: The route's shared secret (Slack calls it the signing secret,
            Stripe the endpoint secret).
        payload: The scheme's constructed signing payload — NOT necessarily
            the request body.

    Returns:
        The hex digest.
    """
    return hmac.new(secret.encode('utf-8'), payload, hashlib.sha256).hexdigest()


def parse_unix_timestamp(raw: str) -> Optional[int]:
    """Parse a Unix-seconds timestamp exactly as Slack and Stripe send one.

    Deliberately strict: both senders emit integer seconds, so a float, an
    ISO-8601 string, an empty value or anything else is ``None`` — which the
    caller turns into a refusal, never into a skipped window check.

    Args:
        raw: The raw header (or header-field) value.

    Returns:
        The timestamp as an int, or None when it is not integer seconds.
    """
    if not raw:
        return None
    try:
        return int(raw.strip())
    except (TypeError, ValueError):
        return None


def is_fresh(timestamp: Optional[int], now: float, max_age_seconds: int) -> bool:
    """Whether a signed timestamp falls inside the freshness window.

    The comparison is two-sided (``abs``): a timestamp far in the FUTURE is as
    much evidence of a problem as a stale one — a replay recorded against a
    skewed clock, or a sender whose clock is wrong — and accepting it would
    hand an attacker an arbitrarily long-lived signature.

    Args:
        timestamp: The signed Unix timestamp, or None when the scheme has none.
        now: Current Unix time.  Injected rather than read here so a test can
            state the window it means instead of sleeping through it.
        max_age_seconds: The window in seconds.  ``0`` (or negative) disables
            the check — the operator's explicit, announced opt-out.

    Returns:
        True when the request is inside the window (or the window is off).
    """
    if max_age_seconds <= 0:
        return True
    if timestamp is None:
        return False
    return abs(now - timestamp) <= max_age_seconds


def verify_slack_v0(
    body: bytes,
    secret: str,
    signature_header: str,
    timestamp_raw: str,
) -> SignatureVerdict:
    """Verify Slack's ``v0`` request signature.

    Slack's documented scheme: concatenate the version, the request timestamp
    and the raw body with colons — ``v0:{timestamp}:{body}`` — HMAC-SHA256 it
    with the app's signing secret, and compare the hex digest prefixed with
    ``v0=`` against ``X-Slack-Signature``.

    The ``v0=`` prefix is part of the comparison rather than stripped first:
    it is what the sender transmits, and accepting a bare digest would accept
    a signature built by a scheme nobody configured.

    Args:
        body: The raw request body, byte-for-byte as received.
        secret: The Slack signing secret.
        signature_header: Value of ``X-Slack-Signature``.
        timestamp_raw: Value of the route's ``timestamp_header``
            (``X-Slack-Request-Timestamp``).

    Returns:
        A SignatureVerdict.  ``timestamp`` is set whenever it parsed, so the
        caller can log a stale delivery precisely; it is only *trustworthy*
        when ``ok`` is True, because the basestring that verified contains it.
    """
    if not signature_header:
        return SignatureVerdict(False, REASON_NO_SIGNATURE)
    if not timestamp_raw:
        return SignatureVerdict(False, REASON_NO_TIMESTAMP)

    timestamp = parse_unix_timestamp(timestamp_raw)
    if timestamp is None:
        return SignatureVerdict(False, REASON_BAD_TIMESTAMP)

    # The basestring uses the timestamp EXACTLY as sent: re-rendering the
    # parsed int would diverge from the sender on any spelling it accepts and
    # we normalise (a leading zero, surrounding whitespace).
    basestring = b'v0:' + timestamp_raw.encode('utf-8') + b':' + body
    expected = 'v0=' + hmac_sha256_hex(secret, basestring)

    if not constant_time_equals(expected, signature_header):
        return SignatureVerdict(False, REASON_BAD_SIGNATURE, timestamp=timestamp)

    return SignatureVerdict(
        True, REASON_OK, timestamp=timestamp, replay_material=signature_header,
    )


def _parse_stripe_signature_header(header: str) -> Tuple[Optional[str], List[str]]:
    """Split a ``Stripe-Signature`` header into its timestamp and v1 digests.

    The header is a comma-separated list of ``key=value`` pairs, e.g.
    ``t=1492774577,v1=5257a869...,v1=aa4bd2d1...``.  Several ``v1`` values
    appear while an endpoint secret is being rotated, and Stripe's own
    libraries accept the request if ANY of them matches.

    ``v0`` entries (Stripe's legacy test-mode scheme) are deliberately ignored
    rather than accepted: this route was configured for ``stripe-v1``.

    Args:
        header: The raw ``Stripe-Signature`` value.

    Returns:
        ``(timestamp_raw_or_None, [v1_digest, ...])``.  A malformed pair is
        skipped; a header with no ``t`` or no ``v1`` yields None / an empty
        list, which the caller turns into a refusal.
    """
    timestamp_raw: Optional[str] = None
    v1_digests: List[str] = []
    for part in header.split(','):
        key, sep, value = part.strip().partition('=')
        if not sep:
            continue
        if key == 't':
            timestamp_raw = value
        elif key == 'v1':
            v1_digests.append(value)
    return timestamp_raw, v1_digests


def verify_stripe_v1(
    body: bytes,
    secret: str,
    signature_header: str,
) -> SignatureVerdict:
    """Verify Stripe's ``v1`` webhook signature.

    Stripe's documented scheme: take the ``t`` value out of the
    ``Stripe-Signature`` header, build the signed payload as
    ``{timestamp}.{body}``, HMAC-SHA256 it with the endpoint secret, and
    compare the hex digest against each ``v1`` entry in the same header.

    The timestamp therefore comes from the signature header itself, which is
    why a route on this scheme must NOT also name a ``timestamp_header``:
    a second source could disagree with the signed one, and the unsigned one
    would win the window check.

    Args:
        body: The raw request body, byte-for-byte as received.
        secret: The Stripe endpoint secret (``whsec_...``).
        signature_header: Value of ``Stripe-Signature``.

    Returns:
        A SignatureVerdict.  ``replay_material`` is the matching ``v1``
        digest, which varies with both the body and the timestamp.
    """
    if not signature_header:
        return SignatureVerdict(False, REASON_NO_SIGNATURE)

    timestamp_raw, v1_digests = _parse_stripe_signature_header(signature_header)
    if timestamp_raw is None:
        return SignatureVerdict(False, REASON_NO_TIMESTAMP)

    timestamp = parse_unix_timestamp(timestamp_raw)
    if timestamp is None:
        return SignatureVerdict(False, REASON_BAD_TIMESTAMP)
    if not v1_digests:
        return SignatureVerdict(False, REASON_NO_SIGNATURE, timestamp=timestamp)

    signed_payload = timestamp_raw.encode('utf-8') + b'.' + body
    expected = hmac_sha256_hex(secret, signed_payload)

    # Every candidate is compared in constant time, and the loop does not break
    # early on a match, so the work done is a function of how many digests the
    # sender supplied rather than of which one (if any) was right.
    matched = False
    for candidate in v1_digests:
        if constant_time_equals(expected, candidate):
            matched = True
    if not matched:
        return SignatureVerdict(False, REASON_BAD_SIGNATURE, timestamp=timestamp)

    return SignatureVerdict(
        True, REASON_OK, timestamp=timestamp, replay_material=expected,
    )
