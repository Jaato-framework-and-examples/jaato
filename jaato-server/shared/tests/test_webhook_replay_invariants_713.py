"""The three webhook replay invariants that a refactor could silently undo (#713).

The behavioural suite for this feature lives beside the plugin
(``shared/plugins/webhook/tests/test_replay_protection_713.py``) and is the
thorough one.  What is here is the subset whose *reversion* is invisible — a
change that leaves every other test green, the listener running and the config
accepted, while the protection is gone:

1. **The freshness window is actually consulted.**  Drop the call and a stale
   capture is accepted again; nothing else changes.
2. **A replay is actually refused.**  Drop the cache check and every delivery is
   accepted; the cache still fills, the counters still exist, the logs are
   unchanged.
3. **Only an AUTHENTICATED delivery is recorded.**  Record on arrival instead
   and replay refusal still works perfectly for honest traffic — while anyone
   who can reach the port can register a guessed delivery id and have the
   *genuine* delivery refused as a replay.  This is the one where the sabotage
   makes the system look MORE protective, not less.

Plus the constant-time comparison (#713 ask 4), which is a pure source
property: reverting it to ``==`` changes no behaviour at all, so only a source
guard can see it.

This module sits in ``shared/tests`` rather than beside the plugin because
that is where :mod:`~shared.tests.test_every_guard_detects_its_own_reversion`
discovers guards, and a guard nothing re-checks is the thing that suite exists
to prevent.
"""

import hashlib
import hmac
import inspect
import json

from shared.plugins.webhook.config import RouteConfig
from shared.plugins.webhook.replay import ReplayCache
from shared.plugins.webhook.routes import parse_webhook_request
from shared.plugins.webhook import routes as _routes
from shared.plugins.webhook import signature_schemes as _schemes
from shared.tests.reversion import Reversion

_ROUTES = "jaato-server/shared/plugins/webhook/routes.py"
_SCHEMES = "jaato-server/shared/plugins/webhook/signature_schemes.py"

# Fabricated. Never a real credential.
SECRET = "fabricated-invariant-secret"
BODY = json.dumps({"event": "push"}).encode()
NOW = 1_700_000_000.0


REVERSIONS = [
    Reversion(
        target=_ROUTES,
        find="""    if route.binds_timestamp() and not schemes.is_fresh(
        verdict.timestamp, now, route.max_age_seconds
    ):""",
        replace="""    if False and not schemes.is_fresh(
        verdict.timestamp, now, route.max_age_seconds
    ):""",
        test="test_a_stale_delivery_is_refused",
        because="the freshness window is no longer consulted, so a capture "
                "from any point in the past authenticates again — the #713 "
                "state, with the config keys still accepted",
    ),
    Reversion(
        target=_ROUTES,
        find="    if replay_cache.check_and_record(key, now, ttl):",
        replace="    if False and replay_cache.check_and_record(key, now, ttl):",
        test="test_a_replay_inside_the_window_is_refused",
        because="a replay inside the freshness window is accepted again, which "
                "the timestamp check by construction cannot catch",
    ),
    Reversion(
        target=_ROUTES,
        find="""        return 403, "Signature verification failed\"""",
        replace="""        _check_replay(
            route_name, route, lower_headers, verdict, now, replay_cache,
        )
        return 403, "Signature verification failed\"""",
        test="test_a_forged_request_cannot_poison_the_replay_cache",
        because="an unauthenticated request writes to the replay cache, so "
                "anyone who can reach the port can pre-register a delivery id "
                "and have the genuine delivery refused as a replay",
    ),
    Reversion(
        target=_SCHEMES,
        find="    return hmac.compare_digest(a.encode('utf-8'), b.encode('utf-8'))",
        replace="    return a == b",
        test="test_credential_comparison_is_constant_time",
        because="credential comparison is back to a short-circuiting '==', "
                "which leaks the matching prefix through timing and changes "
                "no observable behaviour",
    ),
]


def _github_route(**overrides) -> RouteConfig:
    """A GitHub-shaped route with delivery-id deduplication."""
    data = {
        "path": "/webhook/github",
        "secret_header": "X-Hub-Signature-256",
        "secret_algo": "hmac-sha256",
        "replay_key_header": "X-GitHub-Delivery",
    }
    data.update(overrides)
    return RouteConfig.from_dict(data)


def _slack_route() -> RouteConfig:
    """A Slack-shaped route: timestamp bound into the signature."""
    return RouteConfig.from_dict({
        "path": "/webhook/slack",
        "secret_header": "X-Slack-Signature",
        "secret_algo": "hmac-sha256",
        "signature_scheme": "slack-v0",
        "timestamp_header": "X-Slack-Request-Timestamp",
    })


def _slack_signature(timestamp: str) -> str:
    """Slack's ``v0`` signature, spelled out from the vendor's documentation."""
    basestring = b"v0:" + timestamp.encode() + b":" + BODY
    return "v0=" + hmac.new(
        SECRET.encode(), basestring, hashlib.sha256,
    ).hexdigest()


def _deliver(route, headers, now=NOW, cache=None, name="r", body=BODY):
    """Drive one request through the real request path."""
    full = {"Content-Type": "application/json"}
    full.update(headers)
    return parse_webhook_request(
        body, full, name, route, SECRET, now=now, replay_cache=cache,
    )


def test_a_stale_delivery_is_refused():
    """A signed timestamp outside the window does not authenticate.

    Reverting the window leaves the route config valid, the signature
    verification correct and every other test green.
    """
    old = str(int(NOW) - 86_400)
    headers = {
        "X-Slack-Request-Timestamp": old,
        "X-Slack-Signature": _slack_signature(old),
    }
    event, status, message = _deliver(_slack_route(), headers)
    assert event is None, "a day-old capture authenticated"
    assert status == 403
    assert "freshness window" in message


def test_a_replay_inside_the_window_is_refused():
    """The second copy of an accepted delivery is refused.

    Inside the window the timestamp check cannot tell the copies apart — both
    carry the same signed timestamp and both are equally fresh — so this is
    the cache's job alone.
    """
    cache = ReplayCache()
    route = _slack_route()
    ts = str(int(NOW))
    headers = {
        "X-Slack-Request-Timestamp": ts,
        "X-Slack-Signature": _slack_signature(ts),
    }

    assert _deliver(route, headers, cache=cache)[1] is None

    event, status, message = _deliver(route, headers, now=NOW + 1, cache=cache)
    assert event is None, "a byte-identical replay was accepted"
    assert status == 409
    assert "already been accepted" in message


def test_a_forged_request_cannot_poison_the_replay_cache():
    """An unauthenticated request never writes to the cache.

    The sabotage for this one makes the system look *more* careful, which is
    why it needs a guard: replay refusal keeps working for honest traffic
    while an attacker who can reach the port pre-registers the delivery id of
    a webhook they expect and has the genuine one refused as a duplicate.
    """
    cache = ReplayCache()
    route = _github_route()
    guid = "11111111-2222-3333-4444-555555555555"

    forged, status, _ = _deliver(route, {
        "X-Hub-Signature-256": "sha256=" + "0" * 64,
        "X-GitHub-Delivery": guid,
    }, cache=cache)
    assert forged is None and status == 403
    assert len(cache) == 0, "a refused request was recorded"

    genuine, status, _ = _deliver(route, {
        "X-Hub-Signature-256": "sha256=" + hmac.new(
            SECRET.encode(), BODY, hashlib.sha256,
        ).hexdigest(),
        "X-GitHub-Delivery": guid,
    }, now=NOW + 1, cache=cache)
    assert status is None, (
        "the genuine delivery was refused as a replay — the forged request "
        "pre-registered its delivery id"
    )
    assert genuine is not None


def test_credential_comparison_is_constant_time():
    """Every credential comparison routes through ``hmac.compare_digest``.

    A pure source property: ``==`` verifies identically and leaks the matching
    prefix through timing, so no behavioural test can see the reversion.  The
    WS bearer check established the pattern; this asserts the webhook path did
    not grow a naive comparison beside it.
    """
    # The COMPILED code object, not the source text.  Reading the source would
    # be satisfied by this function's own docstring, which names
    # ``hmac.compare_digest`` — and a guard a docstring can satisfy is exactly
    # the decorative kind the reversion suite exists to catch.  It caught this
    # one: the first draft here asserted against ``inspect.getsource`` and
    # passed with the body replaced by ``return a == b``.
    assert "compare_digest" in _schemes.constant_time_equals.__code__.co_names, (
        "the shared comparison helper no longer calls hmac.compare_digest"
    )

    for module in (_schemes, _routes):
        source = inspect.getsource(module)
        for marker in ("== signature_header", "signature_header ==",
                       "== expected", "expected ==",
                       "== candidate", "candidate =="):
            assert marker not in source, (
                f"{module.__name__} compares a credential with '=='"
            )
