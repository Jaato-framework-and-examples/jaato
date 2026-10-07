"""Replay protection for webhook signatures (#713).

Before this, a webhook signature was an HMAC over the request BODY and nothing
else, so it authenticated the same bytes forever: anyone who observed one
delivery could replay it verbatim, indefinitely.  On a listener whose purpose
is to drive agent sessions that is a re-triggered turn each time — tool calls,
ledger spend, side effects.

Two properties every test here is built around:

**The vendor schemes are derived from the vendors' documentation, not from
this implementation.**  A test that builds Slack's signature by calling
:func:`verify_slack_v0`'s own helper proves only that the code agrees with
itself.  So the concatenations are spelled out inline
(``b'v0:' + ts + b':' + body``; ``ts + b'.' + body``) and, for Slack, the
module carries the **published worked example** — secret, timestamp, body and
expected digest copied from Slack's docs — which no implementation detail of
ours can satisfy by accident.

**No test sleeps.**  Both the freshness window and the replay TTL take ``now``
as a parameter, so a test states the instant it means.  A TTL asserted with
``time.sleep`` is a timing bet, and this tree has paid for one before (#996).
"""

import hashlib
import hmac
import json
import logging

import pytest

from jaato_server.shared.plugins.webhook.config import RouteConfig, validate_config
from jaato_server.shared.plugins.webhook.http_server import WebhookHTTPServer
from jaato_server.shared.plugins.webhook.config import WebhookConfig
from jaato_server.shared.plugins.webhook.replay import ReplayCache, replay_key_for
from jaato_server.shared.plugins.webhook.routes import parse_webhook_request
from jaato_server.shared.plugins.webhook.signature_schemes import (
    DEFAULT_MAX_AGE_SECONDS,
    REASON_BAD_SIGNATURE,
    REASON_BAD_TIMESTAMP,
    REASON_NO_SIGNATURE,
    REASON_NO_TIMESTAMP,
    verify_slack_v0,
    verify_stripe_v1,
)

# Fabricated throughout — never a real credential.
SECRET = "fabricated-test-secret-not-a-real-credential"
NOW = 1_700_000_000.0
JSON_BODY = json.dumps({"event": "push", "n": 1}).encode()
OTHER_BODY = json.dumps({"event": "push", "n": 2}).encode()


# ---------------------------------------------------------------------------
# Signature construction, derived from the vendors' own documentation
# ---------------------------------------------------------------------------

def slack_signature(secret: str, timestamp: str, body: bytes) -> str:
    """Build a Slack ``v0`` signature the way Slack's documentation states it.

    Written out here rather than imported so this file is an INDEPENDENT
    derivation: "Concatenate the version, the timestamp and the body with
    colons; HMAC-SHA256 it with the signing secret; prefix the hex digest with
    ``v0=``."

    Args:
        secret: The app's signing secret.
        timestamp: ``X-Slack-Request-Timestamp``, as sent.
        body: The raw request body.

    Returns:
        The ``X-Slack-Signature`` value.
    """
    basestring = b"v0:" + timestamp.encode() + b":" + body
    digest = hmac.new(secret.encode(), basestring, hashlib.sha256).hexdigest()
    return "v0=" + digest


def stripe_signature_header(
    secret: str, timestamp: str, body: bytes, scheme: str = "v1",
) -> str:
    """Build a ``Stripe-Signature`` header the way Stripe's documentation states it.

    "The ``signed_payload`` string is created by concatenating: the timestamp
    (as a string), the character ``.``, the actual JSON payload."  HMAC-SHA256
    with the endpoint's signing secret as the key.

    Args:
        secret: The endpoint signing secret.
        timestamp: The ``t`` value.
        body: The raw request body.
        scheme: The signature scheme prefix — ``'v1'`` for the real one, or
            ``'v0'`` to build the legacy test-mode signature Stripe tells
            receivers to ignore.

    Returns:
        A single-line ``Stripe-Signature`` header value.
    """
    signed_payload = timestamp.encode() + b"." + body
    digest = hmac.new(secret.encode(), signed_payload, hashlib.sha256).hexdigest()
    return f"t={timestamp},{scheme}={digest}"


def github_signature(secret: str, body: bytes) -> str:
    """Build GitHub's body-only signature — the pre-#713 shape, unchanged."""
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


# ---------------------------------------------------------------------------
# Route builders
# ---------------------------------------------------------------------------

def slack_route(**overrides) -> RouteConfig:
    """A Slack route, built through ``from_dict`` so config coercion is covered."""
    data = {
        "path": "/webhook/slack",
        "secret_header": "X-Slack-Signature",
        "secret_algo": "hmac-sha256",
        "signature_scheme": "slack-v0",
        "timestamp_header": "X-Slack-Request-Timestamp",
    }
    data.update(overrides)
    return RouteConfig.from_dict(data)


def stripe_route(**overrides) -> RouteConfig:
    """A Stripe route."""
    data = {
        "path": "/webhook/stripe",
        "secret_header": "Stripe-Signature",
        "secret_algo": "hmac-sha256",
        "signature_scheme": "stripe-v1",
    }
    data.update(overrides)
    return RouteConfig.from_dict(data)


def github_route(**overrides) -> RouteConfig:
    """A GitHub route — the pre-#713 shape, declaring no new key by default."""
    data = {
        "path": "/webhook/github",
        "secret_header": "X-Hub-Signature-256",
        "secret_algo": "hmac-sha256",
    }
    data.update(overrides)
    return RouteConfig.from_dict(data)


def deliver(route, headers, body=JSON_BODY, now=NOW, cache=None, name="r"):
    """Drive one request through the real request path.

    Args:
        route: The matched route.
        headers: Request headers (content-type is added).
        body: Raw body bytes.
        now: Injected Unix time.
        cache: The listener's ReplayCache, or None.
        name: Route name.

    Returns:
        ``(event, status, message)`` exactly as ``parse_webhook_request`` does.
    """
    full = {"Content-Type": "application/json"}
    full.update(headers)
    return parse_webhook_request(
        body, full, name, route, SECRET, now=now, replay_cache=cache,
    )


# ===========================================================================
# 1. The vendor constructions
# ===========================================================================

class TestSlackConstruction:
    """Slack's ``v0:{ts}:{body}`` basestring — #713 ask 3."""

    # Slack's own published worked example.  Nothing in this repository can
    # make this pass by agreeing with itself: the digest is fixed by the
    # vendor's construction, so if the basestring were assembled any other way
    # (body-only, a different separator, no version prefix, the timestamp
    # re-rendered) it would not match.
    DOC_SECRET = "8f742231b10e8888abcd99yyyzzz85a5"
    DOC_TIMESTAMP = "1531420618"
    DOC_BODY = (
        b"token=xyzz0WbapA4vBCDEFasx0q6G&team_id=T1DC2JH3J"
        b"&team_domain=testteamnow&channel_id=G8PSS9T3V&channel_name=foobar"
        b"&user_id=U2CERLKJA&user_name=roadrunner&command=%2Fwebhook-collect"
        b"&text=&response_url=https%3A%2F%2Fhooks.slack.com%2Fcommands"
        b"%2FT1DC2JH3J%2F397700885554%2F96rGlfmibIGlgcZRskXaIFfN"
        b"&trigger_id=398738663015.47445629121.803a0bc887a14d10d2c447fce8b6703c"
    )
    DOC_SIGNATURE = (
        "v0=a2114d57b48eac39b9ad189dd8316235a7b4a8d21a10bd27519666489c69b503"
    )

    def test_slacks_published_worked_example_verifies(self):
        """The vendor's own vector is accepted, digest for digest."""
        verdict = verify_slack_v0(
            self.DOC_BODY, self.DOC_SECRET, self.DOC_SIGNATURE, self.DOC_TIMESTAMP,
        )
        assert verdict.ok, verdict.reason
        assert verdict.timestamp == 1531420618

    def test_the_helper_in_this_file_reproduces_the_published_vector(self):
        """The independent derivation used by every other test is itself right.

        If this fails, every "accepted" assertion below is worthless, because
        the signatures they feed in were built wrong.
        """
        assert slack_signature(
            self.DOC_SECRET, self.DOC_TIMESTAMP, self.DOC_BODY,
        ) == self.DOC_SIGNATURE

    def test_an_hmac_over_the_body_alone_is_refused(self):
        """No cross-mode leniency: the ``body`` construction is not accepted here.

        This is the shape that made a Slack route unconfigurable before #713 —
        the plugin could only verify "HMAC over body", so it either rejected
        Slack outright or would have had to accept a construction Slack never
        sends.
        """
        body_only = "v0=" + hmac.new(
            self.DOC_SECRET.encode(), self.DOC_BODY, hashlib.sha256,
        ).hexdigest()
        verdict = verify_slack_v0(
            self.DOC_BODY, self.DOC_SECRET, body_only, self.DOC_TIMESTAMP,
        )
        assert not verdict.ok
        assert verdict.reason == REASON_BAD_SIGNATURE

    def test_a_bare_digest_without_the_v0_prefix_is_refused(self):
        """The ``v0=`` prefix is part of what the sender transmits."""
        verdict = verify_slack_v0(
            self.DOC_BODY, self.DOC_SECRET,
            self.DOC_SIGNATURE[len("v0="):], self.DOC_TIMESTAMP,
        )
        assert not verdict.ok

    def test_a_missing_or_unparseable_timestamp_is_refused_not_skipped(self):
        """A route that asked for a timestamp and got junk REFUSES.

        Falling through would turn a malformed header into a bypass of the
        very window it configures.
        """
        sig = slack_signature(SECRET, "1700000000", JSON_BODY)
        assert verify_slack_v0(JSON_BODY, SECRET, sig, "").reason == REASON_NO_TIMESTAMP
        assert verify_slack_v0(
            JSON_BODY, SECRET, sig, "not-a-number",
        ).reason == REASON_BAD_TIMESTAMP
        assert verify_slack_v0(
            JSON_BODY, SECRET, "", "1700000000",
        ).reason == REASON_NO_SIGNATURE


class TestStripeConstruction:
    """Stripe's ``t=…,v1=…`` header over ``{ts}.{body}`` — #713 ask 3."""

    def test_the_documented_signed_payload_verifies(self):
        header = stripe_signature_header(SECRET, "1700000000", JSON_BODY)
        verdict = verify_stripe_v1(JSON_BODY, SECRET, header)
        assert verdict.ok, verdict.reason
        assert verdict.timestamp == 1700000000

    def test_an_hmac_over_the_body_alone_is_refused(self):
        """No cross-mode leniency, the other direction."""
        digest = hmac.new(SECRET.encode(), JSON_BODY, hashlib.sha256).hexdigest()
        header = f"t=1700000000,v1={digest}"
        assert not verify_stripe_v1(JSON_BODY, SECRET, header).ok

    def test_a_v0_scheme_signature_alone_is_refused(self):
        """Stripe: "ignore all schemes that aren't v1" (downgrade protection).

        ``v0`` is a real, correctly-computed signature over the same payload —
        Stripe sends one for test events — and it must not authenticate.
        """
        header = stripe_signature_header(
            SECRET, "1700000000", JSON_BODY, scheme="v0",
        )
        verdict = verify_stripe_v1(JSON_BODY, SECRET, header)
        assert not verdict.ok
        assert verdict.reason == REASON_NO_SIGNATURE

    def test_any_of_several_v1_signatures_matches_during_secret_rotation(self):
        """Stripe emits one signature per active secret while a secret rolls."""
        good = stripe_signature_header(SECRET, "1700000000", JSON_BODY)
        stale_digest = hmac.new(
            b"previous-fabricated-secret",
            b"1700000000." + JSON_BODY,
            hashlib.sha256,
        ).hexdigest()
        header = f"{good},v1={stale_digest}"
        assert verify_stripe_v1(JSON_BODY, SECRET, header).ok

        reordered = f"t=1700000000,v1={stale_digest}," + good.split(",", 1)[1]
        assert verify_stripe_v1(JSON_BODY, SECRET, reordered).ok

    def test_a_header_with_no_timestamp_or_no_v1_is_refused(self):
        digest = hmac.new(
            SECRET.encode(), b"1700000000." + JSON_BODY, hashlib.sha256,
        ).hexdigest()
        assert verify_stripe_v1(
            JSON_BODY, SECRET, f"v1={digest}",
        ).reason == REASON_NO_TIMESTAMP
        assert verify_stripe_v1(
            JSON_BODY, SECRET, "t=1700000000",
        ).reason == REASON_NO_SIGNATURE
        assert verify_stripe_v1(
            JSON_BODY, SECRET, f"t=nonsense,v1={digest}",
        ).reason == REASON_BAD_TIMESTAMP


# ===========================================================================
# 2. The freshness window — #713 ask 1
# ===========================================================================

class TestFreshnessWindow:
    """A signed timestamp outside the window is refused; a fresh one is not."""

    def _slack(self, offset, **route_kw):
        ts = str(int(NOW) + offset)
        headers = {
            "X-Slack-Request-Timestamp": ts,
            "X-Slack-Signature": slack_signature(SECRET, ts, JSON_BODY),
        }
        return deliver(slack_route(**route_kw), headers)

    def test_a_fresh_delivery_is_accepted(self):
        event, status, _ = self._slack(0)
        assert status is None and event is not None

    def test_a_delivery_just_inside_the_window_is_accepted(self):
        event, status, _ = self._slack(-(DEFAULT_MAX_AGE_SECONDS - 1))
        assert status is None and event is not None

    def test_a_stale_delivery_is_refused(self):
        """The captured-request-replayed-tomorrow case."""
        event, status, msg = self._slack(-(DEFAULT_MAX_AGE_SECONDS + 1))
        assert event is None
        assert status == 403
        assert "freshness window" in msg

    def test_a_delivery_one_day_old_is_refused(self):
        assert self._slack(-86400)[1] == 403

    def test_a_future_timestamp_beyond_the_window_is_refused(self):
        """The window is two-sided.

        A one-sided check would let an attacker hold a signature whose
        timestamp is a year ahead and replay it for a year.
        """
        assert self._slack(DEFAULT_MAX_AGE_SECONDS + 1)[1] == 403

    def test_a_custom_window_is_honoured(self):
        assert self._slack(-30, max_age_seconds=10)[1] == 403
        assert self._slack(-5, max_age_seconds=10)[1] is None

    def test_zero_disables_the_window(self):
        """The explicit opt-out — Stripe's docs call this out too."""
        event, status, _ = self._slack(-86400, max_age_seconds=0)
        assert status is None and event is not None

    def test_stripe_freshness_uses_the_signed_t_value(self):
        ts = str(int(NOW) - DEFAULT_MAX_AGE_SECONDS - 1)
        headers = {
            "Stripe-Signature": stripe_signature_header(SECRET, ts, JSON_BODY),
        }
        assert deliver(stripe_route(), headers)[1] == 403

    def test_a_body_scheme_route_has_no_window_at_all(self):
        """Stated as a fact, not an oversight.

        A ``body`` signature covers no timestamp, so there is nothing to test
        freshness against — which is precisely the weakness #713 could not fix
        for these routes and announces instead.
        """
        headers = {"X-Hub-Signature-256": github_signature(SECRET, JSON_BODY)}
        event, status, _ = deliver(
            github_route(), headers, now=NOW + 86400 * 365,
        )
        assert status is None and event is not None


# ===========================================================================
# 3. The replay cache — #713 ask 2
# ===========================================================================

class TestReplayRefusal:
    """A second copy of an accepted delivery is refused, inside the window."""

    def test_slack_replay_inside_the_window_is_refused(self):
        """Fresh accepted, byte-identical replay refused — the core of #713."""
        cache = ReplayCache()
        ts = str(int(NOW))
        headers = {
            "X-Slack-Request-Timestamp": ts,
            "X-Slack-Signature": slack_signature(SECRET, ts, JSON_BODY),
        }
        route = slack_route()

        event, status, _ = deliver(route, headers, cache=cache)
        assert status is None and event is not None

        # Same bytes, same headers, one second later — still inside the
        # 300s window, so freshness alone would accept it.
        event, status, msg = deliver(route, headers, now=NOW + 1, cache=cache)
        assert event is None
        assert status == 409
        assert "already been accepted" in msg

    def test_a_genuinely_new_delivery_is_still_accepted_after_a_replay(self):
        """The cache refuses repeats, not traffic."""
        cache = ReplayCache()
        route = slack_route()

        def send(ts_offset, body, now):
            ts = str(int(NOW) + ts_offset)
            return deliver(route, {
                "X-Slack-Request-Timestamp": ts,
                "X-Slack-Signature": slack_signature(SECRET, ts, body),
            }, body=body, now=now, cache=cache)

        assert send(0, JSON_BODY, NOW)[1] is None
        assert send(0, JSON_BODY, NOW + 1)[1] == 409      # replay
        assert send(1, JSON_BODY, NOW + 1)[1] is None     # new timestamp
        assert send(0, OTHER_BODY, NOW + 1)[1] is None    # new body

    def test_stripe_replay_inside_the_window_is_refused(self):
        cache = ReplayCache()
        ts = str(int(NOW))
        headers = {"Stripe-Signature": stripe_signature_header(SECRET, ts, JSON_BODY)}
        route = stripe_route()

        assert deliver(route, headers, cache=cache)[1] is None
        assert deliver(route, headers, now=NOW + 60, cache=cache)[1] == 409

    def test_past_the_ttl_the_freshness_window_refuses_it_instead(self):
        """The two mechanisms hand off, and neither leaves a gap.

        Once the cache entry expires the delivery is no longer remembered —
        and by construction it is also no longer fresh, so it is refused as
        stale rather than accepted.  A cache TTL sized to the window is what
        makes that true.
        """
        cache = ReplayCache()
        ts = str(int(NOW))
        headers = {
            "X-Slack-Request-Timestamp": ts,
            "X-Slack-Signature": slack_signature(SECRET, ts, JSON_BODY),
        }
        route = slack_route()

        assert deliver(route, headers, cache=cache)[1] is None
        later = NOW + DEFAULT_MAX_AGE_SECONDS + 1
        assert len(cache) == 1
        event, status, msg = deliver(route, headers, now=later, cache=cache)
        assert event is None
        assert status == 403          # stale, not 409 — the entry has expired
        assert "freshness window" in msg

    def test_a_refused_request_is_never_recorded(self):
        """No cache poisoning.

        If arrival recorded a key, anyone who could reach the port could
        register a guessed delivery id and have the genuine delivery refused
        as a replay — a denial of service built out of the anti-replay
        control.  So only a delivery that passed BOTH earlier checks is
        written.
        """
        cache = ReplayCache()
        route = slack_route()
        ts = str(int(NOW))
        good_sig = slack_signature(SECRET, ts, JSON_BODY)

        # A forgery carrying the genuine signature over a different body.
        assert deliver(route, {
            "X-Slack-Request-Timestamp": ts,
            "X-Slack-Signature": good_sig,
        }, body=OTHER_BODY, cache=cache)[1] == 403
        assert len(cache) == 0

        # A stale-but-authentic delivery.
        old = str(int(NOW) - 86400)
        assert deliver(route, {
            "X-Slack-Request-Timestamp": old,
            "X-Slack-Signature": slack_signature(SECRET, old, JSON_BODY),
        }, cache=cache)[1] == 403
        assert len(cache) == 0

        # The genuine delivery still gets through.
        assert deliver(route, {
            "X-Slack-Request-Timestamp": ts,
            "X-Slack-Signature": good_sig,
        }, cache=cache)[1] is None

    def test_delivery_id_dedupes_a_body_scheme_route(self):
        """GitHub's shape: no signed timestamp, but a delivery GUID."""
        cache = ReplayCache()
        route = github_route(replay_key_header="X-GitHub-Delivery")
        sig = github_signature(SECRET, JSON_BODY)

        def send(guid, now):
            return deliver(route, {
                "X-Hub-Signature-256": sig,
                "X-GitHub-Delivery": guid,
            }, now=now, cache=cache)

        assert send("11111111-2222-3333-4444-555555555555", NOW)[1] is None
        assert send("11111111-2222-3333-4444-555555555555", NOW + 5)[1] == 409
        assert send("99999999-8888-7777-6666-555555555555", NOW + 5)[1] is None

    def test_a_body_route_without_a_delivery_id_accepts_the_replay(self):
        """The limit of what is achievable, asserted rather than implied.

        This is the state every pre-#713 route is in and the honest cost of
        the backward-compatibility choice: with no signed timestamp and no
        delivery id there is nothing that distinguishes a replay from the
        original, so it is accepted — and a startup WARNING says so (see
        ``TestWeakPostureIsAnnounced``).
        """
        cache = ReplayCache()
        route = github_route()
        headers = {"X-Hub-Signature-256": github_signature(SECRET, JSON_BODY)}

        assert deliver(route, headers, cache=cache)[1] is None
        assert deliver(route, headers, now=NOW + 1, cache=cache)[1] is None
        assert len(cache) == 0

    def test_token_mode_dedupes_only_with_a_delivery_id(self):
        """What replay protection means for the WEAKER mode (#930).

        ``token`` carries the shared secret verbatim, so the credential is
        identical on every request and there is no signed payload to bind a
        timestamp into.  Keying a cache on the credential would refuse the
        second legitimate delivery, so a delivery id is the only thing
        available — and it bounds a verbatim replay, not an attacker who holds
        the token and mints fresh requests.  That is credential compromise,
        which this mode concedes by design.
        """
        cache = ReplayCache()
        route = RouteConfig.from_dict({
            "path": "/webhook/gitlab",
            "secret_header": "X-Gitlab-Token",
            "secret_algo": "token",
            "replay_key_header": "X-Gitlab-Event-UUID",
        })
        base = {"X-Gitlab-Token": SECRET}

        assert deliver(route, {**base, "X-Gitlab-Event-UUID": "a"}, cache=cache)[1] is None
        assert deliver(
            route, {**base, "X-Gitlab-Event-UUID": "a"}, now=NOW + 1, cache=cache,
        )[1] == 409
        assert deliver(
            route, {**base, "X-Gitlab-Event-UUID": "b"}, now=NOW + 1, cache=cache,
        )[1] is None

        # Without the id header there is nothing to key on, and the same
        # credential authenticates every request for ever.
        bare = RouteConfig.from_dict({
            "path": "/webhook/gitlab",
            "secret_header": "X-Gitlab-Token",
            "secret_algo": "token",
        })
        assert deliver(bare, base, cache=cache)[1] is None
        assert deliver(bare, base, now=NOW + 1, cache=cache)[1] is None

    def test_two_routes_do_not_collide_on_a_shared_delivery_id_space(self):
        cache = ReplayCache()
        route = github_route(replay_key_header="X-GitHub-Delivery")
        headers = {
            "X-Hub-Signature-256": github_signature(SECRET, JSON_BODY),
            "X-GitHub-Delivery": "shared-id",
        }
        assert deliver(route, headers, cache=cache, name="alpha")[1] is None
        assert deliver(route, headers, cache=cache, name="beta")[1] is None
        assert deliver(route, headers, cache=cache, name="alpha")[1] == 409


class TestReplayCacheItself:
    """The cache's own bounds, with the clock injected."""

    def test_an_entry_expires_exactly_at_its_ttl(self):
        cache = ReplayCache()
        assert cache.check_and_record("k", 100.0, ttl=10) is False
        assert cache.check_and_record("k", 109.9, ttl=10) is True
        assert cache.check_and_record("k", 110.0, ttl=10) is False

    def test_expired_entries_are_purged_rather_than_accumulating(self):
        cache = ReplayCache()
        for i in range(50):
            cache.check_and_record(f"k{i}", 100.0, ttl=10)
        assert len(cache) == 50
        cache.check_and_record("later", 200.0, ttl=10)
        assert len(cache) == 1

    def test_the_ceiling_evicts_oldest_first_and_counts_it(self):
        """An eviction under the ceiling re-opens a replay window; it is counted."""
        cache = ReplayCache(max_entries=3)
        for i in range(5):
            cache.check_and_record(f"k{i}", 100.0, ttl=300)
        assert len(cache) == 3
        assert cache.evictions == 2
        # k0 was evicted before its TTL, so its replay is no longer caught.
        assert cache.check_and_record("k0", 100.0, ttl=300) is False
        assert cache.check_and_record("k4", 100.0, ttl=300) is True

    def test_a_zero_or_negative_ceiling_is_raised_to_one(self):
        """A stray 0 must not silently accept every replay."""
        assert ReplayCache(max_entries=0).max_entries == 1
        assert ReplayCache(max_entries=-5).max_entries == 1

    def test_replays_refused_is_counted(self):
        cache = ReplayCache()
        cache.check_and_record("k", 100.0, ttl=10)
        cache.check_and_record("k", 101.0, ttl=10)
        cache.check_and_record("k", 102.0, ttl=10)
        assert cache.get_stats()["replays_refused"] == 2

    def test_key_selection_follows_the_scheme_regime(self):
        """A timestamp-bound scheme keys on the unforgeable signature.

        Preferring a delivery id there would be worse: no sender signs its
        headers, so a replayer rewrites the id and walks past the cache.
        """
        assert replay_key_for("r", "id-1", "sig-1", prefer_signature=True) \
            == "r:sig:sig-1"
        assert replay_key_for("r", "id-1", None, prefer_signature=True) \
            == "r:id:id-1"
        assert replay_key_for("r", "id-1", "sig-1", prefer_signature=False) \
            == "r:id:id-1"
        assert replay_key_for("r", None, "sig-1", prefer_signature=False) is None
        assert replay_key_for("r", None, None, prefer_signature=True) is None


# ===========================================================================
# 4. Tampering
# ===========================================================================

class TestTampering:
    """A modified request does not authenticate, on any scheme."""

    def test_slack_tampered_body_is_refused(self):
        ts = str(int(NOW))
        headers = {
            "X-Slack-Request-Timestamp": ts,
            "X-Slack-Signature": slack_signature(SECRET, ts, JSON_BODY),
        }
        assert deliver(slack_route(), headers, body=OTHER_BODY)[1] == 403

    def test_slack_tampered_timestamp_is_refused(self):
        """Rewriting the timestamp to refresh a stale capture breaks the digest.

        This is the property that makes the window meaningful rather than
        decorative: the timestamp is INSIDE the signed basestring.
        """
        old = str(int(NOW) - 86400)
        headers = {
            # The signature is genuine — for yesterday's timestamp.
            "X-Slack-Signature": slack_signature(SECRET, old, JSON_BODY),
            # The attacker refreshes the header to get inside the window.
            "X-Slack-Request-Timestamp": str(int(NOW)),
        }
        event, status, msg = deliver(slack_route(), headers)
        assert event is None
        assert status == 403
        assert "Signature verification failed" in msg

    def test_stripe_tampered_body_and_timestamp_are_refused(self):
        ts = str(int(NOW))
        good = stripe_signature_header(SECRET, ts, JSON_BODY)
        assert deliver(stripe_route(), {"Stripe-Signature": good},
                       body=OTHER_BODY)[1] == 403

        digest = good.split("v1=")[1]
        refreshed = f"t={int(NOW)},v1={digest}"
        old_header = stripe_signature_header(SECRET, str(int(NOW) - 86400), JSON_BODY)
        assert old_header != refreshed
        assert deliver(stripe_route(), {"Stripe-Signature": refreshed})[1] is None
        # ...and moving the t= of a genuinely old signature does not work:
        old_digest = old_header.split("v1=")[1]
        assert deliver(stripe_route(), {
            "Stripe-Signature": f"t={int(NOW)},v1={old_digest}",
        })[1] == 403

    def test_a_wrong_secret_is_refused(self):
        ts = str(int(NOW))
        headers = {
            "X-Slack-Request-Timestamp": ts,
            "X-Slack-Signature": slack_signature("a-different-secret", ts, JSON_BODY),
        }
        assert deliver(slack_route(), headers)[1] == 403


# ===========================================================================
# 5. Fail-closed: expressible-but-wrong configurations are REFUSED
# ===========================================================================

class TestFailClosedConfiguration:
    """A route that cannot honour its own keys is a 500, never a downgrade.

    The plugin's standing rule: widening the vocabulary widens what a route
    may SAY, never what it may omit.  Every case here is also a
    ``validate_config`` error (see ``TestValidateConfig``); this is the
    runtime half, for a route built around that check.
    """

    @pytest.mark.parametrize("overrides,why", [
        ({"signature_scheme": "slack_v0"}, "a typo must not fall back to 'body'"),
        ({"signature_scheme": "hmac-sha256"}, "the algo name is not a scheme"),
    ])
    def test_an_unknown_scheme_refuses(self, overrides, why):
        route = github_route(**overrides)
        headers = {"X-Hub-Signature-256": github_signature(SECRET, JSON_BODY)}
        event, status, msg = deliver(route, headers)
        assert event is None, why
        assert status == 500
        assert "signature_scheme" in msg

    def test_slack_without_a_timestamp_header_refuses(self):
        route = slack_route(timestamp_header=None)
        ts = str(int(NOW))
        headers = {
            "X-Slack-Request-Timestamp": ts,
            "X-Slack-Signature": slack_signature(SECRET, ts, JSON_BODY),
        }
        assert deliver(route, headers)[1] == 500

    def test_stripe_with_a_timestamp_header_refuses(self):
        """Two sources for one timestamp: the unsigned one could win."""
        route = stripe_route(timestamp_header="X-When")
        ts = str(int(NOW))
        headers = {
            "Stripe-Signature": stripe_signature_header(SECRET, ts, JSON_BODY),
            "X-When": ts,
        }
        assert deliver(route, headers)[1] == 500

    def test_a_timestamp_header_on_the_body_scheme_refuses(self):
        """The theatre case.

        Nothing signs the header, so whoever replays the request sets it to
        now.  Accepting the configuration would ship a freshness window that
        checks an attacker-controlled value — worse than none, because it
        reads like protection.
        """
        route = github_route(timestamp_header="X-Delivery-Time")
        headers = {
            "X-Hub-Signature-256": github_signature(SECRET, JSON_BODY),
            "X-Delivery-Time": str(int(NOW)),
        }
        event, status, msg = deliver(route, headers)
        assert event is None
        assert status == 500
        assert "timestamp_header" in msg

    def test_a_timestamp_scheme_with_token_algo_refuses(self):
        """No cross-mode leniency between the scheme and the MAC."""
        route = slack_route(secret_algo="token")
        ts = str(int(NOW))
        headers = {
            "X-Slack-Request-Timestamp": ts,
            "X-Slack-Signature": SECRET,
        }
        event, status, msg = deliver(route, headers)
        assert event is None
        assert status == 500
        assert "hmac-sha256" in msg

    def test_a_broken_scheme_config_is_500_not_403(self):
        """A broken server and a bad request must not read the same."""
        route = slack_route(timestamp_header=None)
        assert deliver(route, {"X-Slack-Signature": "v0=deadbeef"})[1] == 500


# ===========================================================================
# 6. Configuration validation
# ===========================================================================

class TestValidateConfig:
    """``validate_config`` reports each expressible-but-wrong route (#713)."""

    def _errors(self, route):
        ok, errors = validate_config({"routes": {"r": {"path": "/w", **route}}})
        return ok, " | ".join(errors)

    def test_an_unknown_signature_scheme_is_an_error(self):
        ok, errors = self._errors({"signature_scheme": "slack-v1"})
        assert not ok and "signature_scheme" in errors

    def test_slack_without_a_timestamp_header_is_an_error(self):
        ok, errors = self._errors({
            "secret_header": "X-Slack-Signature", "secret_algo": "hmac-sha256",
            "signature_scheme": "slack-v0",
        })
        assert not ok and "timestamp_header is required" in errors

    def test_stripe_with_a_timestamp_header_is_an_error(self):
        ok, errors = self._errors({
            "secret_header": "Stripe-Signature", "secret_algo": "hmac-sha256",
            "signature_scheme": "stripe-v1", "timestamp_header": "X-When",
        })
        assert not ok and "must not be set" in errors

    def test_a_timestamp_header_on_the_body_scheme_is_an_error(self):
        ok, errors = self._errors({"timestamp_header": "X-When"})
        assert not ok and "replay_key_header" in errors

    def test_a_timestamp_scheme_with_token_algo_is_an_error(self):
        ok, errors = self._errors({
            "secret_header": "X-Slack-Signature", "secret_algo": "token",
            "signature_scheme": "slack-v0",
            "timestamp_header": "X-Slack-Request-Timestamp",
        })
        assert not ok and "hmac-sha256" in errors

    @pytest.mark.parametrize("value", [-1, "300", 1.5, True])
    def test_a_bad_max_age_is_an_error(self, value):
        ok, _ = self._errors({"max_age_seconds": value})
        assert not ok

    def test_a_well_formed_slack_route_validates(self):
        ok, errors = self._errors({
            "secret_header": "X-Slack-Signature", "secret_algo": "hmac-sha256",
            "signature_scheme": "slack-v0",
            "timestamp_header": "X-Slack-Request-Timestamp",
            "max_age_seconds": 120,
        })
        assert ok, errors

    def test_a_pre_713_route_still_validates(self):
        ok, errors = self._errors({
            "secret_header": "X-Hub-Signature-256", "secret_algo": "hmac-sha256",
        })
        assert ok, errors

    @pytest.mark.parametrize("value", [0, -1, "big", 1.5])
    def test_a_bad_replay_cache_size_is_an_error(self, value):
        ok, errors = validate_config({"replay_cache_size": value})
        assert not ok and "replay_cache_size" in " | ".join(errors)

    def test_an_unparseable_max_age_falls_back_to_the_default_not_to_zero(self):
        """A typo must never land on the disabled posture."""
        route = RouteConfig.from_dict({"path": "/w", "max_age_seconds": "oops"})
        assert route.max_age_seconds == DEFAULT_MAX_AGE_SECONDS

    def test_an_expanded_env_var_string_is_still_a_number(self):
        route = RouteConfig.from_dict({"path": "/w", "max_age_seconds": "120"})
        assert route.max_age_seconds == 120


# ===========================================================================
# 7. Constant-time comparison — #713 ask 4
# ===========================================================================

class TestConstantTimeComparison:
    """Every credential comparison goes through ``hmac.compare_digest``."""

    def test_no_scheme_compares_a_credential_with_a_plain_equality(self):
        """A source-level guard: ``==`` on a digest is the defect, not a style.

        The WS bearer check established the pattern; this asserts the webhook
        path did not grow a second, naive comparison beside it.
        """
        import inspect
        from jaato_server.shared.plugins.webhook import routes, signature_schemes

        for module in (signature_schemes, routes):
            source = inspect.getsource(module)
            for marker in ("== signature_header", "== expected", "== candidate",
                           "signature_header ==", "expected ==", "candidate =="):
                assert marker not in source, (
                    f"{module.__name__} compares a credential with '=='; use "
                    f"constant_time_equals (hmac.compare_digest)"
                )
        # The compiled code object, not the source: reading the source is
        # satisfied by the function's own docstring, which names
        # ``hmac.compare_digest``.
        assert "compare_digest" in (
            signature_schemes.constant_time_equals.__code__.co_names
        )

    def test_a_non_ascii_header_is_refused_rather_than_raising(self):
        """``compare_digest`` raises TypeError on a non-ASCII str.

        Both operands come off the wire, so a crafted header must not turn a
        verification failure into a 500.
        """
        for route, headers in (
            (slack_route(), {"X-Slack-Request-Timestamp": str(int(NOW)),
                             "X-Slack-Signature": "v0=café–ü"}),
            (stripe_route(), {"Stripe-Signature": f"t={int(NOW)},v1=café–ü"}),
            (github_route(), {"X-Hub-Signature-256": "sha256=café–ü"}),
        ):
            event, status, _ = deliver(route, headers)
            assert event is None
            assert status == 403


# ===========================================================================
# 8. Backward compatibility and the announced posture
# ===========================================================================

class TestBackwardCompatibility:
    """A route configured before #713 behaves exactly as it did."""

    def test_a_github_route_declaring_no_new_key_still_works(self):
        headers = {"X-Hub-Signature-256": github_signature(SECRET, JSON_BODY)}
        event, status, _ = deliver(github_route(), headers, cache=ReplayCache())
        assert status is None
        assert event["source"] == "r"
        assert event["payload"] == {"event": "push", "n": 1}

    def test_a_gitlab_token_route_declaring_no_new_key_still_works(self):
        route = RouteConfig.from_dict({
            "path": "/w", "secret_header": "X-Gitlab-Token", "secret_algo": "token",
        })
        event, status, _ = deliver(
            route, {"X-Gitlab-Token": SECRET}, cache=ReplayCache(),
        )
        assert status is None and event is not None

    def test_a_caller_passing_no_cache_and_no_clock_is_unchanged(self):
        """``parse_webhook_request`` keeps its pre-#713 call shape."""
        headers = {
            "Content-Type": "application/json",
            "X-Hub-Signature-256": github_signature(SECRET, JSON_BODY),
        }
        event, status, _ = parse_webhook_request(
            JSON_BODY, headers, "r", github_route(), SECRET,
        )
        assert status is None and event is not None

    def test_the_defaults_of_a_bare_route_are_the_old_behaviour(self):
        route = RouteConfig.from_dict({"path": "/w"})
        assert route.signature_scheme == "body"
        assert route.timestamp_header is None
        assert route.replay_key_header is None
        assert route.binds_timestamp() is False
        assert route.replay_protected() is False


class TestWeakPostureIsAnnounced:
    """A weakened posture announces itself — the tree's standing rule.

    ``scrub_secret_env: none``, ``--ws-unsafe-no-auth`` and ``secret_algo:
    'token'`` all do this.  The backward-compatibility choice for #713 means
    the protection could NOT be switched on for a route whose sender signs no
    timestamp — so what is switched on instead is saying so, per route, at
    listener startup.
    """

    def _start_and_capture(self, caplog, route_data):
        config = WebhookConfig.from_dict({
            "port": 9199, "secret": SECRET, "routes": {"r": route_data},
        })
        server = WebhookHTTPServer(config, lambda *a: None)
        with caplog.at_level(logging.INFO,
                             logger="jaato_server.shared.plugins.webhook.http_server"):
            server._warn_on_weak_posture()
        return caplog.text

    def test_a_body_scheme_route_is_warned_about(self, caplog):
        text = self._start_and_capture(caplog, {
            "path": "/w", "secret_header": "X-Hub-Signature-256",
            "secret_algo": "hmac-sha256",
        })
        assert "NO replay protection" in text
        assert "authenticates forever" in text
        assert "replay_key_header" in text

    def test_a_delivery_id_route_states_its_bound(self, caplog):
        text = self._start_and_capture(caplog, {
            "path": "/w", "secret_header": "X-Hub-Signature-256",
            "secret_algo": "hmac-sha256",
            "replay_key_header": "X-GitHub-Delivery",
        })
        assert "refused for" in text
        assert "rewrites that header is not caught" in text

    def test_a_disabled_window_announces_itself(self, caplog):
        text = self._start_and_capture(caplog, {
            "path": "/w", "secret_header": "X-Slack-Signature",
            "secret_algo": "hmac-sha256", "signature_scheme": "slack-v0",
            "timestamp_header": "X-Slack-Request-Timestamp",
            "max_age_seconds": 0,
        })
        assert "freshness window is DISABLED" in text

    def test_the_strong_posture_says_nothing(self, caplog):
        text = self._start_and_capture(caplog, {
            "path": "/w", "secret_header": "X-Slack-Signature",
            "secret_algo": "hmac-sha256", "signature_scheme": "slack-v0",
            "timestamp_header": "X-Slack-Request-Timestamp",
        })
        assert "replay" not in text.lower()

    def test_stats_report_the_replay_counters(self):
        config = WebhookConfig.from_dict({"port": 9199, "routes": {}})
        server = WebhookHTTPServer(config, lambda *a: None)
        stats = server.get_stats()
        assert stats["requests_refused_replay"] == 0
        assert stats["replay_cache"]["max_entries"] > 0
