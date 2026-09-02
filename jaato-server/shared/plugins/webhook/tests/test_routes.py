"""Tests for webhook route matching and HMAC verification."""

import hashlib
import hmac
import json

from shared.plugins.webhook.config import RouteConfig
from shared.plugins.webhook.routes import (
    match_route,
    parse_webhook_request,
    verify_signature,
)


class TestMatchRoute:
    """Tests for route matching."""

    def test_exact_match(self):
        routes = {
            "github": RouteConfig(path="/webhook/github"),
            "slack": RouteConfig(path="/webhook/slack"),
        }
        result = match_route("/webhook/github", routes)
        assert result is not None
        name, route = result
        assert name == "github"
        assert route.path == "/webhook/github"

    def test_no_match(self):
        routes = {"github": RouteConfig(path="/webhook/github")}
        assert match_route("/webhook/unknown", routes) is None

    def test_empty_routes(self):
        assert match_route("/webhook", {}) is None


class TestVerifySignature:
    """Tests for HMAC signature verification."""

    def test_valid_signature(self):
        body = b'{"action": "push"}'
        secret = "mysecret"
        sig = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()

        assert verify_signature(body, secret, sig, "hmac-sha256") is True

    def test_valid_signature_with_prefix(self):
        body = b'{"action": "push"}'
        secret = "mysecret"
        sig = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()

        assert verify_signature(body, secret, sig, "hmac-sha256") is True

    def test_invalid_signature(self):
        body = b'{"action": "push"}'
        assert verify_signature(body, "secret", "bad_signature", "hmac-sha256") is False

    def test_empty_signature(self):
        assert verify_signature(b"body", "secret", "", "hmac-sha256") is False

    def test_unsupported_algo(self):
        assert verify_signature(b"body", "secret", "sig", "hmac-md5") is False


class TestParseWebhookRequest:
    """Tests for full request parsing and validation."""

    def _make_body(self, payload):
        return json.dumps(payload).encode('utf-8')

    def test_valid_request(self):
        body = self._make_body({"action": "opened"})
        headers = {
            "Content-Type": "application/json",
            "X-GitHub-Event": "pull_request",
        }
        route = RouteConfig(
            path="/webhook/github",
            event_type_header="X-GitHub-Event",
            allow_unauthenticated=True,  # this test covers parsing, not auth
        )

        event, status, msg = parse_webhook_request(
            body, headers, "github", route
        )
        assert event is not None
        assert status is None
        assert event["source"] == "github"
        assert event["event_type"] == "pull_request"
        assert event["payload"] == {"action": "opened"}

    def test_wrong_content_type(self):
        body = b"not json"
        headers = {"Content-Type": "text/plain"}
        route = RouteConfig(path="/webhook")

        event, status, msg = parse_webhook_request(
            body, headers, "generic", route
        )
        assert event is None
        assert status == 415

    def test_invalid_json(self):
        body = b"not valid json"
        headers = {"Content-Type": "application/json"}
        route = RouteConfig(path="/webhook", allow_unauthenticated=True)

        event, status, msg = parse_webhook_request(
            body, headers, "generic", route
        )
        assert event is None
        assert status == 400

    def test_hmac_verification_success(self):
        body = self._make_body({"ref": "refs/heads/main"})
        secret = "webhook_secret"
        sig = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()

        headers = {
            "Content-Type": "application/json",
            "X-Hub-Signature-256": f"sha256={sig}",
        }
        route = RouteConfig(
            path="/webhook/github",
            secret_header="X-Hub-Signature-256",
            secret_algo="hmac-sha256",
        )

        event, status, msg = parse_webhook_request(
            body, headers, "github", route, global_secret=secret
        )
        assert event is not None
        assert status is None

    def test_hmac_verification_failure(self):
        body = self._make_body({"ref": "refs/heads/main"})
        headers = {
            "Content-Type": "application/json",
            "X-Hub-Signature-256": "sha256=invalid",
        }
        route = RouteConfig(
            path="/webhook/github",
            secret_header="X-Hub-Signature-256",
            secret_algo="hmac-sha256",
        )

        event, status, msg = parse_webhook_request(
            body, headers, "github", route, global_secret="real_secret"
        )
        assert event is None
        assert status == 403

    def test_no_secret_configured(self):
        body = self._make_body({})
        headers = {"Content-Type": "application/json"}
        route = RouteConfig(
            path="/webhook",
            secret_header="X-Signature",
            secret_algo="hmac-sha256",
        )

        event, status, msg = parse_webhook_request(
            body, headers, "generic", route, global_secret=None
        )
        assert event is None
        assert status == 500

    def test_event_type_defaults_to_unknown(self):
        body = self._make_body({"data": 1})
        headers = {"Content-Type": "application/json"}
        route = RouteConfig(path="/webhook", allow_unauthenticated=True)

        event, status, msg = parse_webhook_request(
            body, headers, "generic", route
        )
        assert event is not None
        assert event["event_type"] == "unknown"

    def test_metadata_included(self):
        body = self._make_body({})
        headers = {"Content-Type": "application/json"}
        route = RouteConfig(
            path="/webhook",
            metadata={"source": "test", "env": "staging"},
            allow_unauthenticated=True,
        )

        event, status, msg = parse_webhook_request(
            body, headers, "generic", route
        )
        assert event is not None
        assert event["metadata"] == {"source": "test", "env": "staging"}


class TestUnsignedRouteFailClosed:
    """A route without an HMAC secret is refused unless auth is provided."""

    @staticmethod
    def _body():
        return json.dumps({"x": 1}).encode()

    _HEADERS = {"Content-Type": "application/json"}

    def test_unsigned_route_refused_by_default(self):
        # No secret, no transport auth, no opt-in → 401 (fail-closed).
        route = RouteConfig(path="/webhook")
        event, status, msg = parse_webhook_request(
            self._body(), self._HEADERS, "generic", route
        )
        assert event is None
        assert status == 401
        assert "no signature verification" in msg.lower()

    def test_unsigned_route_allowed_with_explicit_optin(self):
        route = RouteConfig(path="/webhook", allow_unauthenticated=True)
        event, status, msg = parse_webhook_request(
            self._body(), self._HEADERS, "generic", route
        )
        assert event is not None
        assert status is None

    def test_unsigned_route_allowed_when_transport_authenticated(self):
        # mTLS / IP-allowlist deployment: caller passes transport_authenticated.
        route = RouteConfig(path="/webhook")
        event, status, msg = parse_webhook_request(
            self._body(), self._HEADERS, "generic", route,
            transport_authenticated=True,
        )
        assert event is not None
        assert status is None

    def test_signed_route_still_enforced_even_with_transport_auth(self):
        # A declared secret is verified regardless of transport auth: a bad
        # signature is 403 even when the deployment has mTLS/allowlist.
        route = RouteConfig(
            path="/webhook", secret_header="X-Sig", secret_algo="hmac-sha256",
        )
        event, status, msg = parse_webhook_request(
            self._body(), {**self._HEADERS, "X-Sig": "sha256=deadbeef"},
            "generic", route, global_secret="s3cr3t",
            transport_authenticated=True,
        )
        assert event is None
        assert status == 403

    def test_incomplete_hmac_config_fails_closed(self):
        # secret_header without secret_algo must NOT silently downgrade to
        # unsigned — even under transport auth it is a 500 misconfiguration.
        route = RouteConfig(path="/webhook", secret_header="X-Sig")  # no algo
        event, status, msg = parse_webhook_request(
            self._body(), self._HEADERS, "generic", route,
            global_secret="s3cr3t", transport_authenticated=True,
        )
        assert event is None
        assert status == 500
        assert "not both" in msg.lower()

    def test_incomplete_hmac_config_algo_only_fails_closed(self):
        route = RouteConfig(path="/webhook", secret_algo="hmac-sha256")  # no header
        event, status, msg = parse_webhook_request(
            self._body(), self._HEADERS, "generic", route,
        )
        assert event is None
        assert status == 500
