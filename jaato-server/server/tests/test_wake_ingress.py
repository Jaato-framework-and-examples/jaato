"""wake_ingress.process_wake — Stage-B auth + routing decision flow.

Injects fake resolve_binding / wake_fn so the security-critical path (verify
over raw body, replay window, status mapping) is tested without a socket or a
live SessionManager.  Also covers WakeIngressConfig parsing.
"""

import base64
import json

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from server.session_manager import WakeOutcome
from server.wake_binding_registry import WakeBinding
from server.wake_ingress import WakeIngressConfig, process_wake


NOW = 1000.0
WINDOW = 300


def _pub(priv) -> str:
    return priv.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()


def _request(priv, payload, sign_body=None):
    """Return (raw_body, headers) with an Ed25519 sig over the raw body."""
    body = json.dumps(payload).encode()
    sig = base64.b64encode(priv.sign(sign_body if sign_body is not None else body))
    return body, {"X-Jaato-Wake-Signature": sig.decode()}


class _Wake:
    def __init__(self, outcome=WakeOutcome.OK):
        self.calls = []
        self.outcome = outcome

    def __call__(self, session_id, text, source, event_id,
                 wake_ref=None, cascade_driver_id=None):
        self.calls.append((session_id, text, source, event_id))
        self.last_wake_ref = wake_ref
        self.last_cid = cascade_driver_id
        return (self.outcome, self.outcome.value)


def _resolver(priv, session_id="s1", ref="github-pr:o/r#1", cid=None):
    binding = WakeBinding(session_id=session_id, workspace_path="/ws",
                          trust_keys=[_pub(priv)], expires_at=9e12,
                          cascade_driver_id=cid)
    return lambda wr: binding if wr == ref else None


def _payload(**over):
    p = {"wake_ref": "github-pr:o/r#1", "text": "review body",
         "source": "github-pr", "event_id": "rev_42", "ts": NOW}
    p.update(over)
    return p


# ---- happy path ---------------------------------------------------------------

def test_valid_wake_drives_session():
    priv = Ed25519PrivateKey.generate()
    wake = _Wake(WakeOutcome.OK)
    body, headers = _request(priv, _payload())
    status, _ = process_wake(body, headers, _resolver(priv), wake, WINDOW, NOW)
    assert status == 200
    assert wake.calls == [("s1", "review body", "github-pr", "rev_42")]


def test_source_defaults_when_absent():
    priv = Ed25519PrivateKey.generate()
    wake = _Wake()
    p = _payload()
    del p["source"]
    body, headers = _request(priv, p)
    process_wake(body, headers, _resolver(priv), wake, WINDOW, NOW)
    assert wake.calls[0][2] == "wake"  # default provenance tag


# ---- auth failures (4xx, no drive) --------------------------------------------

def test_missing_signature_401():
    priv = Ed25519PrivateKey.generate()
    wake = _Wake()
    body, _ = _request(priv, _payload())
    status, _ = process_wake(body, {}, _resolver(priv), wake, WINDOW, NOW)
    assert status == 401
    assert wake.calls == []


def test_unknown_wake_ref_404():
    priv = Ed25519PrivateKey.generate()
    wake = _Wake()
    body, headers = _request(priv, _payload(wake_ref="github-pr:o/r#999"))
    status, _ = process_wake(body, headers, _resolver(priv), wake, WINDOW, NOW)
    assert status == 404
    assert wake.calls == []


def test_bad_signature_401():
    signer = Ed25519PrivateKey.generate()
    resolver_key = Ed25519PrivateKey.generate()   # binding trusts a DIFFERENT key
    wake = _Wake()
    body, headers = _request(signer, _payload())
    status, _ = process_wake(body, headers, _resolver(resolver_key), wake, WINDOW, NOW)
    assert status == 401
    assert wake.calls == []


def test_malformed_body_400():
    priv = Ed25519PrivateKey.generate()
    wake = _Wake()
    sig = base64.b64encode(priv.sign(b"{ not json")).decode()
    status, _ = process_wake(b"{ not json", {"X-Jaato-Wake-Signature": sig},
                             _resolver(priv), wake, WINDOW, NOW)
    assert status == 400
    assert wake.calls == []


def test_stale_ts_401():
    priv = Ed25519PrivateKey.generate()
    wake = _Wake()
    body, headers = _request(priv, _payload(ts=NOW))
    # request signed at ts=NOW, but arrives far outside the window
    status, _ = process_wake(body, headers, _resolver(priv), wake, WINDOW, NOW + 10_000)
    assert status == 401
    assert wake.calls == []


def test_missing_text_400_after_verify():
    priv = Ed25519PrivateKey.generate()
    wake = _Wake()
    p = _payload()
    del p["text"]
    body, headers = _request(priv, p)
    status, _ = process_wake(body, headers, _resolver(priv), wake, WINDOW, NOW)
    assert status == 400
    assert wake.calls == []


# ---- wake_fn outcome → status mapping -----------------------------------------

def test_duplicate_is_200():
    priv = Ed25519PrivateKey.generate()
    wake = _Wake(WakeOutcome.DUPLICATE)
    body, headers = _request(priv, _payload())
    status, _ = process_wake(body, headers, _resolver(priv), wake, WINDOW, NOW)
    assert status == 200


def test_transient_failure_is_503():
    priv = Ed25519PrivateKey.generate()
    body, headers = _request(priv, _payload())
    for oc in (WakeOutcome.REVIVE_FAILED, WakeOutcome.NOT_DRIVABLE):
        status, _ = process_wake(body, headers, _resolver(priv), _Wake(oc), WINDOW, NOW)
        assert status == 503


def test_permanent_failure_is_400():
    priv = Ed25519PrivateKey.generate()
    body, headers = _request(priv, _payload())
    for oc in (WakeOutcome.INVALID, WakeOutcome.UNRESOLVED):
        status, _ = process_wake(body, headers, _resolver(priv), _Wake(oc), WINDOW, NOW)
        assert status == 400


def test_deferred_is_200():
    priv = Ed25519PrivateKey.generate()
    wake = _Wake(WakeOutcome.DEFERRED)
    body, headers = _request(priv, _payload())
    status, _ = process_wake(body, headers, _resolver(priv), wake, WINDOW, NOW)
    assert status == 200  # revived cold + deferred = accepted


def test_wake_ref_and_cid_passed_to_wake_fn():
    priv = Ed25519PrivateKey.generate()
    wake = _Wake()
    body, headers = _request(priv, _payload())
    process_wake(body, headers, _resolver(priv, cid="bot-cid-9"), wake, WINDOW, NOW)
    assert wake.last_wake_ref == "github-pr:o/r#1"
    assert wake.last_cid == "bot-cid-9"


# ---- config -------------------------------------------------------------------

def test_config_from_dict():
    c = WakeIngressConfig.from_dict({
        "enabled": True, "host": "0.0.0.0", "port": 9200, "path": "/w",
        "allowed_ips": ["10.0.0.0/8"], "rate_limit_per_second": 5,
        "replay_window_seconds": 120, "max_body_size": 4096,
        "tls": {"enabled": True, "certfile": "/c.pem", "keyfile": "/k.pem"},
    })
    assert c.enabled and c.host == "0.0.0.0" and c.port == 9200
    assert c.allowed_ips == ["10.0.0.0/8"] and c.rate_limit_per_second == 5
    assert c.replay_window_seconds == 120
    assert c.tls_certfile == "/c.pem" and c.tls_keyfile == "/k.pem"


def test_config_defaults_and_disabled():
    c = WakeIngressConfig.from_dict({})
    assert c.enabled is False
    assert c.replay_window_seconds == 300  # server-config knob default
    assert c.public_url == ""


def test_config_parses_public_url():
    c = WakeIngressConfig.from_dict({
        "enabled": True, "public_url": "  https://bot.example.com/wake  "})
    assert c.public_url == "https://bot.example.com/wake"  # stripped


def test_config_whitespace_or_nonstr_public_url_is_empty():
    assert WakeIngressConfig.from_dict({"public_url": "   "}).public_url == ""
    assert WakeIngressConfig.from_dict({"public_url": 123}).public_url == ""


def test_config_bad_int_field_falls_back_not_raises():
    # a non-int port must not raise — falls back to the default
    c = WakeIngressConfig.from_dict({"enabled": True, "port": "not-a-number"})
    assert c.port == 9110  # default, no exception


def test_config_tls_incomplete_fails_closed():
    # tls.enabled but no certfile/keyfile → ingress DISABLED (no plaintext)
    c = WakeIngressConfig.from_dict({
        "enabled": True, "tls": {"enabled": True, "certfile": "/c.pem"}})
    assert c.enabled is False
    assert c.tls_certfile is None
