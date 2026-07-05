"""WakeBindingRegistry — owner-guarded wake_ref→binding store for the wake ingress.

Pins: create/resolve, validation (no-session/no-keys/too-many/malformed),
owner-guarded upsert (same-session rotation refresh vs different-session
UNAUTHORIZED), TTL expiry (resolve→None, expired ref re-bindable by a new
owner), unbind (owner/non-owner/unknown), and persistence round-trip.
"""

import pytest

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from server.wake_binding_registry import (
    BindOutcome, WakeBindingRegistry, _MAX_TRUST_KEYS,
)


def _pub_pem() -> str:
    pub = Ed25519PrivateKey.generate().public_key()
    return pub.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()


def _reg(tmp_path, **kw):
    return WakeBindingRegistry(path=tmp_path / "wb.json", **kw)


# ---- create / resolve ---------------------------------------------------------

def test_bind_and_resolve(tmp_path):
    r = _reg(tmp_path)
    k = _pub_pem()
    assert r.bind("github-pr:o/r#1", "sess1", "/ws/a", [k]) == BindOutcome.OK
    b = r.resolve("github-pr:o/r#1")
    assert b is not None
    assert b.session_id == "sess1" and b.workspace_path == "/ws/a"
    assert b.trust_keys == [k]


def test_resolve_unknown_is_none(tmp_path):
    assert _reg(tmp_path).resolve("nope") is None


# ---- validation ---------------------------------------------------------------

def test_bind_no_session(tmp_path):
    assert _reg(tmp_path).bind("w", "", "/ws", [_pub_pem()]) == BindOutcome.NO_SESSION


def test_bind_no_keys(tmp_path):
    assert _reg(tmp_path).bind("w", "s", "/ws", []) == BindOutcome.NO_KEYS


def test_bind_too_many_keys(tmp_path):
    keys = [_pub_pem() for _ in range(_MAX_TRUST_KEYS + 1)]
    assert _reg(tmp_path).bind("w", "s", "/ws", keys) == BindOutcome.TOO_MANY_KEYS


def test_bind_malformed_key(tmp_path):
    assert _reg(tmp_path).bind("w", "s", "/ws", ["not a key"]) == BindOutcome.MALFORMED_KEY


# ---- owner-guarded upsert -----------------------------------------------------

def test_same_session_refreshes_keys(tmp_path):
    r = _reg(tmp_path)
    old, new = _pub_pem(), _pub_pem()
    r.bind("w", "s", "/ws", [old])
    # rotation: same owner re-binds with [old, new]
    assert r.bind("w", "s", "/ws", [old, new]) == BindOutcome.OK
    assert r.resolve("w").trust_keys == [old, new]
    # then drop old
    assert r.bind("w", "s", "/ws", [new]) == BindOutcome.OK
    assert r.resolve("w").trust_keys == [new]


def test_different_session_cannot_overwrite(tmp_path):
    r = _reg(tmp_path)
    r.bind("w", "owner", "/ws/owner", [_pub_pem()])
    # a squatter / other session cannot rebind a live ref
    assert r.bind("w", "attacker", "/ws/atk", [_pub_pem()]) == BindOutcome.UNAUTHORIZED
    # ownership + workspace unchanged
    assert r.resolve("w").session_id == "owner"


# ---- TTL expiry ---------------------------------------------------------------

def test_expired_binding_resolves_none(tmp_path):
    r = _reg(tmp_path)
    r.bind("w", "s", "/ws", [_pub_pem()], ttl_seconds=-1)  # already expired
    assert r.resolve("w") is None


def test_expired_ref_rebindable_by_new_owner(tmp_path):
    r = _reg(tmp_path)
    r.bind("w", "owner", "/ws/o", [_pub_pem()], ttl_seconds=-1)  # expired
    # since expired == absent, a different session may claim the ref
    assert r.bind("w", "newowner", "/ws/n", [_pub_pem()]) == BindOutcome.OK
    assert r.resolve("w").session_id == "newowner"


# ---- unbind -------------------------------------------------------------------

def test_unbind_owner_ok(tmp_path):
    r = _reg(tmp_path)
    r.bind("w", "s", "/ws", [_pub_pem()])
    assert r.unbind("w", "s") == BindOutcome.OK
    assert r.resolve("w") is None


def test_unbind_non_owner_rejected(tmp_path):
    r = _reg(tmp_path)
    r.bind("w", "owner", "/ws", [_pub_pem()])
    assert r.unbind("w", "attacker") == BindOutcome.UNAUTHORIZED
    assert r.resolve("w") is not None  # untouched


def test_unbind_unknown(tmp_path):
    assert _reg(tmp_path).unbind("nope", "s") == BindOutcome.UNKNOWN


# ---- persistence --------------------------------------------------------------

def test_persistence_round_trip(tmp_path):
    k = _pub_pem()
    r1 = _reg(tmp_path)
    r1.bind("w", "s", "/ws", [k])
    r2 = WakeBindingRegistry(path=tmp_path / "wb.json")
    b = r2.resolve("w")
    assert b is not None and b.session_id == "s" and b.trust_keys == [k]


def test_corrupt_file_starts_empty(tmp_path):
    p = tmp_path / "wb.json"
    p.write_text("{ not json", encoding="utf-8")
    r = WakeBindingRegistry(path=p)  # must not raise
    assert r.resolve("w") is None
    assert r.bind("w", "s", "/ws", [_pub_pem()]) == BindOutcome.OK
