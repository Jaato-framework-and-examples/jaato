"""wake_verify — trust-key parsing/validation + signature verify (mode B)."""

import base64

import pytest

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from server.wake_verify import (
    InvalidTrustKey, load_trust_key, validate_trust_keys, verify_wake_signature,
)


def _pub_pem_of(priv) -> str:
    return priv.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()


def _ed_sign(priv, msg: bytes) -> str:
    return base64.b64encode(priv.sign(msg)).decode()


def _rsa_sign(priv, msg: bytes) -> str:
    sig = priv.sign(
        msg,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH),
        hashes.SHA256())
    return base64.b64encode(sig).decode()


def _ed25519_pub_pem() -> str:
    pub = Ed25519PrivateKey.generate().public_key()
    return pub.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()


def _rsa_pub_pem() -> str:
    pub = rsa.generate_private_key(public_exponent=65537, key_size=2048).public_key()
    return pub.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()


def test_loads_ed25519():
    assert load_trust_key(_ed25519_pub_pem()) is not None


def test_loads_rsa():
    assert load_trust_key(_rsa_pub_pem()) is not None


@pytest.mark.parametrize("bad", [
    "", "   ", "not a key", "-----BEGIN PUBLIC KEY-----\ngarbage\n-----END PUBLIC KEY-----",
])
def test_rejects_malformed(bad):
    with pytest.raises(InvalidTrustKey):
        load_trust_key(bad)


def test_rejects_non_str():
    with pytest.raises(InvalidTrustKey):
        load_trust_key(None)  # type: ignore[arg-type]


def test_validate_all_valid_passes():
    validate_trust_keys([_ed25519_pub_pem(), _rsa_pub_pem()])  # no raise


def test_validate_one_bad_raises():
    with pytest.raises(InvalidTrustKey):
        validate_trust_keys([_ed25519_pub_pem(), "not a key"])


# ---- signature verify (Stage B) ----------------------------------------------

def test_ed25519_signature_verifies():
    priv = Ed25519PrivateKey.generate()
    body = b'{"wake_ref":"w","text":"hi","ts":1}'
    assert verify_wake_signature([_pub_pem_of(priv)], body, _ed_sign(priv, body))


def test_rsa_signature_verifies():
    priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    body = b'{"wake_ref":"w"}'
    assert verify_wake_signature([_pub_pem_of(priv)], body, _rsa_sign(priv, body))


def test_wrong_key_fails():
    signer = Ed25519PrivateKey.generate()
    other = Ed25519PrivateKey.generate()
    body = b"payload"
    assert not verify_wake_signature([_pub_pem_of(other)], body, _ed_sign(signer, body))


def test_tampered_body_fails():
    priv = Ed25519PrivateKey.generate()
    sig = _ed_sign(priv, b"original")
    assert not verify_wake_signature([_pub_pem_of(priv)], b"tampered", sig)


def test_or_verify_matches_any_key_rotation():
    old = Ed25519PrivateKey.generate()
    new = Ed25519PrivateKey.generate()
    body = b"rotating"
    # signed with NEW; binding carries [old_pub, new_pub] during overlap
    sig = _ed_sign(new, body)
    assert verify_wake_signature([_pub_pem_of(old), _pub_pem_of(new)], body, sig)


def test_one_malformed_key_does_not_veto_good_one():
    priv = Ed25519PrivateKey.generate()
    body = b"x"
    keys = ["not a key", _pub_pem_of(priv)]
    assert verify_wake_signature(keys, body, _ed_sign(priv, body))


def test_bad_base64_signature_fails():
    priv = Ed25519PrivateKey.generate()
    assert not verify_wake_signature([_pub_pem_of(priv)], b"x", "!!!not-base64!!!")


def test_empty_inputs_fail():
    priv = Ed25519PrivateKey.generate()
    assert not verify_wake_signature([], b"x", _ed_sign(priv, b"x"))
    assert not verify_wake_signature([_pub_pem_of(priv)], b"x", "")
