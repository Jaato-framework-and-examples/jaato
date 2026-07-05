"""wake_verify — trust-key parsing/validation for the mode-B wake ingress."""

import pytest

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.asymmetric import rsa

from server.wake_verify import InvalidTrustKey, load_trust_key, validate_trust_keys


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
