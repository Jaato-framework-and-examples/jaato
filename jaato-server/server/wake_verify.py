"""Crypto for the mode-B wake ingress.

Mode B (Daniel 2026-07-05): a wake relay signs the forwarded payload with a
store PRIVATE key; the daemon verifies the signature against the target
session's declared PUBLIC ``trust_keys``.  Trust is SESSION-scoped — the keys
live on the session's :class:`WakeBindingRegistry` entry, never a daemon-global
config — so a session can only make ITSELF wakeable (zero cross-session blast
radius; see ``project_session_wake_primitive``).

This module is the single crypto home for that ingress:

- :func:`load_trust_key` / :func:`validate_trust_keys` — parse/validate the
  PEM public keys at ``bind_wake`` time (fail-fast: a malformed key is rejected
  when it is declared, not when a wake later fails).
- Signature verification (Stage B) is added here once the canonical-signing
  spec — the exact bytes the relay signs — is pinned jointly with the relay
  author.  Kept out of this PR so the relay and the verifier commit to the
  same canonical bytes before either is built.

Uses ``cryptography`` (already present in every jaato-server install via
``google-auth`` / ``pyspnego`` — no new install dependency).  Ed25519 is the
recommended key type; RSA / EC public keys also load, and the (future) verify
step dispatches on the concrete key type.
"""
from __future__ import annotations

import base64
import logging
from typing import List

from cryptography.exceptions import InvalidSignature, UnsupportedAlgorithm
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicKey
from cryptography.hazmat.primitives.serialization import load_pem_public_key

logger = logging.getLogger(__name__)


class InvalidTrustKey(ValueError):
    """A declared ``trust_key`` is not a well-formed PEM public key."""


def load_trust_key(pem: str):
    """Parse a PEM ``SubjectPublicKeyInfo`` public key.

    Returns the loaded public-key object (its concrete type — Ed25519 / RSA /
    EC — is what the verify step will dispatch on).  Raises
    :class:`InvalidTrustKey` on an empty, non-string, or unparseable value.
    """
    if not isinstance(pem, str) or not pem.strip():
        raise InvalidTrustKey("empty or non-string trust_key")
    try:
        return load_pem_public_key(pem.encode("utf-8"))
    except (ValueError, TypeError, UnsupportedAlgorithm) as exc:
        raise InvalidTrustKey(f"not a valid PEM public key: {exc}") from exc


def validate_trust_keys(keys: List[str]) -> None:
    """Raise :class:`InvalidTrustKey` unless EVERY entry parses as a PEM public
    key.  Called at ``bind_wake`` time so a malformed key is refused up front."""
    for k in keys:
        load_trust_key(k)


def verify_wake_signature(
    trust_keys: List[str], message: bytes, signature_b64: str,
) -> bool:
    """Stage B: OR-verify a wake signature over the RAW request body.

    Returns ``True`` iff ``signature_b64`` (base64) over ``message`` (the exact
    bytes the relay signed and POSTed) validates against ANY key in
    ``trust_keys``.  This is the mode-B authoritative gate — the caller resolves
    ``wake_ref`` → binding first, then passes ``binding.trust_keys`` here.

    Canonical spec (sealed 2026-07-05): the relay signs the RAW body bytes; the
    daemon verifies over the raw bytes it received (never a re-serialization).
    Ed25519 is the default; RSA public keys verify with **PSS / SHA-256** (the
    documented RSA scheme — a relay using RSA must sign with the same).  A
    malformed key or a key type we don't verify is skipped, not fatal, so one
    bad key in a rotation set can't veto a good one.
    """
    if not signature_b64 or not trust_keys:
        return False
    try:
        sig = base64.b64decode(signature_b64, validate=True)
    except (ValueError, TypeError):
        return False
    for pem in trust_keys:
        try:
            key = load_trust_key(pem)
        except InvalidTrustKey:
            continue
        try:
            if isinstance(key, Ed25519PublicKey):
                key.verify(sig, message)  # raises InvalidSignature on mismatch
                return True
            if isinstance(key, RSAPublicKey):
                key.verify(
                    sig, message,
                    padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                                salt_length=padding.PSS.MAX_LENGTH),
                    hashes.SHA256(),
                )
                return True
            # Other key types (EC, DSA, ...) are not part of the wake spec.
            logger.debug("wake verify: unsupported key type %s — skipping",
                         type(key).__name__)
        except InvalidSignature:
            continue
        except Exception:  # noqa: BLE001 — a broken key must not abort the OR-loop
            logger.debug("wake verify: error checking a trust key — skipping",
                         exc_info=True)
            continue
    return False
