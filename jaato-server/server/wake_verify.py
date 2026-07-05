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

from typing import List

from cryptography.exceptions import UnsupportedAlgorithm
from cryptography.hazmat.primitives.serialization import load_pem_public_key


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
