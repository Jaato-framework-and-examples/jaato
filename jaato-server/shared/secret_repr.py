"""Secret-safe ``repr`` for credential-bearing dataclasses.

Credential objects are ordinary dataclasses, so Python hands them a
generated ``__repr__`` that prints every field verbatim.  That is fine
until something *else* prints the object — and plenty does:

- a failing ``assert creds is None`` in pytest, which renders the
  left-hand operand into the failure message (issue #721: a live
  ``sk-or-v1-…`` key was printed into terminal scrollback and would
  have reached any CI log capturing pytest output);
- ``logger.debug("loaded %s", creds)``;
- an unhandled exception whose traceback carries a local.

None of those callers know they are handling a secret.  The object
does, so the object is where the redaction belongs — a repr that never
carries the secret cannot be made to leak it by a caller downstream.

Usage::

    @dataclass
    class OpenRouterCredentials:
        api_key: str
        created_at: float

        __repr__ = secret_safe_repr("api_key")

The returned function is a ``__repr__`` implementation; the names
passed to :func:`secret_safe_repr` are the fields to redact.  Every
other field is rendered normally, so the repr stays useful for
debugging (``created_at``, ``base_url``, expiry timestamps, the
account e-mail).

Empty and ``None`` values are rendered as-is: "the key is missing" is
not a secret, and hiding the difference between an absent credential
and a present one is what would send someone hunting.

Serialisation is deliberately untouched — ``to_dict()`` must keep
returning the real secret, because that is what gets written to the
0600 credential file.  This module guards the *display* path only.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable

#: Rendered in place of a non-empty secret.  Carries no prefix, no
#: suffix and no length — a key fingerprint is still key material, and
#: a length narrows a brute-force search.
REDACTED = "<redacted>"

#: Field / mapping-key names that hold a secret.  Two consumers share
#: it, which is the point: :func:`redact_mapping` uses it to decide
#: which entries of a free-form dict to hide, and
#: ``shared/tests/test_credential_disclosure.py`` uses it to find every
#: dataclass in the tree that carries one and assert the value cannot
#: be printed.  A new provider whose credential field is named from
#: this set is covered the day it lands; one that invents a new name
#: adds it here.
SECRET_FIELD_NAMES = frozenset({
    "access_token",
    "api_key",
    "api_token",
    "auth_token",
    "client_secret",
    "credentials",
    "device_code",
    "oauth_token",
    "password",
    "private_key",
    "refresh_token",
    "secret",
    "secret_key",
    "token",
    "ws_token",
})


def redact(value: Any) -> Any:
    """Return ``value`` if it carries no secret, else :data:`REDACTED`.

    ``None`` and empty strings pass through unchanged: they say
    "nothing is stored here", which is a fact worth seeing and not a
    disclosure.
    """
    if value is None or value == "":
        return value
    return REDACTED


def redact_mapping(mapping: Any) -> Any:
    """Return a copy of ``mapping`` with secret-named entries redacted.

    For free-form config dicts — ``ProviderConfig.extra`` carries a
    provider's whole ``plugin_configs`` block, so ``host``,
    ``context_length`` and ``routing`` sit in the same dict as
    ``api_token``.  Redacting the field wholesale would cost the
    debuggability the repr exists for, so only the keys named in
    :data:`SECRET_FIELD_NAMES` are hidden.

    Non-mapping values are returned unchanged, so a field that is
    sometimes a dict and sometimes ``None`` needs no special case.
    """
    if not isinstance(mapping, dict):
        return mapping
    return {
        key: (redact(value) if key in SECRET_FIELD_NAMES else value)
        for key, value in mapping.items()
    }


def secret_safe_repr(
    *secret_fields: str,
    mappings: tuple = (),
) -> Callable[[Any], str]:
    """Build a ``__repr__`` that redacts the named dataclass fields.

    Args:
        *secret_fields: Names of the fields holding secrets.  A name
            that is not a field of the instance is ignored, so a field
            can be renamed or dropped without breaking the repr — the
            guard in ``shared/tests/test_credential_disclosure.py`` is
            what catches a secret field that stops being covered.
        mappings: Names of dict-valued fields whose secret-named *keys*
            should be redacted while the rest of the dict is rendered
            (see :func:`redact_mapping`).

    Returns:
        A function suitable for assignment to ``__repr__`` on a
        dataclass.  Non-dataclass instances fall back to the default
        ``object`` repr rather than raising, so a subclass that drops
        ``@dataclass`` degrades to something safe rather than to a
        crash inside an error path.
    """
    redacted = frozenset(secret_fields)
    scrubbed_mappings = frozenset(mappings)

    def __repr__(self: Any) -> str:
        if not dataclasses.is_dataclass(self):
            return object.__repr__(self)
        parts = []
        for field in dataclasses.fields(self):
            if not field.repr:
                continue
            value = getattr(self, field.name, None)
            if field.name in redacted:
                value = redact(value)
            elif field.name in scrubbed_mappings:
                value = redact_mapping(value)
            parts.append(f"{field.name}={value!r}")
        return f"{type(self).__name__}({', '.join(parts)})"

    __repr__.__doc__ = (
        "Return a repr with "
        + ", ".join(sorted(redacted | scrubbed_mappings))
        + " redacted, so printing this object cannot disclose a "
        "credential (#721)."
    )
    return __repr__
