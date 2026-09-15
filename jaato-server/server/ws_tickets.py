"""Per-connection identity for the WebSocket transport (#1074).

A WS client used to be attributed to a user by a **message** — the
``auth.token`` frame a daemon extension validates against one
hard-configured OIDC realm — which makes two things true that need not
be:

1. one daemon serves one realm, so a second application with its own
   userbase cannot share it; and
2. **identity is opt-in**, so declining to present one is the
   permissive path: a client that completes the bearer handshake and
   never sends ``auth.token`` is attributed to nobody, and every
   ownership guard written as ``if user_id and ...`` short-circuits.

This module is the other route.  The application has *already*
authenticated the user in whatever realm it owns; it is the authority
on who they are.  So it mints a short-lived, single-use **ticket**
bound to that user, hands the ticket to that user's browser, and the
browser presents it exactly where the shared bearer token is presented
today — on the Upgrade request.  The daemon resolves it **at connection
establishment**, before the first frame, and stamps the identity on the
``ClientConnection``.  Declining to present an identity stops being
representable rather than being patched.

Two credential kinds, both presented the same way:

======================  ================  =============  ======================
kind                    held by           lifetime       authorises
======================  ================  =============  ======================
app credential          the application   long-lived,    calling ``ticket.bind``
                        backend           configured     / ``ticket.revoke``
user ticket             one user's        minted per     opening ONE attributed
                        browser           login, short   connection
======================  ================  =============  ======================

**No realm vocabulary reaches this module**, which is the point: there
is no JWKS fetch, no ``aud``/``iss`` validation, no per-tenant
``SSOAuth``.  Stdlib only, so it can be imported wherever the transport
can.

Qualification
-------------

``app_id`` comes from the credential that called ``bind`` — an
authenticated fact the caller cannot assert — and the identity the
daemon stamps is ``BoundIdentity.qualified``, ``f"{app_id}:{user}"``.
Without that, two applications each holding an ``alice`` would collide
in ``Session.created_by`` and the ownership guards would silently pass
*across* the application boundary.  Qualifying it in the daemon means
no integrator can forget to, which is why ``app_id`` is not a field of
the bind request.

``app_id`` is therefore refused at load if it contains ``:``.  The
qualified form is a concatenation, so an app id spelling ``a:b`` with
user ``c`` and an app id ``a`` with user ``b:c`` would produce one
string for two identities — an ambiguity in the value the ownership
guards compare.

What is stored
--------------

Never a credential.  Both stores are keyed by ``sha256(credential)``:
``bind`` returns the plaintext ticket to its caller and keeps only the
digest, and ``load_app_credentials`` discards the plaintext after
hashing.  So a heap dump, a ``repr`` or a traceback renders no usable
credential, and the lookup on the connect path is a dict lookup keyed
by a digest rather than a comparison loop over secrets.

A dict lookup is not constant-time, and that is deliberate rather than
overlooked: what is compared is a SHA-256 *digest*, and a timing signal
about a digest is not a timing signal about the credential that
produced it (which would need a preimage).  Where a single expected
value IS compared directly — the daemon-wide shared token, in
:mod:`server.websocket` — the comparison stays
:func:`hmac.compare_digest`.

Lifetime
--------

Bindings live **in memory**, in one registry owned by the WS server:

* a daemon restart invalidates every outstanding ticket (users
  re-login, which is the same thing a restart already does to every
  connection); and
* a ticket bound on one node does not resolve on another.

Both are consequences of the storage choice, not of the protocol —
:class:`TicketRegistry` is the whole persistence surface, so a shared
store is a substitution here and nowhere else.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import stat
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional, Tuple

__all__ = [
    "AppCredentialStore",
    "AppCredentialsError",
    "BoundIdentity",
    "TicketCapacityError",
    "TicketRegistry",
    "credential_digest",
    "load_app_credentials",
    "DEFAULT_TICKET_TTL_SECONDS",
    "MAX_TICKET_TTL_SECONDS",
    "MAX_OUTSTANDING_TICKETS",
    "MIN_APP_CREDENTIAL_CHARS",
]


#: Default ticket lifetime.  Long enough for a page load and a
#: WebSocket handshake, short enough that a ticket captured from a URL
#: (browsers cannot set headers on ``new WebSocket()``, so ``?token=``
#: is the browser form and lands in ``document.location``) is stale
#: before it is useful.
DEFAULT_TICKET_TTL_SECONDS = 300

#: Ceiling on a requested TTL.  A ticket is a *connect* credential, not
#: a session credential — the connection it opens outlives it — so an
#: hour is already generous.  A larger request is refused rather than
#: clamped: silently issuing something other than what was asked for is
#: how an integrator comes to believe a ticket lasts a day.
MAX_TICKET_TTL_SECONDS = 3600

#: Ceiling on outstanding (unconsumed, unexpired) tickets.  ``bind`` is
#: remote-triggerable by an authenticated application, so an unbounded
#: registry is a memory sink reachable by one misbehaving integrator.
#: Expired entries are swept before the ceiling is consulted, so this
#: only binds genuinely-live tickets.
MAX_OUTSTANDING_TICKETS = 4096

#: Refuse an app credential shorter than this.  It is long-lived and
#: sits in a config file, so it is the one credential here that a
#: guessing attacker has time to work on.  32 URL-safe bytes
#: (``secrets.token_urlsafe(32)``, what ``--ws-token-file`` generates)
#: is 43 characters, comfortably above.
MIN_APP_CREDENTIAL_CHARS = 16

#: Longest ``user`` a bind may assert.  The value becomes
#: ``Session.created_by`` and is compared by the ownership guards; an
#: unbounded string there is a storage question the transport should
#: not be answering.
MAX_USER_CHARS = 256

#: An app id is an operator-chosen label, and it is half of the
#: qualified identity.  ``:`` is excluded because the qualified form
#: concatenates on it (see the module docstring); control characters
#: and whitespace because the value is logged.
_APP_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class AppCredentialsError(Exception):
    """An app-credentials file could not be loaded.

    Raised by :func:`load_app_credentials` for a missing file, a file
    other principals can read, malformed JSON, or an entry that fails
    validation.  Never raised for a *correct* file that happens to be
    empty of a particular app — that is a lookup miss, not an error.

    The caller (``server/__main__.py``) turns this into an exit, not a
    degraded start: a credentials file that cannot be read is a
    security configuration that did not take effect, and starting
    anyway would serve the pre-#1074 posture while the operator
    believes otherwise.
    """


class TicketCapacityError(RuntimeError):
    """The registry is at :data:`MAX_OUTSTANDING_TICKETS` live tickets.

    Refusing the bind is the fail-closed answer: the alternative is
    evicting somebody else's valid ticket, which turns one
    misbehaving application into failed logins for another.
    """


def credential_digest(credential: str) -> bytes:
    """Return ``sha256(credential)`` as raw bytes.

    The one hashing site for both credential kinds, so the ticket
    registry, the app-credential store and the WS server's connect
    path cannot come to disagree about the encoding.  UTF-8, matching
    :meth:`server.websocket.JaatoWSServer._check_ws_token`, which has
    hashed the shared bearer token this way since the flag existed.
    """
    return hashlib.sha256(credential.encode("utf-8")).digest()


@dataclass(frozen=True)
class BoundIdentity:
    """Who a resolved ticket says the connection belongs to.

    Immutable, and constructed only by :meth:`TicketRegistry.bind` —
    which takes ``app_id`` from the credential that authenticated the
    bind channel rather than from the request body, so neither field is
    ever a caller's assertion about itself.

    Attributes:
        app_id: The application that bound the ticket.  An authenticated
            fact: it names which entry of the app-credentials file
            presented itself on the bind connection.
        user: The identity that application asserts, verbatim, in
            whatever spelling its own realm uses (a ``preferred_username``,
            an email, an internal id).  The daemon never validates it —
            that is the whole point — and never renders it alone.
    """

    app_id: str
    user: str

    @property
    def qualified(self) -> str:
        """``f"{app_id}:{user}"`` — the form the daemon attributes to.

        This, not :attr:`user`, is what reaches
        ``ClientConnection.user_id`` and therefore ``Session.created_by``:
        ``preferred_username`` is unique only within a realm, so two
        applications each holding an ``alice`` must not compare equal.
        """
        return f"{self.app_id}:{self.user}"


class AppCredentialStore:
    """Digest → ``app_id`` for the configured application credentials.

    Built once at daemon start by :func:`load_app_credentials` and read
    on every connection that did not present the shared token.  Holds
    no plaintext: the credentials are hashed at load and discarded, so
    the object is safe to render.

    An **empty** store is meaningful and distinct from ``None``: it is
    the shape every deployment that configured no app credentials has,
    and :meth:`__bool__` answers ``False`` so the WS server's auth
    resolution collapses to exactly its pre-#1074 form.  (In practice
    :func:`load_app_credentials` refuses to *produce* an empty store
    from a file — a file with no entries is a flag that did nothing —
    so an empty store is only ever the unconfigured default.)
    """

    def __init__(self, credentials: Mapping[str, str]) -> None:
        """Hash ``credentials`` (``app_id`` → plaintext) into the store.

        Validation lives in :func:`load_app_credentials` rather than
        here, so that a file's problems are reported with the file's
        name and line-free clarity.  This constructor is the in-memory
        path used by tests and by embedders that hold their credentials
        somewhere other than a file.

        Raises:
            ValueError: on a duplicate credential across two app ids.
                Two applications sharing one credential means the
                resolved ``app_id`` depends on dict order, and the
                qualified identity with it.
        """
        by_digest: Dict[bytes, str] = {}
        for app_id, credential in credentials.items():
            digest = credential_digest(credential)
            if digest in by_digest and by_digest[digest] != app_id:
                raise ValueError(
                    "two applications share one credential "
                    f"({by_digest[digest]!r} and {app_id!r}); identity "
                    "would depend on iteration order"
                )
            by_digest[digest] = app_id
        self._by_digest = by_digest

    def lookup(self, digest: bytes) -> Optional[str]:
        """Return the ``app_id`` for ``digest``, or ``None``.

        ``digest`` is the raw ``sha256`` of the presented credential
        (:func:`credential_digest`).  See the module docstring for why a
        dict lookup on a digest is the right primitive here and
        :func:`hmac.compare_digest` is still the right one for the
        single shared token.
        """
        return self._by_digest.get(digest)

    def app_ids(self) -> Tuple[str, ...]:
        """Every configured ``app_id``, sorted.  For startup logging."""
        return tuple(sorted(set(self._by_digest.values())))

    def __len__(self) -> int:
        return len(self._by_digest)

    def __bool__(self) -> bool:
        return bool(self._by_digest)

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"AppCredentialStore(apps={list(self.app_ids())!r})"


@dataclass(frozen=True)
class _Ticket:
    """One outstanding binding.  Internal; never leaves the registry.

    Attributes:
        identity: What :meth:`TicketRegistry.resolve` returns.
        deadline: A *monotonic* instant, not a wall clock one.  TTL
            enforcement must survive an NTP step or a suspended host:
            a clock jumped backwards would otherwise extend every
            outstanding ticket, and jumped forwards would expire them
            mid-login.
        expires_at: The same deadline as an ISO-8601 UTC string, kept
            only to report to the binder.  A monotonic float is
            meaningless in another process, and the binder's whole use
            for it is deciding when to mint the next one.
        single_use: Whether :meth:`TicketRegistry.resolve` consumes it.
    """

    identity: BoundIdentity
    deadline: float
    expires_at: str
    single_use: bool


class TicketRegistry:
    """In-memory, digest-keyed store of outstanding user tickets.

    Lifecycle of one ticket:

    1. **bound** — :meth:`bind` mints 32 random URL-safe bytes, stores
       ``sha256`` of them against a :class:`BoundIdentity`, and returns
       the plaintext to the application's backend.  The plaintext
       exists in this process only for the duration of that call.
    2. **live** — resolvable until its deadline.
    3. **resolved** — :meth:`resolve` returns the identity.  A
       single-use ticket (the default) is deleted in the same call, so
       a captured ticket cannot open a second connection.  The
       connection it opened is unaffected: identity was copied onto the
       ``ClientConnection`` at accept time and is not re-derived.
    4. **gone** — consumed, expired, or revoked.  All three are the
       same answer to a later :meth:`resolve`: ``None``.  They are
       deliberately indistinguishable, so the verb is not an oracle for
       which tickets ever existed.

    Thread safety: every public method takes one lock.  The WS server
    drives this from its event loop, but ``bind`` is also reachable
    from a daemon extension's own thread, and the registry is one
    shared mutable object — the argument ``PluginRegistry`` makes in
    #938, answered here with a lock because no call under it reaches
    foreign code.
    """

    def __init__(
        self,
        *,
        max_outstanding: int = MAX_OUTSTANDING_TICKETS,
        clock: Optional[Callable[[], float]] = None,
        wall_clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        """Construct an empty registry.

        Args:
            max_outstanding: Ceiling on live tickets; see
                :data:`MAX_OUTSTANDING_TICKETS`.
            clock: Monotonic time source, for tests that need to state
                the instant they mean rather than sleep for it (#996).
                Defaults to :func:`time.monotonic`.
            wall_clock: Source of the reported ``expires_at``.  Split
                from ``clock`` because the two answer different
                questions — enforcement must not follow a clock step,
                reporting must be readable in another process.
        """
        self._lock = threading.Lock()
        self._tickets: Dict[bytes, _Ticket] = {}
        self._max_outstanding = max_outstanding
        self._clock = clock or time.monotonic
        self._wall_clock = wall_clock or (lambda: datetime.now(timezone.utc))

    # -- minting -----------------------------------------------------

    def bind(
        self,
        *,
        app_id: str,
        user: str,
        ttl_seconds: int = DEFAULT_TICKET_TTL_SECONDS,
        single_use: bool = True,
    ) -> Tuple[str, str]:
        """Mint a ticket for ``user`` on behalf of ``app_id``.

        Args:
            app_id: The binding application.  **Never taken from the
                request body** — the caller passes what the bind
                connection's own credential resolved to.
            user: The identity that application asserts.
            ttl_seconds: Lifetime, ``1..``:data:`MAX_TICKET_TTL_SECONDS`.
            single_use: When ``True`` (the default) the first
                :meth:`resolve` consumes it.

        Returns:
            ``(ticket, expires_at)`` — the plaintext credential to hand
            to that user's client, and an ISO-8601 UTC instant for the
            binder's own scheduling.  The plaintext is not retained.

        Raises:
            ValueError: ``app_id`` or ``user`` is unusable, or
                ``ttl_seconds`` is outside the permitted range.  A
                refused bind mints nothing.
            TicketCapacityError: the registry is full of live tickets.
        """
        if not _APP_ID_RE.match(app_id or ""):
            raise ValueError(f"invalid app_id: {app_id!r}")
        _validate_user(user)
        if not isinstance(ttl_seconds, int) or isinstance(ttl_seconds, bool):
            raise ValueError(f"ttl_seconds must be an int, got {ttl_seconds!r}")
        if not 1 <= ttl_seconds <= MAX_TICKET_TTL_SECONDS:
            raise ValueError(
                f"ttl_seconds must be 1..{MAX_TICKET_TTL_SECONDS}, "
                f"got {ttl_seconds}"
            )

        ticket = secrets.token_urlsafe(32)
        digest = credential_digest(ticket)
        expires_at = (
            self._wall_clock() + timedelta(seconds=ttl_seconds)
        ).isoformat()
        entry = _Ticket(
            identity=BoundIdentity(app_id=app_id, user=user),
            deadline=self._clock() + ttl_seconds,
            expires_at=expires_at,
            single_use=bool(single_use),
        )
        with self._lock:
            self._purge_expired_locked()
            if len(self._tickets) >= self._max_outstanding:
                raise TicketCapacityError(
                    f"{len(self._tickets)} outstanding tickets at the "
                    f"ceiling of {self._max_outstanding}"
                )
            self._tickets[digest] = entry
        return ticket, expires_at

    # -- resolving ---------------------------------------------------

    def resolve(
        self, credential: str, *, consume: bool = True
    ) -> Optional[BoundIdentity]:
        """Resolve a presented ticket to the identity it was bound to.

        Args:
            credential: The plaintext ticket as presented on the
                Upgrade request.
            consume: When ``False``, a single-use ticket is *peeked* at
                rather than spent.  Exists for the non-destructive
                predicate (``_check_ws_token``): a bool-returning
                helper that silently consumed a credential would be a
                trap for every later caller.

        Returns:
            The :class:`BoundIdentity`, or ``None`` when the ticket is
            unknown, expired, consumed or revoked — one answer for all
            four, so this is not an existence oracle.
        """
        if not credential:
            return None
        return self.resolve_digest(credential_digest(credential), consume=consume)

    def resolve_digest(
        self, digest: bytes, *, consume: bool = True
    ) -> Optional[BoundIdentity]:
        """:meth:`resolve` for a caller that already hashed the credential.

        The WS connect path hashes once and asks each tier in turn, so
        a presented credential is hashed exactly once however many
        tiers it falls through.
        """
        with self._lock:
            entry = self._tickets.get(digest)
            if entry is None:
                return None
            if entry.deadline <= self._clock():
                # Expired: drop it here rather than waiting for a sweep,
                # so a ticket cannot be resolved by a caller that races
                # the purge.
                self._tickets.pop(digest, None)
                return None
            if consume and entry.single_use:
                self._tickets.pop(digest, None)
            return entry.identity

    # -- revoking ----------------------------------------------------

    def revoke(self, credential: str, *, app_id: Optional[str] = None) -> bool:
        """Revoke one outstanding ticket.

        Args:
            credential: The plaintext ticket, as returned by
                :meth:`bind`.
            app_id: When given, the ticket is revoked only if it was
                bound by that application.  A ticket belonging to
                *another* application answers ``False`` — the same
                answer an unknown ticket gives, so a caller cannot use
                this verb to discover that another application's ticket
                exists.

        Returns:
            Whether a ticket was removed.
        """
        if not credential:
            return False
        digest = credential_digest(credential)
        with self._lock:
            entry = self._tickets.get(digest)
            if entry is None:
                return False
            if app_id is not None and entry.identity.app_id != app_id:
                return False
            del self._tickets[digest]
            return True

    def revoke_user(self, app_id: str, user: str) -> int:
        """Revoke every outstanding ticket for one user of one app.

        The logout path: the application's backend ends a session in
        its own realm and tells the daemon that any ticket it minted
        for that user is void.  Scoped by ``app_id`` for the same
        reason :meth:`revoke` is — an application must not be able to
        log out another application's ``alice``.

        Returns:
            How many tickets were removed; ``0`` when the user had none
            outstanding, which is the ordinary case for a logout that
            follows a completed login (the ticket was consumed at
            connect).
        """
        with self._lock:
            doomed = [
                digest
                for digest, entry in self._tickets.items()
                if entry.identity.app_id == app_id
                and entry.identity.user == user
            ]
            for digest in doomed:
                del self._tickets[digest]
            return len(doomed)

    # -- housekeeping ------------------------------------------------

    def purge_expired(self) -> int:
        """Drop every expired entry; return how many.  Idempotent."""
        with self._lock:
            return self._purge_expired_locked()

    def _purge_expired_locked(self) -> int:
        """:meth:`purge_expired` for a caller already holding the lock.

        The lock is not reentrant (``threading.Lock``), so the two are
        separate rather than one delegating to the other — the #683
        rule that call sites under a lock stay flat.
        """
        now = self._clock()
        doomed = [
            digest
            for digest, entry in self._tickets.items()
            if entry.deadline <= now
        ]
        for digest in doomed:
            del self._tickets[digest]
        return len(doomed)

    def __len__(self) -> int:
        """Outstanding entries, expired ones included until swept."""
        with self._lock:
            return len(self._tickets)

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"TicketRegistry(outstanding={len(self)})"


def _validate_user(user: str) -> None:
    """Refuse a ``user`` the daemon should not attribute a session to.

    Three refusals, each for a reason that is not style:

    * empty — ``created_by=""`` is falsy, so every ownership guard
      written ``if user_id and ...`` would short-circuit exactly as it
      does for an unauthenticated client.  That is the fail-open this
      whole mechanism exists to close, arriving through the front door.
    * over :data:`MAX_USER_CHARS`.
    * carrying a control character — the value is logged and persisted
      as ``Session.created_by``; a newline in it forges a log line.
    """
    if not isinstance(user, str) or not user:
        raise ValueError("user must be a non-empty string")
    if len(user) > MAX_USER_CHARS:
        raise ValueError(f"user exceeds {MAX_USER_CHARS} characters")
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in user):
        raise ValueError("user contains control characters")


def load_app_credentials(path: "str | os.PathLike[str]") -> AppCredentialStore:
    """Read the app-credentials file at ``path``.

    Format — a JSON object mapping ``app_id`` to that application's
    credential::

        {
          "acme-portal": "nQ3...43-chars...",
          "internal-ops": "Zk8...43-chars..."
        }

    One shape, deliberately: a richer per-application form (a
    ``workspace_root``, a ``config_root``) is an additive change to
    this loader and to nothing else, and picking its spelling now would
    be guessing at a requirement that has not arrived.  This function
    is the whole format surface.

    Mode is enforced exactly as ``--ws-token-file`` enforces it: a file
    any other principal can read is refused rather than read, because
    an app credential is a credential for *every* identity that
    application can assert.

    Raises:
        AppCredentialsError: missing, unreadable, group/other
            accessible, malformed, or carrying an entry that fails
            validation.  Every failure is fail-closed: the function
            returns a store or it raises, never a partial one.  A file
            with no entries raises too — a configured flag that
            authorises nobody is a mistake, not a posture.
    """
    file_path = Path(path).expanduser()
    _refuse_loose_mode(file_path)
    try:
        raw = file_path.read_text()
    except OSError as exc:
        raise AppCredentialsError(
            f"cannot read app credentials file {file_path}: {exc}"
        ) from exc
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AppCredentialsError(
            f"app credentials file {file_path} is not valid JSON: {exc}"
        ) from exc
    if not isinstance(parsed, dict):
        raise AppCredentialsError(
            f"app credentials file {file_path} must be a JSON object "
            'mapping app_id to credential, e.g. {"acme": "<token>"}'
        )
    credentials = _validated_entries(parsed, file_path)
    if not credentials:
        raise AppCredentialsError(
            f"app credentials file {file_path} declares no applications; "
            "remove the flag or add an entry"
        )
    try:
        return AppCredentialStore(credentials)
    except ValueError as exc:
        raise AppCredentialsError(f"{file_path}: {exc}") from exc


def _validated_entries(
    parsed: Mapping[str, object], file_path: Path
) -> Dict[str, str]:
    """Return ``{app_id: credential}`` or raise ``AppCredentialsError``.

    Split out of :func:`load_app_credentials` so the per-entry rules
    read as one list rather than as nesting inside the file handling.
    """
    credentials: Dict[str, str] = {}
    for app_id, credential in parsed.items():
        if not isinstance(app_id, str) or not _APP_ID_RE.match(app_id):
            raise AppCredentialsError(
                f"{file_path}: invalid app id {app_id!r} — must match "
                f"{_APP_ID_RE.pattern} (a ':' would make the qualified "
                "identity ambiguous)"
            )
        if not isinstance(credential, str):
            raise AppCredentialsError(
                f"{file_path}: credential for {app_id!r} must be a string"
            )
        if len(credential) < MIN_APP_CREDENTIAL_CHARS:
            raise AppCredentialsError(
                f"{file_path}: credential for {app_id!r} is shorter than "
                f"{MIN_APP_CREDENTIAL_CHARS} characters"
            )
        credentials[app_id] = credential
    return credentials


def _refuse_loose_mode(file_path: Path) -> None:
    """Refuse a credentials file other principals can read.

    The same check ``server/__main__.py:_load_token_file`` applies to
    ``--ws-token-file``, and the same one ``ssh`` applies to private
    keys.  Skipped on Windows, where the POSIX mode bits do not carry
    the meaning being tested.
    """
    try:
        mode = file_path.stat().st_mode
    except OSError as exc:
        raise AppCredentialsError(
            f"cannot read app credentials file {file_path}: {exc}"
        ) from exc
    if sys.platform != "win32" and mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise AppCredentialsError(
            f"app credentials file {file_path} is group/other accessible "
            f"(mode {oct(mode & 0o777)}); restrict to 0600"
        )
