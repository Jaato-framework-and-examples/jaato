"""One lock and one re-read, for credentials that rotate when refreshed.

WHAT THIS IS FOR.  An OAuth refresh token **rotates**: the token that
buys a new access token is itself replaced by the response, and the old
one is void the moment the provider answers.  So the sequence

    load -> is it stale? -> refresh -> save

is a read-modify-write on a shared file, and running two of them at once
loses one of the writes.  The loser does not fail at the time; it fails
at its *next* refresh, holding a superseded token, and the user is
logged out with nothing in the failure naming the cause.

WHY THIS TREE IS MORE EXPOSED THAN A ONE-PROCESS CLI.  Claude Code is one
process per session and still needed a cross-process lock for this.  A
jaato daemon runs many sessions in one process, spawns runner
subprocesses that refresh independently of it, and serves sessions from
a pre-warm pool whose slots ``fork()`` from a template that has already
imported the auth plugins.  A cascade fanning out after an idle period
is N stages waking at once against one expired token -- the normal
shape here, not an edge case.  An in-process ``threading.Lock`` is
therefore *not sufficient*: it does not exist across the runner
boundary, and a lock object held at fork time is inherited in a state
the child cannot reason about.

THE MECHANISM.  A ``flock`` on a sibling ``<credential>.lock`` file.
Three properties make it the right primitive here:

- **It spans processes.**  Daemon, runner and pool slot contend on the
  same inode, which is what the refresh actually shares.
- **It spans threads of one process too.**  ``fcntl.flock`` locks the
  open file *description*, and each acquisition here opens its own, so
  two threads of one daemon contend exactly as two processes do.  That
  is why there is no second, in-process lock: a lock is easier to reason
  about than two locks, and two locks are how a deadlock is built.
- **Nothing is held at import time.**  The descriptor is opened and
  closed inside one call, so a pool slot forked from the template
  inherits no lock state.  For the residual case -- a ``fork()`` that
  lands while another thread holds the lock, where the child would
  inherit a descriptor holding a lock it never took -- the live
  descriptors are tracked and closed in the child by an
  ``os.register_at_fork`` hook.

THE RE-READ IS THE FIX.  A lock alone turns a race into a queue: N
sessions still perform N refreshes, each rotating the token out from
under the next, and the last one wins for reasons nobody can see.
:func:`refresh_under_lock` re-reads the credential *after* acquiring, so
a session that blocked adopts the token the winner just wrote and does
not refresh at all.  That is what converts the queue into one refresh.

A TRANSIENT FAILURE IS NOT A LOGOUT.  "The server says this token is
invalid" and "this request failed" have opposite correct responses, and
the code that cannot tell them apart turns a network blip into a
re-login prompt.  :func:`raise_for_refresh_failure` classifies, and the
asymmetry is deliberate: only an explicit OAuth error code
(``invalid_grant`` and friends) is read as a dead credential, and
everything else -- a proxy's bare 401, a 502, a timeout -- is transient.
Mistaking a dead grant for a transient costs one wasted retry;
mistaking a transient for a dead grant costs the user their session.

Usage::

    tokens = refresh_under_lock(
        credential_path=path,
        load=lambda: load_tokens(...),
        needs_refresh=lambda t: t.is_expired,
        still_usable=lambda t: time.time() < t.expires_at,
        refresh=lambda t: refresh_tokens(t.refresh_token),
        save=lambda t: save_tokens(t, ...),
        label="anthropic",
    )
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator, NoReturn, Optional, TypeVar, Union

logger = logging.getLogger(__name__)


#: How long a caller waits for another process to finish its refresh
#: before giving up.  A refresh is one HTTPS round trip with a 30s
#: timeout of its own, so 60s covers a slow one plus the queue behind
#: it.  The bound exists so a stale lock file cannot hang a session
#: forever -- a wait nobody bounded is the failure mode this module was
#: written against, in a different disguise.
DEFAULT_LOCK_TIMEOUT_SECONDS = 60.0

#: How far ahead of real expiry a token is treated as stale.  Refreshing
#: early is what keeps the *common* path off the lock entirely: by the
#: time a herd of sessions wakes, one of them has already replaced the
#: token and the rest read a fresh one.
#:
#: Note what a margin does NOT do, since it is easy to over-claim: it
#: does not disperse a herd.  Every process computes the same threshold
#: from the same ``expires_at``, so N sessions cross it at the same
#: instant exactly as they crossed real expiry -- the margin only moves
#: the instant earlier.  What turns the herd into one refresh is the
#: lock plus the re-read; the margin's job is to make sure that happens
#: while the old token is still *valid*, so the losers have something
#: usable to fall back on if the refresh fails.
DEFAULT_REFRESH_MARGIN_SECONDS = 300.0

LOCK_TIMEOUT_ENV = "JAATO_CREDENTIAL_LOCK_TIMEOUT"
REFRESH_MARGIN_ENV = "JAATO_OAUTH_REFRESH_MARGIN"

#: Poll interval while waiting for the lock.  ``flock`` can block
#: natively, but a blocking wait cannot be given a timeout without
#: signals (not thread-safe) so acquisition polls instead.  Correctness
#: does not depend on this number -- ``flock`` decides who wins -- only
#: how promptly a waiter notices it can proceed.
_POLL_INTERVAL_SECONDS = 0.02

T = TypeVar("T")
PathLike = Union[str, Path]


class CredentialRefreshError(RuntimeError):
    """Base for refresh failures.

    Subclasses :class:`RuntimeError` deliberately: every existing caller
    of the refresh functions catches ``RuntimeError``, and this is a
    narrowing of that contract rather than a replacement for it.
    """


class TransientRefreshError(CredentialRefreshError):
    """The refresh request failed; the stored credential is not implicated.

    A timeout, a connection reset, a 5xx, a 429, a proxy's bare 401.
    Nothing about the stored refresh token has been disproved, so it MUST
    NOT be cleared or overwritten -- and if it is still inside its real
    validity window the caller should go on using it.
    """


class InvalidGrantError(CredentialRefreshError):
    """The server said this refresh token is no longer valid.

    Raised only for an explicit OAuth error code.  This is the one case
    where re-authentication is genuinely required.
    """


class CredentialLockTimeout(TransientRefreshError):
    """Waiting for another process's refresh exceeded the timeout.

    A transient by construction: the holder was mid-refresh, so the
    stored credential is untouched and about to be replaced by someone
    else.
    """


# ---------------------------------------------------------------------------
# fork safety
# ---------------------------------------------------------------------------

#: Descriptors currently holding a lock in THIS process.  Only ever
#: touched under the GIL by single ``add``/``discard`` calls, so it needs
#: no lock of its own -- and must not have one, since a lock here would
#: be a second lock object exposed to ``fork()``.
_HELD_FDS: "set[int]" = set()


def _drop_inherited_locks() -> None:
    """Close lock descriptors inherited by a forked child.

    ``fork()`` duplicates descriptors, and a duplicate shares the open
    file description -- so a child forked while another thread held the
    lock inherits a descriptor that *is* holding it.  The child never
    took that lock and must not hold it; closing releases the child's
    reference without disturbing the parent, which still has its own.

    Runs in the child, which is single-threaded at that point, so the
    set cannot change underneath this.
    """
    inherited = list(_HELD_FDS)
    _HELD_FDS.clear()
    for fd in inherited:
        try:
            os.close(fd)
        except OSError:
            pass


if hasattr(os, "register_at_fork"):  # POSIX only
    os.register_at_fork(after_in_child=_drop_inherited_locks)


# ---------------------------------------------------------------------------
# knobs
# ---------------------------------------------------------------------------

def _positive_float(raw: Optional[str], name: str, default: float) -> float:
    """Parse a positive float, falling back to ``default`` on nonsense.

    An unparseable or non-positive value falls back rather than
    disabling the bound: "unbounded" is the bug these numbers exist to
    prevent, so it must not be reachable by typo.

    Pure — the ``os.environ`` read is left at the call sites below so
    each var name appears as a literal argument there.  ``explain env``
    and ``test_env_scope_catalog`` both derive the var list by AST-
    scanning read sites and resolving same-file constants; a name that
    only ever reaches ``os.environ`` through a function parameter is
    invisible to them, which would make the knob undocumented and the
    catalog entry read as stale.
    """
    if not raw:
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning("%s=%r is not a number; using %s", name, raw, default)
        return default
    if value <= 0:
        logger.warning("%s=%r is not positive; using %s", name, raw, default)
        return default
    return value


def lock_timeout_seconds() -> float:
    """Effective lock wait bound (``JAATO_CREDENTIAL_LOCK_TIMEOUT``)."""
    return _positive_float(
        os.environ.get(LOCK_TIMEOUT_ENV),
        LOCK_TIMEOUT_ENV,
        DEFAULT_LOCK_TIMEOUT_SECONDS,
    )


def refresh_margin_seconds() -> float:
    """Effective early-refresh margin (``JAATO_OAUTH_REFRESH_MARGIN``)."""
    return _positive_float(
        os.environ.get(REFRESH_MARGIN_ENV),
        REFRESH_MARGIN_ENV,
        DEFAULT_REFRESH_MARGIN_SECONDS,
    )


# ---------------------------------------------------------------------------
# the lock
# ---------------------------------------------------------------------------

def lock_path_for(credential_path: PathLike) -> Path:
    """The lock file guarding ``credential_path``.

    A sibling ``<name>.lock`` rather than the credential file itself, so
    an ``flock`` is never held on a descriptor that some other code path
    might replace out from under it -- the credential file is rewritten
    by ``os.replace``, which swaps the inode and would silently detach
    the lock from the thing it guards.
    """
    return Path(str(credential_path) + ".lock")


def _open_lock_file(path: Path) -> int:
    """Open (creating) the lock file and return its descriptor."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    return fd


def _try_flock(fd: int) -> bool:
    """One non-blocking exclusive ``flock`` attempt.

    Returns ``True`` when the lock is held.  On a platform without
    ``fcntl`` (Windows) returns ``True`` unconditionally: the lock
    degrades to advisory-only, which is the same fallback
    ``references/reconcile.py`` takes, and is honest about the fact that
    no boundary exists there rather than pretending to one.
    """
    try:
        import fcntl  # type: ignore
    except ImportError:
        return True
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except (BlockingIOError, OSError):
        return False


def _unflock(fd: int) -> None:
    """Release an ``flock``; closing would do it too, belt and braces."""
    try:
        import fcntl  # type: ignore
    except ImportError:
        return
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    except OSError:
        pass


@contextmanager
def credential_lock(
    credential_path: PathLike,
    timeout: Optional[float] = None,
) -> Iterator[None]:
    """Hold an exclusive cross-process lock on ``credential_path``.

    Covers the whole read-check-refresh-write sequence, not just the
    write -- locking only the write would still let two processes decide
    to refresh, which is the defect.

    Raises :class:`CredentialLockTimeout` if the lock is not acquired
    within ``timeout`` (default :func:`lock_timeout_seconds`).

    **Not reentrant.**  ``flock`` attaches to the open file description,
    and each acquisition here opens its own — which is exactly what makes
    two threads of one process contend like two processes, and is also
    why nesting two acquisitions on one path self-deadlocks until the
    timeout fires.  Keep the call sites flat: a helper called from inside
    the lock must not take it again.
    """
    wait = lock_timeout_seconds() if timeout is None else timeout
    path = lock_path_for(credential_path)
    fd = _open_lock_file(path)
    deadline = time.monotonic() + wait
    acquired = False
    try:
        while True:
            if _try_flock(fd):
                acquired = True
                break
            if time.monotonic() >= deadline:
                raise CredentialLockTimeout(
                    f"timed out after {wait:g}s waiting for the credential "
                    f"lock at {path}; another process is refreshing"
                )
            time.sleep(_POLL_INTERVAL_SECONDS)
        _HELD_FDS.add(fd)
        yield
    finally:
        if acquired:
            _HELD_FDS.discard(fd)
            _unflock(fd)
        try:
            os.close(fd)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# failure classification
# ---------------------------------------------------------------------------

#: OAuth 2.0 error codes (RFC 6749 §5.2) that mean the credential itself
#: is dead.  Anything not on this list is read as transient -- see the
#: module docstring for why the asymmetry runs this way.
_DEAD_GRANT_CODES = (
    "invalid_grant",
    "invalid_client",
    "unauthorized_client",
    "invalid_token",
)


def classify_refresh_failure(
    status_code: Optional[int],
    body: str,
) -> "type[CredentialRefreshError]":
    """Decide whether a failed refresh disproves the stored credential.

    ``status_code`` is ``None`` for a failure with no HTTP response at
    all (DNS, connect, timeout), which is transient by definition.
    """
    if status_code is None:
        return TransientRefreshError
    haystack = (body or "").lower()
    if any(code in haystack for code in _DEAD_GRANT_CODES):
        return InvalidGrantError
    return TransientRefreshError


def raise_for_refresh_failure(
    status_code: Optional[int],
    body: str,
    label: str,
) -> NoReturn:
    """Raise the classified error for a failed refresh.

    Declared ``NoReturn`` so a call in an ``except`` block reads as
    terminal: the code after it is genuinely unreachable, and a reader
    (or a checker) should not have to prove that from the body.  ``label`` names the provider so the message says
    which credential is involved; the body is included because it
    carries the provider's own error code, and an OAuth token endpoint
    error body carries no token -- the tokens travel in the *request*
    and in a *successful* response.
    """
    kind = classify_refresh_failure(status_code, body)
    where = f"HTTP {status_code}" if status_code is not None else "no response"
    raise kind(f"{label} token refresh failed ({where}): {body}")


# ---------------------------------------------------------------------------
# the sequence
# ---------------------------------------------------------------------------

def _refresh_and_save(
    state: T,
    refresh: Callable[[T], T],
    save: Callable[[T], None],
    label: str,
) -> T:
    """Refresh, then persist -- and persist nothing if the refresh failed.

    Split out so the ordering is stated in one place: ``save`` is
    unreachable unless ``refresh`` returned.  A refresh that raises must
    leave the stored credential exactly as it was, because the stored
    one may still be valid and is certainly better than a partial write.
    """
    new_state = refresh(state)
    save(new_state)
    logger.debug("%s credential refreshed and stored", label)
    return new_state


def refresh_under_lock(
    *,
    credential_path: PathLike,
    load: Callable[[], Optional[T]],
    needs_refresh: Callable[[T], bool],
    refresh: Callable[[T], T],
    save: Callable[[T], None],
    still_usable: Optional[Callable[[T], bool]] = None,
    timeout: Optional[float] = None,
    label: str = "credential",
) -> Optional[T]:
    """Run load-check-refresh-write once, under a cross-process lock.

    The four asks of #683, in one place:

    1. The lock spans the whole sequence, and spans processes.
    2. The credential is **re-read after acquiring**, so a caller that
       waited adopts the winner's token instead of refreshing again.
    3. ``needs_refresh`` is expected to carry a margin, so the refresh
       happens while the old token is still valid.
    4. A :class:`TransientRefreshError` never overwrites anything, and
       when ``still_usable`` says the current token has real validity
       left, it is returned rather than raised -- a network blip during
       the margin window is not a logout.

    ``load`` returning ``None`` (no credential stored) returns ``None``:
    that is "not logged in", which is not this function's problem.
    """
    state = load()
    if state is None:
        return None
    if not needs_refresh(state):
        # The overwhelmingly common path.  Deliberately outside the
        # lock: taking one on every token read would serialise every
        # request in the daemon behind a file lock, to guard a sequence
        # that is not happening.
        return state

    with credential_lock(credential_path, timeout=timeout):
        # Ask 2.  Whoever held the lock may have just written a fresh
        # token; adopting it is what makes N waiters cost one refresh.
        state = load()
        if state is None:
            return None
        if not needs_refresh(state):
            logger.debug(
                "%s credential was refreshed by another holder; reusing it",
                label,
            )
            return state
        try:
            return _refresh_and_save(state, refresh, save, label)
        except TransientRefreshError as exc:
            if still_usable is not None and still_usable(state):
                logger.warning(
                    "%s token refresh failed transiently (%s); the stored "
                    "token has not expired, continuing with it",
                    label, exc,
                )
                return state
            raise


def with_margin(expires_at: float, margin: Optional[float] = None) -> bool:
    """``True`` when ``expires_at`` is within the early-refresh margin.

    One definition of "stale", so the providers cannot drift apart on
    the number or on whether the comparison is strict.
    """
    window = refresh_margin_seconds() if margin is None else margin
    return time.time() > (expires_at - window)


def has_expired(expires_at: float) -> bool:
    """``True`` when ``expires_at`` is in the past -- no margin.

    The ``still_usable`` half of the pair above: a token inside the
    margin is stale but perfectly valid, and that distinction is what
    lets a transient refresh failure be survivable.
    """
    return time.time() >= expires_at


__all__ = [
    "CredentialRefreshError",
    "TransientRefreshError",
    "InvalidGrantError",
    "CredentialLockTimeout",
    "DEFAULT_LOCK_TIMEOUT_SECONDS",
    "DEFAULT_REFRESH_MARGIN_SECONDS",
    "LOCK_TIMEOUT_ENV",
    "REFRESH_MARGIN_ENV",
    "classify_refresh_failure",
    "credential_lock",
    "has_expired",
    "lock_path_for",
    "lock_timeout_seconds",
    "raise_for_refresh_failure",
    "refresh_margin_seconds",
    "refresh_under_lock",
    "with_margin",
]
