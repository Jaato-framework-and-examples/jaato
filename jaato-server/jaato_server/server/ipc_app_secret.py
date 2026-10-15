"""``app://`` resolution over IPC, by the account that owns the workspace (#1609).

On WebSocket the daemon resolves ``NAME=app://<ref>`` by asking the
application that owns the workspace over its bind channel (#1226,
:mod:`server.app_secret`).  IPC has no application credential and no bind
channel, so a session the daemon started on its own -- a reactor stage, a
``session.wake`` from something other than the application, a revive from
disk -- re-resolved its environment with nobody to ask and dropped every
reference.  The only IPC route was ``pass://`` / ``vault://``, which needs a
store the daemon's own credential can read for EVERY user.

This module is the IPC half.  An IPC connection sends
``secret.resolver.register``; from then until it disconnects it is asked
``secret.resolve`` for the workspaces its OS account owns.

The rules it holds to:

* **Ownership is the workspace directory's owner uid** (``os.stat`` of the
  resolved path), and the identity is the connection's ``SO_PEERCRED`` uid,
  read by the kernel at ``connect(2)``.  The two must be equal.  Nothing in
  the request names a uid, so a client cannot answer for another account,
  and an account can only supply a value to a session in a tree it already
  controls (it can write that tree's ``.env`` anyway).
* **WebSocket ownership wins.**  This resolver is consulted only for a
  workspace the WS server says is unowned (:class:`ChainedAppSecretResolver`),
  so a workspace an application owns is never answered by a local account.
* **No live registration is ``unreachable``**, which drops the reference
  (or, for ``app://<ref>?required``, refuses the bootstrap) exactly as an
  unreachable WS application does.  The WARNING names the uid that would
  have to register.
* **One answer per request, from the connection asked.**  A result whose
  ``request_id`` was sent to another connection is ignored, so a second
  client of the same account cannot answer for the first.
"""

from __future__ import annotations

import logging
import os
import stat
import threading
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from jaato_server.server.app_secret import (
    DEFAULT_RESOLVE_TIMEOUT_SECONDS,
    AppSecretAnswer,
)
from jaato_server.shared.plugins.subagent.config import (
    APP_SECRET_SCHEME,
    AppSecretReference,
    SecretResolutionError,
    SecretResolveContext,
)

logger = logging.getLogger(__name__)

#: Prefix of the owner string this resolver answers ``owner_for`` with.  Not
#: an ``app:user`` qualification: it never reaches the WS owner checks
#: (memory curation, ``secret.reload``), which read the WS resolver alone.
IPC_OWNER_PREFIX = "unix-uid:"

#: ``(client_id, user, workspace, name, timeout) -> AppSecretAnswer``.  Sends
#: ``secret.resolve`` to one IPC connection and awaits its answer; never
#: raises for a timeout or a closed connection (those are statuses).
IPCSecretTransport = Callable[[str, str, str, str, float], AppSecretAnswer]


@dataclass(frozen=True)
class _Registration:
    client_id: str
    uid: int
    user: str


def workspace_owner_uid(workspace_path: Optional[str]) -> Optional[int]:
    """The uid owning the workspace directory, resolved through symlinks.

    ``None`` for an empty path, a path that does not resolve to a directory,
    or a platform with no POSIX ownership: such a workspace has no owner to
    ask, so its references are dropped.
    """
    if not workspace_path:
        return None
    try:
        st = os.stat(os.path.realpath(workspace_path))
    except OSError:
        return None
    if not stat.S_ISDIR(st.st_mode):
        return None
    return getattr(st, "st_uid", None)


class IPCAppSecretResolver:
    """Answers ``app://`` references through registered IPC connections (#1609).

    Exposes the two methods ``JaatoServer._apply_app_secret_references``
    calls on a resolver (:meth:`owner_for`, :meth:`resolve_reference`).
    Registrations are added by ``secret.resolver.register`` and removed when
    the connection closes; the transport is injected by the IPC server.
    """

    def __init__(
        self,
        transport: IPCSecretTransport,
        *,
        owner_uid_of: Callable[[Optional[str]], Optional[int]] = workspace_owner_uid,
        timeout: float = DEFAULT_RESOLVE_TIMEOUT_SECONDS,
    ) -> None:
        self._transport = transport
        self._owner_uid_of = owner_uid_of
        self._timeout = timeout
        self._lock = threading.Lock()
        self._registrations: List[_Registration] = []

    # -- registration ------------------------------------------------------

    def register(self, client_id: str, uid: int, user: str) -> None:
        """Make ``client_id`` the resolver for workspaces owned by ``uid``.

        Re-registering a connection replaces its entry.  Several connections
        of one account may register; the earliest live one is asked.
        """
        with self._lock:
            self._registrations = [
                r for r in self._registrations if r.client_id != client_id
            ]
            self._registrations.append(_Registration(client_id, uid, user))
        logger.info(
            "app:// resolver registered over IPC: client=%s uid=%d (%s)",
            client_id, uid, user,
        )

    def unregister(self, client_id: str) -> None:
        """Forget ``client_id``'s registration (its connection closed)."""
        with self._lock:
            before = len(self._registrations)
            self._registrations = [
                r for r in self._registrations if r.client_id != client_id
            ]
            removed = before != len(self._registrations)
        if removed:
            logger.info("app:// resolver unregistered: client=%s", client_id)

    def _registration_for(self, uid: int) -> Optional[_Registration]:
        with self._lock:
            for reg in self._registrations:
                if reg.uid == uid:
                    return reg
        return None

    # -- the resolver interface -------------------------------------------

    def owner_for(self, workspace_path: Optional[str]) -> Optional[str]:
        """``unix-uid:<n>`` for the account owning the workspace, else ``None``.

        Answered whether or not that account registered, so an unanswered
        reference is reported as "nobody registered for uid N" rather than
        as an unowned workspace.
        """
        try:
            uid = self._owner_uid_of(workspace_path)
        except Exception:  # noqa: BLE001 -- a lookup failure is "unowned"
            logger.warning(
                "app:// IPC owner lookup for %r raised; treating as unowned",
                workspace_path, exc_info=True,
            )
            return None
        return None if uid is None else f"{IPC_OWNER_PREFIX}{uid}"

    def resolve_reference(
        self,
        ref: AppSecretReference,
        context: Optional[SecretResolveContext],
        *,
        timeout: Optional[float] = None,
    ) -> AppSecretAnswer:
        """Ask the connection registered for the owning uid, or answer why not."""
        if context is None:
            raise SecretResolutionError(
                f"{APP_SECRET_SCHEME}://{ref.name}",
                "no resolve context (app:// is resolvable only at session-env "
                "resolution, where the workspace owner is known)",
            )
        owner = context.workspace_owner or ""
        if not owner.startswith(IPC_OWNER_PREFIX):
            return AppSecretAnswer(status="unowned", detail="workspace has no owner")
        try:
            uid = int(owner[len(IPC_OWNER_PREFIX):])
        except ValueError:
            return AppSecretAnswer(status="unowned", detail=f"owner {owner!r} unreadable")
        reg = self._registration_for(uid)
        if reg is None:
            return AppSecretAnswer(
                status="unreachable",
                detail=(
                    f"no IPC connection of the workspace owner (uid {uid}) has "
                    "registered an app:// resolver (secret.resolver.register)"
                ),
            )
        deadline = self._timeout if timeout is None else timeout
        try:
            return self._transport(
                reg.client_id, reg.user, context.workspace_path or "",
                ref.name, deadline,
            )
        except Exception as exc:  # noqa: BLE001 -- transport error is a drop
            logger.warning(
                "app://%s resolution over IPC for uid %d failed: %s",
                ref.name, uid, exc, exc_info=True,
            )
            return AppSecretAnswer(status="unreachable", detail=str(exc))


class PendingSecretResolves:
    """Outstanding ``secret.resolve`` requests over IPC, by ``request_id``.

    Records which connection each request went to, so :meth:`deliver` takes
    an answer only from that connection.  Thread-safe: requests are made
    from executor threads and answers arrive on the event loop.
    """

    def __init__(self) -> None:
        import concurrent.futures  # noqa: PLC0415 -- local, like websocket.py

        self._futures = concurrent.futures
        self._lock = threading.Lock()
        self._pending: Dict[str, tuple] = {}

    def open(self, request_id: str, client_id: str):
        """Register a request and return the future its answer resolves."""
        fut = self._futures.Future()
        with self._lock:
            self._pending[request_id] = (client_id, fut)
        return fut

    def close(self, request_id: str) -> None:
        """Forget a request that timed out or failed to send."""
        with self._lock:
            self._pending.pop(request_id, None)

    def deliver(self, client_id: str, request_id: str, result) -> bool:
        """Resolve the request with ``result`` if ``client_id`` was asked."""
        with self._lock:
            entry = self._pending.get(request_id)
            if entry is None or entry[0] != client_id:
                return False
            self._pending.pop(request_id, None)
        fut = entry[1]
        if not fut.done():
            fut.set_result(result)
        return True

    def drop_client(self, client_id: str) -> None:
        """Fail every request sent to a connection that closed."""
        with self._lock:
            gone = [rid for rid, (cid, _) in self._pending.items() if cid == client_id]
            futs = [self._pending.pop(rid)[1] for rid in gone]
        for fut in futs:
            if not fut.done():
                fut.set_exception(ConnectionError("the resolver connection closed"))
