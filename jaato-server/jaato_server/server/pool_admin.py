"""Read and resize the pre-warm runner pool on a running daemon (protocol 1.35).

The pool's floor (``JAATO_RUNNER_POOL_SIZE``) and ceiling
(``JAATO_RUNNER_POOL_MAX_SIZE``) were read once, when the daemon started, so
the only way to add warm runners was a restart -- which unloads every session
to change a number the replenish loop re-reads on every pass anyway.
:meth:`PoolManager.resize` changes the numbers; this module decides WHO may
ask, turns the answer into a :class:`PoolStatusEvent`, and reports whether
``--restart`` will keep the new sizes.

Who may ask
-----------
The pool is a property of the daemon PROCESS: it bounds memory on the host
and is shared by every session.  ``env_scope`` already classifies both knobs
``host``.  So the verb is answered only for a connection whose OS account the
KERNEL vouches for (``SO_PEERCRED``, :mod:`shared.peer_identity`) and which is
the daemon's own uid or root -- the accounts that could already stop and
restart the daemon with a different environment.  Everything else is refused
by name, never silently:

* no peer credential: a WS connection (a bearer token or a ticket names an
  application user, not an OS account), a Windows pipe, a non-Linux socket.
  An absent credential is never read as permission.
* another local account on a shared socket (``--socket-mode 666``).

The rule reads the peer the TRANSPORT reports and nothing in the request
body, so a client cannot claim an identity.

The three entry points -- ``PoolStatusRequest`` (the SDK),
``pool.status`` / ``pool.resize <target> [<max>]`` (typed, from the TUI's
prompt or ``--cmd``) and ``python -m jaato_server --pool-size`` -- all reach
:meth:`PoolAdmin.answer`, so they cannot disagree about the rule.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

#: ``PoolStatusEvent.category`` values for a refused request.
NOT_AUTHORIZED = "not_authorized"
INVALID_REQUEST = "invalid_request"
NO_POOL = "no_pool"

#: The typed verbs (TUI prompt, ``rich_client.py --cmd``).  The router
#: dispatches on these and ``jaato-scaffold explain pool`` prints them, so
#: the two cannot name different commands.
VERB_STATUS = "pool.status"
VERB_RESIZE = "pool.resize"

#: The daemon CLI flags that send a ``PoolStatusRequest``.  Read by the
#: argparse definition and by ``explain pool``.
CLI_SIZE_FLAG = "--pool-size"
CLI_MAX_FLAG = "--pool-max"

#: The env knobs the pool is configured from at startup, with their
#: defaults.  The reads themselves stay literal (the env-scope catalog is
#: derived by an AST scan for literal reads); a guard checks every name
#: here is read where this says it is.
STARTUP_KNOBS = (
    ("JAATO_RUNNER_POOL_SIZE", "2",
     "floor: unreserved idle runners any arriving session may take"),
    ("JAATO_RUNNER_POOL_MAX_SIZE", "2 x pool size",
     "ceiling: total idle runners, cascade reservations included"),
    ("JAATO_RUNNER_POOL_ENABLED", "true",
     "whether sessions are routed to the pool at all"),
)

#: Categories of ``PoolStatusEvent`` refusals, for ``explain pool``.
REFUSAL_CATEGORIES = (
    (NOT_AUTHORIZED, "the connection is not the daemon's own account or "
                     "root, or carries no OS account (WS, Windows)"),
    (INVALID_REQUEST, "a size that is not a non-negative integer"),
    (NO_POOL, "the daemon runs without a runner pool"),
)


def pool_admin_refusal(peer: Any, daemon_uid: Optional[int] = None) -> Optional[str]:
    """Why ``peer`` may not read or resize the pool, or ``None`` if it may.

    Args:
        peer: The connection's :class:`PeerCredentials`, or ``None`` when
            the transport has none (WS, Windows, non-Linux).
        daemon_uid: The daemon's uid; ``os.getuid()`` when omitted.  A
            parameter so the rule can be checked without being root.

    Returns:
        A sentence for ``PoolStatusEvent.error``, or ``None``.
    """
    if peer is None:
        return ("the runner pool is managed over the daemon's IPC socket by "
                "the account that runs the daemon (or root); this connection "
                "carries no OS account the kernel vouches for")
    uid = getattr(peer, "uid", None)
    if not isinstance(uid, int):
        return "the connection's OS account could not be determined"
    own = os.getuid() if daemon_uid is None else daemon_uid
    if uid in (own, 0):
        return None
    return (f"the runner pool may be managed only by the account that runs "
            f"the daemon (uid {own}) or root; this connection is "
            f"{getattr(peer, 'identity', f'uid:{uid}')}")


def parse_resize_args(args: List[str]) -> Tuple[Optional[int], Optional[int]]:
    """``(target_size, max_size)`` from ``pool.resize <target> [<max>]``.

    Raises:
        ValueError: No target, too many arguments, or a value that is not
            a non-negative integer.  The message is the usage line.
    """
    usage = f"usage: {VERB_RESIZE} <target_size> [<max_size>]"
    if not args or len(args) > 2:
        raise ValueError(usage)
    values = []
    for raw in args:
        try:
            value = int(str(raw).strip())
        except ValueError:
            raise ValueError(f"{usage} -- {raw!r} is not an integer") from None
        if value < 0:
            raise ValueError(f"{usage} -- sizes must be >= 0, got {value}")
        values.append(value)
    return values[0], (values[1] if len(values) > 1 else None)


def describe(event: Any) -> str:
    """One line for a person: what the pool is, and what changed.

    Used as the ``SystemMessageEvent`` beside a typed command's
    ``PoolStatusEvent``, and by ``--pool-size`` / the TUI's ``--cmd``.
    """
    if not event.ok:
        return f"pool: refused ({event.category}): {event.error}"
    head = (f"pool: target_size {event.previous_target_size} -> "
            f"{event.target_size}, max_size {event.previous_max_size} -> "
            f"{event.max_size}" if event.changed else
            f"pool: target_size {event.target_size}, max_size "
            f"{event.max_size}")
    tail = (f"; idle {event.idle} (unreserved {event.unreserved}, reserved "
            f"{event.reserved})")
    if event.pending_teardown:
        tail += f", {event.pending_teardown} awaiting teardown"
    notes = []
    if not event.routing_enabled:
        notes.append("sessions are NOT routed to the pool "
                     "(JAATO_RUNNER_POOL_ENABLED)")
    if not event.template_alive:
        notes.append("the runner template is down")
    if event.changed and not event.persisted:
        notes.append("--restart will NOT keep these sizes")
    return head + tail + ("" if not notes else " -- " + "; ".join(notes))


class PoolAdmin:
    """The daemon's one answer to "what is the pool / make it this size".

    Args:
        pool_manager: The daemon's :class:`PoolManager`, or ``None`` when
            it runs without one (answers ``no_pool``).
        on_resized: Called with ``(target_size, max_size)`` after a resize,
            ``max_size`` being ``None`` when the ceiling is derived
            (``2 * target_size``) rather than chosen, so a restart derives
            it again; returns whether the sizes were recorded for
            ``--restart``.
            ``None`` means nothing records them.
        routing_enabled: Reports ``JAATO_RUNNER_POOL_ENABLED`` as the
            session router reads it.
    """

    def __init__(
        self,
        pool_manager: Any,
        *,
        on_resized: Optional[Callable[[int, Optional[int]], bool]] = None,
        routing_enabled: Optional[Callable[[], bool]] = None,
    ) -> None:
        self._pool = pool_manager
        self._on_resized = on_resized
        if routing_enabled is None:
            from jaato_server.server.runner_spawn import _pool_enabled
            routing_enabled = _pool_enabled
        self._routing_enabled = routing_enabled
        #: Whether ``--restart`` reproduces the current sizes.  True until a
        #: resize fails to record itself; the env vars a restart re-reads
        #: are what produced the startup sizes.
        self._persisted = True

    def answer(
        self,
        peer: Any,
        *,
        request_id: str = "",
        target_size: Optional[int] = None,
        max_size: Optional[int] = None,
    ) -> Any:
        """Authorise, apply (when sizes are given), and describe the pool.

        Never raises: a refusal or a bad size is an ``ok=False`` event.

        Args:
            peer: The connection's kernel-reported credential, or ``None``.
            request_id: Echoed on the answer.
            target_size: New floor, or ``None`` to keep it.
            max_size: New ceiling, or ``None``.

        Returns:
            A :class:`jaato_sdk.events.PoolStatusEvent`.
        """
        from jaato_sdk.events import PoolStatusEvent

        def refused(category: str, error: str) -> Any:
            logger.warning("pool admin: refused (%s): %s", category, error)
            return PoolStatusEvent(request_id=request_id, ok=False,
                                   category=category, error=error)

        refusal = pool_admin_refusal(peer)
        if refusal:
            return refused(NOT_AUTHORIZED, refusal)
        if self._pool is None:
            return refused(NO_POOL, "this daemon runs without a runner pool")
        change: Dict[str, Any] = {}
        if target_size is not None or max_size is not None:
            target = (self._pool.target_size if target_size is None
                      else target_size)
            try:
                change = self._pool.resize(target, max_size)
            except ValueError as exc:
                return refused(INVALID_REQUEST, str(exc))
            self._persisted = self._record(change["current"])
            logger.info(
                "pool admin: %s resized the pool: %s -> %s (persisted=%s)",
                getattr(peer, "identity", "?"), change["previous"],
                change["current"], self._persisted,
            )
        snap = self._pool.snapshot()
        previous = change.get("previous", {})
        return PoolStatusEvent(
            request_id=request_id,
            changed=bool(change),
            previous_target_size=previous.get("target_size"),
            previous_max_size=previous.get("max_size"),
            routing_enabled=bool(self._routing_enabled()),
            persisted=self._persisted,
            **{k: snap[k] for k in (
                "target_size", "max_size", "max_size_explicit", "idle",
                "unreserved", "reserved", "pending_teardown", "replenishing",
                "template_alive", "telemetry")},
        )

    def _record(self, current: Dict[str, Any]) -> bool:
        """Hand the new sizes to ``on_resized``; ``False`` when not kept."""
        if self._on_resized is None:
            return False
        try:
            ceiling = (current["max_size"]
                       if current.get("max_size_explicit") else None)
            return bool(self._on_resized(current["target_size"], ceiling))
        except Exception as exc:  # noqa: BLE001 — the resize itself stands
            logger.warning("pool admin: could not record the new sizes for "
                           "--restart: %s", exc)
            return False
