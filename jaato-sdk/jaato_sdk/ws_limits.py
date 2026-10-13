"""The WebSocket message ceiling, shared by the daemon and the SDK client.

``websockets`` refuses any message over its ``max_size`` by closing the
connection with code 1009, and its default is 1 MiB.  The daemon raised its
own ceiling to :data:`DEFAULT_WS_MAX_MESSAGE_SIZE` long ago
(``--ws-max-message-size``), but the SDK client never passed ``max_size`` at
all, so it kept the library's 1 MiB for what it RECEIVES.  Any daemon -> client
message between 1 MiB and 16 MiB -- a ``session.list`` with two thousand rows,
the user echo of a long voice note -- made the CLIENT close the connection
with 1009, and the call waiting for that message waited out its timeout
(jaato #1279, #1586).

The two ends read the default from here so they cannot drift apart again: the
daemon imports it as its own default, the client passes it to
``websockets.connect`` unless told otherwise.

Stdlib only: the daemon imports this before anything else of the SDK's.
"""

from __future__ import annotations

from typing import Any, Optional

__all__ = [
    "DEFAULT_WS_MAX_MESSAGE_SIZE",
    "LEGACY_WS_MAX_MESSAGE_SIZE",
    "MESSAGE_TOO_BIG_CLOSE_CODE",
    "MESSAGE_TOO_LARGE_ERROR_TYPE",
    "close_frame_of",
]

#: The largest single WebSocket message either end accepts by default, in
#: bytes.  The daemon's ``--ws-max-message-size`` overrides its side; a
#: client's ``max_message_size=`` overrides the client's.
DEFAULT_WS_MAX_MESSAGE_SIZE = 16 * 1024 * 1024

#: What ``websockets`` applies when no ``max_size`` is passed, and so what
#: every daemon before ``max_message_size`` was advertised enforced -- and
#: what every SDK client before #1279 enforced on what it received.
LEGACY_WS_MAX_MESSAGE_SIZE = 1024 * 1024

#: RFC 6455 "message too big".
MESSAGE_TOO_BIG_CLOSE_CODE = 1009

#: ``ErrorEvent.error_type`` of the event the SDK client puts on its own event
#: stream when a connection ended on a refused message, so an event-driven
#: waiter (the facade's ``ask``, a driver awaiting ``SESSION_LIST``) learns
#: why instead of waiting out its timeout.  Never sent by the daemon.
MESSAGE_TOO_LARGE_ERROR_TYPE = "MessageTooLarge"


def close_frame_of(exc: Any) -> "tuple[Optional[Any], Optional[str]]":
    """Return ``(close_frame, initiator)`` for a ``ConnectionClosed``.

    ``initiator`` is ``"self"`` when THIS end started the close and
    ``"peer"`` when the other end did; ``close_frame`` is the frame the
    initiator sent.  ``(None, None)`` when no close frame was exchanged
    (a dropped connection).

    The distinction matters for 1009 and is easy to get wrong: the end that
    receives a close echoes it, so after a refusal BOTH ``sent`` and ``rcvd``
    carry 1009.  Preferring ``sent`` -- what the daemon's log line did --
    reports the echo as the refusal, and so blamed the daemon's 16 MiB limit
    for a frame the client's 1 MiB limit had refused (#1586).
    ``rcvd_then_sent`` is what says who went first.  Tolerates objects
    without the attributes, so a reshaped ``websockets`` degrades to
    "unknown" rather than raising inside an ``except``.
    """
    sent = getattr(exc, "sent", None)
    rcvd = getattr(exc, "rcvd", None)
    order = getattr(exc, "rcvd_then_sent", None)
    if rcvd is not None and (order is True or sent is None):
        return rcvd, "peer"
    if sent is not None:
        return sent, "self"
    return None, None
