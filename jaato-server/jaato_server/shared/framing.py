"""Length-prefixed JSON-frame transport (shared between IPC and runner-RPC).

Wire format:
    | 4-byte big-endian length | UTF-8 payload |

Used by:
- ``server/ipc.py`` for the daemon ↔ client (TUI / SDK) channel.
- ``server/runner/rpc.py`` and ``server/runner_rpc.py`` for the daemon
  ↔ runner channel (Phase 2 onwards — see
  ``docs/design/per_session_confined_runner.md`` §4.1).

Two API flavours are exposed because the two callers have different
loops:

- **Async** (``read_frame`` / ``write_frame``) — used daemon-side, where
  asyncio is in charge of the event loop.
- **Synchronous** (``read_frame_sync`` / ``write_frame_sync``) — used
  runner-side, where the RPC dispatcher serves on a blocking socket in
  a worker thread.

The bytes on the wire are identical between the two flavours; only the
I/O machinery differs.  Same ``HEADER_SIZE`` / ``MAX_MESSAGE_SIZE``
constants apply on both sides.
"""

from __future__ import annotations

import asyncio
import logging
import socket
import struct
from typing import Optional


logger = logging.getLogger(__name__)


HEADER_SIZE = 4
MAX_MESSAGE_SIZE = 10 * 1024 * 1024  # 10 MB max — same as legacy IPC.


class FrameTooLargeError(ValueError):
    """Frame length-prefix exceeds ``MAX_MESSAGE_SIZE``.

    Raised on read to surface "the peer sent us a malformed/oversized
    frame" distinctly from "the peer closed the connection cleanly".
    """


# ----------------------------------------------------------------------
# Async (asyncio.StreamReader / StreamWriter)
# ----------------------------------------------------------------------


async def read_frame(reader: asyncio.StreamReader) -> Optional[str]:
    """Read one length-prefixed UTF-8 frame from *reader*.

    Returns:
        The decoded payload as ``str``, or ``None`` if the peer closed
        the stream cleanly between frames.

    Raises:
        FrameTooLargeError: header reports a length over
            ``MAX_MESSAGE_SIZE``.

    Other transport errors (``ConnectionResetError`` etc.) are returned
    as ``None`` — the legacy IPC behaviour treated peer-reset as a
    clean disconnect, and we preserve that contract.
    """
    try:
        header = await reader.readexactly(HEADER_SIZE)
    except asyncio.IncompleteReadError:
        return None
    except ConnectionResetError:
        return None

    length = struct.unpack(">I", header)[0]
    if length > MAX_MESSAGE_SIZE:
        raise FrameTooLargeError(
            f"Message too large: {length} bytes (cap {MAX_MESSAGE_SIZE})"
        )

    try:
        payload = await reader.readexactly(length)
    except asyncio.IncompleteReadError:
        return None
    except ConnectionResetError:
        return None

    return payload.decode("utf-8")


async def write_frame(writer: asyncio.StreamWriter, message: str) -> None:
    """Write *message* to *writer* as one length-prefixed UTF-8 frame.

    Drains the underlying transport before returning so the caller can
    treat the write as complete-or-failed (no half-buffered state).
    """
    payload = message.encode("utf-8")
    header = struct.pack(">I", len(payload))
    writer.write(header + payload)
    await writer.drain()


# ----------------------------------------------------------------------
# Synchronous (blocking socket / fileobj — used in runner worker thread)
# ----------------------------------------------------------------------


def _recv_exact(sock: socket.socket, n: int) -> Optional[bytes]:
    """Read exactly *n* bytes from a blocking socket, or None on EOF."""
    out = bytearray()
    while len(out) < n:
        try:
            chunk = sock.recv(n - len(out))
        except ConnectionResetError:
            return None
        if not chunk:
            # Peer closed the connection.  When ``out`` is empty this is
            # a clean between-frames close; when ``out`` is partial we
            # hit EOF mid-frame — both surface to the caller as
            # ``None`` (per §6.7 EOF-mid-frame is benign on both sides).
            return None
        out.extend(chunk)
    return bytes(out)


def read_frame_sync(sock: socket.socket) -> Optional[str]:
    """Synchronous read of one length-prefixed UTF-8 frame.

    Mirror of :func:`read_frame` for blocking-socket callers (the
    runner-side RPC dispatcher).  Returns ``None`` on clean EOF or
    EOF-mid-frame; raises :class:`FrameTooLargeError` on oversized
    header.
    """
    header = _recv_exact(sock, HEADER_SIZE)
    if header is None:
        return None

    length = struct.unpack(">I", header)[0]
    if length > MAX_MESSAGE_SIZE:
        raise FrameTooLargeError(
            f"Message too large: {length} bytes (cap {MAX_MESSAGE_SIZE})"
        )

    payload = _recv_exact(sock, length)
    if payload is None:
        return None

    return payload.decode("utf-8")


def write_frame_sync(sock: socket.socket, message: str) -> None:
    """Synchronous write of one length-prefixed UTF-8 frame.

    Uses :meth:`socket.sendall` so the call returns only after the full
    header + payload has been pushed to the kernel buffer.
    """
    payload = message.encode("utf-8")
    header = struct.pack(">I", len(payload))
    sock.sendall(header + payload)
