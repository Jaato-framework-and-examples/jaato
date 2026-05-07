"""Tests for ``shared.framing`` — length-prefixed JSON-frame transport.

Covers both the async (asyncio) and synchronous (blocking-socket) APIs
because both wire codepaths must produce/consume identical bytes.
"""

from __future__ import annotations

import asyncio
import socket
import struct
from typing import List, Optional

import pytest

from shared.framing import (
    HEADER_SIZE,
    MAX_MESSAGE_SIZE,
    FrameTooLargeError,
    read_frame,
    write_frame,
    read_frame_sync,
    write_frame_sync,
)


# ----------------------------------------------------------------------
# Async (asyncio.StreamReader / StreamWriter)
#
# We exercise read_frame against a plain StreamReader fed via
# feed_data / feed_eof (lets us forge bytes-on-the-wire without a real
# transport).  write_frame is exercised against a CapturingWriter mock
# implementing only the StreamWriter surface read_frame uses.
# ----------------------------------------------------------------------


class _CapturingWriter:
    """Minimal StreamWriter stand-in capturing bytes via ``write`` and
    asserting ``drain`` was awaited.  Sufficient for ``write_frame``,
    which only calls ``write`` + ``drain``.
    """

    def __init__(self) -> None:
        self.buf = bytearray()
        self.drains = 0

    def write(self, data: bytes) -> None:
        self.buf.extend(data)

    async def drain(self) -> None:
        self.drains += 1


@pytest.mark.asyncio
async def test_async_write_frame_emits_legacy_format() -> None:
    """write_frame produces the same bytes the legacy IPC format wrote.

    Wire format: 4-byte big-endian length || UTF-8 payload.
    """
    writer = _CapturingWriter()
    await write_frame(writer, '{"hello": "world"}')

    payload = b'{"hello": "world"}'
    expected = struct.pack(">I", len(payload)) + payload
    assert bytes(writer.buf) == expected
    assert writer.drains == 1


@pytest.mark.asyncio
async def test_async_read_frame_round_trip_ascii() -> None:
    reader = asyncio.StreamReader()
    payload = '{"hello": "world"}'.encode("utf-8")
    reader.feed_data(struct.pack(">I", len(payload)) + payload)
    reader.feed_eof()
    out = await read_frame(reader)
    assert out == '{"hello": "world"}'


@pytest.mark.asyncio
async def test_async_read_frame_utf8() -> None:
    reader = asyncio.StreamReader()
    payload = '{"emoji": "✓", "kanji": "日本語"}'.encode("utf-8")
    reader.feed_data(struct.pack(">I", len(payload)) + payload)
    reader.feed_eof()
    out = await read_frame(reader)
    assert out == '{"emoji": "✓", "kanji": "日本語"}'


@pytest.mark.asyncio
async def test_async_eof_between_frames_returns_none() -> None:
    reader = asyncio.StreamReader()
    reader.feed_eof()
    out = await read_frame(reader)
    assert out is None


@pytest.mark.asyncio
async def test_async_eof_mid_header_returns_none() -> None:
    """Half a length prefix on the wire = treated as clean disconnect."""
    reader = asyncio.StreamReader()
    reader.feed_data(b"\x00\x01")  # 2 bytes of a 4-byte header
    reader.feed_eof()
    out = await read_frame(reader)
    assert out is None


@pytest.mark.asyncio
async def test_async_eof_mid_payload_returns_none() -> None:
    """EOF after the header but before the payload completes — clean None.

    Per §6.7 EOF-mid-frame is benign on both sides.
    """
    reader = asyncio.StreamReader()
    reader.feed_data(struct.pack(">I", 100))  # claim 100 bytes
    reader.feed_data(b"only ten ")              # 9 bytes, then EOF
    reader.feed_eof()
    out = await read_frame(reader)
    assert out is None


@pytest.mark.asyncio
async def test_async_oversize_header_raises() -> None:
    """Length prefix exceeding MAX_MESSAGE_SIZE raises FrameTooLargeError."""
    reader = asyncio.StreamReader()
    reader.feed_data(struct.pack(">I", MAX_MESSAGE_SIZE + 1))
    reader.feed_eof()
    with pytest.raises(FrameTooLargeError):
        await read_frame(reader)


@pytest.mark.asyncio
async def test_async_back_to_back_frames() -> None:
    """Two frames concatenated on the wire come back as two reads."""
    reader = asyncio.StreamReader()
    p1 = b'{"first": 1}'
    p2 = b'{"second": 2}'
    reader.feed_data(struct.pack(">I", len(p1)) + p1)
    reader.feed_data(struct.pack(">I", len(p2)) + p2)
    reader.feed_eof()

    out1 = await read_frame(reader)
    out2 = await read_frame(reader)
    out3 = await read_frame(reader)
    assert out1 == '{"first": 1}'
    assert out2 == '{"second": 2}'
    assert out3 is None


# ----------------------------------------------------------------------
# Sync (blocking socket — runner-side code path)
# ----------------------------------------------------------------------


def test_sync_round_trip_ascii() -> None:
    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        write_frame_sync(a, '{"k": 1}')
        out = read_frame_sync(b)
        assert out == '{"k": 1}'
    finally:
        a.close()
        b.close()


def test_sync_round_trip_utf8() -> None:
    payload = '{"日本": "OK"}'
    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        write_frame_sync(a, payload)
        out = read_frame_sync(b)
        assert out == payload
    finally:
        a.close()
        b.close()


def test_sync_eof_between_frames_returns_none() -> None:
    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    a.close()
    out = read_frame_sync(b)
    assert out is None
    b.close()


def test_sync_eof_mid_frame_returns_none() -> None:
    """EOF mid-frame is benign on both sides per §6.7 — surfaces as None."""
    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    a.sendall(struct.pack(">I", 100))  # header claims 100 bytes
    a.close()                           # never sends payload
    out = read_frame_sync(b)
    assert out is None
    b.close()


def test_sync_oversize_header_raises() -> None:
    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        a.sendall(struct.pack(">I", MAX_MESSAGE_SIZE + 1))
        with pytest.raises(FrameTooLargeError):
            read_frame_sync(b)
    finally:
        a.close()
        b.close()


def test_sync_back_to_back_frames() -> None:
    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        write_frame_sync(a, '{"first": 1}')
        write_frame_sync(a, '{"second": 2}')
        a.shutdown(socket.SHUT_WR)
        out1 = read_frame_sync(b)
        out2 = read_frame_sync(b)
        out3 = read_frame_sync(b)
        assert out1 == '{"first": 1}'
        assert out2 == '{"second": 2}'
        assert out3 is None
    finally:
        a.close()
        b.close()


# ----------------------------------------------------------------------
# Cross-flavour: bytes-on-the-wire identical between async and sync
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_writer_async_reader_round_trip() -> None:
    """Bytes written via write_frame_sync are decoded by read_frame.

    Confirms wire compatibility — daemon writes async, runner reads
    sync, and vice versa, all over the same on-the-wire format.
    """
    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        write_frame_sync(a, '{"sync": "writes"}')
        a.shutdown(socket.SHUT_WR)
        # Drain the wire bytes through a StreamReader.
        reader = asyncio.StreamReader()
        reader.feed_data(b.recv(4096))
        reader.feed_eof()
        out = await read_frame(reader)
        assert out == '{"sync": "writes"}'
    finally:
        a.close()
        b.close()


@pytest.mark.asyncio
async def test_async_writer_sync_reader_round_trip() -> None:
    """Inverse cross-flavour: async-written bytes decode via sync reader."""
    writer = _CapturingWriter()
    await write_frame(writer, '{"async": "writes"}')

    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        a.sendall(bytes(writer.buf))
        a.shutdown(socket.SHUT_WR)
        out = read_frame_sync(b)
        assert out == '{"async": "writes"}'
    finally:
        a.close()
        b.close()
