"""Streaming tool results infrastructure.

This module provides infrastructure for tools to stream incremental results
to the model, enabling reactive workflows where the model can act on partial
results while tools continue executing.

Key components:
- StreamingCapable: Protocol for plugins that support streaming
- StreamChunk: Individual chunk of streaming output
- StreamState: Tracks active streams and pending chunks
- StreamManager: Manages active streams and provides dismiss_stream tool
"""

from .protocol import (
    StreamStatus,
    StreamChunk,
    StreamHandle,
    StreamState,
    StreamUpdate,
    ChunkCallback,
    StreamingCapable,
)
from .manager import StreamManager

__all__ = [
    'StreamStatus',
    'StreamChunk',
    'StreamHandle',
    'StreamState',
    'StreamUpdate',
    'ChunkCallback',
    'StreamingCapable',
    'StreamManager',
]
