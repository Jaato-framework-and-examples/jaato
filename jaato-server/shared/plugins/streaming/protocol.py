"""Protocol definitions for tool result streaming.

This module defines the StreamingCapable protocol that plugins can implement
to support streaming incremental results, along with supporting data structures.

Streaming tools allow the model to receive partial results as they're generated,
enabling reactive workflows where the model can act on early results while
the tool continues producing more.
"""

import asyncio
import base64
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    runtime_checkable,
)


class StreamStatus(Enum):
    """Status of a streaming tool execution."""
    STARTING = "starting"     # Stream is being set up
    STREAMING = "streaming"   # Actively producing chunks
    PAUSED = "paused"         # Temporarily paused
    COMPLETED = "completed"   # Finished successfully
    FAILED = "failed"         # Finished with error
    DISMISSED = "dismissed"   # Model dismissed the stream


class Audience(str, Enum):
    """Who a chunk is *for*.

    Audience selects whether a chunk's payload enters THIS session's
    conversation history -- not whether the event is published.  Every
    chunk is published to all three subscription surfaces (SDK client,
    ``subscribeToEvents`` agent tool, in-process ``EventBus``) regardless;
    a parent agent watching a child therefore sees a ``CLIENT`` chunk, and
    is subject to its own modality gate if it wants to consume the bytes.

    ``str`` mixin so the value serializes as a plain string in JSON frames
    and compares equal to its wire form.

    Attributes:
        MODEL: Into the conversation.  The historical behaviour of tool
            streaming (wrapped in ``<hidden>`` so the model reads it and
            the user does not).  Default, so existing producers are
            unchanged.
        CLIENT: To subscribed clients only; never enters history.  What
            model-unconsumable media becomes -- a TTS tool's audio, or an
            attachment the modality gate withheld.
        BOTH: Into the conversation *and* to clients.  A screenshot a
            vision tier can read and a person may also want to look at.
    """
    MODEL = "model"
    CLIENT = "client"
    BOTH = "both"

    def reaches_model(self) -> bool:
        """Whether a chunk with this audience enters the conversation."""
        return self in (Audience.MODEL, Audience.BOTH)

    def reaches_client(self) -> bool:
        """Whether a chunk with this audience is rendered for a viewer."""
        return self in (Audience.CLIENT, Audience.BOTH)


@dataclass
class StreamChunk:
    """A single chunk of incremental output from a streaming tool.

    A chunk carries text, binary content, or both.  ``inline_data``
    deliberately mirrors :attr:`Part.inline_data` so one ``{mime_type,
    data}`` shape is used on every path -- inbound message parts, tool
    result attachments, and chunks -- rather than a parallel media type.

    New fields are appended, so positional construction
    (``StreamChunk("text", "match", 5)``) is unchanged, as is every
    existing producer: a text-only chunk leaves ``inline_data`` ``None``
    and takes the default ``MODEL`` audience.

    Attributes:
        content: The actual content of this chunk.  Defaults to ``""`` so
            a pure-media chunk need not pass an empty string; for a media
            chunk this is an optional human-readable label, not the payload.
        chunk_type: Type hint for the chunk (e.g., "match", "progress", "result").
        sequence: Sequence number for ordering (auto-assigned if not provided).
        timestamp: When this chunk was produced.
        metadata: Optional additional metadata about this chunk.
        inline_data: Optional binary payload as ``{"mime_type": str,
            "data": bytes}``.  When set, consumers MUST route the chunk
            around any text formatter -- a formatter that reflows text
            corrupts bytes (see ``server/core.py`` ``on_tool_output``).
        audience: Who the chunk is for; see :class:`Audience`.  Defaults
            to ``MODEL``, preserving the historical behaviour exactly.
    """
    content: str = ""
    chunk_type: str = "result"
    sequence: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = None
    inline_data: Optional[Dict[str, Any]] = None
    audience: Audience = Audience.MODEL

    def is_media(self) -> bool:
        """Whether this chunk carries a binary payload.

        The single predicate every routing decision should use, so
        "has bytes" is defined in one place rather than re-derived as
        ``chunk.inline_data and chunk.inline_data.get("data")`` at each
        call site.
        """
        return bool(self.inline_data and self.inline_data.get("data"))

    @property
    def mime_type(self) -> Optional[str]:
        """MIME type of the binary payload, or ``None`` for a text chunk."""
        if not self.inline_data:
            return None
        return self.inline_data.get("mime_type")

    def data_b64(self) -> Optional[str]:
        """Binary payload base64-encoded for a UTF-8 JSON frame.

        Returns ``None`` when the chunk carries no bytes.  Accepts ``str``
        in ``inline_data["data"]`` as already-encoded base64 and passes it
        through, so a producer that encoded upstream is not double-encoded.
        """
        if not self.is_media():
            return None
        data = self.inline_data["data"]
        if isinstance(data, str):
            return data
        return base64.b64encode(data).decode("ascii")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Binary payloads are base64-encoded (+33% size) because the frame is
        UTF-8 JSON.  ``inline_data`` is omitted entirely for a text-only
        chunk rather than serialized as ``None``, keeping the common frame
        byte-identical to before this field existed.
        """
        out: Dict[str, Any] = {
            "content": self.content,
            "chunk_type": self.chunk_type,
            "sequence": self.sequence,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "audience": self.audience.value,
        }
        if self.is_media():
            out["inline_data"] = {
                "mime_type": self.mime_type,
                "data": self.data_b64(),
            }
        return out


@dataclass
class StreamHandle:
    """Handle returned when a streaming tool is started.

    Attributes:
        stream_id: Unique identifier for this stream.
        plugin_name: Name of the plugin that owns this stream.
        tool_name: Name of the tool being executed.
        created_at: Timestamp when the stream was created.
        initial_chunks: First batch of chunks (returned immediately).
        status: Current status of the stream.
    """
    stream_id: str
    plugin_name: str
    tool_name: str
    created_at: datetime
    initial_chunks: List[StreamChunk] = field(default_factory=list)
    status: StreamStatus = StreamStatus.STARTING

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "stream_id": self.stream_id,
            "plugin_name": self.plugin_name,
            "tool_name": self.tool_name,
            "created_at": self.created_at.isoformat(),
            "initial_chunks": [c.to_dict() for c in self.initial_chunks],
            "status": self.status.value,
        }


@dataclass
class StreamState:
    """Current state of an active stream.

    Used to track streaming tools and collect updates for injection
    into the conversation when the model becomes idle.

    Attributes:
        stream_id: Unique identifier for this stream.
        plugin_name: Name of the plugin that owns this stream.
        tool_name: Name of the tool being executed.
        call_id: The original function call ID (for correlation).
        created_at: When the stream was started.
        chunks: All chunks received so far.
        chunks_delivered: Number of chunks already delivered to model.
        status: Current status of the stream.
        final_result: Final aggregated result (when completed).
        error: Error message if stream failed.
    """
    stream_id: str
    plugin_name: str
    tool_name: str
    call_id: str
    created_at: datetime
    chunks: List[StreamChunk] = field(default_factory=list)
    chunks_delivered: int = 0
    status: StreamStatus = StreamStatus.STREAMING
    final_result: Optional[str] = None
    error: Optional[str] = None

    def get_new_chunks(self) -> List[StreamChunk]:
        """Get chunks that haven't been delivered to the model yet."""
        return self.chunks[self.chunks_delivered:]

    def mark_delivered(self, count: int) -> None:
        """Mark chunks as delivered to the model."""
        self.chunks_delivered += count

    def is_active(self) -> bool:
        """Check if the stream is still active (may produce more chunks)."""
        return self.status in (StreamStatus.STARTING, StreamStatus.STREAMING, StreamStatus.PAUSED)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "stream_id": self.stream_id,
            "plugin_name": self.plugin_name,
            "tool_name": self.tool_name,
            "call_id": self.call_id,
            "created_at": self.created_at.isoformat(),
            "total_chunks": len(self.chunks),
            "chunks_delivered": self.chunks_delivered,
            "chunks_pending": len(self.chunks) - self.chunks_delivered,
            "status": self.status.value,
            "has_final_result": self.final_result is not None,
            "error": self.error,
        }


@dataclass
class StreamUpdate:
    """An update from one or more active streams.

    Collected and injected into the conversation when the model is idle.

    Attributes:
        stream_id: Which stream this update is from.
        tool_name: Name of the tool producing chunks.
        new_chunks: New chunks since last delivery.
        is_complete: Whether the stream has finished.
        total_chunks: Total chunks produced so far.
        final_result: Aggregated result if stream is complete.
    """
    stream_id: str
    tool_name: str
    new_chunks: List[StreamChunk]
    is_complete: bool
    total_chunks: int
    final_result: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "stream_id": self.stream_id,
            "tool_name": self.tool_name,
            "new_chunks": [c.to_dict() for c in self.new_chunks],
            "is_complete": self.is_complete,
            "total_chunks": self.total_chunks,
            "final_result": self.final_result,
        }


# Type alias for the chunk callback
ChunkCallback = Callable[[StreamChunk], None]


@runtime_checkable
class StreamingCapable(Protocol):
    """Protocol for plugins that support streaming tool results.

    Plugins implementing this protocol can stream incremental results
    to the model, enabling reactive workflows where the model acts on
    partial results while the tool continues executing.

    The streaming flow:
    1. Model calls tool_name:stream (auto-generated variant)
    2. Plugin starts streaming execution
    3. Initial chunks are collected and returned immediately
    4. Stream continues in background, collecting more chunks
    5. When model becomes idle, new chunks are injected
    6. Model can call dismiss_stream to stop receiving updates
    7. Stream completes when tool finishes or is dismissed

    Implementation notes:
    - execute_streaming() should be an async generator yielding StreamChunks
    - Plugins should handle cancellation gracefully
    - The final chunk can include aggregated results in metadata
    """

    def supports_streaming(self, tool_name: str) -> bool:
        """Check if a specific tool supports streaming execution.

        Not all tools benefit from streaming. Tools that produce results
        incrementally (like grep, file search) are good candidates.
        Tools that produce atomic results (like file write) are not.

        Args:
            tool_name: Name of the tool to check.

        Returns:
            True if the tool supports streaming execution.
        """
        ...

    async def execute_streaming(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        on_chunk: Optional[ChunkCallback] = None,
    ) -> AsyncIterator[StreamChunk]:
        """Execute a tool and yield incremental results.

        This method is an async generator that yields StreamChunks as
        they become available. The final chunk should have metadata
        indicating completion.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Arguments to pass to the tool.
            on_chunk: Optional callback invoked for each chunk (for UI).

        Yields:
            StreamChunk objects as results become available.

        Raises:
            ValueError: If tool doesn't support streaming.
            RuntimeError: If streaming execution fails.
        """
        ...
        # Make this a generator (required for Protocol)
        if False:
            yield StreamChunk(content="")

    def get_streaming_tool_names(self) -> List[str]:
        """Get list of tools that support streaming.

        Used by the registry to auto-generate :stream variants.

        Returns:
            List of tool names that support streaming.
        """
        ...


__all__ = [
    'StreamStatus',
    'StreamChunk',
    'StreamHandle',
    'StreamState',
    'StreamUpdate',
    'ChunkCallback',
    'StreamingCapable',
]
