"""Stream manager for coordinating streaming tool executions.

This module provides the StreamManager class which manages active streams,
coordinates chunk collection, and provides updates for session injection.
"""

import asyncio
import logging
import threading
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from .protocol import (
    StreamStatus,
    StreamChunk,
    StreamHandle,
    StreamState,
    StreamUpdate,
    StreamingCapable,
    ChunkCallback,
)

if TYPE_CHECKING:
    from ..registry import PluginRegistry

logger = logging.getLogger(__name__)


class StreamManager:
    """Manages active streaming tool executions.

    The StreamManager is responsible for:
    - Starting streaming tool executions
    - Collecting chunks as they arrive
    - Tracking which chunks have been delivered to the model
    - Providing updates when the model becomes idle
    - Handling dismiss requests

    Thread safety:
    - All state is protected by a lock
    - Async streaming runs in a dedicated event loop
    - Chunk callbacks are thread-safe
    """

    def __init__(self):
        """Initialize the stream manager."""
        self._streams: Dict[str, StreamState] = {}
        self._lock = threading.Lock()
        self._registry: Optional['PluginRegistry'] = None

        # Event loop for async streaming operations
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None

        # Callback for notifying session of new chunks
        self._on_chunks_available: Optional[Callable[[], None]] = None

        # Configuration
        self._initial_chunk_count = 5  # Chunks to return immediately
        self._chunk_wait_timeout = 2.0  # Seconds to wait for initial chunks

    def set_registry(self, registry: 'PluginRegistry') -> None:
        """Set the plugin registry for accessing streaming-capable plugins.

        Args:
            registry: The plugin registry.
        """
        self._registry = registry

    def set_chunks_available_callback(self, callback: Callable[[], None]) -> None:
        """Set callback to invoke when new chunks are available.

        This callback is used to notify the session that it should check
        for streaming updates (e.g., when the model is idle).

        Args:
            callback: Callback function (no arguments).
        """
        self._on_chunks_available = callback

    def _ensure_event_loop(self) -> asyncio.AbstractEventLoop:
        """Ensure the async event loop is running."""
        if self._loop is None or not self._loop.is_running():
            self._loop = asyncio.new_event_loop()

            def run_loop():
                asyncio.set_event_loop(self._loop)
                self._loop.run_forever()

            self._loop_thread = threading.Thread(target=run_loop, daemon=True)
            self._loop_thread.start()

        return self._loop

    def start_stream(
        self,
        plugin: StreamingCapable,
        plugin_name: str,
        tool_name: str,
        arguments: Dict[str, Any],
        call_id: str,
        on_ui_chunk: Optional[ChunkCallback] = None,
    ) -> StreamHandle:
        """Start a streaming tool execution.

        Args:
            plugin: The streaming-capable plugin.
            plugin_name: Name of the plugin.
            tool_name: Name of the tool (without :stream suffix).
            arguments: Arguments for the tool.
            call_id: The original function call ID.
            on_ui_chunk: Optional callback for UI display of chunks.

        Returns:
            StreamHandle with stream_id and initial chunks.
        """
        stream_id = f"stream_{uuid.uuid4().hex[:12]}"
        now = datetime.now()

        # Create stream state
        state = StreamState(
            stream_id=stream_id,
            plugin_name=plugin_name,
            tool_name=tool_name,
            call_id=call_id,
            created_at=now,
            status=StreamStatus.STARTING,
        )

        with self._lock:
            self._streams[stream_id] = state

        # Start async streaming in background
        loop = self._ensure_event_loop()

        async def collect_chunks():
            try:
                sequence = 0
                async for chunk in plugin.execute_streaming(tool_name, arguments):
                    # Assign sequence number if not set
                    if chunk.sequence == 0:
                        chunk.sequence = sequence
                        sequence += 1

                    with self._lock:
                        if stream_id not in self._streams:
                            # Stream was dismissed
                            break
                        stream = self._streams[stream_id]
                        if stream.status == StreamStatus.DISMISSED:
                            break
                        stream.chunks.append(chunk)
                        stream.status = StreamStatus.STREAMING

                    # Notify UI
                    if on_ui_chunk:
                        try:
                            on_ui_chunk(chunk)
                        except Exception as e:
                            logger.debug(f"UI chunk callback error: {e}")

                    # Notify session that chunks are available
                    if self._on_chunks_available:
                        try:
                            self._on_chunks_available()
                        except Exception as e:
                            logger.debug(f"Chunks available callback error: {e}")

                # Mark as completed
                with self._lock:
                    if stream_id in self._streams:
                        stream = self._streams[stream_id]
                        if stream.status != StreamStatus.DISMISSED:
                            stream.status = StreamStatus.COMPLETED
                            # Build final result from all chunks
                            stream.final_result = "\n".join(
                                c.content for c in stream.chunks if c.content
                            )

            except Exception as e:
                logger.exception(f"Stream {stream_id} failed: {e}")
                with self._lock:
                    if stream_id in self._streams:
                        stream = self._streams[stream_id]
                        stream.status = StreamStatus.FAILED
                        stream.error = str(e)

        # Schedule the coroutine
        asyncio.run_coroutine_threadsafe(collect_chunks(), loop)

        # Wait briefly for initial chunks
        initial_chunks = self._wait_for_initial_chunks(stream_id)

        with self._lock:
            state = self._streams[stream_id]
            return StreamHandle(
                stream_id=stream_id,
                plugin_name=plugin_name,
                tool_name=tool_name,
                created_at=now,
                initial_chunks=initial_chunks,
                status=state.status,
            )

    def _wait_for_initial_chunks(self, stream_id: str) -> List[StreamChunk]:
        """Wait briefly for initial chunks to arrive.

        Args:
            stream_id: The stream to wait for.

        Returns:
            List of initial chunks (may be empty).
        """
        import time
        start = time.time()
        chunks: List[StreamChunk] = []

        while time.time() - start < self._chunk_wait_timeout:
            with self._lock:
                if stream_id not in self._streams:
                    break
                state = self._streams[stream_id]

                # Get new chunks
                new_chunks = state.get_new_chunks()
                if new_chunks:
                    # Take up to initial_chunk_count
                    to_take = min(len(new_chunks), self._initial_chunk_count - len(chunks))
                    chunks.extend(new_chunks[:to_take])
                    state.mark_delivered(to_take)

                # Check if we have enough or stream is done
                if len(chunks) >= self._initial_chunk_count:
                    break
                if state.status in (StreamStatus.COMPLETED, StreamStatus.FAILED):
                    # Get any remaining chunks
                    remaining = state.get_new_chunks()
                    chunks.extend(remaining)
                    state.mark_delivered(len(remaining))
                    break

            time.sleep(0.05)  # Brief sleep to avoid busy-wait

        return chunks

    def get_pending_updates(self) -> List[StreamUpdate]:
        """Get updates from all active streams with pending chunks.

        Called by the session when the model becomes idle.

        Returns:
            List of StreamUpdate objects with new chunks.
        """
        updates: List[StreamUpdate] = []

        with self._lock:
            for stream_id, state in list(self._streams.items()):
                new_chunks = state.get_new_chunks()

                if new_chunks or state.status == StreamStatus.COMPLETED:
                    update = StreamUpdate(
                        stream_id=stream_id,
                        tool_name=state.tool_name,
                        new_chunks=list(new_chunks),
                        is_complete=state.status == StreamStatus.COMPLETED,
                        total_chunks=len(state.chunks),
                        final_result=state.final_result if state.status == StreamStatus.COMPLETED else None,
                    )
                    updates.append(update)

                    # Mark chunks as delivered
                    state.mark_delivered(len(new_chunks))

                    # Clean up completed streams after delivering final update
                    if state.status == StreamStatus.COMPLETED:
                        # Keep for a bit in case of retry, but mark as delivered
                        pass

        return updates

    def has_active_streams(self) -> bool:
        """Check if there are any active (non-complete) streams.

        Returns:
            True if there are active streams that may produce more chunks.
        """
        with self._lock:
            return any(state.is_active() for state in self._streams.values())

    def has_pending_chunks(self) -> bool:
        """Check if there are undelivered chunks in any stream.

        Returns:
            True if there are chunks waiting to be delivered.
        """
        with self._lock:
            for state in self._streams.values():
                if state.get_new_chunks():
                    return True
                # Also return True if stream just completed but not yet reported
                if state.status == StreamStatus.COMPLETED and state.chunks_delivered < len(state.chunks):
                    return True
            return False

    def wait_for_updates(self, timeout: float = 5.0) -> List[StreamUpdate]:
        """Wait for new chunks to arrive from active streams.

        Blocks until new chunks are available or timeout is reached.

        Args:
            timeout: Maximum time to wait in seconds.

        Returns:
            List of StreamUpdate objects (may be empty on timeout).
        """
        import time
        start = time.time()

        while time.time() - start < timeout:
            # Check for pending chunks
            if self.has_pending_chunks():
                return self.get_pending_updates()

            # Check if all streams are complete
            if not self.has_active_streams():
                # Return any final updates
                return self.get_pending_updates()

            time.sleep(0.1)

        # Timeout - return whatever we have
        return self.get_pending_updates()

    def dismiss_stream(self, stream_id: str) -> bool:
        """Dismiss a stream (stop receiving updates).

        The model calls this when it has seen enough chunks.

        Args:
            stream_id: Stream to dismiss, or "*" for all streams.

        Returns:
            True if stream(s) were dismissed.
        """
        with self._lock:
            if stream_id == "*":
                # Dismiss all active streams
                dismissed = False
                for state in self._streams.values():
                    if state.is_active():
                        state.status = StreamStatus.DISMISSED
                        dismissed = True
                return dismissed
            elif stream_id in self._streams:
                state = self._streams[stream_id]
                if state.is_active():
                    state.status = StreamStatus.DISMISSED
                    return True
            return False

    def get_stream_state(self, stream_id: str) -> Optional[StreamState]:
        """Get the current state of a stream.

        Args:
            stream_id: Stream to query.

        Returns:
            StreamState or None if not found.
        """
        with self._lock:
            return self._streams.get(stream_id)

    def list_active_streams(self) -> List[StreamState]:
        """List all active streams.

        Returns:
            List of active StreamState objects.
        """
        with self._lock:
            return [state for state in self._streams.values() if state.is_active()]

    def cleanup_completed(self, max_age_seconds: float = 300) -> int:
        """Clean up old completed/dismissed streams.

        Args:
            max_age_seconds: Remove streams older than this.

        Returns:
            Number of streams cleaned up.
        """
        import time
        now = datetime.now()
        cleaned = 0

        with self._lock:
            to_remove = []
            for stream_id, state in self._streams.items():
                if not state.is_active():
                    age = (now - state.created_at).total_seconds()
                    if age > max_age_seconds:
                        to_remove.append(stream_id)

            for stream_id in to_remove:
                del self._streams[stream_id]
                cleaned += 1

        return cleaned

    # ==================== Core Tool Interface ====================

    def get_tool_schemas(self) -> list:
        """Return tool schemas for streaming control.

        This allows the session to register dismiss_stream as a core tool
        without needing a separate plugin.

        Returns:
            List containing the dismiss_stream ToolSchema.
        """
        from ..model_provider.types import ToolSchema

        return [
            ToolSchema(
                name="dismiss_stream",
                description=(
                    "Stop receiving streaming updates from a tool. "
                    "Call this when you have enough results from a streaming tool "
                    "and don't need more updates. Use stream_id='*' to dismiss all active streams."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "stream_id": {
                            "type": "string",
                            "description": (
                                "ID of the stream to dismiss (from the streaming tool response), "
                                "or '*' to dismiss all active streams."
                            ),
                        },
                    },
                    "required": ["stream_id"],
                },
                category="system",
                discoverability="discoverable",
            ),
        ]

    def get_executors(self) -> dict:
        """Return executor functions for streaming control tools.

        Returns:
            Dict mapping tool name to executor function.
        """
        return {
            "dismiss_stream": self._execute_dismiss_stream,
        }

    def get_auto_approved_tools(self) -> list:
        """Return list of auto-approved tools.

        dismiss_stream is auto-approved because it's a control tool
        with no security implications - it just stops receiving updates.

        Returns:
            List of tool names that should be auto-approved.
        """
        return ["dismiss_stream"]

    def _execute_dismiss_stream(self, args: dict) -> dict:
        """Execute the dismiss_stream tool.

        Args:
            args: Tool arguments containing stream_id.

        Returns:
            Dict with success status and message.
        """
        stream_id = args.get("stream_id", "")
        if not stream_id:
            return {
                "success": False,
                "error": "stream_id is required",
            }

        dismissed = self.dismiss_stream(stream_id)

        if stream_id == "*":
            return {
                "success": True,
                "message": "All active streams dismissed" if dismissed else "No active streams to dismiss",
            }
        else:
            if dismissed:
                return {
                    "success": True,
                    "message": f"Stream {stream_id} dismissed",
                }
            else:
                return {
                    "success": False,
                    "error": f"Stream {stream_id} not found or already completed",
                }

    def shutdown(self) -> None:
        """Shutdown the stream manager and clean up resources."""
        # Dismiss all active streams
        self.dismiss_stream("*")

        # Stop the event loop
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)

        if self._loop_thread:
            self._loop_thread.join(timeout=2.0)

        self._loop = None
        self._loop_thread = None


__all__ = ['StreamManager']
