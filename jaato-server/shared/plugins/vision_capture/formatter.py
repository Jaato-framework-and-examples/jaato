# shared/plugins/vision_capture/formatter.py
"""Vision capture formatter plugin for the output pipeline.

This formatter observes output but passes it through unchanged.
It triggers captures on configurable events (turn end, periodic, etc.) via callbacks.
"""

import time
from typing import Callable, Iterator, Optional

from .protocol import CaptureContext


class VisionCaptureFormatter:
    """Formatter that observes output stream and triggers vision captures.

    This plugin sits at the end of the formatter pipeline (high priority)
    and observes output without modifying it. It can trigger captures
    on turn boundaries, at intervals during streaming, or other events.

    The actual capture is performed by the callback, which has access
    to the full rendered panel (not just the text stream).

    Capture modes:
    - Manual: Call trigger_capture() directly
    - Turn end: Auto-capture when flush() is called
    - Periodic: Capture every N milliseconds during streaming
    """

    name = "vision_capture"
    priority = 95  # Run near the end, after all formatting

    def __init__(
        self,
        capture_callback: Optional[Callable[[CaptureContext, int], None]] = None,
        auto_capture_on_turn_end: bool = False,
        capture_interval_ms: int = 0,
    ):
        """Initialize the vision capture formatter.

        Args:
            capture_callback: Called when a capture should be triggered.
                Signature: (context: CaptureContext, turn_index: int) -> None
            auto_capture_on_turn_end: If True, automatically trigger capture
                when flush() is called (end of model turn).
            capture_interval_ms: If > 0, capture periodically during streaming
                at this interval (milliseconds). 0 = disabled.
        """
        self._capture_callback = capture_callback
        self._auto_capture_on_turn_end = auto_capture_on_turn_end
        self._capture_interval_ms = capture_interval_ms
        self._turn_index = 0
        self._buffer: list[str] = []
        self._last_capture_time: float = 0
        self._periodic_capture_count: int = 0

    def set_capture_callback(
        self,
        callback: Callable[[CaptureContext, int], None]
    ) -> None:
        """Set or update the capture callback.

        Args:
            callback: Called when a capture should be triggered.
        """
        self._capture_callback = callback

    def set_auto_capture(self, enabled: bool) -> None:
        """Enable or disable automatic capture on turn end.

        Args:
            enabled: If True, capture automatically when turn ends.
        """
        self._auto_capture_on_turn_end = enabled

    def set_capture_interval(self, interval_ms: int) -> None:
        """Set periodic capture interval during streaming.

        Args:
            interval_ms: Capture every N milliseconds. 0 = disabled.
        """
        self._capture_interval_ms = interval_ms
        if interval_ms > 0:
            # Reset timer when enabling
            self._last_capture_time = time.time()
            self._periodic_capture_count = 0

    @property
    def capture_interval_ms(self) -> int:
        """Current capture interval in milliseconds."""
        return self._capture_interval_ms

    def process_chunk(self, chunk: str) -> Iterator[str]:
        """Process a chunk of output (pass through unchanged).

        Also checks if periodic capture is due.

        Args:
            chunk: Text chunk from the output stream.

        Yields:
            The same chunk, unchanged.
        """
        self._buffer.append(chunk)

        # Check for periodic capture
        if self._capture_interval_ms > 0 and self._capture_callback:
            now = time.time()
            elapsed_ms = (now - self._last_capture_time) * 1000
            if elapsed_ms >= self._capture_interval_ms:
                self._last_capture_time = now
                self._periodic_capture_count += 1
                self._capture_callback(CaptureContext.PERIODIC, self._turn_index)

        yield chunk

    def flush(self) -> Iterator[str]:
        """Called at turn end - trigger capture if auto-capture enabled.

        Yields:
            Nothing (no additional output).
        """
        if self._auto_capture_on_turn_end and self._capture_callback:
            self._capture_callback(CaptureContext.TURN_END, self._turn_index)

        self._turn_index += 1
        self._buffer = []
        # Reset periodic capture state for next turn
        self._periodic_capture_count = 0
        return
        yield  # Make this a generator

    def reset(self) -> None:
        """Reset state for new conversation."""
        self._buffer = []
        self._periodic_capture_count = 0
        # Don't reset turn_index - preserve across resets for context

    def trigger_capture(self, context: CaptureContext = CaptureContext.USER_REQUESTED) -> None:
        """Manually trigger a capture.

        Args:
            context: The context/reason for the capture.
        """
        if self._capture_callback:
            self._capture_callback(context, self._turn_index)

    @property
    def turn_index(self) -> int:
        """Current turn index."""
        return self._turn_index

    @property
    def periodic_capture_count(self) -> int:
        """Number of periodic captures in current turn."""
        return self._periodic_capture_count


def create_formatter(
    capture_callback: Optional[Callable[[CaptureContext, int], None]] = None,
    auto_capture_on_turn_end: bool = False,
    capture_interval_ms: int = 0,
) -> VisionCaptureFormatter:
    """Create a vision capture formatter.

    Args:
        capture_callback: Called when a capture should be triggered.
        auto_capture_on_turn_end: If True, automatically capture on turn end.
        capture_interval_ms: If > 0, capture periodically during streaming.

    Returns:
        Configured VisionCaptureFormatter instance.
    """
    return VisionCaptureFormatter(
        capture_callback=capture_callback,
        auto_capture_on_turn_end=auto_capture_on_turn_end,
        capture_interval_ms=capture_interval_ms,
    )
