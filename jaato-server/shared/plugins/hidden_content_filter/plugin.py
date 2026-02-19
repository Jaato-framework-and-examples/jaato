# shared/plugins/hidden_content_filter/plugin.py
"""Streaming hidden content filter plugin.

This plugin strips <hidden>...</hidden> tagged content from the output
stream. It runs very early in the pipeline (priority 5) to ensure hidden
content is removed before any other formatting is applied.

The plugin handles streaming by buffering when it sees a partial <hidden>
tag, ensuring content is only stripped when complete tags are found.

Usage:
    from shared.plugins.hidden_content_filter import create_plugin

    formatter = create_plugin()

    # Streaming mode
    for chunk in model_output:
        for output in formatter.process_chunk(chunk):
            print(output, end='')
    for output in formatter.flush():
        print(output, end='')
"""

import re
from typing import Iterator


# Priority for pipeline ordering (0-19 = pre-processing)
DEFAULT_PRIORITY = 5

# Pattern to match complete <hidden>...</hidden> tags
_HIDDEN_PATTERN = re.compile(r'<hidden>.*?</hidden>', re.DOTALL)

# Pattern to detect partial opening tag at end of buffer
_PARTIAL_OPEN_PATTERN = re.compile(r'<(?:h(?:i(?:d(?:d(?:e(?:n)?)?)?)?)?)?$')

# Pattern to detect we're inside an unclosed hidden tag
_OPEN_TAG = '<hidden>'
_CLOSE_TAG = '</hidden>'


class HiddenContentFilterPlugin:
    """Streaming plugin that filters out <hidden>...</hidden> content.

    Implements the FormatterPlugin protocol. Buffers content when partial
    hidden tags are detected to ensure complete tags are stripped.
    """

    def __init__(self):
        self._priority = DEFAULT_PRIORITY
        self._buffer = ""
        self._in_hidden = False  # True when we've seen <hidden> but not </hidden>

    # ==================== FormatterPlugin Protocol ====================

    @property
    def name(self) -> str:
        """Unique identifier for this formatter."""
        return "hidden_content_filter"

    @property
    def priority(self) -> int:
        """Execution priority (5 = pre-processing, runs early)."""
        return self._priority

    def process_chunk(self, chunk: str) -> Iterator[str]:
        """Process a chunk, stripping hidden content.

        Handles streaming by buffering when partial tags are detected.
        """
        self._buffer += chunk

        while True:
            if self._in_hidden:
                # We're inside a hidden block, look for closing tag
                close_idx = self._buffer.find(_CLOSE_TAG)
                if close_idx == -1:
                    # No closing tag yet, keep buffering (discard hidden content)
                    # But check if we might have partial </hidden> at the end
                    # Keep last 8 chars in case of partial closing tag
                    if len(self._buffer) > 9:
                        self._buffer = self._buffer[-9:]
                    return
                else:
                    # Found closing tag, discard everything up to and including it
                    self._buffer = self._buffer[close_idx + len(_CLOSE_TAG):]
                    self._in_hidden = False
                    # Continue processing remaining buffer
            else:
                # Not in hidden block, look for opening tag
                open_idx = self._buffer.find(_OPEN_TAG)
                if open_idx == -1:
                    # No opening tag, but check for partial tag at end
                    match = _PARTIAL_OPEN_PATTERN.search(self._buffer)
                    if match:
                        # Potential partial tag at end, yield everything before it
                        safe_text = self._buffer[:match.start()]
                        self._buffer = self._buffer[match.start():]
                        if safe_text:
                            yield safe_text
                    else:
                        # No partial tag, yield entire buffer
                        if self._buffer:
                            yield self._buffer
                        self._buffer = ""
                    return
                else:
                    # Found opening tag
                    # Yield text before the hidden tag
                    if open_idx > 0:
                        yield self._buffer[:open_idx]
                    # Enter hidden mode and continue processing
                    self._buffer = self._buffer[open_idx + len(_OPEN_TAG):]
                    self._in_hidden = True
                    # Continue loop to look for closing tag

    def flush(self) -> Iterator[str]:
        """Flush any remaining buffered content.

        If we're still inside an unclosed hidden tag, the content is discarded.
        Any remaining non-hidden content is yielded.
        """
        if self._in_hidden:
            # Unclosed hidden tag - discard the buffered hidden content
            self._buffer = ""
            self._in_hidden = False
        elif self._buffer:
            # Remaining content (possibly partial tag that never completed)
            yield self._buffer
            self._buffer = ""

    def reset(self) -> None:
        """Reset state for a new turn."""
        self._buffer = ""
        self._in_hidden = False


def create_plugin() -> HiddenContentFilterPlugin:
    """Factory function to create a HiddenContentFilterPlugin instance."""
    return HiddenContentFilterPlugin()
