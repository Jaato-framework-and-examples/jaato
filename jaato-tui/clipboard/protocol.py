"""Clipboard provider protocol."""

from typing import Protocol


class ClipboardProvider(Protocol):
    """Protocol for clipboard implementations."""

    def copy(self, text: str) -> bool:
        """Copy text to clipboard.

        Args:
            text: The text to copy.

        Returns:
            Success hint (may be unreliable for some mechanisms).
        """
        ...

    @property
    def name(self) -> str:
        """Human-readable name for status messages."""
        ...
