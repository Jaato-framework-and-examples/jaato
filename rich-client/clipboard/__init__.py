"""Clipboard module for rich-client.

Provides configurable clipboard mechanisms with chrome-free copy support.
"""

from .config import ClipboardConfig, ClipboardMechanism
from .protocol import ClipboardProvider
from .osc52 import OSC52Provider


def create_provider(config: ClipboardConfig) -> ClipboardProvider:
    """Factory to create clipboard provider from config."""
    if config.mechanism == ClipboardMechanism.OSC52:
        return OSC52Provider()
    raise ValueError(f"Unknown clipboard mechanism: {config.mechanism}")


__all__ = [
    "ClipboardConfig",
    "ClipboardMechanism",
    "ClipboardProvider",
    "OSC52Provider",
    "create_provider",
]
