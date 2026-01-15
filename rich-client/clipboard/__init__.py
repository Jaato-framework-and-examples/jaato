"""Clipboard module for rich-client.

Provides configurable clipboard mechanisms with chrome-free copy support.
"""

from .config import ClipboardConfig, ClipboardMechanism
from .protocol import ClipboardProvider
from .osc52 import OSC52Provider
from .native import NativeProvider
from .image import copy_image_to_clipboard


class AutoProvider:
    """Clipboard provider that auto-selects based on environment.

    - If DISPLAY is set: try native tools (xclip/xsel), fallback to OSC52
    - If no DISPLAY (SSH/tmux): use OSC52 directly (the right choice for terminals)
    """

    def __init__(self):
        import os
        self._native = NativeProvider()
        self._osc52 = OSC52Provider()
        # Prefer OSC52 when no display (SSH/tmux scenario)
        self._has_display = bool(os.environ.get("DISPLAY"))

    @property
    def name(self) -> str:
        if self._has_display and self._native.available:
            return f"Auto ({self._native.name})"
        return f"Auto ({self._osc52.name})"

    def copy(self, text: str) -> bool:
        """Use native if display available, otherwise OSC52."""
        if self._has_display and self._native.available:
            if self._native.copy(text):
                return True
            # Native failed, try OSC52
        # No display or native failed - use OSC52 (best for SSH/tmux)
        return self._osc52.copy(text)


def create_provider(config: ClipboardConfig) -> ClipboardProvider:
    """Factory to create clipboard provider from config."""
    if config.mechanism == ClipboardMechanism.OSC52:
        return OSC52Provider()
    elif config.mechanism == ClipboardMechanism.NATIVE:
        provider = NativeProvider()
        if not provider.available:
            raise ValueError("No native clipboard tool available (xclip/xsel/wl-copy)")
        return provider
    elif config.mechanism == ClipboardMechanism.AUTO:
        return AutoProvider()
    raise ValueError(f"Unknown clipboard mechanism: {config.mechanism}")


__all__ = [
    "ClipboardConfig",
    "ClipboardMechanism",
    "ClipboardProvider",
    "OSC52Provider",
    "NativeProvider",
    "create_provider",
    "copy_image_to_clipboard",
]
