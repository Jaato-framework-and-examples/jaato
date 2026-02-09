# shared/plugins/mermaid_formatter/backends/__init__.py
"""Terminal graphics backends for rendering Mermaid diagrams.

Auto-selects the best backend based on terminal capabilities detected
by shared.terminal_caps. Users can override via JAATO_MERMAID_BACKEND.

Backend priority:
    1. Kitty graphics protocol (kitty, ghostty)
    2. iTerm2 inline image protocol (iTerm2, WezTerm, mintty)
    3. Sixel bitmap protocol (foot, mlterm)
    4. rich-pixels Unicode half-block fallback (everywhere)
"""

import os
from typing import Optional, Protocol

from shared.terminal_caps import detect as detect_terminal_caps


class GraphicsBackend(Protocol):
    """Protocol for terminal graphics backends."""

    @property
    def name(self) -> str:
        """Backend identifier."""
        ...

    def render(self, png_data: bytes, max_width: int = 80) -> str:
        """Render PNG image data as terminal output.

        Args:
            png_data: Raw PNG image bytes.
            max_width: Maximum width in terminal columns.

        Returns:
            String containing terminal escape sequences or Unicode art
            that renders the image when printed.
        """
        ...


def select_backend(max_width: int = 80) -> GraphicsBackend:
    """Select the best available graphics backend.

    Checks JAATO_MERMAID_BACKEND env var first, then falls back to
    auto-detection via terminal_caps.

    Returns:
        A GraphicsBackend instance ready to render images.
    """
    # User override
    override = os.environ.get("JAATO_MERMAID_BACKEND", "").lower()
    if override == "off":
        # "off" means don't render diagrams at all - caller handles this
        from .rich_pixels import RichPixelsBackend
        return RichPixelsBackend(max_width)

    if override in ("kitty", "iterm", "sixel", "ascii"):
        return _create_backend(override, max_width)

    # Auto-detect from terminal capabilities
    caps = detect_terminal_caps()
    graphics = caps.get("graphics")

    if graphics:
        return _create_backend(graphics, max_width)

    # Universal fallback
    from .rich_pixels import RichPixelsBackend
    return RichPixelsBackend(max_width)


def _create_backend(protocol: str, max_width: int) -> GraphicsBackend:
    """Create a backend instance for the given protocol."""
    if protocol == "kitty":
        from .kitty import KittyBackend
        return KittyBackend(max_width)
    elif protocol == "iterm":
        from .iterm import ITermBackend
        return ITermBackend(max_width)
    elif protocol == "sixel":
        from .sixel import SixelBackend
        return SixelBackend(max_width)
    else:
        from .rich_pixels import RichPixelsBackend
        return RichPixelsBackend(max_width)
