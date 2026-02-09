# shared/plugins/mermaid_formatter/backends/iterm.py
"""iTerm2 inline image protocol backend.

Renders PNG images inline using iTerm2's proprietary escape sequence.
Supported by iTerm2, WezTerm, and Mintty.

Protocol: https://iterm2.com/documentation-images.html
"""

import base64
from io import BytesIO


class ITermBackend:
    """Renders PNG images via iTerm2 inline image protocol."""

    def __init__(self, max_width: int = 80):
        self._max_width = max_width

    @property
    def name(self) -> str:
        return "iterm"

    def render(self, png_data: bytes, max_width: int = 0) -> str:
        """Render PNG using iTerm2 inline image protocol.

        Format:
            OSC 1337 ; File=[args] : base64data BEL
        Where args include:
            inline=1     - display inline (vs download)
            size=N       - file size in bytes
            width=Ncols  - display width in terminal columns
            preserveAspectRatio=1

        Args:
            png_data: Raw PNG image bytes.
            max_width: Override max width (0 = use instance default).

        Returns:
            String with iTerm2 inline image escape sequence.
        """
        width = max_width or self._max_width

        try:
            png_data = self._resize_if_needed(png_data, width)
        except Exception:
            pass

        encoded = base64.standard_b64encode(png_data).decode("ascii")
        size = len(png_data)

        # OSC 1337 with ST (string terminator) for wider compatibility
        args = (
            f"inline=1"
            f";size={size}"
            f";width={width}"
            f";preserveAspectRatio=1"
        )
        return f"\x1b]1337;File={args}:{encoded}\x07\n"

    def _resize_if_needed(self, png_data: bytes, max_cols: int) -> bytes:
        """Resize image to fit terminal width."""
        try:
            from PIL import Image
        except ImportError:
            return png_data

        img = Image.open(BytesIO(png_data))
        max_px = max_cols * 8

        if img.width <= max_px:
            return png_data

        ratio = max_px / img.width
        new_w = int(img.width * ratio)
        new_h = int(img.height * ratio)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
