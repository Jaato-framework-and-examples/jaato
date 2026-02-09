# shared/plugins/mermaid_formatter/backends/rich_pixels.py
"""Universal fallback backend using Unicode half-block characters.

Works on any terminal with 256-color or truecolor support.
Uses Pillow to resize the image and maps pixels to Unicode
half-block characters (▀▄█) with ANSI foreground/background colors.

No external dependencies beyond Pillow (which is a transitive dep
via several packages in requirements.txt).
"""

from io import BytesIO

from rich.console import Console
from rich.text import Text


class RichPixelsBackend:
    """Renders PNG images as Unicode half-block art."""

    def __init__(self, max_width: int = 80):
        self._max_width = max_width

    @property
    def name(self) -> str:
        return "rich_pixels"

    def render(self, png_data: bytes, max_width: int = 0) -> str:
        """Render PNG as Unicode half-block art.

        Each terminal cell encodes two vertical pixels using the upper
        half-block character (▀) with separate foreground (top pixel)
        and background (bottom pixel) colors.

        Args:
            png_data: Raw PNG image bytes.
            max_width: Override max width (0 = use instance default).

        Returns:
            ANSI-escaped string of Unicode half-block characters.
        """
        width = max_width or self._max_width

        try:
            from PIL import Image
        except ImportError:
            return "[mermaid: install Pillow for diagram rendering]\n"

        try:
            img = Image.open(BytesIO(png_data))
        except Exception:
            return "[mermaid: failed to decode image]\n"

        # Convert to RGBA for consistent handling
        img = img.convert("RGBA")

        # Resize to fit terminal width, maintaining aspect ratio.
        # Each terminal cell is roughly 1:2 (width:height), and we use
        # half-blocks to get 2 vertical pixels per cell.
        orig_w, orig_h = img.size
        scale = width / orig_w
        new_w = width
        new_h = int(orig_h * scale)

        # Ensure height is even for half-block pairing
        if new_h % 2 != 0:
            new_h += 1

        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        pixels = img.load()

        lines = []
        for y in range(0, new_h, 2):
            line = Text()
            for x in range(new_w):
                # Top pixel (foreground) and bottom pixel (background)
                r1, g1, b1, a1 = pixels[x, y]
                if y + 1 < new_h:
                    r2, g2, b2, a2 = pixels[x, y + 1]
                else:
                    r2, g2, b2, a2 = r1, g1, b1, a1

                # Handle transparency - blend with white background
                if a1 < 128:
                    r1, g1, b1 = 255, 255, 255
                if a2 < 128:
                    r2, g2, b2 = 255, 255, 255

                fg = f"rgb({r1},{g1},{b1})"
                bg = f"rgb({r2},{g2},{b2})"
                line.append("▀", style=f"{fg} on {bg}")
            lines.append(line)

        # Render to ANSI string
        console = Console(
            width=width + 2,
            force_terminal=True,
            no_color=False,
        )
        result_parts = []
        for line in lines:
            with console.capture() as capture:
                console.print(line, end="")
            result_parts.append(capture.get())

        return "\n".join(result_parts) + "\x1b[0m\n"
