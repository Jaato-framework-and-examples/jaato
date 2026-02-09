# shared/plugins/mermaid_formatter/backends/kitty.py
"""Kitty graphics protocol backend.

Renders PNG images inline using Kitty's terminal graphics protocol.
Supported by Kitty and Ghostty terminals.

Protocol: https://sw.kovidgoyal.net/kitty/graphics-protocol/
"""

import base64
from io import BytesIO


class KittyBackend:
    """Renders PNG images via Kitty graphics protocol."""

    CHUNK_SIZE = 4096  # Max base64 payload per chunk

    def __init__(self, max_width: int = 80):
        self._max_width = max_width

    @property
    def name(self) -> str:
        return "kitty"

    def render(self, png_data: bytes, max_width: int = 0) -> str:
        """Render PNG using Kitty graphics protocol.

        Sends the image in chunks via APC escape sequences. The terminal
        handles scaling and display.

        The protocol uses:
            ESC_G<payload>ESC\\
        where payload is key=value pairs followed by ;base64data

        Args:
            png_data: Raw PNG image bytes.
            max_width: Override max width (0 = use instance default).

        Returns:
            String with Kitty graphics protocol escape sequences.
        """
        width = max_width or self._max_width

        try:
            png_data = self._resize_if_needed(png_data, width)
        except Exception:
            pass  # Send original if resize fails

        encoded = base64.standard_b64encode(png_data).decode("ascii")
        chunks = [encoded[i:i + self.CHUNK_SIZE]
                  for i in range(0, len(encoded), self.CHUNK_SIZE)]

        parts = []
        for i, chunk in enumerate(chunks):
            is_last = (i == len(chunks) - 1)
            # m=0 for last chunk, m=1 for "more data follows"
            m = 0 if is_last else 1

            if i == 0:
                # First chunk: include format and action
                # f=100 = PNG, a=T = transmit and display, c/r = columns/rows
                parts.append(f"\x1b_Gf=100,a=T,m={m};{chunk}\x1b\\")
            else:
                # Continuation chunks
                parts.append(f"\x1b_Gm={m};{chunk}\x1b\\")

        return "".join(parts) + "\n"

    def _resize_if_needed(self, png_data: bytes, max_cols: int) -> bytes:
        """Resize image if it exceeds the target column width.

        Kitty displays images at their pixel size, so we resize to
        roughly fit the terminal column width. Assumes ~8px per column.
        """
        try:
            from PIL import Image
        except ImportError:
            return png_data

        img = Image.open(BytesIO(png_data))
        max_px = max_cols * 8  # Approximate pixels per column

        if img.width <= max_px:
            return png_data

        ratio = max_px / img.width
        new_w = int(img.width * ratio)
        new_h = int(img.height * ratio)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
