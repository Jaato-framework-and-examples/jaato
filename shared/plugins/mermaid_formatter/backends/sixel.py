# shared/plugins/mermaid_formatter/backends/sixel.py
"""Sixel graphics protocol backend.

Renders PNG images inline using the Sixel bitmap protocol.
Supported by foot, mlterm, xterm (with sixel compiled), and others.

Sixel encodes images as 6-pixel-high rows using printable ASCII
characters, with a palette of up to 256 colors.

Protocol: https://en.wikipedia.org/wiki/Sixel
"""

from io import BytesIO


class SixelBackend:
    """Renders PNG images via Sixel escape sequences."""

    MAX_COLORS = 256

    def __init__(self, max_width: int = 80):
        self._max_width = max_width

    @property
    def name(self) -> str:
        return "sixel"

    def render(self, png_data: bytes, max_width: int = 0) -> str:
        """Render PNG as Sixel escape sequences.

        Converts the image to a 256-color palette, then encodes each
        6-pixel-high row as Sixel data.

        Args:
            png_data: Raw PNG image bytes.
            max_width: Override max width (0 = use instance default).

        Returns:
            String with Sixel escape sequences.
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

        # Resize to fit terminal width (~8px per column)
        max_px = width * 8
        if img.width > max_px:
            ratio = max_px / img.width
            new_w = int(img.width * ratio)
            new_h = int(img.height * ratio)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Convert to RGB and quantize to 256 colors
        img = img.convert("RGB")
        quantized = img.quantize(colors=self.MAX_COLORS)
        palette = quantized.getpalette()  # Flat list [R, G, B, R, G, B, ...]
        # get_flattened_data() is the non-deprecated replacement in Pillow 14+
        if hasattr(quantized, 'get_flattened_data'):
            pixels = list(quantized.get_flattened_data())
        else:
            pixels = list(quantized.getdata())
        w, h = quantized.size

        # Build Sixel output
        parts = []

        # DCS (Device Control String) introducer
        # P1=0 (pixel aspect 1:1), P2=1 (no background), P3=0 (horiz grid 0)
        parts.append("\x1bP0;1;0q")

        # Set raster attributes: width x height
        parts.append(f"\"1;1;{w};{h}")

        # Define color palette
        for i in range(self.MAX_COLORS):
            if palette and i * 3 + 2 < len(palette):
                r = int(palette[i * 3] / 255 * 100)
                g = int(palette[i * 3 + 1] / 255 * 100)
                b = int(palette[i * 3 + 2] / 255 * 100)
                parts.append(f"#{i};2;{r};{g};{b}")

        # Encode pixels in 6-row bands
        for band_y in range(0, h, 6):
            for color_idx in range(self.MAX_COLORS):
                # Check if this color appears in this band
                band_data = []
                has_color = False

                for x in range(w):
                    sixel_val = 0
                    for bit in range(6):
                        y = band_y + bit
                        if y < h:
                            px_idx = y * w + x
                            if px_idx < len(pixels) and pixels[px_idx] == color_idx:
                                sixel_val |= (1 << bit)
                                has_color = True
                    band_data.append(sixel_val)

                if not has_color:
                    continue

                # Select color and emit sixel data
                parts.append(f"#{color_idx}")

                # RLE-compress the band data
                parts.append(self._rle_encode(band_data))

                # Graphics carriage return (stay on same band for next color)
                parts.append("$")

            # Graphics new line (move to next 6-row band)
            parts.append("-")

        # String terminator
        parts.append("\x1b\\")

        return "".join(parts) + "\n"

    @staticmethod
    def _rle_encode(data: list) -> str:
        """RLE-compress sixel data.

        Sixel uses !N<char> for run-length encoding, where N is the
        repeat count and <char> is the sixel character (value + 63).
        """
        if not data:
            return ""

        parts = []
        i = 0
        while i < len(data):
            val = data[i]
            char = chr(val + 63)

            # Count consecutive identical values
            count = 1
            while i + count < len(data) and data[i + count] == val:
                count += 1

            if count >= 3:
                parts.append(f"!{count}{char}")
            else:
                parts.append(char * count)

            i += count

        return "".join(parts)
