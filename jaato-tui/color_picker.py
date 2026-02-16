"""Color picker widget for theme editor.

Provides an RGB slider-based color picker with hex input support.
"""

from enum import Enum
from typing import Optional, Tuple

from rich.text import Text


class ColorChannel(Enum):
    """RGB color channels."""
    RED = 0
    GREEN = 1
    BLUE = 2


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color to RGB tuple.

    Args:
        hex_color: Hex color string like "#RRGGBB".

    Returns:
        Tuple of (red, green, blue) values 0-255.
    """
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_hex(r: int, g: int, b: int) -> str:
    """Convert RGB values to hex color.

    Args:
        r: Red value 0-255.
        g: Green value 0-255.
        b: Blue value 0-255.

    Returns:
        Hex color string like "#RRGGBB".
    """
    return f"#{r:02X}{g:02X}{b:02X}"


class ColorPicker:
    """RGB slider widget for color selection.

    Provides three sliders (R/G/B) with visual feedback and hex input.

    Usage:
        picker = ColorPicker("#5fd7ff")
        text = picker.render()  # Rich Text for display

        # Handle key events
        result = picker.handle_key("right")  # Increase current channel
        result = picker.handle_key("up")     # Increase by 5
        result = picker.handle_key("tab")    # Switch channel
        result = picker.handle_key("enter")  # Confirm -> returns hex color
        result = picker.handle_key("escape") # Cancel -> returns None
    """

    SLIDER_WIDTH = 16  # Width of the slider bar

    def __init__(self, initial_color: str):
        """Initialize color picker with a starting color.

        Args:
            initial_color: Initial hex color (e.g., "#5fd7ff").
        """
        self._original_color = initial_color
        self._r, self._g, self._b = hex_to_rgb(initial_color)
        self._channel = ColorChannel.RED
        self._hex_input_mode = False
        self._hex_input_buffer = ""

    @property
    def current_color(self) -> str:
        """Get current color as hex string."""
        return rgb_to_hex(self._r, self._g, self._b)

    @property
    def original_color(self) -> str:
        """Get original color as hex string."""
        return self._original_color

    @property
    def is_modified(self) -> bool:
        """Check if color has been modified from original."""
        return self.current_color != self._original_color

    def set_channel(self, channel: ColorChannel) -> None:
        """Set the active channel.

        Args:
            channel: ColorChannel to make active.
        """
        self._channel = channel

    def get_channel_value(self, channel: ColorChannel) -> int:
        """Get value for a specific channel.

        Args:
            channel: ColorChannel to query.

        Returns:
            Value 0-255.
        """
        if channel == ColorChannel.RED:
            return self._r
        elif channel == ColorChannel.GREEN:
            return self._g
        else:
            return self._b

    def set_channel_value(self, channel: ColorChannel, value: int) -> None:
        """Set value for a specific channel.

        Args:
            channel: ColorChannel to modify.
            value: Value 0-255 (clamped if out of range).
        """
        value = max(0, min(255, value))
        if channel == ColorChannel.RED:
            self._r = value
        elif channel == ColorChannel.GREEN:
            self._g = value
        else:
            self._b = value

    def _adjust_current_channel(self, delta: int) -> None:
        """Adjust current channel by delta.

        Args:
            delta: Amount to add (can be negative).
        """
        current = self.get_channel_value(self._channel)
        self.set_channel_value(self._channel, current + delta)

    def _next_channel(self) -> None:
        """Cycle to next channel."""
        channels = [ColorChannel.RED, ColorChannel.GREEN, ColorChannel.BLUE]
        idx = channels.index(self._channel)
        self._channel = channels[(idx + 1) % 3]

    def _prev_channel(self) -> None:
        """Cycle to previous channel."""
        channels = [ColorChannel.RED, ColorChannel.GREEN, ColorChannel.BLUE]
        idx = channels.index(self._channel)
        self._channel = channels[(idx - 1) % 3]

    def render(self, width: int = 30) -> Text:
        """Render the color picker as Rich Text.

        Args:
            width: Width of the widget.

        Returns:
            Rich Text object with the color picker UI.
        """
        output = Text()

        # Color swatch comparison (original vs current)
        output.append("Original: ")
        output.append("\u2588\u2588\u2588\u2588", style=self._original_color)
        output.append("  Current: ")
        output.append("\u2588\u2588\u2588\u2588", style=self.current_color)
        output.append("\n\n")

        # RGB sliders
        for channel in [ColorChannel.RED, ColorChannel.GREEN, ColorChannel.BLUE]:
            is_selected = channel == self._channel
            value = self.get_channel_value(channel)

            # Channel label
            label = channel.name[0]
            if is_selected:
                output.append(f"[{label}]", style="bold reverse")
            else:
                output.append(f" {label} ", style="dim")

            output.append(" ")

            # Slider bar
            filled = int((value / 255) * self.SLIDER_WIDTH)
            empty = self.SLIDER_WIDTH - filled

            # Color the slider based on channel
            if channel == ColorChannel.RED:
                fill_color = "#ff0000"
            elif channel == ColorChannel.GREEN:
                fill_color = "#00ff00"
            else:
                fill_color = "#0000ff"

            output.append("[")
            output.append("\u2588" * filled, style=fill_color)
            output.append("\u2591" * empty, style="dim")
            output.append("]")

            # Value
            output.append(f" {value:3d}", style="bold" if is_selected else "")
            output.append("\n")

        # Hex display
        output.append("\nHex: ")
        output.append(self.current_color, style=f"bold {self.current_color}")
        output.append("\n")

        # Instructions
        output.append("\n")
        output.append("\u2190\u2192", style="bold")
        output.append(" adjust  ")
        output.append("Tab", style="bold")
        output.append(" channel  ")
        output.append("Enter", style="bold")
        output.append(" confirm  ")
        output.append("Esc", style="bold")
        output.append(" cancel")

        return output

    def render_compact(self) -> Text:
        """Render a compact single-line color picker.

        Returns:
            Rich Text with compact color display.
        """
        output = Text()

        # Color swatch
        output.append("\u2588\u2588", style=self.current_color)
        output.append(" ")

        # RGB values
        output.append(f"R:{self._r:3d} ", style="#ff0000" if self._channel == ColorChannel.RED else "dim")
        output.append(f"G:{self._g:3d} ", style="#00ff00" if self._channel == ColorChannel.GREEN else "dim")
        output.append(f"B:{self._b:3d} ", style="#0000ff" if self._channel == ColorChannel.BLUE else "dim")

        # Hex
        output.append(self.current_color, style="bold")

        return output

    def handle_key(self, key: str) -> Optional[str]:
        """Handle a key press.

        Args:
            key: Key name (e.g., "left", "right", "up", "down", "tab", "enter", "escape").

        Returns:
            - Hex color string if confirmed (enter pressed)
            - None if cancelled (escape pressed)
            - Empty string "" if key was handled but no result yet
            - The key string unchanged if not handled
        """
        key = key.lower()

        if key in ("left", "c-b"):
            # Decrease current channel
            self._adjust_current_channel(-1)
            return ""

        elif key in ("right", "c-f"):
            # Increase current channel
            self._adjust_current_channel(1)
            return ""

        elif key in ("up", "c-p"):
            # Increase by 5
            self._adjust_current_channel(5)
            return ""

        elif key in ("down", "c-n"):
            # Decrease by 5
            self._adjust_current_channel(-5)
            return ""

        elif key == "tab":
            # Cycle to next channel
            self._next_channel()
            return ""

        elif key == "s-tab":
            # Cycle to previous channel
            self._prev_channel()
            return ""

        elif key == "pageup":
            # Large increase (+25)
            self._adjust_current_channel(25)
            return ""

        elif key == "pagedown":
            # Large decrease (-25)
            self._adjust_current_channel(-25)
            return ""

        elif key == "home":
            # Set to 0
            self.set_channel_value(self._channel, 0)
            return ""

        elif key == "end":
            # Set to 255
            self.set_channel_value(self._channel, 255)
            return ""

        elif key == "enter":
            # Confirm selection
            return self.current_color

        elif key == "escape":
            # Cancel - return None to indicate cancellation
            return None

        elif key == "r":
            # Reset to original
            self._r, self._g, self._b = hex_to_rgb(self._original_color)
            return ""

        # Not handled
        return key

    def set_color(self, hex_color: str) -> bool:
        """Set the current color.

        Args:
            hex_color: Hex color string like "#RRGGBB".

        Returns:
            True if valid color, False otherwise.
        """
        if not hex_color.startswith("#") or len(hex_color) != 7:
            return False
        try:
            self._r, self._g, self._b = hex_to_rgb(hex_color)
            return True
        except (ValueError, IndexError):
            return False

    def reset(self) -> None:
        """Reset to original color."""
        self._r, self._g, self._b = hex_to_rgb(self._original_color)
        self._channel = ColorChannel.RED


class HexColorInput:
    """Direct hex color input field.

    Allows typing hex colors directly with validation.
    """

    def __init__(self, initial_color: str = "#000000"):
        """Initialize hex input.

        Args:
            initial_color: Initial hex color.
        """
        self._buffer = initial_color
        self._cursor = len(initial_color)

    @property
    def value(self) -> str:
        """Get current buffer value."""
        return self._buffer

    @property
    def is_valid(self) -> bool:
        """Check if current buffer is a valid hex color."""
        if not self._buffer.startswith("#"):
            return False
        hex_part = self._buffer[1:]
        if len(hex_part) != 6:
            return False
        try:
            int(hex_part, 16)
            return True
        except ValueError:
            return False

    def render(self) -> Text:
        """Render the input field.

        Returns:
            Rich Text with input display.
        """
        output = Text()

        # Input field with cursor
        output.append("Hex: ")

        for i, char in enumerate(self._buffer):
            if i == self._cursor:
                output.append(char, style="reverse")
            else:
                output.append(char)

        if self._cursor == len(self._buffer):
            output.append(" ", style="reverse")

        # Validation indicator
        output.append("  ")
        if self.is_valid:
            output.append("\u2713", style="green")
            output.append(" ", style=self._buffer)
            output.append("\u2588\u2588\u2588\u2588", style=self._buffer)
        else:
            output.append("\u2717", style="red")
            output.append(" invalid")

        return output

    def handle_key(self, key: str) -> Optional[str]:
        """Handle key input.

        Args:
            key: Key or character.

        Returns:
            Hex color if enter pressed and valid, None if escape, "" if handled.
        """
        if key == "enter" and self.is_valid:
            return self._buffer.upper()
        elif key == "escape":
            return None
        elif key == "backspace" and self._cursor > 0:
            self._buffer = self._buffer[:self._cursor-1] + self._buffer[self._cursor:]
            self._cursor -= 1
            return ""
        elif key == "delete" and self._cursor < len(self._buffer):
            self._buffer = self._buffer[:self._cursor] + self._buffer[self._cursor+1:]
            return ""
        elif key == "left" and self._cursor > 0:
            self._cursor -= 1
            return ""
        elif key == "right" and self._cursor < len(self._buffer):
            self._cursor += 1
            return ""
        elif key == "home":
            self._cursor = 0
            return ""
        elif key == "end":
            self._cursor = len(self._buffer)
            return ""
        elif len(key) == 1 and key in "0123456789abcdefABCDEF#":
            # Insert character
            if len(self._buffer) < 7 or (key == "#" and "#" not in self._buffer):
                self._buffer = self._buffer[:self._cursor] + key + self._buffer[self._cursor:]
                self._cursor += 1
            return ""

        return key  # Not handled
