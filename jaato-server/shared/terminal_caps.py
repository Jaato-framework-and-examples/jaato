# shared/terminal_caps.py
"""Shared terminal capability detection with process-wide caching.

Provides terminal environment detection (color depth, emulator, multiplexer,
graphics protocol support) as a shared utility that any plugin can import
directly without cross-plugin coupling.

Usage:
    from shared.terminal_caps import detect

    caps = detect()
    caps["graphics"]       # "kitty" | "iterm" | "sixel" | None
    caps["color_depth"]    # "24bit" | "256" | "basic" | "none"
    caps["multiplexer"]    # "tmux" | "screen" | None
    caps["emulator"]       # "iTerm.app" | "kitty" | "xterm-compatible" | ...
"""

import os
import sys
from typing import Any, Dict, Optional

_cached: Optional[Dict[str, Any]] = None


def detect() -> Dict[str, Any]:
    """Detect terminal capabilities. Result is cached process-wide.

    Returns:
        Dict with keys: interactive, term, term_program, colorterm,
        multiplexer, color_depth, emulator, graphics.
    """
    global _cached
    if _cached is None:
        _cached = _detect()
    return _cached


def invalidate_cache() -> None:
    """Clear the cached result. Useful for testing or after env changes."""
    global _cached
    _cached = None


def _detect() -> Dict[str, Any]:
    """Perform full terminal capability detection."""
    is_interactive = sys.stdout.isatty()

    term = os.environ.get("TERM")
    term_program = os.environ.get("TERM_PROGRAM")
    colorterm = os.environ.get("COLORTERM")

    info: Dict[str, Any] = {
        "interactive": is_interactive,
        "term": term,
        "term_program": term_program,
        "colorterm": colorterm,
    }

    # Detect terminal multiplexers
    info["multiplexer"] = _detect_multiplexer(term)

    # Detect color capability
    info["color_depth"] = _detect_color_depth(
        is_interactive, term, colorterm
    )

    # Detect terminal emulator
    info["emulator"] = _detect_emulator(term_program, term)

    # Detect graphics protocol support
    info["graphics"] = _detect_graphics(
        term_program, term, info["multiplexer"], is_interactive
    )

    return info


def _detect_multiplexer(term: Optional[str]) -> Optional[str]:
    """Detect if running inside a terminal multiplexer."""
    if os.environ.get("TMUX"):
        return "tmux"
    if os.environ.get("STY"):
        return "screen"
    if term and "screen" in term:
        return "screen"
    return None


def _detect_color_depth(
    is_interactive: bool,
    term: Optional[str],
    colorterm: Optional[str],
) -> str:
    """Detect terminal color depth."""
    if not is_interactive:
        return "none"

    term_lower = (term or "").lower()
    colorterm_lower = (colorterm or "").lower()

    if colorterm_lower in ("truecolor", "24bit") or "truecolor" in colorterm_lower:
        return "24bit"
    if "256color" in term_lower or "256" in colorterm_lower:
        return "256"
    if term_lower and term_lower != "dumb":
        return "basic"
    return "none"


def _detect_emulator(
    term_program: Optional[str],
    term: Optional[str],
) -> Optional[str]:
    """Detect the terminal emulator."""
    if term_program:
        return term_program
    term_lower = (term or "").lower()
    if "xterm" in term_lower:
        return "xterm-compatible"
    if "linux" in term_lower:
        return "linux-console"
    return None


def _detect_graphics(
    term_program: Optional[str],
    term: Optional[str],
    multiplexer: Optional[str],
    is_interactive: bool = False,
) -> Optional[str]:
    """Detect the best available terminal graphics protocol.

    Multiplexers (tmux, screen) strip graphics escape sequences, so we
    return None when one is detected. Users can override via the
    JAATO_GRAPHICS_PROTOCOL env var if they know their setup supports
    passthrough (e.g. tmux with allow-passthrough).

    Args:
        term_program: Value of TERM_PROGRAM env var.
        term: Value of TERM env var.
        multiplexer: Detected multiplexer ("tmux", "screen", or None).
        is_interactive: Whether stdout is a TTY.

    Returns:
        "kitty" - Kitty graphics protocol (Kitty, Ghostty)
        "iterm" - iTerm2 inline image protocol (iTerm2, WezTerm, Mintty)
        "sixel" - Sixel bitmap protocol (foot, mlterm, xterm w/ sixel)
        None    - No graphics support detected (use rich-pixels fallback)
    """
    # User override always wins
    override = os.environ.get("JAATO_GRAPHICS_PROTOCOL")
    if override:
        if override.lower() in ("kitty", "iterm", "sixel"):
            return override.lower()
        # "none" or any other value disables graphics
        return None

    # Multiplexers break graphics protocols by default
    if multiplexer:
        return None

    # Non-interactive terminals can't display graphics
    if not is_interactive:
        return None

    tp = (term_program or "").lower()
    term_lower = (term or "").lower()

    # Kitty graphics protocol
    if tp in ("kitty", "ghostty"):
        return "kitty"

    # iTerm2 inline image protocol
    # WezTerm supports both iTerm2 and Sixel; iTerm2 protocol is higher fidelity
    if tp in ("iterm.app", "wezterm"):
        return "iterm"

    # Mintty supports iTerm2 inline images
    if tp == "mintty":
        return "iterm"

    # Sixel support
    # foot and mlterm have reliable sixel support
    if term_lower.startswith("foot") or tp == "mlterm":
        return "sixel"

    # xterm *can* support sixel but only if compiled with --enable-sixel-graphics.
    # We can't know that from env vars alone, so we don't assume it.
    # Users can set JAATO_GRAPHICS_PROTOCOL=sixel to opt in.

    return None
