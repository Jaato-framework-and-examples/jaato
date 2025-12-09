"""Rich TUI client for jaato.

A terminal UI client using Rich's Live+Layout for a sticky plan display
with scrolling output below.
"""

from .rich_client import RichClient, main

__all__ = ["RichClient", "main"]
