"""Rich TUI client for jaato.

A terminal UI client using Rich's Live+Layout for a sticky plan display
with scrolling output below.
"""

# Guard against pytest importing this file outside of package context.
# When pytest discovers __init__.py at the project root, it tries to
# load it as a standalone module where relative imports are not available.
if __package__:
    from .app import RichClient, main

    __all__ = ["RichClient", "main"]
