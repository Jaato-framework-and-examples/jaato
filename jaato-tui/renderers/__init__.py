# jaato-tui/renderers/__init__.py
"""Pluggable renderers for jaato client output."""

from .base import Renderer
from .headless import HeadlessFileRenderer

__all__ = ["Renderer", "HeadlessFileRenderer"]
