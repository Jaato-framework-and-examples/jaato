"""Clipboard configuration."""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Set


class ClipboardMechanism(Enum):
    """Supported clipboard mechanisms."""
    OSC52 = "osc52"
    # Future: PYPERCLIP = "pyperclip"
    # Future: NATIVE = "native"  # pbcopy/xclip/clip.exe


# Default sources to include when copying (chrome-free)
DEFAULT_COPY_SOURCES = {"model"}


def parse_sources(sources_str: str) -> Set[str]:
    """Parse sources string like 'model&user&tool' into a set."""
    if not sources_str:
        return DEFAULT_COPY_SOURCES.copy()
    return {s.strip() for s in sources_str.split("&") if s.strip()}


@dataclass
class ClipboardConfig:
    """Configuration for clipboard operations."""
    mechanism: ClipboardMechanism = ClipboardMechanism.OSC52
    sources: Set[str] = field(default_factory=lambda: DEFAULT_COPY_SOURCES.copy())

    @classmethod
    def from_env(cls) -> "ClipboardConfig":
        """Create config from environment variables.

        Env vars:
            JAATO_COPY_MECHANISM: osc52 (default)
            JAATO_COPY_SOURCES: model&user&tool format (default: model)
        """
        mechanism_str = os.environ.get("JAATO_COPY_MECHANISM", "osc52").lower()
        try:
            mechanism = ClipboardMechanism(mechanism_str)
        except ValueError:
            mechanism = ClipboardMechanism.OSC52

        sources_str = os.environ.get("JAATO_COPY_SOURCES", "")
        sources = parse_sources(sources_str)

        return cls(mechanism=mechanism, sources=sources)

    def should_include_source(self, source: str) -> bool:
        """Check if a source type should be included in copy.

        Handles exact matches and prefix matches for tool sources.
        E.g., "tool" in sources matches "tool:grep", "tool:read", etc.
        """
        if source in self.sources:
            return True
        # Check prefix match for tool sources (e.g., "tool" matches "tool:grep")
        if ":" in source:
            prefix = source.split(":")[0]
            if prefix in self.sources:
                return True
        return False
