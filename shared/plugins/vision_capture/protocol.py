# shared/plugins/vision_capture/protocol.py
"""Protocol definitions for vision capture plugins."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Protocol, runtime_checkable
from datetime import datetime


class CaptureFormat(Enum):
    """Supported capture output formats."""
    PNG = "png"
    SVG = "svg"
    HTML = "html"


class CaptureContext(Enum):
    """What triggered the capture."""
    USER_REQUESTED = "user_requested"
    TURN_END = "turn_end"
    TOOL_END = "tool_end"
    ERROR = "error"
    PERIODIC = "periodic"


@dataclass
class CaptureResult:
    """Result of a vision capture operation."""
    path: str
    format: CaptureFormat
    timestamp: datetime
    context: CaptureContext
    width: int
    height: int
    turn_index: Optional[int] = None
    agent_id: Optional[str] = None
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None

    def to_user_message(self, workspace_root: Optional[str] = None) -> str:
        """Generate a user message informing the model about the screenshot.

        Args:
            workspace_root: If provided, the path will be made relative to this root.
        """
        import os

        # Make path relative to workspace if provided
        display_path = self.path
        if workspace_root and self.path.startswith(workspace_root):
            display_path = os.path.relpath(self.path, workspace_root)

        return f"I took a screenshot of the TUI, saved at {display_path}"


@dataclass
class CaptureConfig:
    """Configuration for vision capture."""
    output_dir: str = "/tmp/jaato_vision"
    format: CaptureFormat = CaptureFormat.SVG  # SVG is default (no dependencies)
    width: int = 120
    height: int = 50
    title: str = "Jaato TUI"
    auto_cleanup_hours: int = 24  # Delete captures older than this


@runtime_checkable
class VisionCapturePlugin(Protocol):
    """Protocol for vision capture implementations."""

    @property
    def name(self) -> str:
        """Plugin name."""
        ...

    def initialize(self, config: Optional[CaptureConfig] = None) -> None:
        """Initialize the capture plugin with configuration."""
        ...

    def capture(
        self,
        renderable,
        context: CaptureContext = CaptureContext.USER_REQUESTED,
        turn_index: Optional[int] = None,
        agent_id: Optional[str] = None,
    ) -> CaptureResult:
        """Capture a Rich renderable to an image file."""
        ...

    def capture_ansi(
        self,
        ansi_text: str,
        context: CaptureContext = CaptureContext.USER_REQUESTED,
        turn_index: Optional[int] = None,
        agent_id: Optional[str] = None,
    ) -> CaptureResult:
        """Capture ANSI text to an image file."""
        ...

    def get_last_capture(self) -> Optional[CaptureResult]:
        """Get the most recent capture result."""
        ...

    def cleanup_old_captures(self) -> int:
        """Remove captures older than configured threshold. Returns count removed."""
        ...
