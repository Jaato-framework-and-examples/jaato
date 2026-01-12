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

    def to_system_message(self) -> str:
        """Generate system message for injection."""
        attrs = [
            f'path="{self.path}"',
            f'format="{self.format.value}"',
            f'context="{self.context.value}"',
            f'timestamp="{self.timestamp.isoformat()}"',
        ]
        if self.turn_index is not None:
            attrs.append(f'turn="{self.turn_index}"')
        if self.agent_id:
            attrs.append(f'agent="{self.agent_id}"')

        return f'<tui-screenshot {" ".join(attrs)}/>'


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
