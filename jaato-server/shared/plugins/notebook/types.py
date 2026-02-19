"""Type definitions for the notebook plugin."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ExecutionStatus(str, Enum):
    """Status of a notebook execution."""
    PENDING = "pending"      # Not yet started
    QUEUED = "queued"        # Queued for execution (cloud backends)
    RUNNING = "running"      # Currently executing
    COMPLETED = "completed"  # Successfully completed
    FAILED = "failed"        # Execution failed
    CANCELLED = "cancelled"  # Cancelled by user/timeout


class OutputType(str, Enum):
    """Type of cell output."""
    STDOUT = "stdout"        # Standard output (print)
    STDERR = "stderr"        # Standard error
    RESULT = "result"        # Execution result (last expression)
    DISPLAY = "display"      # Rich display (HTML, images, etc.)
    ERROR = "error"          # Exception traceback


@dataclass
class CellOutput:
    """Output from a cell execution.

    Attributes:
        output_type: Type of output (stdout, stderr, result, display, error)
        content: The output content (text, base64 image, HTML, etc.)
        mime_type: MIME type for rich outputs (e.g., 'image/png', 'text/html')
        metadata: Additional metadata (e.g., image dimensions)
    """
    output_type: OutputType
    content: str
    mime_type: str = "text/plain"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "output_type": self.output_type.value,
            "content": self.content,
            "mime_type": self.mime_type,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CellOutput':
        """Create from dictionary."""
        return cls(
            output_type=OutputType(data["output_type"]),
            content=data["content"],
            mime_type=data.get("mime_type", "text/plain"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ExecutionResult:
    """Result of executing code in a notebook.

    Attributes:
        status: Execution status
        outputs: List of outputs from the execution
        execution_count: Cell execution counter
        duration_seconds: How long the execution took
        error_name: Exception class name if failed
        error_message: Exception message if failed
        traceback: Full traceback if failed
        variables: Snapshot of defined variables (names and types)
    """
    status: ExecutionStatus
    outputs: List[CellOutput] = field(default_factory=list)
    execution_count: Optional[int] = None
    duration_seconds: Optional[float] = None
    error_name: Optional[str] = None
    error_message: Optional[str] = None
    traceback: Optional[str] = None
    variables: Dict[str, str] = field(default_factory=dict)  # name -> type

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "status": self.status.value,
            "outputs": [o.to_dict() for o in self.outputs],
            "execution_count": self.execution_count,
            "duration_seconds": self.duration_seconds,
            "error_name": self.error_name,
            "error_message": self.error_message,
            "traceback": self.traceback,
            "variables": self.variables,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExecutionResult':
        """Create from dictionary."""
        return cls(
            status=ExecutionStatus(data["status"]),
            outputs=[CellOutput.from_dict(o) for o in data.get("outputs", [])],
            execution_count=data.get("execution_count"),
            duration_seconds=data.get("duration_seconds"),
            error_name=data.get("error_name"),
            error_message=data.get("error_message"),
            traceback=data.get("traceback"),
            variables=data.get("variables", {}),
        )

    def get_text_output(self) -> str:
        """Get combined text output (stdout + result)."""
        texts = []
        for output in self.outputs:
            if output.output_type in (OutputType.STDOUT, OutputType.RESULT):
                texts.append(output.content)
        return "\n".join(texts)

    def has_error(self) -> bool:
        """Check if execution resulted in an error."""
        return self.status == ExecutionStatus.FAILED


@dataclass
class NotebookInfo:
    """Information about a notebook.

    Attributes:
        notebook_id: Unique identifier for the notebook
        name: Human-readable name
        backend: Backend used (local, kaggle, lightning)
        gpu_enabled: Whether GPU is enabled
        created_at: Creation timestamp (ISO format)
        last_executed_at: Last execution timestamp
        execution_count: Total executions in this notebook
        variables: Currently defined variables
    """
    notebook_id: str
    name: str
    backend: str
    gpu_enabled: bool = False
    created_at: Optional[str] = None
    last_executed_at: Optional[str] = None
    execution_count: int = 0
    variables: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "notebook_id": self.notebook_id,
            "name": self.name,
            "backend": self.backend,
            "gpu_enabled": self.gpu_enabled,
            "created_at": self.created_at,
            "last_executed_at": self.last_executed_at,
            "execution_count": self.execution_count,
            "variables": self.variables,
        }


@dataclass
class BackendCapabilities:
    """Capabilities of a notebook backend.

    Attributes:
        name: Backend name (local, kaggle, lightning)
        supports_gpu: Whether GPU is available
        gpu_type: GPU type if available (e.g., 'T4', 'P100')
        max_runtime_hours: Maximum runtime per session
        weekly_quota_hours: Weekly GPU quota (None = unlimited)
        supports_packages: Whether pip install is supported
        is_async: Whether execution is asynchronous (requires polling)
        requires_auth: Whether authentication is required
    """
    name: str
    supports_gpu: bool = False
    gpu_type: Optional[str] = None
    max_runtime_hours: Optional[float] = None
    weekly_quota_hours: Optional[float] = None
    supports_packages: bool = True
    is_async: bool = False
    requires_auth: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "supports_gpu": self.supports_gpu,
            "gpu_type": self.gpu_type,
            "max_runtime_hours": self.max_runtime_hours,
            "weekly_quota_hours": self.weekly_quota_hours,
            "supports_packages": self.supports_packages,
            "is_async": self.is_async,
            "requires_auth": self.requires_auth,
        }
