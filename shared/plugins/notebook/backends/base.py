"""Base interface for notebook backends."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

from ..types import (
    BackendCapabilities,
    ExecutionResult,
    NotebookInfo,
)


class NotebookBackend(ABC):
    """Abstract base class for notebook execution backends.

    Backends provide the actual execution environment for Python code.
    Different backends offer different tradeoffs:
    - Local: Instant execution, no GPU, unlimited
    - Kaggle: Free GPU (30h/week), async execution
    - Lightning.ai: Free GPU (35h/month), SDK-based
    """

    @property
    @abstractmethod
    def capabilities(self) -> BackendCapabilities:
        """Return the capabilities of this backend."""
        ...

    @abstractmethod
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the backend.

        Args:
            config: Backend-specific configuration
        """
        ...

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the backend and release resources."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the backend is available and properly configured.

        Returns:
            True if the backend can be used, False otherwise.
        """
        ...

    @abstractmethod
    def create_notebook(
        self,
        name: str,
        gpu_enabled: bool = False,
    ) -> NotebookInfo:
        """Create a new notebook/kernel.

        Args:
            name: Human-readable name for the notebook
            gpu_enabled: Whether to enable GPU (if supported)

        Returns:
            NotebookInfo for the created notebook
        """
        ...

    @abstractmethod
    def execute(
        self,
        notebook_id: str,
        code: str,
        timeout_seconds: Optional[int] = None,
    ) -> ExecutionResult:
        """Execute code in a notebook.

        For synchronous backends (local), this blocks until completion.
        For asynchronous backends (Kaggle), this may return a QUEUED or
        RUNNING status, requiring polling via get_execution_status().

        Args:
            notebook_id: ID of the notebook to execute in
            code: Python code to execute
            timeout_seconds: Optional timeout (backend-specific default if None)

        Returns:
            ExecutionResult with status and outputs
        """
        ...

    @abstractmethod
    def get_execution_status(
        self,
        notebook_id: str,
        execution_id: Optional[str] = None,
    ) -> ExecutionResult:
        """Get the status of an execution (for async backends).

        Args:
            notebook_id: ID of the notebook
            execution_id: Optional specific execution ID

        Returns:
            ExecutionResult with current status
        """
        ...

    @abstractmethod
    def get_variables(self, notebook_id: str) -> Dict[str, str]:
        """Get currently defined variables in the notebook.

        Args:
            notebook_id: ID of the notebook

        Returns:
            Dict mapping variable names to their type descriptions
        """
        ...

    @abstractmethod
    def reset_notebook(self, notebook_id: str) -> str:
        """Reset the notebook state (clear all variables).

        For backends where kernel IDs must be unique per execution (like Kaggle),
        this may generate a new notebook_id to avoid conflicts with the
        previously pushed kernel.

        Args:
            notebook_id: ID of the notebook to reset

        Returns:
            The notebook_id to use for subsequent operations (may be different
            from the input if a new ID was generated)
        """
        ...

    @abstractmethod
    def delete_notebook(self, notebook_id: str) -> None:
        """Delete a notebook and release its resources.

        Args:
            notebook_id: ID of the notebook to delete
        """
        ...

    @abstractmethod
    def list_notebooks(self) -> List[NotebookInfo]:
        """List all active notebooks.

        Returns:
            List of NotebookInfo for active notebooks
        """
        ...

    def install_package(
        self,
        notebook_id: str,
        package: str,
    ) -> ExecutionResult:
        """Install a Python package in the notebook environment.

        Default implementation uses pip install via execute().

        Args:
            notebook_id: ID of the notebook
            package: Package specification (e.g., 'numpy', 'pandas>=2.0')

        Returns:
            ExecutionResult from the installation
        """
        code = f"!pip install -q {package}"
        return self.execute(notebook_id, code)
