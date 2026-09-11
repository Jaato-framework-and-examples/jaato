"""Base interface for notebook backends."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple

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

    def execution_boundary(self) -> Tuple[bool, str]:
        """State the boundary this backend executes model-authored code inside.

        Every notebook backend runs code the *model* wrote, so each one owes
        the plugin an answer to "what stops this cell reaching the rest of the
        host?".  ``NotebookPlugin`` asks before dispatching a cell and refuses
        the execution when the answer is no (issue #710) — the gate is a
        property of notebook execution rather than of one backend, so a
        backend that cannot state a boundary declines instead of quietly
        running uncontained.

        The default is a refusal, deliberately: a backend added later is
        contained only once its author has decided how, and inheriting
        "allowed" would reproduce the exact defect this method exists to
        close.  Implementations today:

        - ``LocalJupyterBackend`` — the in-process gate
          (``_inprocess_exec_allowed``): AppArmor, or an explicit opt-in.
        - ``SubprocessKernelBackend`` — AppArmor inherited by the kernel
          subprocess, else the kernel's audit-hook workspace containment, else
          an explicit opt-out.
        - ``KaggleBackend`` — the code never touches this host.

        Returns:
            ``(allowed, description)``.  ``description`` names the boundary
            when allowed, and is the refusal message (naming the escape
            hatch) when not.
        """
        return False, (
            f"Notebook execution refused: the {type(self).__name__} backend "
            "states no filesystem boundary for model-authored code. A backend "
            "must implement execution_boundary() before it may run cells."
        )

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
