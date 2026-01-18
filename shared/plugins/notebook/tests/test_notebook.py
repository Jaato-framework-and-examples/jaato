"""Tests for the notebook plugin."""

import pytest
from ..plugin import NotebookPlugin, create_plugin
from ..backends.local import LocalJupyterBackend
from ..types import ExecutionStatus, OutputType


class TestLocalBackend:
    """Tests for the local Jupyter backend."""

    def test_create_notebook(self):
        """Test creating a notebook."""
        backend = LocalJupyterBackend()
        backend.initialize()

        info = backend.create_notebook("test-notebook")

        assert info.notebook_id is not None
        assert info.name == "test-notebook"
        assert info.backend == "local"
        assert info.gpu_enabled is False
        assert info.execution_count == 0

        backend.shutdown()

    def test_execute_simple_expression(self):
        """Test executing a simple expression."""
        backend = LocalJupyterBackend()
        backend.initialize()

        info = backend.create_notebook("test")
        result = backend.execute(info.notebook_id, "1 + 1")

        assert result.status == ExecutionStatus.COMPLETED
        assert result.execution_count == 1
        assert len(result.outputs) == 1
        assert result.outputs[0].output_type == OutputType.RESULT
        assert "2" in result.outputs[0].content

        backend.shutdown()

    def test_execute_print_statement(self):
        """Test executing a print statement."""
        backend = LocalJupyterBackend()
        backend.initialize()

        info = backend.create_notebook("test")
        result = backend.execute(info.notebook_id, "print('hello world')")

        assert result.status == ExecutionStatus.COMPLETED
        assert any(
            o.output_type == OutputType.STDOUT and "hello world" in o.content
            for o in result.outputs
        )

        backend.shutdown()

    def test_variable_persistence(self):
        """Test that variables persist across executions."""
        backend = LocalJupyterBackend()
        backend.initialize()

        info = backend.create_notebook("test")

        # Define a variable
        result1 = backend.execute(info.notebook_id, "x = 42")
        assert result1.status == ExecutionStatus.COMPLETED

        # Use the variable
        result2 = backend.execute(info.notebook_id, "x * 2")
        assert result2.status == ExecutionStatus.COMPLETED
        assert any("84" in o.content for o in result2.outputs)

        # Check variables
        variables = backend.get_variables(info.notebook_id)
        assert "x" in variables
        assert variables["x"] == "int"

        backend.shutdown()

    def test_execute_error(self):
        """Test executing code that raises an error."""
        backend = LocalJupyterBackend()
        backend.initialize()

        info = backend.create_notebook("test")
        result = backend.execute(info.notebook_id, "1 / 0")

        assert result.status == ExecutionStatus.FAILED
        assert result.error_name == "ZeroDivisionError"
        assert result.traceback is not None

        backend.shutdown()

    def test_reset_notebook(self):
        """Test resetting a notebook clears variables."""
        backend = LocalJupyterBackend()
        backend.initialize()

        info = backend.create_notebook("test")

        # Define variables
        backend.execute(info.notebook_id, "x = 1; y = 2")
        variables = backend.get_variables(info.notebook_id)
        assert "x" in variables
        assert "y" in variables

        # Reset
        backend.reset_notebook(info.notebook_id)
        variables = backend.get_variables(info.notebook_id)
        assert "x" not in variables
        assert "y" not in variables

        backend.shutdown()

    def test_delete_notebook(self):
        """Test deleting a notebook."""
        backend = LocalJupyterBackend()
        backend.initialize()

        info = backend.create_notebook("test")
        notebook_id = info.notebook_id

        # Verify it exists
        notebooks = backend.list_notebooks()
        assert any(n.notebook_id == notebook_id for n in notebooks)

        # Delete
        backend.delete_notebook(notebook_id)

        # Verify it's gone
        notebooks = backend.list_notebooks()
        assert not any(n.notebook_id == notebook_id for n in notebooks)

        backend.shutdown()

    def test_list_notebooks(self):
        """Test listing notebooks."""
        backend = LocalJupyterBackend()
        backend.initialize()

        backend.create_notebook("nb1")
        backend.create_notebook("nb2")

        notebooks = backend.list_notebooks()
        assert len(notebooks) == 2
        names = [n.name for n in notebooks]
        assert "nb1" in names
        assert "nb2" in names

        backend.shutdown()

    def test_multiline_code(self):
        """Test executing multiline code."""
        backend = LocalJupyterBackend()
        backend.initialize()

        info = backend.create_notebook("test")
        code = """
def greet(name):
    return f"Hello, {name}!"

result = greet("World")
print(result)
"""
        result = backend.execute(info.notebook_id, code)

        assert result.status == ExecutionStatus.COMPLETED
        assert any("Hello, World!" in o.content for o in result.outputs)

        backend.shutdown()

    def test_import_modules(self):
        """Test importing standard library modules."""
        backend = LocalJupyterBackend()
        backend.initialize()

        info = backend.create_notebook("test")

        # Import and use a module
        result = backend.execute(info.notebook_id, "import math; math.sqrt(16)")

        assert result.status == ExecutionStatus.COMPLETED
        assert any("4.0" in o.content for o in result.outputs)

        backend.shutdown()


class TestNotebookPlugin:
    """Tests for the notebook plugin."""

    def test_create_plugin(self):
        """Test plugin creation via factory."""
        plugin = create_plugin()
        assert plugin.name == "notebook"

    def test_initialize(self):
        """Test plugin initialization."""
        plugin = NotebookPlugin()
        plugin.initialize({"enable_kaggle": False})

        assert plugin._initialized
        assert "local" in plugin._backends

        plugin.shutdown()

    def test_get_tool_schemas(self):
        """Test getting tool schemas."""
        plugin = NotebookPlugin()
        plugin.initialize({"enable_kaggle": False})

        schemas = plugin.get_tool_schemas()

        # Should have multiple tools
        assert len(schemas) >= 4

        # Check for key tools
        tool_names = [s.name for s in schemas]
        assert "notebook_execute" in tool_names
        assert "notebook_create" in tool_names
        assert "notebook_variables" in tool_names
        assert "notebook_reset" in tool_names

        plugin.shutdown()

    def test_get_executors(self):
        """Test getting executors."""
        plugin = NotebookPlugin()
        plugin.initialize({"enable_kaggle": False})

        executors = plugin.get_executors()

        assert "notebook_execute" in executors
        assert callable(executors["notebook_execute"])

        plugin.shutdown()

    def test_execute_auto_creates_notebook(self):
        """Test that execute auto-creates a notebook if needed."""
        plugin = NotebookPlugin()
        plugin.initialize({"enable_kaggle": False})

        # Execute without creating notebook first
        result = plugin._execute_code({"code": "1 + 1"})

        assert "error" not in result
        assert result["status"] == "completed"
        assert "notebook_id" in result

        plugin.shutdown()

    def test_create_and_execute(self):
        """Test creating a notebook and executing code."""
        plugin = NotebookPlugin()
        plugin.initialize({"enable_kaggle": False})

        # Create notebook
        create_result = plugin._create_notebook({"name": "test-nb"})
        assert "error" not in create_result
        notebook_id = create_result["notebook_id"]

        # Execute code
        exec_result = plugin._execute_code({
            "code": "x = 10; x ** 2",
            "notebook_id": notebook_id,
        })

        assert exec_result["status"] == "completed"
        assert "100" in exec_result.get("output", "")

        plugin.shutdown()

    def test_list_backends(self):
        """Test listing backends."""
        plugin = NotebookPlugin()
        plugin.initialize({"enable_kaggle": False})

        result = plugin._list_backends({})

        assert "backends" in result
        assert len(result["backends"]) >= 1

        local_backend = next(b for b in result["backends"] if b["name"] == "local")
        assert local_backend["available"] is True
        assert local_backend["supports_gpu"] is False

        plugin.shutdown()

    def test_get_system_instructions(self):
        """Test getting system instructions."""
        plugin = NotebookPlugin()
        plugin.initialize({"enable_kaggle": False})

        instructions = plugin.get_system_instructions()

        assert instructions is not None
        assert "notebook_execute" in instructions
        assert "Python" in instructions

        plugin.shutdown()

    def test_auto_approved_tools(self):
        """Test auto-approved tools list."""
        plugin = NotebookPlugin()
        plugin.initialize({"enable_kaggle": False})

        auto_approved = plugin.get_auto_approved_tools()

        # Read-only tools should be auto-approved
        assert "notebook_variables" in auto_approved
        assert "notebook_list" in auto_approved
        assert "notebook_backends" in auto_approved

        # Execute should NOT be auto-approved
        assert "notebook_execute" not in auto_approved

        plugin.shutdown()


class TestNotebookTypes:
    """Tests for notebook types."""

    def test_execution_result_to_dict(self):
        """Test ExecutionResult serialization."""
        from ..types import ExecutionResult, CellOutput, ExecutionStatus, OutputType

        result = ExecutionResult(
            status=ExecutionStatus.COMPLETED,
            outputs=[
                CellOutput(
                    output_type=OutputType.STDOUT,
                    content="hello",
                )
            ],
            execution_count=1,
            duration_seconds=0.5,
            variables={"x": "int"},
        )

        d = result.to_dict()

        assert d["status"] == "completed"
        assert d["execution_count"] == 1
        assert len(d["outputs"]) == 1
        assert d["outputs"][0]["content"] == "hello"

    def test_execution_result_from_dict(self):
        """Test ExecutionResult deserialization."""
        from ..types import ExecutionResult, ExecutionStatus

        d = {
            "status": "completed",
            "outputs": [
                {"output_type": "stdout", "content": "hello"}
            ],
            "execution_count": 1,
        }

        result = ExecutionResult.from_dict(d)

        assert result.status == ExecutionStatus.COMPLETED
        assert result.execution_count == 1
        assert len(result.outputs) == 1
        assert result.outputs[0].content == "hello"

    def test_backend_capabilities(self):
        """Test BackendCapabilities."""
        from ..types import BackendCapabilities

        caps = BackendCapabilities(
            name="test",
            supports_gpu=True,
            gpu_type="T4",
            weekly_quota_hours=30.0,
        )

        d = caps.to_dict()

        assert d["name"] == "test"
        assert d["supports_gpu"] is True
        assert d["gpu_type"] == "T4"
        assert d["weekly_quota_hours"] == 30.0
