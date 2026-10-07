"""Tests for Notebook Tool Bindings — stub generation, bridge, and integration."""

import pytest
from jaato_sdk.plugins.model_provider.types import ToolSchema

from ..tool_stubs import (
    ToolBridge,
    ToolExecutionError,
    ToolCallRecord,
    generate_tools_module,
    generate_tool_signatures,
    _build_docstring,
    _make_tool_function,
)
from ..backends.local import LocalJupyterBackend, INPROCESS_OPT_IN_ENV
from ..types import ExecutionStatus, OutputType


@pytest.fixture(autouse=True)
def _allow_inprocess_exec(monkeypatch):
    """Opt into in-process cell execution (the test runner is unconfined)."""
    monkeypatch.setenv(INPROCESS_OPT_IN_ENV, "1")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _sample_schemas():
    """Return a small set of ToolSchema objects for testing."""
    return [
        ToolSchema(
            name="web_search",
            description="Search the web for information.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "description": "Max results to return"},
                },
                "required": ["query"],
            },
        ),
        ToolSchema(
            name="file_read",
            description="Read a file's contents.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read"},
                },
                "required": ["path"],
            },
        ),
        ToolSchema(
            name="notebook_execute",
            description="Execute Python code in a notebook.",
            parameters={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Code to execute"},
                },
                "required": ["code"],
            },
        ),
    ]


def _mock_executor(name, args, **kwargs):
    """Mock executor that returns canned responses based on tool name."""
    if name == "web_search":
        return True, {"results": [{"title": "Result 1", "url": "http://example.com"}]}
    elif name == "file_read":
        path = args.get("path", "")
        if path == "missing.txt":
            return False, {"error": f"File not found: {path}"}
        return True, {"content": f"contents of {path}", "path": path}
    elif name == "fail_tool":
        return False, {"error": "Intentional failure"}
    return False, {"error": f"Unknown tool: {name}"}


# ---------------------------------------------------------------------------
# ToolBridge tests
# ---------------------------------------------------------------------------

class TestToolBridge:
    """Tests for the ToolBridge adapter."""

    def test_call_success(self):
        """Successful tool call returns result and logs the call."""
        bridge = ToolBridge(_mock_executor)
        result = bridge.call("web_search", {"query": "test"})

        assert "results" in result
        assert len(bridge.call_log) == 1
        assert bridge.call_log[0].tool_name == "web_search"
        assert bridge.call_log[0].success is True
        assert bridge.call_log[0].duration_seconds >= 0

    def test_call_failure_raises(self):
        """Failed tool call raises ToolExecutionError."""
        bridge = ToolBridge(_mock_executor)

        with pytest.raises(ToolExecutionError) as exc_info:
            bridge.call("file_read", {"path": "missing.txt"})

        assert exc_info.value.tool_name == "file_read"
        assert "File not found" in exc_info.value.message
        assert len(bridge.call_log) == 1
        assert bridge.call_log[0].success is False
        assert bridge.call_log[0].error is not None

    def test_call_failure_with_string_result(self):
        """Failed call where executor returns a plain string error."""
        def string_error_executor(name, args, **kwargs):
            return False, "plain error message"

        bridge = ToolBridge(string_error_executor)

        with pytest.raises(ToolExecutionError) as exc_info:
            bridge.call("some_tool", {})

        assert "plain error message" in exc_info.value.message

    def test_call_log_accumulates(self):
        """Multiple calls accumulate in the call log."""
        bridge = ToolBridge(_mock_executor)

        bridge.call("web_search", {"query": "q1"})
        bridge.call("file_read", {"path": "foo.txt"})
        try:
            bridge.call("file_read", {"path": "missing.txt"})
        except ToolExecutionError:
            pass

        assert len(bridge.call_log) == 3
        assert bridge.call_log[0].tool_name == "web_search"
        assert bridge.call_log[1].tool_name == "file_read"
        assert bridge.call_log[1].success is True
        assert bridge.call_log[2].tool_name == "file_read"
        assert bridge.call_log[2].success is False


# ---------------------------------------------------------------------------
# ToolExecutionError tests
# ---------------------------------------------------------------------------

class TestToolExecutionError:
    """Tests for the ToolExecutionError exception."""

    def test_attributes(self):
        err = ToolExecutionError("my_tool", "something broke", {"detail": "info"})
        assert err.tool_name == "my_tool"
        assert err.message == "something broke"
        assert err.details == {"detail": "info"}
        assert "my_tool" in str(err)
        assert "something broke" in str(err)

    def test_default_details(self):
        err = ToolExecutionError("t", "msg")
        assert err.details == {}


# ---------------------------------------------------------------------------
# Stub generation tests
# ---------------------------------------------------------------------------

class TestGenerateToolsModule:
    """Tests for generate_tools_module()."""

    def test_module_has_tool_functions(self):
        """Generated module has one function per schema."""
        schemas = _sample_schemas()
        bridge = ToolBridge(_mock_executor)
        module = generate_tools_module(schemas, bridge)

        assert hasattr(module, "web_search")
        assert hasattr(module, "file_read")
        assert callable(module.web_search)
        assert callable(module.file_read)

    def test_module_excludes_specified_tools(self):
        """Tools in exclude_tools are not added to the module."""
        schemas = _sample_schemas()
        bridge = ToolBridge(_mock_executor)
        module = generate_tools_module(
            schemas, bridge,
            exclude_tools=frozenset({"notebook_execute"}),
        )

        assert hasattr(module, "web_search")
        assert hasattr(module, "file_read")
        assert not hasattr(module, "notebook_execute")

    def test_list_tools_helper(self):
        """Module has list_tools() that returns tool names and descriptions."""
        schemas = _sample_schemas()
        bridge = ToolBridge(_mock_executor)
        module = generate_tools_module(
            schemas, bridge,
            exclude_tools=frozenset({"notebook_execute"}),
        )

        tools = module.list_tools()
        assert "web_search" in tools
        assert "file_read" in tools
        assert "notebook_execute" not in tools
        assert "Search" in tools["web_search"]

    def test_module_exposes_exception_class(self):
        """Module exposes ToolExecutionError for catch blocks."""
        schemas = _sample_schemas()
        bridge = ToolBridge(_mock_executor)
        module = generate_tools_module(schemas, bridge)

        assert module.ToolExecutionError is ToolExecutionError

    def test_module_exposes_call_log(self):
        """Module exposes the bridge's call_log."""
        schemas = _sample_schemas()
        bridge = ToolBridge(_mock_executor)
        module = generate_tools_module(schemas, bridge)

        assert module.call_log is bridge.call_log

    def test_calling_stub_function(self):
        """Calling a generated function routes through the bridge."""
        schemas = _sample_schemas()
        bridge = ToolBridge(_mock_executor)
        module = generate_tools_module(schemas, bridge)

        result = module.web_search(query="hello")
        assert "results" in result
        assert len(bridge.call_log) == 1

    def test_stub_function_raises_on_missing_required(self):
        """Missing required param raises TypeError."""
        schemas = _sample_schemas()
        bridge = ToolBridge(_mock_executor)
        module = generate_tools_module(schemas, bridge)

        with pytest.raises(TypeError, match="missing required argument"):
            module.web_search()  # 'query' is required

    def test_stub_function_raises_on_tool_failure(self):
        """Tool failure propagates as ToolExecutionError."""
        schemas = _sample_schemas()
        bridge = ToolBridge(_mock_executor)
        module = generate_tools_module(schemas, bridge)

        with pytest.raises(ToolExecutionError):
            module.file_read(path="missing.txt")

    def test_stub_function_optional_params(self):
        """Optional params can be omitted."""
        schemas = _sample_schemas()
        bridge = ToolBridge(_mock_executor)
        module = generate_tools_module(schemas, bridge)

        # max_results is optional
        result = module.web_search(query="test")
        assert "results" in result

    def test_stub_function_docstring(self):
        """Generated functions have docstrings."""
        schemas = _sample_schemas()
        bridge = ToolBridge(_mock_executor)
        module = generate_tools_module(schemas, bridge)

        assert module.web_search.__doc__ is not None
        assert "Search" in module.web_search.__doc__
        assert "query" in module.web_search.__doc__

    def test_empty_schemas(self):
        """Empty schema list produces a module with just list_tools()."""
        bridge = ToolBridge(_mock_executor)
        module = generate_tools_module([], bridge)

        assert module.list_tools() == {}
        assert module.ToolExecutionError is ToolExecutionError


# ---------------------------------------------------------------------------
# generate_tool_signatures tests
# ---------------------------------------------------------------------------

class TestGenerateToolSignatures:
    """Tests for generate_tool_signatures()."""

    def test_basic_signatures(self):
        schemas = _sample_schemas()
        sigs = generate_tool_signatures(schemas, exclude_tools=frozenset({"notebook_execute"}))

        assert "tools.web_search" in sigs
        assert "tools.file_read" in sigs
        assert "notebook_execute" not in sigs
        assert "query: str" in sigs
        assert "max_results: int = None" in sigs

    def test_empty_schemas(self):
        sigs = generate_tool_signatures([])
        assert sigs == ""


# ---------------------------------------------------------------------------
# _build_docstring tests
# ---------------------------------------------------------------------------

class TestBuildDocstring:
    """Tests for _build_docstring()."""

    def test_includes_description(self):
        schema = _sample_schemas()[0]
        doc = _build_docstring(schema)
        assert "Search the web" in doc

    def test_includes_params(self):
        schema = _sample_schemas()[0]
        doc = _build_docstring(schema)
        assert "query (str)" in doc
        assert "max_results (int, optional)" in doc

    def test_includes_returns_and_raises(self):
        schema = _sample_schemas()[0]
        doc = _build_docstring(schema)
        assert "Returns:" in doc
        assert "Raises:" in doc
        assert "ToolExecutionError" in doc


# ---------------------------------------------------------------------------
# Integration: LocalJupyterBackend + tools module
# ---------------------------------------------------------------------------

class TestBackendToolsInjection:
    """Tests for tool bindings module injection into notebook namespaces."""

    def _make_backend_with_tools(self):
        """Create a backend with a tools module injected."""
        schemas = _sample_schemas()
        bridge = ToolBridge(_mock_executor)
        module = generate_tools_module(
            schemas, bridge,
            exclude_tools=frozenset({"notebook_execute"}),
        )

        backend = LocalJupyterBackend()
        backend.initialize()
        backend.inject_tools_module(module)

        return backend, bridge, module

    def test_inject_into_new_notebook(self):
        """New notebooks created after injection have tools in namespace."""
        backend, bridge, module = self._make_backend_with_tools()

        info = backend.create_notebook("test")
        namespace = backend._notebooks[info.notebook_id]

        assert "tools" in namespace
        assert namespace["tools"] is module

        backend.shutdown()

    def test_inject_into_existing_notebooks(self):
        """inject_tools_module injects into already-existing namespaces."""
        backend = LocalJupyterBackend()
        backend.initialize()

        info = backend.create_notebook("existing")
        namespace = backend._notebooks[info.notebook_id]
        assert "tools" not in namespace

        # Now inject
        schemas = _sample_schemas()
        bridge = ToolBridge(_mock_executor)
        module = generate_tools_module(schemas, bridge)
        backend.inject_tools_module(module)

        assert "tools" in namespace
        assert namespace["tools"] is module

        backend.shutdown()

    def test_tools_survive_reset(self):
        """Notebook reset preserves the tools module in namespace."""
        backend, bridge, module = self._make_backend_with_tools()

        info = backend.create_notebook("test")
        backend.execute(info.notebook_id, "x = 42")
        backend.reset_notebook(info.notebook_id)

        namespace = backend._notebooks[info.notebook_id]
        assert "tools" in namespace
        assert namespace["tools"] is module

        # Variables should be cleared
        variables = backend.get_variables(info.notebook_id)
        assert "x" not in variables

        backend.shutdown()

    def test_execute_with_tools_in_script(self):
        """Scripts can call tools from within notebook execution."""
        backend, bridge, module = self._make_backend_with_tools()

        info = backend.create_notebook("test")
        code = 'result = tools.web_search(query="test")\nprint(result["results"][0]["title"])'
        result = backend.execute(info.notebook_id, code)

        assert result.status == ExecutionStatus.COMPLETED
        assert any("Result 1" in o.content for o in result.outputs)
        assert len(bridge.call_log) == 1
        assert bridge.call_log[0].tool_name == "web_search"

        backend.shutdown()

    def test_execute_with_tool_error_handling(self):
        """Scripts can catch ToolExecutionError."""
        backend, bridge, module = self._make_backend_with_tools()

        info = backend.create_notebook("test")
        code = """
try:
    result = tools.file_read(path="missing.txt")
except tools.ToolExecutionError as e:
    error_msg = f"Caught: {e.tool_name} - {e.message}"
    print(error_msg)
"""
        result = backend.execute(info.notebook_id, code)

        assert result.status == ExecutionStatus.COMPLETED
        assert any(
            "Caught: file_read" in o.content and "File not found" in o.content
            for o in result.outputs
        )

        backend.shutdown()

    def test_execute_list_tools(self):
        """Scripts can call tools.list_tools()."""
        backend, bridge, module = self._make_backend_with_tools()

        info = backend.create_notebook("test")
        code = "available = tools.list_tools()\nprint(list(available.keys()))"
        result = backend.execute(info.notebook_id, code)

        assert result.status == ExecutionStatus.COMPLETED
        assert any("web_search" in o.content for o in result.outputs)
        assert any("file_read" in o.content for o in result.outputs)

        backend.shutdown()

    def test_tools_persist_across_executions(self):
        """Tools module remains available across multiple executions."""
        backend, bridge, module = self._make_backend_with_tools()

        info = backend.create_notebook("test")

        # First execution: use tools
        result1 = backend.execute(info.notebook_id, 'r1 = tools.web_search(query="first")')
        assert result1.status == ExecutionStatus.COMPLETED

        # Second execution: tools still available, and r1 persists
        result2 = backend.execute(info.notebook_id, 'r2 = tools.file_read(path="test.py")\nprint(r2["content"])')
        assert result2.status == ExecutionStatus.COMPLETED
        assert any("contents of test.py" in o.content for o in result2.outputs)

        assert len(bridge.call_log) == 2

        backend.shutdown()


# ---------------------------------------------------------------------------
# Integration: NotebookPlugin tool bindings wiring
# ---------------------------------------------------------------------------

class TestNotebookPluginToolBindings:
    """Tests for tool bindings wiring in the NotebookPlugin."""

    def test_tool_bindings_enabled_by_default(self):
        """Tool bindings are enabled when JAATO_TOOL_BINDINGS is not set."""
        import os
        # Ensure env var is not set (default = enabled)
        env_backup = os.environ.pop("JAATO_TOOL_BINDINGS", None)
        try:
            from ..plugin import NotebookPlugin
            plugin = NotebookPlugin()
            assert plugin._tool_bindings_enabled is True
            # Module is not built yet (no registry or executor)
            assert plugin._tool_bindings_module is None
        finally:
            if env_backup is not None:
                os.environ["JAATO_TOOL_BINDINGS"] = env_backup

    def test_tool_bindings_disabled_via_env(self):
        """Tool bindings can be disabled with JAATO_TOOL_BINDINGS=false."""
        import os
        env_backup = os.environ.get("JAATO_TOOL_BINDINGS")
        os.environ["JAATO_TOOL_BINDINGS"] = "false"
        try:
            from ..plugin import NotebookPlugin
            plugin = NotebookPlugin()
            assert plugin._tool_bindings_enabled is False
            assert plugin._tool_bindings_module is None
        finally:
            if env_backup is not None:
                os.environ["JAATO_TOOL_BINDINGS"] = env_backup
            else:
                os.environ.pop("JAATO_TOOL_BINDINGS", None)
