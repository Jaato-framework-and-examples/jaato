"""Notebook Tool Bindings — stub generator and bridge.

Generates Python functions from ToolSchema objects so that notebook code
can call jaato tools directly from scripts:

    result = tools.web_search(query="python dataclasses")
    content = tools.file_read(path="src/main.py")

Architecture:
    ToolSchema (registry) → generate_tools_module() → types.ModuleType
    Each generated function → ToolBridge.call() → ToolExecutor.execute()

The generated module is injected into the notebook namespace as ``tools``,
giving scripts direct access to every exposed tool without JSON serialization
or model round-trips.
"""

import time
import types
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Tuple

from jaato_sdk.plugins.model_provider.types import ToolSchema


class ToolExecutionError(Exception):
    """Raised when a tool call via the bridge fails.

    Scripts can catch this to handle tool failures gracefully::

        try:
            result = tools.file_read(path="missing.txt")
        except tools.ToolExecutionError as e:
            print(f"Tool {e.tool_name} failed: {e.message}")

    Attributes:
        tool_name: Name of the tool that failed.
        message: Error description from the executor.
        details: Optional full error dict from the executor.
    """

    def __init__(self, tool_name: str, message: str, details: Optional[Dict[str, Any]] = None):
        self.tool_name = tool_name
        self.message = message
        self.details = details or {}
        super().__init__(f"{tool_name}: {message}")


@dataclass
class ToolCallRecord:
    """Record of a single tool call made through the bridge.

    Attributes:
        tool_name: Name of the tool that was called.
        args: Arguments passed to the tool.
        success: Whether the call succeeded.
        duration_seconds: Wall-clock time for the call.
        error: Error message if the call failed, None otherwise.
    """
    tool_name: str
    args: Dict[str, Any]
    success: bool
    duration_seconds: float
    error: Optional[str] = None


class ToolBridge:
    """Thin adapter between generated stub functions and ToolExecutor.execute().

    Translates the stub's ``call(name, args)`` into the executor's
    ``execute(name, args) → (success, result)`` protocol, raising
    ToolExecutionError on failure so scripts can use try/except naturally.

    Attributes:
        call_log: List of ToolCallRecord for every call made through this bridge.
    """

    def __init__(self, executor_fn: Callable[[str, Dict[str, Any]], Tuple[bool, Any]]):
        """Initialize the bridge.

        Args:
            executor_fn: A callable matching ToolExecutor.execute(name, args) → (bool, Any).
                Typically ``executor.execute`` or a partial binding thereof.
        """
        self._executor_fn = executor_fn
        self.call_log: List[ToolCallRecord] = []

    def call(self, tool_name: str, args: Dict[str, Any]) -> Any:
        """Call a tool and return its result, or raise on failure.

        Args:
            tool_name: Registered tool name.
            args: Arguments dict matching the tool's parameter schema.

        Returns:
            The result dict from the executor (success case).

        Raises:
            ToolExecutionError: If the executor returns success=False.
        """
        start = time.monotonic()
        success, result = self._executor_fn(tool_name, args)
        elapsed = time.monotonic() - start

        if success:
            self.call_log.append(ToolCallRecord(
                tool_name=tool_name,
                args=args,
                success=True,
                duration_seconds=elapsed,
            ))
            return result
        else:
            error_msg = "Unknown error"
            if isinstance(result, dict):
                error_msg = result.get("error", str(result))
            elif isinstance(result, str):
                error_msg = result
            self.call_log.append(ToolCallRecord(
                tool_name=tool_name,
                args=args,
                success=False,
                duration_seconds=elapsed,
                error=error_msg,
            ))
            raise ToolExecutionError(
                tool_name,
                error_msg,
                details=result if isinstance(result, dict) else {"error": error_msg},
            )


# JSON Schema type → Python type name mapping (for docstrings only)
_JSON_TYPE_MAP = {
    "string": "str",
    "integer": "int",
    "number": "float",
    "boolean": "bool",
    "array": "list",
    "object": "dict",
}


def _resolve_json_type(raw_type) -> str:
    """Map a JSON Schema type to a Python type name.

    Handles union types (``["string", "null"]``) by picking the first
    non-null entry.  Falls back to ``"Any"`` for unknown types.
    """
    if isinstance(raw_type, list):
        raw_type = next((t for t in raw_type if t != "null"), raw_type[0] if raw_type else "")
    return _JSON_TYPE_MAP.get(raw_type, "Any")


def _build_docstring(schema: ToolSchema) -> str:
    """Build a Python docstring from a ToolSchema.

    Includes the tool description and parameter documentation derived
    from the JSON Schema ``parameters`` field.

    Args:
        schema: The tool schema to document.

    Returns:
        A formatted docstring string.
    """
    lines = [schema.description, ""]

    props = schema.parameters.get("properties", {})
    required = set(schema.parameters.get("required", []))

    if props:
        lines.append("Args:")
        for param_name, param_spec in props.items():
            ptype = _resolve_json_type(param_spec.get("type", ""))
            desc = param_spec.get("description", "")
            opt = "" if param_name in required else ", optional"
            lines.append(f"    {param_name} ({ptype}{opt}): {desc}")

    lines.append("")
    lines.append("Returns:")
    lines.append("    dict: Tool result.")
    lines.append("")
    lines.append("Raises:")
    lines.append("    ToolExecutionError: If the tool call fails.")

    return "\n".join(lines)


def _make_tool_function(schema: ToolSchema, bridge: ToolBridge) -> Callable[..., Any]:
    """Create a Python function that calls a tool via the bridge.

    The generated function accepts keyword arguments matching the tool's
    JSON Schema parameters. Positional arguments are not supported to
    avoid ambiguity.

    Args:
        schema: Tool schema describing the function signature.
        bridge: ToolBridge instance to route calls through.

    Returns:
        A callable function with the tool's name, docstring, and
        keyword-only parameters.
    """
    tool_name = schema.name
    props = schema.parameters.get("properties", {})
    required = set(schema.parameters.get("required", []))
    param_names = list(props.keys())

    def tool_fn(**kwargs: Any) -> Any:
        # Validate required parameters
        for req in required:
            if req not in kwargs:
                raise TypeError(f"{tool_name}() missing required argument: '{req}'")
        # Filter out None values for optional params not explicitly passed
        args = {k: v for k, v in kwargs.items() if v is not None or k in required}
        return bridge.call(tool_name, args)

    # Set function metadata
    tool_fn.__name__ = tool_name
    tool_fn.__qualname__ = tool_name
    tool_fn.__doc__ = _build_docstring(schema)

    return tool_fn


def generate_tools_module(
    schemas: List[ToolSchema],
    bridge: ToolBridge,
    exclude_tools: Optional[FrozenSet[str]] = None,
) -> types.ModuleType:
    """Generate a Python module with one callable function per tool.

    The returned module is intended to be injected into a notebook namespace
    as ``tools``, so scripts can call ``tools.web_search(query="...")``.

    The module also exposes:
    - ``tools.list_tools()`` → dict of {name: description}
    - ``tools.ToolExecutionError`` → the exception class for catch blocks
    - ``tools.call_log`` → list of ToolCallRecord from the bridge

    Args:
        schemas: List of ToolSchema objects to generate functions for.
        bridge: ToolBridge instance that routes calls to the executor.
        exclude_tools: Optional set of tool names to exclude (e.g., notebook
            tools themselves to prevent recursion).

    Returns:
        A types.ModuleType with one function attribute per tool schema.
    """
    exclude = exclude_tools or frozenset()

    module = types.ModuleType("jaato_tools")
    module.__doc__ = "Auto-generated tool functions for programmatic access."

    tool_names: Dict[str, str] = {}

    for schema in schemas:
        if schema.name in exclude:
            continue
        fn = _make_tool_function(schema, bridge)
        setattr(module, schema.name, fn)
        tool_names[schema.name] = schema.description

    # Discovery helper
    def list_tools() -> Dict[str, str]:
        """List all available tool functions with their descriptions."""
        return dict(tool_names)

    module.list_tools = list_tools

    # Expose exception class so scripts can catch it
    module.ToolExecutionError = ToolExecutionError

    # Expose call log for observability
    module.call_log = bridge.call_log

    return module


def generate_tool_signatures(schemas: List[ToolSchema], exclude_tools: Optional[FrozenSet[str]] = None) -> str:
    """Generate human-readable function signatures for system prompt inclusion.

    Produces a compact listing of available tool functions that the model
    can reference when writing notebook scripts.

    Args:
        schemas: List of ToolSchema objects.
        exclude_tools: Optional set of tool names to exclude.

    Returns:
        A formatted string with one function signature per line.
    """
    exclude = exclude_tools or frozenset()
    lines = []

    for schema in sorted(schemas, key=lambda s: s.name):
        if schema.name in exclude:
            continue

        props = schema.parameters.get("properties", {})
        required = set(schema.parameters.get("required", []))
        params = []
        for pname, pspec in props.items():
            ptype = _resolve_json_type(pspec.get("type", ""))
            if pname in required:
                params.append(f"{pname}: {ptype}")
            else:
                params.append(f"{pname}: {ptype} = None")

        sig = f"tools.{schema.name}({', '.join(params)}) -> dict"
        desc = schema.description[:80]
        lines.append(f"  {sig}")
        lines.append(f"    # {desc}")

    return "\n".join(lines)
