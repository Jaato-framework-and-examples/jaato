"""MCP tool plugin for executing Model Context Protocol tools."""

import asyncio
import json
import os
import queue
import sys
import threading
import time
from typing import Dict, List, Any, Callable, Optional

from ..base import UserCommand
from ..model_provider.types import ToolSchema


class MCPToolPlugin:
    """Plugin that provides MCP (Model Context Protocol) tool execution.

    This plugin connects to MCP servers defined in .mcp.json and exposes
    their tools to the AI model. It runs a background thread with an
    asyncio event loop to handle the async MCP protocol.
    """

    def __init__(self):
        # Instance state instead of module globals
        self._tool_cache: Dict[str, List[Any]] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._manager: Any = None
        self._request_queue: Optional[queue.Queue] = None
        self._response_queue: Optional[queue.Queue] = None
        self._initialized = False
        self._mcp_patch_applied = False

    @property
    def name(self) -> str:
        return "mcp"

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the MCP plugin by starting the background thread."""
        if self._initialized:
            return
        self._ensure_thread()
        self._initialized = True

    def shutdown(self) -> None:
        """Shutdown the MCP plugin and clean up resources."""
        if self._request_queue:
            self._request_queue.put(None)  # Signal shutdown
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._tool_cache = {}
        self._loop = None
        self._thread = None
        self._manager = None
        self._request_queue = None
        self._response_queue = None
        self._initialized = False

    def get_tool_schemas(self) -> List[ToolSchema]:
        """Return ToolSchemas for all discovered MCP tools."""
        if not self._initialized:
            self.initialize()

        schemas = []
        for server_name, tools in self._tool_cache.items():
            for tool in tools:
                try:
                    cleaned_schema = self._clean_schema_for_vertex(tool.inputSchema)
                    schema = ToolSchema(
                        name=tool.name,
                        description=tool.description,
                        parameters=cleaned_schema
                    )
                    schemas.append(schema)
                except Exception as exc:
                    print(f"[MCPToolPlugin] Error creating schema for {tool.name}: {exc}", file=sys.stderr)

        return schemas

    def get_executors(self) -> Dict[str, Callable[[Dict[str, Any]], Any]]:
        """Return executor mappings for all discovered MCP tools."""
        if not self._initialized:
            self.initialize()

        executors = {}
        for tools in self._tool_cache.values():
            for tool in tools:
                # Create a closure that captures the tool name
                def make_executor(toolname: str):
                    def executor(args: Dict[str, Any]) -> Dict[str, Any]:
                        return self._execute(toolname, args)
                    return executor
                executors[tool.name] = make_executor(tool.name)

        return executors

    def get_system_instructions(self) -> Optional[str]:
        """Return system instructions describing available MCP tools."""
        if not self._initialized:
            self.initialize()

        if not self._tool_cache:
            return None

        lines = ["You have access to the following MCP (Model Context Protocol) tools:"]

        for server_name, tools in self._tool_cache.items():
            lines.append(f"\nFrom '{server_name}' server:")
            for tool in tools:
                desc = tool.description or "No description"
                lines.append(f"  - {tool.name}: {desc}")

        return "\n".join(lines)

    def get_auto_approved_tools(self) -> List[str]:
        """MCP tools require permission - return empty list."""
        return []

    def get_user_commands(self) -> List[UserCommand]:
        """MCP plugin provides model tools only, no user commands."""
        return []

    def _ensure_mcp_patch(self):
        """Lazily import mcp and apply the JSON-RPC validation patch."""
        if self._mcp_patch_applied:
            return

        from mcp import types as mcp_types

        # Store original
        _original_validate_json = mcp_types.JSONRPCMessage.model_validate_json.__func__

        class SkipMessage(Exception):
            """Raised to signal a non-JSON-RPC message that should be skipped."""
            pass

        @classmethod
        def filtered_validate_json(cls, json_data, *args, **kwargs):
            """Wrapper that filters out non-JSON-RPC messages before validation."""
            if isinstance(json_data, bytes):
                json_data = json_data.decode('utf-8', errors='replace')

            line = json_data.strip()

            # Quick checks before expensive parsing
            if not line or not line.startswith('{'):
                print(f"[Filtered non-JSON]: {line}", file=sys.stderr)
                raise SkipMessage(line)

            # Validate it's actually JSON-RPC 2.0
            try:
                data = json.loads(line)
                if not isinstance(data, dict) or data.get('jsonrpc') != '2.0':
                    print(f"[Filtered non-JSONRPC]: {line}", file=sys.stderr)
                    raise SkipMessage(line)
            except json.JSONDecodeError:
                print(f"[Filtered invalid JSON]: {line}", file=sys.stderr)
                raise SkipMessage(line)

            # It's valid JSON-RPC, let Pydantic parse it properly
            return _original_validate_json(cls, json_data, *args, **kwargs)

        # Apply patch
        mcp_types.JSONRPCMessage.model_validate_json = filtered_validate_json
        self._mcp_patch_applied = True

    def _load_mcp_registry(self, registry_path: Optional[str] = None) -> Dict[str, Any]:
        """Load MCP registry from specified path or default locations."""
        default_paths = [
            os.path.join(os.getcwd(), '.mcp.json'),
            os.path.expanduser('~/.mcp.json')
        ]
        if registry_path:
            paths = [registry_path] + default_paths
        else:
            paths = default_paths

        for path in paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        return json.load(f)
                except Exception:
                    continue
        return {}

    def _clean_schema_for_vertex(self, schema: dict) -> dict:
        """Remove fields from JSON schema that Vertex AI doesn't support."""
        if not isinstance(schema, dict):
            return schema
        # Fields not supported by Vertex AI's Schema proto
        unsupported = {'$schema', '$id', '$ref', '$defs', 'definitions', 'additionalItems'}
        cleaned = {k: v for k, v in schema.items() if k not in unsupported}
        # Recursively clean nested schemas
        if 'properties' in cleaned and isinstance(cleaned['properties'], dict):
            cleaned['properties'] = {
                k: self._clean_schema_for_vertex(v) for k, v in cleaned['properties'].items()
            }
        if 'items' in cleaned:
            cleaned['items'] = self._clean_schema_for_vertex(cleaned['items'])
        return cleaned

    def _thread_main(self):
        """Background thread running the MCP event loop."""

        async def run_mcp_server():
            self._ensure_mcp_patch()
            from shared.mcp_context_manager import MCPClientManager

            registry = self._load_mcp_registry()
            servers = registry.get('mcpServers', {})

            # Expand env vars
            def expand_env(env_dict):
                result = {}
                for k, v in (env_dict or {}).items():
                    if isinstance(v, str) and v.startswith('${') and v.endswith('}'):
                        result[k] = os.environ.get(v[2:-1], '')
                    else:
                        result[k] = v
                return result

            manager = MCPClientManager()
            async with manager:
                for name, spec in servers.items():
                    try:
                        await manager.connect(
                            name,
                            spec.get('command'),
                            spec.get('args', []),
                            expand_env(spec.get('env', {}))
                        )
                    except Exception as exc:
                        print(f"[MCPToolPlugin] Connection error for {name}: {exc}", file=sys.stderr)

                # Cache tools
                self._tool_cache = {
                    name: list(conn.tools)
                    for name, conn in manager._connections.items()
                }
                self._manager = manager

                # Process requests from main thread
                while True:
                    try:
                        req = self._request_queue.get(timeout=0.1)
                        if req is None:  # Shutdown signal
                            break
                        toolname, args = req
                        try:
                            res = await manager.call_tool_auto(toolname, args)
                            self._response_queue.put(('ok', res))
                        except Exception as exc:
                            self._response_queue.put(('error', str(exc)))
                    except queue.Empty:
                        await asyncio.sleep(0.01)

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(run_mcp_server())
        except Exception as exc:
            print(f"[MCPToolPlugin] Thread error: {exc}", file=sys.stderr)
        finally:
            self._loop.close()

    def _ensure_thread(self):
        """Start the MCP background thread if not already running."""
        if self._thread is not None and self._thread.is_alive():
            return

        self._request_queue = queue.Queue()
        self._response_queue = queue.Queue()
        self._thread = threading.Thread(target=self._thread_main, daemon=True)
        self._thread.start()

        # Wait for tools to be discovered
        for _ in range(100):  # 10 second timeout
            if self._tool_cache:
                break
            time.sleep(0.1)

    def _execute(self, toolname: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an MCP tool via the background connection.

        Args:
            toolname: Name of the MCP tool to invoke.
            args: Dict containing tool parameters.

        Returns:
            Dict with 'result' containing tool invocation details or 'error'.
        """
        # Ensure MCP thread is running
        if not self._initialized:
            self.initialize()

        if not self._tool_cache:
            return {'error': 'MCP tools not available'}

        # Verify tool exists
        found = False
        for tools in self._tool_cache.values():
            for t in tools:
                if t.name == toolname:
                    found = True
                    break
            if found:
                break

        if not found:
            return {'error': f"Tool '{toolname}' not found on any MCP server"}

        try:
            # Send request to MCP thread
            self._request_queue.put((toolname, args))

            # Wait for response (30 second timeout)
            status, result = self._response_queue.get(timeout=30)

            if status == 'error':
                return {'error': result}

            out = {
                'tool': toolname,
                'isError': getattr(result, 'isError', False),
                'structured': getattr(result, 'structuredContent', None),
                'content': [getattr(c, 'text', None) for c in getattr(result, 'content', [])],
            }
            return {'result': out}

        except queue.Empty:
            return {'error': 'MCP tool call timed out'}
        except Exception as exc:
            return {'error': f"{type(exc).__name__}: {exc}"}


def create_plugin() -> MCPToolPlugin:
    """Factory function to create the MCP plugin instance."""
    return MCPToolPlugin()
