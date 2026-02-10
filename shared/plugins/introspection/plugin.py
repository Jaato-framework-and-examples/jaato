"""Tool Discovery & Introspection Plugin implementation.

This plugin provides tools for agents to discover and query available tools
at runtime, enabling dynamic tool selection and self-documentation.

The plugin supports deferred tool loading for token economy:
- Only "core" tools are loaded into initial context
- "discoverable" tools can be queried via list_tools and get_tool_schemas
- Models request schemas on-demand, reducing initial context overhead
"""

import threading
from typing import Any, Callable, Dict, List, Optional, Set

from ..model_provider.types import ToolSchema, TOOL_CATEGORIES
from ..streaming import StreamingCapable

# Thread-local storage for session reference per agent context
# This prevents subagents from overwriting parent's session reference
_thread_local = threading.local()


class IntrospectionPlugin:
    """Plugin that provides tool discovery and introspection capabilities.

    This plugin exposes tools for the LLM to:
    - list_tools: Discover available tools with optional category filtering
    - get_tool_schemas: Get full schemas for specific tools (enables on-demand loading)

    The plugin receives access to the PluginRegistry via set_plugin_registry(),
    which is called automatically by the registry during expose_tool().

    Deferred Loading:
        Tools have a "discoverability" attribute:
        - "core": Always in initial context
        - "discoverable": Schema provided on-demand via get_tool_schemas

        This plugin's tools are marked as "core" since they're needed for discovery.
        The _accessed_tools set tracks which tools the model has requested schemas for,
        useful for telemetry and GC decisions.
    """

    def __init__(self):
        self._initialized = False
        self._registry = None  # Set via set_plugin_registry()
        self._accessed_tools: Set[str] = set()  # Track tools model has requested

    @property
    def name(self) -> str:
        return "introspection"

    @property
    def _session(self):
        """Get the session for the current thread context.

        Uses thread-local storage so each agent (main or subagent) gets
        its own session reference, preventing subagents from overwriting
        the parent's session.
        """
        return getattr(_thread_local, 'session', None)

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the introspection plugin.

        Args:
            config: Optional configuration dict (currently unused).
        """
        self._initialized = True

    def shutdown(self) -> None:
        """Shutdown the introspection plugin."""
        self._initialized = False

    def set_plugin_registry(self, registry) -> None:
        """Receive the plugin registry for tool discovery.

        This is called automatically by the PluginRegistry during expose_tool()
        when it detects this method exists on the plugin.

        Args:
            registry: The PluginRegistry instance.
        """
        self._registry = registry

    def set_session(self, session) -> None:
        """Receive the session for tool activation.

        This is called automatically by the plugin wiring system. When tools
        are discovered via get_tool_schemas, they need to be activated in
        the session so the provider can use them.

        Stores in thread-local storage so each agent context gets its own session.

        Args:
            session: The JaatoSession instance.
        """
        _thread_local.session = session

    def get_tool_schemas(self) -> List[ToolSchema]:
        """Return tool schemas for introspection tools.

        Both tools are marked as 'core' discoverability since they're required
        for the deferred tool loading mechanism to work.
        """
        return [
            ToolSchema(
                name="list_tools",
                description="Discover available tools. "
                           "Without a category: returns available categories with tool counts. "
                           "With a category: returns tools in that category with brief descriptions.",
                parameters={
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "description": f"Category to list tools from. "
                                         f"If omitted, returns category summary. "
                                         f"Categories: {', '.join(TOOL_CATEGORIES)}",
                            "enum": TOOL_CATEGORIES,
                        },
                        "verbose": {
                            "type": "boolean",
                            "description": "If true, include full descriptions. Default is false.",
                            "default": False,
                        },
                    },
                    "required": []
                },
                category="system",
                discoverability="core",
            ),
            ToolSchema(
                name="get_tool_schemas",
                description="Get detailed schemas for specific tools. "
                           "Use this after list_tools to learn how to call tools you need. "
                           "Returns full parameter specifications, types, "
                           "required/optional flags, and descriptions.",
                parameters={
                    "type": "object",
                    "properties": {
                        "names": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Names of the tools to get schemas for."
                        }
                    },
                    "required": ["names"]
                },
                category="system",
                discoverability="core",
            ),
        ]

    def get_executors(self) -> Dict[str, Callable[[Dict[str, Any]], Any]]:
        """Return the executors for introspection tools."""
        return {
            "list_tools": self._execute_list_tools,
            "get_tool_schemas": self._execute_get_tool_schemas,
        }

    def get_accessed_tools(self) -> Set[str]:
        """Get the set of tools the model has requested schemas for.

        This is useful for:
        - Telemetry: Understanding which tools the model uses
        - GC decisions: Preserving schemas the model has accessed

        Returns:
            Set of tool names that have been accessed via get_tool_schemas.
        """
        return self._accessed_tools.copy()

    def clear_accessed_tools(self) -> None:
        """Clear the accessed tools tracking.

        Call this when resetting the session or for fresh tracking.
        """
        self._accessed_tools.clear()

    def evaluate_gc_policy(self, entry: 'SourceEntry', budget: 'InstructionBudget') -> 'GCPolicy':
        """Evaluate GC policy for a discovered tool schema entry.

        This method is called by the GC system to determine whether a tool schema
        with CONDITIONAL policy should be kept (LOCKED) or removed (EPHEMERAL).

        Logic:
        - If the tool has been successfully executed at least once → LOCKED
        - Otherwise → EPHEMERAL (can be GC'd and re-discovered if needed later)

        Args:
            entry: The instruction budget entry for the tool schema
            budget: The full instruction budget (provides session_id)

        Returns:
            GCPolicy.LOCKED if tool has been used successfully, EPHEMERAL otherwise
        """
        from ..instruction_budget import GCPolicy

        # Extract tool name from entry metadata
        tool_name = entry.metadata.get("tool_name") if entry.metadata else None
        if not tool_name:
            return GCPolicy.EPHEMERAL

        # Get session ID from budget
        session_id = budget.session_id
        if not session_id:
            return GCPolicy.EPHEMERAL

        # Query reliability plugin for usage statistics
        if self._registry:
            reliability = self._registry.get_plugin('reliability')
            if reliability and hasattr(reliability, 'has_successful_execution'):
                try:
                    if reliability.has_successful_execution(tool_name, session_id):
                        return GCPolicy.LOCKED
                except Exception:
                    # If query fails, default to ephemeral
                    pass

        return GCPolicy.EPHEMERAL

    def get_system_instructions(self) -> Optional[str]:
        """Return system instructions for the introspection plugin."""
        return (
            "You have a DYNAMIC tool system with discoverable capabilities.\n\n"
            "CRITICAL: Before saying 'I cannot do X', ALWAYS explore available tools first!\n\n"
            "TOOL DISCOVERY WORKFLOW:\n"
            "1. `list_tools()` - See all categories with descriptions and tool counts\n"
            "2. `list_tools(category='...')` - See tools in a specific category\n"
            "3. `get_tool_schemas(names=[...])` - Get full schemas for tools you need\n"
            "4. Call the tools using the schema information\n\n"
            "CATEGORY QUICK REFERENCE:\n"
            "- coordination: Task tracking, TODO, DELEGATE work, SUBAGENTS, parallel execution\n"
            "- filesystem: Read, write, search files\n"
            "- code: Analysis, editing, LSP diagnostics\n"
            "- web: Fetch URLs, web search\n"
            "- system: Shell commands, environment\n"
            "- communication: Ask user questions\n\n"
            "STREAMING TOOLS:\n"
            "Tools with `streaming: true` support incremental results. To use streaming:\n"
            "- Call `<tool_name>:stream` instead of `<tool_name>` (e.g., `grep_content:stream`)\n"
            "- You'll receive a stream_id and initial results immediately\n"
            "- More results arrive automatically as the tool finds them\n"
            "- Call `dismiss_stream(stream_id)` when you have enough results"
        )

    def get_auto_approved_tools(self) -> List[str]:
        """Return introspection tools as auto-approved (read-only, no security implications)."""
        return ["list_tools", "get_tool_schemas"]

    def get_user_commands(self) -> List:
        """Return user commands (none for this plugin)."""
        return []

    def _execute_list_tools(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the list_tools tool.

        Args:
            args: Dictionary with optional 'category' and 'verbose' keys.

        Returns:
            - If no category: returns available categories with tool counts
            - If category specified: returns tools in that category
        """
        if not self._registry:
            return {"error": "Registry not available. Plugin not properly initialized."}

        category = args.get("category")
        verbose = args.get("verbose", False)

        # Get all tool schemas from exposed plugins
        all_schemas = self._registry.get_exposed_tool_schemas()

        # If no category specified, return category summary only
        if not category:
            # Category descriptions to help models understand what's available
            category_hints = {
                "filesystem": "Read, write, search, and navigate files and directories",
                "code": "Code analysis, editing, refactoring, and LSP diagnostics",
                "search": "Search files, content, patterns across the codebase",
                "memory": "Persistent memory, notes, context storage across sessions",
                "coordination": "Task tracking, TODO, DELEGATE work, SUBAGENTS, PARALLEL execution",
                "system": "Shell commands, environment, system operations",
                "web": "Fetch URLs, web search, external API access",
                "communication": "Ask user questions, request clarification, get input",
            }

            category_counts: Dict[str, int] = {}
            for schema in all_schemas:
                cat = schema.category or "uncategorized"
                category_counts[cat] = category_counts.get(cat, 0) + 1

            return {
                "categories": [
                    {
                        "name": cat,
                        "tool_count": count,
                        "description": category_hints.get(cat, ""),
                    }
                    for cat, count in sorted(category_counts.items())
                ],
                "total_tools": len(all_schemas),
                "hint": "Call list_tools(category='<name>') to see tools in a specific category.",
            }

        # Category specified - return tools in that category
        tools = []
        for schema in all_schemas:
            # Apply category filter (treat None as "uncategorized")
            schema_category = schema.category or "uncategorized"
            if schema_category != category:
                continue

            # Find which plugin provides this tool
            plugin = self._registry.get_plugin_for_tool(schema.name)
            plugin_source = plugin.name if plugin else "unknown"

            # Check if tool is enabled
            is_enabled = self._registry.is_tool_enabled(schema.name)

            # Check if tool supports streaming
            supports_streaming = False
            if plugin and isinstance(plugin, StreamingCapable):
                try:
                    supports_streaming = plugin.supports_streaming(schema.name)
                except Exception:
                    pass  # Plugin may not implement the method correctly

            # Build tool entry
            tool_entry = {
                "name": schema.name,
                "plugin_source": plugin_source,
                "enabled": is_enabled,
                "streaming": supports_streaming,
            }

            if verbose:
                tool_entry["description"] = schema.description
            else:
                # Use first sentence or truncate for concise output
                desc = schema.description
                first_sentence_end = desc.find(". ")
                if first_sentence_end > 0:
                    tool_entry["description"] = desc[:first_sentence_end + 1]
                elif len(desc) > 100:
                    tool_entry["description"] = desc[:97] + "..."
                else:
                    tool_entry["description"] = desc

            tools.append(tool_entry)

        # Sort by name for consistent output
        tools.sort(key=lambda t: t["name"])

        result = {
            "category": category,
            "tool_count": len(tools),
            "tools": tools,
        }

        if tools:
            result["hint"] = "Call get_tool_schemas(names=['<tool_name>']) to get full parameter details."

            # Add streaming hint if any tools support streaming
            streaming_tools = [t["name"] for t in tools if t.get("streaming")]
            if streaming_tools:
                result["streaming_hint"] = (
                    f"Tools with streaming=true support incremental results. "
                    f"Call '<tool_name>:stream' (e.g., '{streaming_tools[0]}:stream') "
                    f"to receive results as they're found. Use dismiss_stream(stream_id) when done."
                )

        return result

    def _execute_get_tool_schemas(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the get_tool_schemas tool.

        Args:
            args: Dictionary with required 'names' key (array of tool names).

        Returns:
            Dictionary with schemas for requested tools and tracking info.
        """
        if not self._registry:
            return {"error": "Registry not available. Plugin not properly initialized."}

        names = args.get("names", [])
        if not names:
            return {"error": "names is required (array of tool names)"}

        if not isinstance(names, list):
            names = [names]  # Handle single name as array

        # Get all available schemas for lookup
        all_schemas = self._registry.get_exposed_tool_schemas()
        schema_map = {s.name: s for s in all_schemas}
        available_tools = list(schema_map.keys())

        # Build results
        schemas = []
        not_found = []

        # Collect tools that need activation (discoverable tools not yet in provider)
        tools_to_activate = []

        for tool_name in names:
            if tool_name in schema_map:
                target_schema = schema_map[tool_name]

                # Track this access
                self._accessed_tools.add(tool_name)

                # Check if this is a discoverable tool that needs activation
                if getattr(target_schema, 'discoverability', 'discoverable') == 'discoverable':
                    tools_to_activate.append(tool_name)

                # Find plugin source
                plugin = self._registry.get_plugin_for_tool(tool_name)
                plugin_source = plugin.name if plugin else "unknown"

                # Build detailed schema response
                schema_entry = {
                    "name": target_schema.name,
                    "description": target_schema.description,
                    "plugin_source": plugin_source,
                    "enabled": self._registry.is_tool_enabled(tool_name),
                }

                if target_schema.category:
                    schema_entry["category"] = target_schema.category

                # Format parameters in a more readable way
                params = target_schema.parameters
                if params:
                    schema_entry["parameters"] = self._format_parameters(params)

                schemas.append(schema_entry)
            else:
                not_found.append(tool_name)

        # Build response
        result = {
            "schemas": schemas,
            "count": len(schemas),
        }

        if not_found:
            # Provide helpful suggestions for not found tools
            suggestions = {}
            for tool_name in not_found:
                similar = [t for t in available_tools if tool_name.lower() in t.lower()]
                if similar:
                    suggestions[tool_name] = similar[:3]

            result["not_found"] = not_found
            if suggestions:
                result["suggestions"] = suggestions
            result["hint"] = "Use list_tools() to see available tools."

        # Activate discovered tools so the model can actually call them
        # This adds the tool schemas to the provider's declared tools
        if tools_to_activate and self._session:
            activated = self._session.activate_discovered_tools(tools_to_activate)
            if activated:
                result["activated"] = activated
                result["activation_note"] = "These tools are now available to call."

        return result

    def _format_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Format JSON Schema parameters into a more readable structure.

        Args:
            params: JSON Schema object for parameters.

        Returns:
            Formatted parameter documentation.
        """
        if not params or params.get("type") != "object":
            return params

        properties = params.get("properties", {})
        required = set(params.get("required", []))

        formatted_params = {}
        for name, prop in properties.items():
            param_info = {
                "type": prop.get("type", "any"),
                "required": name in required,
            }

            if "description" in prop:
                param_info["description"] = prop["description"]

            if "enum" in prop:
                param_info["allowed_values"] = prop["enum"]

            if "default" in prop:
                param_info["default"] = prop["default"]

            if "items" in prop:
                param_info["items_type"] = prop["items"].get("type", "any")

            formatted_params[name] = param_info

        return formatted_params


def create_plugin() -> IntrospectionPlugin:
    """Factory function to create the introspection plugin instance."""
    return IntrospectionPlugin()
