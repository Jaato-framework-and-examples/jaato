"""Tool Discovery & Introspection Plugin implementation.

This plugin provides tools for agents to discover and query available tools
at runtime, enabling dynamic tool selection and self-documentation.
"""

from typing import Any, Callable, Dict, List, Optional

from ..model_provider.types import ToolSchema, TOOL_CATEGORIES


class IntrospectionPlugin:
    """Plugin that provides tool discovery and introspection capabilities.

    This plugin exposes tools for the LLM to:
    - list_tools: Discover available tools with optional category filtering
    - get_tool_schema: Get full documentation for a specific tool

    The plugin receives access to the PluginRegistry via set_plugin_registry(),
    which is called automatically by the registry during expose_tool().
    """

    def __init__(self):
        self._initialized = False
        self._registry = None  # Set via set_plugin_registry()

    @property
    def name(self) -> str:
        return "introspection"

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

    def get_tool_schemas(self) -> List[ToolSchema]:
        """Return tool schemas for introspection tools."""
        return [
            ToolSchema(
                name="list_tools",
                description="List all available tools that can be invoked. "
                           "Use this to discover what capabilities are available. "
                           "Returns tool names with brief descriptions. "
                           "Optionally filter by category for focused discovery.",
                parameters={
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "description": f"Optional category to filter tools. "
                                         f"Standard categories: {', '.join(TOOL_CATEGORIES)}",
                            "enum": TOOL_CATEGORIES + ["all"],
                        },
                        "verbose": {
                            "type": "boolean",
                            "description": "If true, include full descriptions instead of summaries. "
                                         "Default is false for concise output.",
                            "default": False,
                        },
                    },
                    "required": []
                },
                category="system",
            ),
            ToolSchema(
                name="get_tool_schema",
                description="Get detailed documentation for a specific tool. "
                           "Returns full parameter specifications, types, "
                           "required/optional flags, and descriptions. "
                           "Use this when you need to understand exactly how to call a tool.",
                parameters={
                    "type": "object",
                    "properties": {
                        "tool_name": {
                            "type": "string",
                            "description": "Name of the tool to get schema for."
                        }
                    },
                    "required": ["tool_name"]
                },
                category="system",
            ),
        ]

    def get_executors(self) -> Dict[str, Callable[[Dict[str, Any]], Any]]:
        """Return the executors for introspection tools."""
        return {
            "list_tools": self._execute_list_tools,
            "get_tool_schema": self._execute_get_tool_schema,
        }

    def get_system_instructions(self) -> Optional[str]:
        """Return system instructions for the introspection plugin."""
        return (
            "You have access to tool introspection capabilities.\n\n"
            "WHEN TO USE:\n"
            "- Use `list_tools` when you need to know what tools are available\n"
            "- Use `list_tools(category=...)` to find tools for a specific purpose\n"
            "- Use `get_tool_schema(tool_name=...)` when you need exact parameter details\n\n"
            "TIPS:\n"
            "- Categories help narrow down relevant tools: filesystem, code, search, etc.\n"
            "- get_tool_schema provides parameter types and required/optional flags\n"
            "- Use verbose=true in list_tools only when you need full descriptions"
        )

    def get_auto_approved_tools(self) -> List[str]:
        """Return introspection tools as auto-approved (read-only, no security implications)."""
        return ["list_tools", "get_tool_schema"]

    def _execute_list_tools(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the list_tools tool.

        Args:
            args: Dictionary with optional 'category' and 'verbose' keys.

        Returns:
            Dictionary with 'tools' list and metadata.
        """
        if not self._registry:
            return {"error": "Registry not available. Plugin not properly initialized."}

        category = args.get("category")
        verbose = args.get("verbose", False)

        # Normalize "all" category to None (no filter)
        if category == "all":
            category = None

        # Get all tool schemas from exposed plugins
        all_schemas = self._registry.get_exposed_tool_schemas()

        # Build tool list with plugin source info
        tools = []
        for schema in all_schemas:
            # Apply category filter if specified
            if category and schema.category != category:
                continue

            # Find which plugin provides this tool
            plugin = self._registry.get_plugin_for_tool(schema.name)
            plugin_source = plugin.name if plugin else "unknown"

            # Check if tool is enabled
            is_enabled = self._registry.is_tool_enabled(schema.name)

            # Build tool entry
            tool_entry = {
                "name": schema.name,
                "plugin_source": plugin_source,
                "enabled": is_enabled,
            }

            if schema.category:
                tool_entry["category"] = schema.category

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

        # Sort by category then name for consistent output
        tools.sort(key=lambda t: (t.get("category", "zzz"), t["name"]))

        # Gather category summary
        categories_found = set()
        for tool in tools:
            if "category" in tool:
                categories_found.add(tool["category"])

        result = {
            "total_count": len(tools),
            "tools": tools,
        }

        if category:
            result["filtered_by"] = category

        if categories_found:
            result["categories_present"] = sorted(categories_found)

        return result

    def _execute_get_tool_schema(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the get_tool_schema tool.

        Args:
            args: Dictionary with required 'tool_name' key.

        Returns:
            Dictionary with full tool schema details.
        """
        if not self._registry:
            return {"error": "Registry not available. Plugin not properly initialized."}

        tool_name = args.get("tool_name", "")
        if not tool_name:
            return {"error": "tool_name is required"}

        # Search for the tool in all exposed plugins
        all_schemas = self._registry.get_exposed_tool_schemas()
        target_schema = None
        for schema in all_schemas:
            if schema.name == tool_name:
                target_schema = schema
                break

        if not target_schema:
            # Provide helpful suggestions
            available_tools = [s.name for s in all_schemas]
            similar_tools = [t for t in available_tools if tool_name.lower() in t.lower()]

            error_msg = f"Tool '{tool_name}' not found."
            if similar_tools:
                error_msg += f" Similar tools: {', '.join(similar_tools[:5])}"
            else:
                error_msg += f" Use list_tools() to see available tools."

            return {"error": error_msg}

        # Find plugin source
        plugin = self._registry.get_plugin_for_tool(tool_name)
        plugin_source = plugin.name if plugin else "unknown"

        # Build detailed schema response
        result = {
            "name": target_schema.name,
            "description": target_schema.description,
            "plugin_source": plugin_source,
            "enabled": self._registry.is_tool_enabled(tool_name),
        }

        if target_schema.category:
            result["category"] = target_schema.category

        # Format parameters in a more readable way
        params = target_schema.parameters
        if params:
            result["parameters"] = self._format_parameters(params)

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
