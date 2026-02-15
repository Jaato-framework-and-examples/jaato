"""Plugin system for tool discovery and management.

This package provides a plugin architecture for managing tool implementations
that can be discovered, exposed/unexposed, and used by the AI tool runner.

Usage:
    from shared.plugins import PluginRegistry

    registry = PluginRegistry()
    registry.discover()

    # List available plugins
    print(registry.list_available())  # ['cli', 'mcp', ...]

    # Expose specific plugins' tools to the model
    registry.expose_tool('cli', config={'extra_paths': ['/usr/local/bin']})

    # Get tools for exposed plugins
    tool_schemas = registry.get_exposed_tool_schemas()
    executors = registry.get_exposed_executors()

    # Unexpose when done
    registry.unexpose_all()

Lazy loading: All imports are deferred via __getattr__ to allow partial
installation (e.g., TUI distribution includes only formatters and todo).
"""

_LAZY_IMPORTS = {
    "ToolPlugin": (".base", "ToolPlugin"),
    "UserCommand": (".base", "UserCommand"),
    "CommandParameter": (".base", "CommandParameter"),
    "PermissionDisplayInfo": (".base", "PermissionDisplayInfo"),
    "parse_command_args": (".base", "parse_command_args"),
    "PluginRegistry": (".registry", "PluginRegistry"),
}


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        import importlib
        module = importlib.import_module(module_path, __name__)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    'ToolPlugin',
    'PluginRegistry',
    'UserCommand',
    'CommandParameter',
    'PermissionDisplayInfo',
    'parse_command_args',
]
