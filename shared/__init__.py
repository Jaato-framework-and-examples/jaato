# Shared modules package
#
# This module provides a unified import surface for the jaato orchestrator framework.
# Clients can import everything they need from a single location:
#
#   from shared import (
#       JaatoClient, ToolExecutor, TokenLedger,
#       PluginRegistry, PermissionPlugin, active_cert_bundle,
#   )
#
# Note: SDK types (genai, types) are no longer exported. Access the AI provider
# through JaatoClient or the model_provider plugin system instead.
#
# Lazy loading: All imports are deferred via __getattr__ to allow partial
# installation (e.g., TUI distribution includes only formatters and trace).

# Mapping from public name -> (module_path, attribute_name)
_LAZY_IMPORTS = {
    # Token accounting
    "TokenLedger": (".token_accounting", "TokenLedger"),
    "generate_with_ledger": (".token_accounting", "generate_with_ledger"),
    # Tool execution
    "ToolExecutor": (".ai_tool_runner", "ToolExecutor"),
    # Core client and runtime
    "JaatoClient": (".jaato_client", "JaatoClient"),
    "JaatoRuntime": (".jaato_runtime", "JaatoRuntime"),
    "JaatoSession": (".jaato_session", "JaatoSession"),
    "ActivityPhase": (".jaato_session", "ActivityPhase"),
    # Plugin system
    "PluginRegistry": (".plugins.registry", "PluginRegistry"),
    "PermissionPlugin": (".plugins.permission", "PermissionPlugin"),
    "TodoPlugin": (".plugins.todo", "TodoPlugin"),
    # Model provider
    "ModelProviderPlugin": (".plugins.model_provider", "ModelProviderPlugin"),
    "ProviderConfig": (".plugins.model_provider", "ProviderConfig"),
    "load_provider": (".plugins.model_provider", "load_provider"),
    "discover_providers": (".plugins.model_provider", "discover_providers"),
    # Provider-agnostic types
    "Message": (".plugins.model_provider.types", "Message"),
    "Part": (".plugins.model_provider.types", "Part"),
    "Role": (".plugins.model_provider.types", "Role"),
    "ToolSchema": (".plugins.model_provider.types", "ToolSchema"),
    "ToolResult": (".plugins.model_provider.types", "ToolResult"),
    "FunctionCall": (".plugins.model_provider.types", "FunctionCall"),
    "ProviderResponse": (".plugins.model_provider.types", "ProviderResponse"),
    "TokenUsage": (".plugins.model_provider.types", "TokenUsage"),
    "FinishReason": (".plugins.model_provider.types", "FinishReason"),
    "Attachment": (".plugins.model_provider.types", "Attachment"),
    # Utilities
    "active_cert_bundle": (".ssl_helper", "active_cert_bundle"),
    "normalize_ca_env_vars": (".ssl_helper", "normalize_ca_env_vars"),
    "configure_utf8_output": (".console_encoding", "configure_utf8_output"),
}

# UTF-8 console configuration is applied lazily: only when someone actually
# imports configure_utf8_output (or when the full server package triggers it).
_utf8_configured = False


def __getattr__(name):
    global _utf8_configured
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        import importlib
        module = importlib.import_module(module_path, __name__)
        value = getattr(module, attr_name)
        globals()[name] = value

        # Side-effect: configure UTF-8 output when console_encoding is first loaded.
        if module_path == ".console_encoding" and not _utf8_configured:
            _utf8_configured = True
            try:
                value()  # configure_utf8_output()
            except Exception:
                pass

        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Token accounting
    "TokenLedger",
    "generate_with_ledger",
    # Tool execution
    "ToolExecutor",
    # Core client and runtime
    "JaatoClient",
    "JaatoRuntime",
    "JaatoSession",
    "ActivityPhase",
    # Plugin system
    "PluginRegistry",
    "PermissionPlugin",
    "TodoPlugin",
    # Model provider
    "ModelProviderPlugin",
    "ProviderConfig",
    "load_provider",
    "discover_providers",
    # Provider-agnostic types
    "Message",
    "Part",
    "Role",
    "ToolSchema",
    "ToolResult",
    "FunctionCall",
    "ProviderResponse",
    "TokenUsage",
    "FinishReason",
    "Attachment",
    # Utilities
    "active_cert_bundle",
    "normalize_ca_env_vars",
    "configure_utf8_output",
]
