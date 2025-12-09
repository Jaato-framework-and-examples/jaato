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

# Token accounting
from .token_accounting import TokenLedger, generate_with_ledger

# Tool execution
from .ai_tool_runner import ToolExecutor

# Core client
from .jaato_client import JaatoClient

# Plugin system
from .plugins.registry import PluginRegistry
from .plugins.permission import PermissionPlugin
from .plugins.todo import TodoPlugin

# Model provider types (provider-agnostic)
from .plugins.model_provider import (
    ModelProviderPlugin,
    ProviderConfig,
    load_provider,
    discover_providers,
)
from .plugins.model_provider.types import (
    Message,
    Part,
    Role,
    ToolSchema,
    ToolResult,
    FunctionCall,
    ProviderResponse,
    TokenUsage,
    FinishReason,
    Attachment,
)

# Utilities
from .ssl_helper import active_cert_bundle, normalize_ca_env_vars

__all__ = [
    # Token accounting
    "TokenLedger",
    "generate_with_ledger",
    # Tool execution
    "ToolExecutor",
    # Core client
    "JaatoClient",
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
]