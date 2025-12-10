"""
jaato - Just Another Agentic Tool Orchestrator

Public API for external plugins.
This module re-exports the core components from the internal 'shared' module.
"""

# Core client and registry
from shared import JaatoClient
from shared import PluginRegistry

# Provider-agnostic types for plugin development
from shared.plugins.model_provider.types import (
    # Enums
    Role,
    FinishReason,

    # Message types
    Message,
    Part,

    # Tool types
    ToolSchema,
    FunctionCall,
    ToolResult,

    # Response types
    ProviderResponse,
    TokenUsage,

    # Attachments
    Attachment,
)

# Plugin base classes
from shared.plugins.base import UserCommand, CommandParameter

# Public API
__all__ = [
    # Client and registry
    "JaatoClient",
    "PluginRegistry",

    # Enums
    "Role",
    "FinishReason",

    # Message types
    "Message",
    "Part",

    # Tool types
    "ToolSchema",
    "FunctionCall",
    "ToolResult",

    # Response types
    "ProviderResponse",
    "TokenUsage",

    # Attachments
    "Attachment",

    # Plugin support
    "UserCommand",
    "CommandParameter",
]

__version__ = "0.1.0"
