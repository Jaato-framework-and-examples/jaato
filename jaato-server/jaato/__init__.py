"""
jaato - Just Another Agentic Tool Orchestrator

Public API for external plugins.
This module re-exports the core components from the internal 'shared' module.
"""

# Core client and registry
from shared import JaatoClient
from shared import PluginRegistry

# In-process convenience facade + the transport-agnostic ``session(mode=...)``
# entry that picks IPCClient vs InProcessClient. See
# docs/design/in-process-facade.md.
from jaato.in_process import InProcessClient, session

# Provider-agnostic types for plugin development
from jaato_sdk.plugins.model_provider.types import (
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

    # Cancellation
    CancelToken,
    CancelledException,
)

# Presentation context for display-aware agents
from jaato_sdk.events import PresentationContext, ClientType

# Plugin base classes
from jaato_sdk.plugins.base import UserCommand, CommandParameter

# Public API
__all__ = [
    # Client and registry
    "JaatoClient",
    "PluginRegistry",
    "InProcessClient",
    "session",

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

    # Cancellation
    "CancelToken",
    "CancelledException",

    # Presentation
    "PresentationContext",
    "ClientType",

    # Plugin support
    "UserCommand",
    "CommandParameter",
]

__version__ = "0.1.0"
