# Shared modules package
#
# This module provides a unified import surface for the jaato orchestrator framework.
# Clients can import everything they need from a single location:
#
#   from shared import (
#       genai, types,  # Google GenAI SDK
#       ToolExecutor, run_function_call_loop, TokenLedger,
#       PluginRegistry, PermissionPlugin, active_cert_bundle,
#   )

# Google GenAI SDK - re-exported for convenience
from google import genai
from google.genai import types

# Token accounting
from .token_accounting import TokenLedger, generate_with_ledger

# Tool execution
from .ai_tool_runner import (
    ToolExecutor,
    run_function_call_loop,
    run_single_prompt,
    extract_function_calls,
    extract_text_from_parts,
)

# Plugin system
from .plugins.registry import PluginRegistry
from .plugins.permission import PermissionPlugin

# Utilities
from .ssl_helper import active_cert_bundle, normalize_ca_env_vars

__all__ = [
    # Google GenAI SDK
    "genai",
    "types",
    # Token accounting
    "TokenLedger",
    "generate_with_ledger",
    # Tool execution
    "ToolExecutor",
    "run_function_call_loop",
    "run_single_prompt",
    "extract_function_calls",
    "extract_text_from_parts",
    # Plugin system
    "PluginRegistry",
    "PermissionPlugin",
    # Utilities
    "active_cert_bundle",
    "normalize_ca_env_vars",
]