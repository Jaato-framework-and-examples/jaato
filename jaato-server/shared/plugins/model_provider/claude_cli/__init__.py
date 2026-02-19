"""Claude CLI Model Provider Plugin.

This provider uses the Claude Code CLI as a backend, communicating via
the stream-json protocol instead of calling the Anthropic API directly.

Benefits:
- Use Claude Pro/Max subscription without API credits
- Leverage Claude Code's built-in tools (Read, Write, Bash, etc.)
- Automatic prompt caching and session management

Modes:
- delegated: CLI handles tool execution (default)
- passthrough: jaato handles tool execution via PluginRegistry

Configuration:
    Environment variables:
        JAATO_CLAUDE_CLI_MODE: "delegated" or "passthrough" (default: delegated)
        JAATO_CLAUDE_CLI_PATH: Path to claude CLI (default: "claude")

    ProviderConfig.extra:
        cli_mode: Override JAATO_CLAUDE_CLI_MODE
        cli_path: Override JAATO_CLAUDE_CLI_PATH
        max_turns: Maximum agentic turns (default: unlimited)
        permission_mode: CLI permission mode (default: None, uses CLI default)

Example:
    from shared.plugins.model_provider.claude_cli import ClaudeCLIProvider
    from shared.plugins.model_provider.base import ProviderConfig

    provider = ClaudeCLIProvider()
    provider.initialize(ProviderConfig(
        extra={
            "cli_mode": "delegated",
            "cli_path": "/usr/local/bin/claude",
        }
    ))
    provider.connect("sonnet")  # or "opus", or full model name

    provider.create_session(
        system_instruction="You are a helpful assistant.",
        tools=[],  # In delegated mode, CLI provides tools
    )

    response = provider.send_message("Hello!")
"""

from .provider import ClaudeCLIProvider
from .types import (
    CLIMode,
    MessageType,
    ContentBlockType,
    SystemMessage,
    AssistantMessage,
    UserMessage,
    ResultMessage,
    StreamEvent,
    TextBlock,
    ToolUseBlock,
    ToolResultBlock,
)
from .env import resolve_cli_path, resolve_cli_mode

__all__ = [
    # Main provider
    "ClaudeCLIProvider",
    # Types
    "CLIMode",
    "MessageType",
    "ContentBlockType",
    "SystemMessage",
    "AssistantMessage",
    "UserMessage",
    "ResultMessage",
    "StreamEvent",
    "TextBlock",
    "ToolUseBlock",
    "ToolResultBlock",
    # Environment
    "resolve_cli_path",
    "resolve_cli_mode",
]


def create_plugin() -> ClaudeCLIProvider:
    """Factory function for plugin discovery."""
    return ClaudeCLIProvider()
