"""Permission plugin for controlling tool execution access.

This plugin provides blacklist/whitelist-based permission control for tool
execution, with support for interactive channel approval when policy is ambiguous.

Also includes sanitization features for:
- Shell injection prevention
- Dangerous command blocking
- Path scope validation (sandbox to cwd)
"""

# Plugin kind identifier for registry discovery
PLUGIN_KIND = "tool"

from .policy import PermissionPolicy, PermissionDecision, PolicyMatch
from .config_loader import load_config, validate_config, PermissionConfig
from .channels import Channel, ConsoleChannel, WebhookChannel, ChannelResponse
from .plugin import PermissionPlugin, create_plugin
from .sanitization import (
    SanitizationConfig,
    SanitizationResult,
    PathScopeConfig,
    sanitize_command,
    check_shell_injection,
    check_dangerous_command,
    check_path_scope,
    create_strict_config,
    create_permissive_config,
)

__all__ = [
    # Policy
    'PermissionPolicy',
    'PermissionDecision',
    'PolicyMatch',
    # Config
    'PermissionConfig',
    'load_config',
    'validate_config',
    # Channels
    'Channel',
    'ConsoleChannel',
    'WebhookChannel',
    'ChannelResponse',
    # Plugin
    'PermissionPlugin',
    'create_plugin',
    # Sanitization
    'SanitizationConfig',
    'SanitizationResult',
    'PathScopeConfig',
    'sanitize_command',
    'check_shell_injection',
    'check_dangerous_command',
    'check_path_scope',
    'create_strict_config',
    'create_permissive_config',
]
