"""Permission plugin for controlling tool execution access.

This plugin provides blacklist/whitelist-based permission control for tool
execution, with support for interactive actor approval when policy is ambiguous.

Also includes sanitization features for:
- Shell injection prevention
- Dangerous command blocking
- Path scope validation (sandbox to cwd)
"""

from .policy import PermissionPolicy, PermissionDecision, PolicyMatch
from .config_loader import load_config, validate_config, PermissionConfig
from .actors import Actor, ConsoleActor, WebhookActor, ActorResponse
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
    # Actors
    'Actor',
    'ConsoleActor',
    'WebhookActor',
    'ActorResponse',
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
