"""Permission plugin for controlling tool execution access.

This plugin provides blacklist/whitelist-based permission control for tool
execution, with support for interactive actor approval when policy is ambiguous.
"""

from .policy import PermissionPolicy, PermissionDecision
from .config_loader import load_config, validate_config, PermissionConfig
from .actors import Actor, ConsoleActor, WebhookActor, ActorResponse
from .plugin import PermissionPlugin, create_plugin

__all__ = [
    'PermissionPolicy',
    'PermissionDecision',
    'PermissionConfig',
    'load_config',
    'validate_config',
    'Actor',
    'ConsoleActor',
    'WebhookActor',
    'ActorResponse',
    'PermissionPlugin',
    'create_plugin',
]
