"""Sandbox Manager plugin for runtime path permission management.

This plugin enables users to dynamically grant or revoke filesystem access
at runtime through a three-tier configuration model:

1. Global (~/.jaato/sandbox_paths.json) - User-wide settings
2. Workspace (<workspace>/.jaato/sandbox.json) - Project-specific settings
3. Session (<workspace>/.jaato/sessions/<id>/sandbox.json) - Runtime overrides

User Commands:
    sandbox list   - Show all effective paths from all three levels
    sandbox add    - Grant temporary access for current session
    sandbox remove - Block a path for this session (even if globally allowed)
"""

from .plugin import SandboxManagerPlugin, SandboxPath, SandboxConfig, create_plugin

# Plugin kind identifier for registry discovery
PLUGIN_KIND = "tool"

__all__ = [
    'SandboxManagerPlugin',
    'SandboxPath',
    'SandboxConfig',
    'create_plugin',
]
