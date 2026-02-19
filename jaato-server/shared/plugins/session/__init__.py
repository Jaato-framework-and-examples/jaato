"""Session persistence plugin for jaato.

This module provides session save/resume functionality, allowing
users to persist conversation history and resume sessions later.

Usage:
    from shared.plugins.session import (
        SessionPlugin,
        SessionConfig,
        SessionState,
        SessionInfo,
        create_plugin,
    )

    # Create and configure plugin
    plugin = create_plugin()
    plugin.initialize({'storage_path': '.jaato/sessions'})

    # Use with JaatoClient
    client.set_session_plugin(plugin, SessionConfig())
"""

from .base import (
    SessionPlugin,
    SessionConfig,
    SessionState,
    SessionInfo,
)
from .config_loader import load_session_config, save_session_config

# Plugin discovery marker
PLUGIN_KIND = "session"


def create_plugin() -> 'FileSessionPlugin':
    """Factory function to create the default session plugin.

    Returns:
        A FileSessionPlugin instance for file-based session persistence.
    """
    from .file_session import FileSessionPlugin
    return FileSessionPlugin()


__all__ = [
    'SessionPlugin',
    'SessionConfig',
    'SessionState',
    'SessionInfo',
    'create_plugin',
    'load_session_config',
    'save_session_config',
    'PLUGIN_KIND',
]
