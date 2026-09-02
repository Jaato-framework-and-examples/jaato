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


# Phase 4 §4.4 tier-flip (audit `phase4_implementation_audits.md`
# Audit 1).  Pre-§4.4: ``daemon``, which meant Path D's runner-
# discover filter (``tier_filter="runner"``) excluded this plugin
# from the runner registry — making ``session_describe`` unreachable
# from the model post-§7c.  Post-§4.4: ``runner`` so the runner-side
# model loop can invoke ``session_describe`` directly; the
# description-callback bridges to the daemon via the
# ``description_updated`` NotificationFrame event_type
# (§4.4 sub-action B).  ``SessionManager._session_plugin`` (an
# independent instance constructed directly in __init__, not via
# PluginRegistry discovery) is unaffected by the tier-flip and
# continues to handle save/load/delete daemon-side.
PLUGIN_TIER = "runner"


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
    'PLUGIN_TIER',
]
