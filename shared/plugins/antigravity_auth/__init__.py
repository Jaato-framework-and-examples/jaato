"""Antigravity authentication plugin.

Provides user commands for Google OAuth authentication with Antigravity backend.

Commands:
    antigravity-auth login        - Open browser for Google OAuth
    antigravity-auth code <code>  - Complete login with authorization code
    antigravity-auth logout       - Clear stored OAuth tokens
    antigravity-auth status       - Show authentication status
    antigravity-auth accounts     - List all authenticated accounts
"""

# Plugin kind for registry discovery
PLUGIN_KIND = "tool"

from .plugin import AntigravityAuthPlugin, create_plugin

__all__ = ["AntigravityAuthPlugin", "create_plugin", "PLUGIN_KIND"]
