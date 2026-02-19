"""GitHub authentication plugin.

Provides user commands for GitHub device code OAuth authentication.
This allows users to authenticate with GitHub for GitHub Copilot/Models access
without needing to manually create a Personal Access Token.

Commands:
    github-auth login        - Start device code OAuth flow
    github-auth poll         - Poll for authorization (if not auto-polling)
    github-auth logout       - Clear stored OAuth tokens
    github-auth status       - Show authentication status
"""

# Plugin kind for registry discovery
PLUGIN_KIND = "tool"

# Auth plugins work without an active session/provider connection.
# The daemon loads them at startup so commands are available immediately.
SESSION_INDEPENDENT = True

from .plugin import GitHubAuthPlugin, create_plugin

__all__ = ["GitHubAuthPlugin", "create_plugin", "PLUGIN_KIND", "SESSION_INDEPENDENT"]
