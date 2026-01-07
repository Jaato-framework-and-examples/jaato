# shared/plugins/hidden_content_filter/__init__.py
"""Hidden content filter plugin for stripping framework-internal content.

This plugin removes <hidden>...</hidden> tagged content from the output
stream before it reaches any client. Used for framework-internal instructions
(like task completion spurs) that should be visible to the model but not users.
"""

from .plugin import HiddenContentFilterPlugin, create_plugin

__all__ = ["HiddenContentFilterPlugin", "create_plugin"]
