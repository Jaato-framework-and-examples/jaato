"""WebMCP plugin -- invoke the tools a web page declares for agents.

See :mod:`shared.plugins.webmcp.plugin` for why the page's toolset is
discovered through two stable tools rather than registered as schemas, and
:mod:`shared.plugins.webmcp.browser` for the measured Chrome 153 API contract
(which differs from the published explainer in six places).

Assessment and rationale: ``docs/design/webmcp.md``.
"""

from .browser import WebMCPPage, WebMCPUnsupportedError
from .plugin import WebMCPPlugin, create_plugin

PLUGIN_KIND = "tool"
PLUGIN_TIER = "runner"

__all__ = [
    "WebMCPPlugin",
    "WebMCPPage",
    "WebMCPUnsupportedError",
    "create_plugin",
]
