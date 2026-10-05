"""References plugin for managing documentation source injection.

This plugin enables users to configure reference sources (documentation, specs,
guides, etc.) that can be injected into the model's context. Sources can be:

- AUTO: Automatically included in system instructions at startup
- SELECTABLE: User chooses which to include via interactive selection

The model is responsible for fetching content using existing tools (CLI, MCP, etc.).
This plugin manages the catalog and handles user selection via three protocols:

- Console: Interactive terminal prompts
- Webhook: HTTP-based external approval systems
- File: Filesystem-based for automation/scripting

Example usage:

    from jaato_server.shared.plugins.references import ReferencesPlugin, create_plugin

    # Create and initialize plugin
    plugin = create_plugin()
    plugin.initialize({
        "channel_type": "console",
    })

    # Use via tool executors (for LLM)
    executors = plugin.get_executors()
    result = executors["selectReferences"]({
        "context": "Need API documentation for endpoint implementation"
    })

    # Or register with plugin registry
    from jaato_server.shared.plugins import PluginRegistry
    registry = PluginRegistry()
    registry.discover()
    registry.expose_tool("references")
"""

# Plugin kind identifier for registry discovery
PLUGIN_KIND = "tool"

PLUGIN_TIER = "runner"

#: Modules this package imports lazily (imported when the entry handler is registered).
#: The pool template imports them before it forks, so a session's
#: slot inherits them shared instead of importing them privately.
#: A literal tuple of strings, read without importing the package;
#: nothing listed may read session-scoped state at import time.
#: See jaato_server/server/runner/template_preload.py.
PLUGIN_PRELOAD = (
    "jaato_server.shared.plugins.references.entry_handler",
)
from .models import (
    SourceType,
    InjectionMode,
    ReferenceSource,
    SelectionRequest,
    SelectionResponse,
)
from .channels import (
    SelectionChannel,
    ConsoleSelectionChannel,
    WebhookSelectionChannel,
    FileSelectionChannel,
    create_channel,
)
from .config_loader import (
    ReferencesConfig,
    ConfigValidationError,
    load_config,
    validate_config,
    create_default_config,
    discover_references,
)
from .plugin import ReferencesPlugin, create_plugin

__all__ = [
    # Models
    'SourceType',
    'InjectionMode',
    'ReferenceSource',
    'SelectionRequest',
    'SelectionResponse',
    # Channels
    'SelectionChannel',
    'ConsoleSelectionChannel',
    'WebhookSelectionChannel',
    'FileSelectionChannel',
    'create_channel',
    # Config
    'ReferencesConfig',
    'ConfigValidationError',
    'load_config',
    'validate_config',
    'create_default_config',
    'discover_references',
    # Plugin
    'ReferencesPlugin',
    'create_plugin',
]
