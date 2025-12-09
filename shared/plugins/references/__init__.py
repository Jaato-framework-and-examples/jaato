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

    from shared.plugins.references import ReferencesPlugin, create_plugin

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
    from shared.plugins import PluginRegistry
    registry = PluginRegistry()
    registry.discover()
    registry.expose_tool("references")
"""

# Plugin kind identifier for registry discovery
PLUGIN_KIND = "tool"

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
    # Plugin
    'ReferencesPlugin',
    'create_plugin',
]
