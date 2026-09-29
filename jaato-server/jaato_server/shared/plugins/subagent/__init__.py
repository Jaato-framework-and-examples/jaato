"""Subagent plugin for task delegation to specialized subagents.

This plugin enables the parent model to spawn subagents with their own
tool configurations, system instructions, and model selection.

Example usage:
    from jaato_server.shared.plugins import PluginRegistry
    from jaato_server.shared.plugins.subagent import SubagentPlugin, SubagentProfile

    # Create and configure the plugin
    plugin = SubagentPlugin()
    plugin.initialize({
        'project': 'my-project',
        'location': 'us-central1',
        'default_model': 'gemini-2.5-flash',
        'profiles': {
            'code_assistant': {
                'description': 'Subagent for code tasks',
                'plugins': ['cli'],
            }
        }
    })

    # Or add profiles programmatically
    plugin.add_profile(SubagentProfile(
        name='research_agent',
        description='Agent for research tasks',
        plugins=['mcp'],
        system_instructions='Focus on finding accurate information.',
    ))

    # Register with plugin registry
    registry = PluginRegistry()
    registry.discover()
    # Plugin will be auto-discovered if in the plugins directory
"""

# Plugin kind identifier for registry discovery
PLUGIN_KIND = "tool"

PLUGIN_TIER = "runner"
# Lazy (#1267).  Importing ``subagent.config`` (the ``SubagentProfile``
# schema, read by every profile resolver and by ``jaato-scaffold``) runs this
# ``__init__`` first.  An eager ``from .plugin import ...`` here made that
# import pull in the whole plugin: GC, message delivery, the thread pool.
# The registry still finds ``create_plugin`` through ``getattr``, which is
# what ``__getattr__`` below answers.
_LAZY_IMPORTS = {
    "SubagentPlugin": ".plugin",
    "create_plugin": ".plugin",
    "SubagentConfig": ".config",
    "SubagentProfile": ".config",
    "SubagentResult": ".config",
    "ProfileDiscoveryResult": ".config",
    "discover_profiles": ".config",
    "SecretResolver": ".config",
    "SecretResolutionError": ".config",
    "SecretResolveContext": ".config",
    "AppSecretReference": ".config",
    "APP_SECRET_SCHEME": ".config",
    "parse_app_secret_reference": ".config",
    "reset_secret_resolvers": ".config",
}


def __getattr__(name):
    module_path = _LAZY_IMPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    value = getattr(importlib.import_module(module_path, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_IMPORTS))


__all__ = [
    'SubagentPlugin',
    'SubagentConfig',
    'SubagentProfile',
    'SubagentResult',
    'ProfileDiscoveryResult',
    'discover_profiles',
    'create_plugin',
    'SecretResolver',
    'SecretResolutionError',
    'SecretResolveContext',
    'AppSecretReference',
    'APP_SECRET_SCHEME',
    'parse_app_secret_reference',
    'reset_secret_resolvers',
]
