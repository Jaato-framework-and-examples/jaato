"""Subagent plugin for task delegation to specialized subagents.

This plugin enables the parent model to spawn subagents with their own
tool configurations, system instructions, and model selection.

Example usage:
    from shared.plugins import PluginRegistry
    from shared.plugins.subagent import SubagentPlugin, SubagentProfile

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
                'max_turns': 5,
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

from .plugin import SubagentPlugin, create_plugin
from .config import SubagentConfig, SubagentProfile, SubagentResult

__all__ = [
    'SubagentPlugin',
    'SubagentConfig',
    'SubagentProfile',
    'SubagentResult',
    'create_plugin',
]
