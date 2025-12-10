"""Moon Phase Calculator Plugin for jaato.

A plugin that calculates moon phases for any date, providing phase names
and illumination percentages using astronomical algorithms.
"""

from .plugin import MoonPhasePlugin

PLUGIN_INFO = {
    "name": "moon_phase",
    "description": "Calculate moon phases and lunar information",
    "version": "1.0.0",
    "author": "External Developer",
}


def create_plugin():
    """Factory function for plugin discovery.

    This function is called by the jaato PluginRegistry during discovery.

    Returns:
        MoonPhasePlugin: A new instance of the plugin.
    """
    return MoonPhasePlugin()
