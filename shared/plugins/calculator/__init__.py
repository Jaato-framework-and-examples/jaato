# shared/plugins/calculator/__init__.py

from .plugin import CalculatorPlugin

PLUGIN_INFO = {
    "name": "calculator",
    "description": "Mathematical calculation tools",
    "version": "1.0.0",
    "author": "External Developer",
}

def create_plugin():
    """Factory function called by registry."""
    return CalculatorPlugin()
