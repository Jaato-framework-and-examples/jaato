"""Interactive shell plugin for driving user-interactive commands.

This plugin provides tools that let the model spawn persistent PTY sessions
and interact with programs requiring back-and-forth input: REPLs, password
prompts, wizards, debuggers, SSH sessions, etc.

The model reads whatever the process outputs and decides what to type next —
no expect patterns needed. The intelligence is in the model, not the tool.
"""

from .plugin import InteractiveShellPlugin, create_plugin

# Plugin kind identifier for registry discovery
PLUGIN_KIND = "tool"

__all__ = [
    'InteractiveShellPlugin',
    'create_plugin',
    'PLUGIN_KIND',
]
