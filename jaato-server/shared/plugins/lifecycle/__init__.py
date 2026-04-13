"""Lifecycle plugin — agent completion signaling.

Provides a ``signal_completion`` tool that lets the main agent declare
its work is done.  This emits ``AgentCompletedEvent`` through the UI
hooks, enabling downstream reactors (e.g. memory-advisor) to trigger.

Subagents get this for free from the subagent plugin, but the main
agent has no host to signal on its behalf — this tool fills that gap.
"""

PLUGIN_KIND = "tool"

from .plugin import LifecyclePlugin, create_plugin

__all__ = ["LifecyclePlugin", "create_plugin", "PLUGIN_KIND"]
