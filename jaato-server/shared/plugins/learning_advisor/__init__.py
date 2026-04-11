"""Learning advisor plugin — institutional memory with quality filtering.

This plugin silently observes sessions, distills "lessons learned" from
successful approaches, and injects that wisdom into future sessions with
similar goals.  Think of it as institutional memory with quality filtering
— only good outcomes propagate.

The advisor is hidden: it exposes no tools for the model to call.  It works
entirely through prompt enrichment (injecting guidance) and session lifecycle
hooks (reflection on session end).

User commands (``advisor show``, ``advisor rate``, ``advisor reflect``,
``advisor clear``) provide transparency into what the advisor has learned.

Usage:
    # Plugin is auto-discovered by PluginRegistry.
    # Alternatively, register manually:
    registry.register_plugin(advisor_plugin, enrichment_only=True)
"""

PLUGIN_KIND = "tool"

from .plugin import LearningAdvisorPlugin, create_plugin

__all__ = ["LearningAdvisorPlugin", "create_plugin", "PLUGIN_KIND"]
