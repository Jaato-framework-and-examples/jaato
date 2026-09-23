"""Webhook plugin for receiving external HTTP events.

This plugin provides an inbound HTTP listener that receives webhooks from
external services (GitHub, Slack, etc.) and publishes them to the TaskEventBus.
Sessions subscribe to events and long-poll for new ones, enabling event-driven
daemon agent sessions.
"""

from .plugin import WebhookPlugin, create_plugin

# Plugin kind identifier for registry discovery
PLUGIN_KIND = "tool"

PLUGIN_TIER = "runner"
__all__ = [
    'WebhookPlugin',
    'create_plugin',
    'PLUGIN_KIND',
]
