"""Top-level bundle management plugin.

Exposes a single ``bundle`` user command whose subcommands operate
across every domain plugin that has registered a
:class:`bundle_common.handler.BundleEntryHandler` (references, agents,
tasks, profiles, services, ...). Pack/unpack are composite by default
— a single archive can carry entries from any combination of kinds —
and entry-level CRUD takes a ``<kind>:<id>`` qualifier so the same
``bundle add`` / ``bundle eject`` / ``bundle remove`` surface works
for every domain.
"""

PLUGIN_KIND = "tool"

PLUGIN_TIER = "runner"
from .plugin import BundlePlugin, create_plugin

__all__ = ["BundlePlugin", "create_plugin", "PLUGIN_KIND", "PLUGIN_TIER"]
