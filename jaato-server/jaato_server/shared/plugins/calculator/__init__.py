# shared/plugins/calculator/__init__.py

from .plugin import CalculatorPlugin

# Declared for the same reason every other plugin package declares them
# (issue #917).  This package shipped with NEITHER constant while being
# listed in jaato-server's ``[project.entry-points."jaato.plugins"]``
# table, and the combination was exactly the reported failure: the
# entry-point path registers it (that path never checks PLUGIN_KIND), so
# ``jaato-scaffold plugins`` listed it and a profile could name it —
# while ``discover(tier_filter="runner")``, which is what the session
# runner calls, excluded it for having no tier.  A session naming
# ``calculator`` came up without its tools, silently.
#
# The build gate in ``test_plugin_tier_partition`` missed it because
# that walk keys on PLUGIN_KIND being present, and this package had
# none; the gate now also covers every package the entry-point table
# names.
PLUGIN_KIND = "tool"
PLUGIN_TIER = "runner"

PLUGIN_INFO = {
    "name": "calculator",
    "description": "Mathematical calculation tools",
    "version": "1.0.0",
    "author": "External Developer",
}

def create_plugin():
    """Factory function called by registry."""
    return CalculatorPlugin()
