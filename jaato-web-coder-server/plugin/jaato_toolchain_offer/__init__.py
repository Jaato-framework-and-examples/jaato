"""The web coder's ``toolchain_offer`` enrichment plugin (#1344).

Installed into the DAEMON's Python environment beside ``jaato-server``; see
``README.md``.  Runner-tier: tool results are enriched in the runner.
"""

PLUGIN_KIND = "enrichment"
PLUGIN_TIER = "runner"


def create_plugin():
    """Entry-point factory (``jaato.enrichment_plugins``)."""
    from .plugin import ToolchainOfferPlugin

    return ToolchainOfferPlugin()
