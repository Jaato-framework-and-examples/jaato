"""The web coder's toolchains plugin (#1344).

Installed into the DAEMON's Python environment beside ``jaato-server``; see
``INSTALL.md``.  Runner-tier: the ``toolchain`` user command, the install
jobs, tool-result enrichment and the instruction section all run in the
session's runner, as its account and under its confinement.
"""

PLUGIN_KIND = "tool"
PLUGIN_TIER = "runner"


def create_plugin():
    """Entry-point factory (``jaato.plugins``)."""
    from .plugin import WebCoderToolchainsPlugin

    return WebCoderToolchainsPlugin()
