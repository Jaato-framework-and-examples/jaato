"""``lsp`` declares the trait that makes its keep-everything reset reachable.

``LSPToolPlugin.reset_for_next_session()`` is a deliberate no-op documented
as "don't touch anything — the next cascade stage reuses these connections".
It ran on an instance the next ``session.bootstrap`` discarded, so the
connections it preserved were unreachable and the jdtls behind them kept
running with no owner (#890).  :data:`TRAIT_SLOT_SCOPED` is what carries the
instance across the boundary; without the declaration the docstring is a
claim about behaviour that does not happen.
"""

from __future__ import annotations

from jaato_sdk.plugins.base import TRAIT_SLOT_SCOPED
from shared.plugins.lsp.plugin import LSPToolPlugin


def test_lsp_declares_slot_scoped() -> None:
    assert TRAIT_SLOT_SCOPED in LSPToolPlugin.plugin_traits


def test_initialize_is_guarded_so_adoption_costs_nothing() -> None:
    """Adoption re-runs ``initialize`` with the arriving session's config.

    The plugin's own ``_initialized`` guard is what turns that into a no-op
    — and therefore what preserves the warm servers.  If the guard were
    removed, adoption would re-connect and the fix would be silently inert.
    """
    plugin = LSPToolPlugin()
    plugin._initialized = True

    plugin.initialize({"workspace_path": "/tmp/other", "session_id": "s2"})

    assert plugin._workspace_path is None, (
        "initialize() must early-return on _initialized"
    )


def test_set_session_id_restamps_without_touching_connections() -> None:
    """A carried instance keeps the first session's id otherwise, so every
    later stage's trace lines were filed under stage one."""
    plugin = LSPToolPlugin()
    plugin._session_id = "20260908_171006"
    sentinel = object()
    plugin._clients = {"jdtls": sentinel}

    plugin.set_session_id("20260908_171228")

    assert plugin._session_id == "20260908_171228"
    assert plugin._clients == {"jdtls": sentinel}
    assert plugin._initialized is False  # untouched by the re-stamp
