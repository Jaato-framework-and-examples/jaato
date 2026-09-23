"""Budget-aware GC Plugin.

This plugin provides policy-aware garbage collection that uses the
InstructionBudget to make smarter removal decisions based on GC policies
(LOCKED, PRESERVABLE, PARTIAL, EPHEMERAL).

Load path — note the deliberate absence of ``PLUGIN_KIND``/``PLUGIN_TIER``:
GC strategies are discovered exclusively through the ``jaato.gc_plugins``
entry-point group (``discover_gc_plugins()`` in ``shared/plugins/gc``;
registered under ``[project.entry-points."jaato.gc_plugins"]`` in
``pyproject.toml``), NOT through the ``PLUGIN_KIND`` directory scan that
tool/enrichment plugins use.  This plugin therefore needs no ``PLUGIN_KIND``
to be loaded.  The ``PLUGIN_KIND = "gc"`` carried by the sibling strategies
(gc_truncate/summarize/hybrid) is not consumed by the GC loader; omitting it
here is harmless, and the tier-partition gate
(``test_plugin_tier_partition.py``) explicitly exempts plugins that declare
no ``PLUGIN_KIND``.
"""

from .plugin import BudgetGCPlugin, create_plugin

__all__ = ["BudgetGCPlugin", "create_plugin"]
