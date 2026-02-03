"""Budget-aware GC Plugin.

This plugin provides policy-aware garbage collection that uses the
InstructionBudget to make smarter removal decisions based on GC policies
(LOCKED, PRESERVABLE, PARTIAL, EPHEMERAL).
"""

from .plugin import BudgetGCPlugin, create_plugin

__all__ = ["BudgetGCPlugin", "create_plugin"]
