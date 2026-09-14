"""``todo`` declares the trait its cross-session plan map depends on (#890).

Both ``shutdown()`` and ``reset_for_next_session()`` deliberately preserve
``_storage`` and ``_current_plan_ids`` because a later cascade stage reads
the plan an earlier one wrote.  Preserving them meant nothing while the next
``session.bootstrap`` built a fresh instance with an empty storage.
"""

from __future__ import annotations

from jaato_sdk.plugins.base import TRAIT_SLOT_SCOPED
from shared.plugins.todo.plugin import TodoPlugin


def test_todo_declares_slot_scoped() -> None:
    assert TRAIT_SLOT_SCOPED in TodoPlugin.plugin_traits
