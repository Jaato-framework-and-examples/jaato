"""Courier plugin — peer-to-peer messaging between sessions.

A courier carries a parcel to a named recipient wherever they are, and
rings until they answer.  This plugin holds every tool by which one
session reaches ANOTHER session as a peer:

* ``send_to_session`` / ``list_group_sessions`` — any-to-any within a
  GROUP (a shared cascade, or a shared authenticated owner — see
  ``server.session_groups``), waking a cold target to process the message.
* ``send_to_sibling`` / ``list_siblings`` — the cascade-scoped pair, moved
  here from the ``subagent`` plugin unchanged: name-addressed within one
  cascade, and a cold sibling is NOT woken.

It is the horizontal counterpart of ``telepathy`` (child → parent, in
process): a courier exists precisely because peers do not share a process.
The parent ↔ child tools (``spawn_subagent``, ``send_to_subagent`` …) stay
in ``subagent``.

**Cross-tier.**  Every executor needs the daemon's ``SessionManager`` —
the group, the roster and the other session all live there — while the
schema must surface runner-side where the model runs.  So this is the
``daemon_callable`` pattern in full (``shared/plugins/CLAUDE.md`` § "Cross-tier
plugins"): discovered on both sides, every executor forwarded through
``daemon.plugin_execute``, the manager wired into the daemon-side instance
by the generic ``set_session_manager`` sweep at session construction.

**Opt-in.**  Not in ``_ALWAYS_INITIALIZE_PLUGINS``: a profile that does not
list ``courier`` has no peer tools at all.  Narrow it with
``courier(tools:[list_group_sessions])`` for a read-only member, and gate
``send_to_session`` through the permission policy — it is not
auto-approved, because it costs ANOTHER session a turn.

.. code-block:: yaml

    plugins:
      - courier
    plugin_configs:
      courier:
        wake_cold: true                 # false: a cold peer answers session_cold
        max_message_bytes: 8192
        max_pending_per_target: 20
        max_exchanges_per_group: 200

Design: ``docs/design/session-group-messaging.md`` (§11 for the plugin).
"""

from .plugin import CourierPlugin, create_plugin

# Plugin kind identifier for registry discovery
PLUGIN_KIND = "tool"

# Cross-tier: schema runner-side, body daemon-side (needs SessionManager).
PLUGIN_TIER = "daemon_callable"

__all__ = [
    "CourierPlugin",
    "create_plugin",
    "PLUGIN_KIND",
    "PLUGIN_TIER",
]
