"""Premium-reactor registration for the example reliability reactor.

REFERENCE EXAMPLE — see this directory's README.  Mirrors
``jaato_premium/drift_monitor/registration.py``.  A tenant (or a premium
example package) returns this from a ``jaato.premium_reactors`` entry point; the
installer writes the rule fragment + action script idempotently to
``~/.jaato/reactors/`` and ``~/.jaato/scripts/`` at extension start.

To adopt: place this package where its entry point can be discovered and add
to that package's ``pyproject.toml``::

    [project.entry-points."jaato.premium_reactors"]
    reliability = "<your_pkg>.reliability.registration:get_reactor_definition"
"""

from __future__ import annotations

from jaato_premium.reactors.installer import PremiumReactor


# Thin action shim — all logic lives in reactor_logic (import-cached across the
# per-dispatch script reload, so per-session state survives between turns).
# Point the import at wherever this package actually lives.
_SCRIPT_SOURCE = '''\
"""Reliability reactor action script (example) — deployed by the installer.

Idempotent overwrite at daemon start; customise via the rule in
~/.jaato/reactors.json or by forking the reactor_logic module.
"""

from <your_pkg>.reliability.reactor_logic import handle_event


def execute(params, event, ctx):
    """Reactor framework entry point. Routes to the per-event handler."""
    handle_event(params, event, ctx)
'''


# One rule per subscribed bus event type; all point at the same shim, which
# re-dispatches internally by event_type.  These are real bus EventType values
# (jaato_sdk.event_bus.EventType) — tool.call_* / agent.output / turn.completed
# / plan.step_updated already exist; the reactor EMITS reliability.escalated /
# reliability.pattern_detected (added in jaato PR #318), it does not subscribe
# to them here.
_RULES = [
    {"id": "reliability.tool_call_started", "enabled": True,
     "match": {"event_type": "tool.call_started"},
     "action": {"script": "scripts/reliability.py", "params": {}}},
    {"id": "reliability.tool_call_completed", "enabled": True,
     "match": {"event_type": "tool.call_completed"},
     "action": {"script": "scripts/reliability.py", "params": {}}},
    {"id": "reliability.agent_output", "enabled": True,
     "match": {"event_type": "agent.output"},
     "action": {"script": "scripts/reliability.py", "params": {}}},
    {"id": "reliability.turn_completed", "enabled": True,
     "match": {"event_type": "turn.completed"},
     "action": {"script": "scripts/reliability.py", "params": {}}},
    {"id": "reliability.plan_step_updated", "enabled": True,
     "match": {"event_type": "plan.step_updated"},
     "action": {"script": "scripts/reliability.py", "params": {}}},
]


def get_reactor_definition() -> PremiumReactor:
    """Entry-point factory called by the installer at extension start."""
    return PremiumReactor(
        name="reliability",
        rules=_RULES,
        script_source=_SCRIPT_SOURCE,
    )
