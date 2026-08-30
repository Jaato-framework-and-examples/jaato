"""The session-persistence contract must be DECLARED, not just discoverable.

``get_persistence_state`` / ``restore_persistence_state`` have worked for
years: ``SessionManager`` snapshots every exposed plugin implementing them
into ``metadata['plugin_states']`` and hands the value back on load.  But
the protocol only described them in a **commented-out block**, so a plugin
author reading :class:`ToolPlugin` could not see the capability existed.

The permission plugin is what that cost.  It held the operator's runtime
grants and denials in an in-memory set, implemented neither method, and so
lost every one of them whenever a session was detached and reattached
(#706) — a lost ``never`` being the sharp end, since a tool the operator
refused became runnable again with no notice.

These tests pin the declaration itself, because the failure mode is
silence: nothing breaks when a plugin omits the methods, it just quietly
forgets.
"""

import inspect

from jaato_sdk.plugins.base import (SessionPersistentPlugin, ToolPlugin,
                                    TRAIT_SESSION_PERSISTENT)

# Plugins that participate in the GENERIC loop.  ``subagent`` and ``todo``
# are excluded on purpose: session_manager's ``_DEDICATED_PLUGINS`` skips
# them because they own file-based storage, so declaring the trait on them
# would advertise a path they do not take.
_GENERIC_PERSISTENT = {
    "shared.plugins.permission.plugin": "PermissionPlugin",
    "shared.plugins.reliability.plugin": "ReliabilityPlugin",
    "shared.plugins.service_connector.plugin": "ServiceConnectorPlugin",
}


def _load(module_path: str, cls_name: str):
    mod = __import__(module_path, fromlist=[cls_name])
    return getattr(mod, cls_name)


def test_persistence_protocol_declares_both_halves() -> None:
    """The contract must be a real, checkable protocol — not a comment."""
    for meth in ("get_persistence_state", "restore_persistence_state"):
        assert hasattr(SessionPersistentPlugin, meth), (
            f"SessionPersistentPlugin does not declare {meth}(). The mechanism "
            f"still works via hasattr, so nothing fails loudly — a plugin "
            f"author simply cannot discover it, which is how #706 happened."
        )


def test_persistence_contract_is_NOT_on_ToolPlugin() -> None:
    """It must stay off ``ToolPlugin``, and this is not a style preference.

    ``ToolPlugin`` is ``@runtime_checkable`` and the registry gates
    discovery on ``isinstance(plugin, ToolPlugin)`` (``registry.py:618``).
    Adding a method there drops every plugin that does not implement it —
    silently, by shortening the discovered list.  Measured while building
    this: moving the pair onto ``ToolPlugin`` cut discovery to the five
    plugins that happened to implement it and broke ~130 registry tests.

    This is why the contract survived for years as a commented-out block.
    """
    for meth in ("get_persistence_state", "restore_persistence_state"):
        assert not hasattr(ToolPlugin, meth), (
            f"{meth}() is declared on ToolPlugin. Because ToolPlugin is "
            f"runtime_checkable and gates discovery, every plugin without "
            f"this method will vanish from the registry."
        )


def test_optional_protocol_does_not_disturb_conformance() -> None:
    """A plugin that opts OUT must stay a fully valid ToolPlugin."""
    from shared.plugins.cli.plugin import CLIToolPlugin
    cli = CLIToolPlugin()
    assert isinstance(cli, ToolPlugin), "opting out must not cost conformance"
    assert not isinstance(cli, SessionPersistentPlugin), (
        "cli persists nothing, so it must not satisfy the persistence protocol"
    )


def test_trait_declarers_implement_both_halves() -> None:
    """Declaring the trait without implementing it persists nothing.

    Half an implementation is the worst outcome: state is snapshotted and
    then silently dropped on load, or never snapshotted while the plugin
    claims it is.
    """
    for module_path, cls_name in _GENERIC_PERSISTENT.items():
        cls = _load(module_path, cls_name)
        traits = getattr(cls, "plugin_traits", frozenset())
        assert TRAIT_SESSION_PERSISTENT in traits, (
            f"{cls_name} participates in the generic persistence loop but "
            f"does not declare TRAIT_SESSION_PERSISTENT"
        )
        for meth in ("get_persistence_state", "restore_persistence_state"):
            assert callable(getattr(cls, meth, None)), (
                f"{cls_name} declares TRAIT_SESSION_PERSISTENT but has no "
                f"{meth}() — it would persist nothing and lose its state on "
                f"the next reload, silently"
            )
        assert isinstance(cls(), SessionPersistentPlugin), (
            f"{cls_name} declares the trait but does not satisfy "
            f"SessionPersistentPlugin"
        )


def test_permission_plugin_persists_runtime_decisions() -> None:
    """Regression guard for #706, at the level that actually broke.

    Not "does the method exist" but "does a grant and a denial survive the
    round trip", which is the operator-visible property.
    """
    from shared.plugins.permission.plugin import PermissionPlugin
    from shared.plugins.permission.policy import PermissionPolicy

    p = PermissionPlugin()
    p._policy = PermissionPolicy()
    assert p.get_persistence_state() is None, (
        "a session that never touched permissions must write no key"
    )

    p._policy.add_session_whitelist("*")
    p._policy.add_session_blacklist("rm -rf *")
    p._policy.session_default_policy = "ask"
    snapshot = p.get_persistence_state()

    import json
    json.dumps(snapshot)  # must be JSON-serialisable; raises if not

    restored = PermissionPlugin()
    restored._policy = PermissionPolicy()
    restored.restore_persistence_state(snapshot)

    assert restored._policy.session_whitelist == {"*"}
    assert restored._policy.session_blacklist == {"rm -rf *"}, (
        "a lost DENIAL is the sharp end of #706: a tool the operator "
        "refused becomes runnable again after a reattach"
    )
    assert restored._policy.session_default_policy == "ask"
    assert restored.get_persistence_state() == snapshot, "round trip must be stable"


def test_permission_restore_tolerates_a_malformed_snapshot() -> None:
    """A stale or hand-edited snapshot must not make a session unloadable."""
    from shared.plugins.permission.plugin import PermissionPlugin
    from shared.plugins.permission.policy import PermissionPolicy

    p = PermissionPlugin()
    p._policy = PermissionPolicy()
    p.restore_persistence_state({
        "session_whitelist": "not-a-list",      # wrong type
        "session_blacklist": [None, "", "ok"],  # partly junk
        "unknown_key_from_a_future_version": 1,
    })
    assert p._policy.session_blacklist == {"ok"}, (
        "malformed entries must be skipped, not raised on: a dropped rule is "
        "visible in `permissions show`, an unloadable session is not "
        "recoverable by the operator"
    )
