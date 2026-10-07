"""A subagent is judged by ITS OWN profile's permission policy (#957).

The issue's demonstration, in-process.  A parent (``escriba``, a voice
agent with no ``file_edit`` at all) and a child (``documentalista``, whose
``plugin_configs.permission`` whitelists exactly ``writeNewFile`` and
``updateFile``) share one runtime and therefore one ``PermissionPlugin``.
Before #957 the child was judged by whatever policy the enforcer had been
seeded with — the parent's — and ``writeNewFile`` was denied
``method=default`` 14 times; adding the tool to the PARENT's whitelist,
with the child's profile byte-identical, was the only edit that changed
the verdict.

The session's half of the fix: ``configure()`` no longer re-initializes
the permission plugin through the registry (the route that landed on an
unread copy on the daemon and runner, and clobbered the parent's enforcer
in-process), it stashes the block; ``set_agent_context("subagent")`` —
the point at which a session becomes a subagent, called by both spawn
paths before the first turn — installs it as a session-scoped policy on
the shared plugin and stamps the scope into the executor's permission
context.  These tests drive real ``JaatoSession`` objects against a real
``PermissionPlugin`` and assert the verdicts, in both directions, from
the context each session's executor actually carries.
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock

from ..jaato_session import JaatoSession
from ..plugins.permission.plugin import PermissionPlugin


PARENT_POLICY: Dict[str, Any] = {
    "defaultPolicy": "deny",
    "whitelist": {"tools": ["store_memory", "retrieve_memories", "enter_tier",
                            "web_fetch", "spawn_subagent",
                            "list_active_subagents"]},
}
CHILD_BLOCK: Dict[str, Any] = {
    "policy": {
        "defaultPolicy": "deny",
        "whitelist": {"tools": ["writeNewFile", "updateFile"]},
    },
}


def _registry(names: List[str]) -> MagicMock:
    reg = MagicMock()
    reg.exposed = []
    reg.list_available.return_value = list(names)

    def _expose(name, config=None):
        reg.exposed.append((name, config))
        return True

    reg.expose_tool.side_effect = _expose
    reg.get_registered_core_tool_names.return_value = set()
    return reg


def _runtime() -> MagicMock:
    """One runtime, one enforcer seeded from the ROOT profile — what the
    daemon, the runner and the embedded client all do at bootstrap."""
    enforcer = PermissionPlugin()
    enforcer.initialize({"policy": PARENT_POLICY, "channel_type": "queue"})
    runtime = MagicMock()
    runtime.registry = _registry(
        ["cli", "file_edit", "memory", "permission", "subagent"])
    runtime.permission_plugin = enforcer
    return runtime


def _parent(runtime) -> JaatoSession:
    session = JaatoSession(runtime, "test-model")
    session.configure(skip_provider=True, plugins=["memory", "subagent"],
                      plugin_configs={"permission": {"policy": PARENT_POLICY}})
    session.set_agent_context(agent_type="main", agent_name="escriba")
    return session


def _spawn_child(runtime, block=CHILD_BLOCK) -> JaatoSession:
    """The in-process spawn path's order: ``create_session`` (which runs
    ``configure``), THEN ``set_agent_context("subagent", ...)``."""
    session = JaatoSession(runtime, "test-model")
    session.configure(skip_provider=True, plugins=["file_edit"],
                      plugin_configs={"permission": block} if block else None)
    session.set_agent_context(agent_type="subagent", agent_name="documentalista")
    return session


def _verdict(session: JaatoSession, tool: str):
    """Check ``tool`` exactly as the session's executor would."""
    executor = session._executor
    return executor._permission_plugin.check_permission(
        tool, {}, executor._permission_context, "call_1")


# ------------------------------------------------------- the demonstration


def test_child_writes_with_its_own_whitelist():
    """#957: ``writeNewFile`` from the child, the child's profile unchanged."""
    runtime = _runtime()
    _parent(runtime)
    child = _spawn_child(runtime)

    allowed, info = _verdict(child, "writeNewFile")

    assert allowed is True
    assert info["method"] == "whitelist"


def test_child_does_not_inherit_the_parents_grants():
    runtime = _runtime()
    _parent(runtime)
    child = _spawn_child(runtime)

    allowed, info = _verdict(child, "store_memory")

    assert allowed is False
    assert info["method"] == "default"


def test_parent_is_not_judged_by_the_childs_policy():
    """The in-process shape of the defect: the child's block used to
    REPLACE the parent's policy through ``registry.expose_tool``."""
    runtime = _runtime()
    parent = _parent(runtime)
    _spawn_child(runtime)

    allowed, info = _verdict(parent, "store_memory")
    assert allowed is True and info["method"] == "whitelist"

    allowed, info = _verdict(parent, "writeNewFile")
    assert allowed is False and info["method"] == "default"


def test_the_enforcer_is_never_reinitialized_by_a_session():
    """Neither the root's nor the child's block goes through the registry:
    the root's was applied at bootstrap, the child's is scoped."""
    runtime = _runtime()
    enforcer = runtime.permission_plugin
    policy_before = enforcer._policy
    channel_before = enforcer._channel

    _parent(runtime)
    _spawn_child(runtime)

    assert "permission" not in {n for n, _ in runtime.registry.exposed}
    assert enforcer._policy is policy_before
    assert enforcer._channel is channel_before


def test_the_scope_is_in_the_executor_context_only_once_installed():
    runtime = _runtime()
    parent = _parent(runtime)
    child = _spawn_child(runtime)

    assert "permission_scope" not in parent._executor._permission_context
    assert child._executor._permission_context["permission_scope"] == \
        child._permission_scope
    assert runtime.permission_plugin.has_scoped_policy(child._permission_scope)


def test_a_child_without_a_block_is_judged_by_the_runtime_policy():
    """Declared nothing → inherits the parent's posture, exactly as before."""
    runtime = _runtime()
    _parent(runtime)
    child = _spawn_child(runtime, block=None)

    assert not child._permission_scoped
    allowed, info = _verdict(child, "store_memory")
    assert allowed is True and info["method"] == "whitelist"


def test_a_block_defining_no_policy_installs_nothing():
    """``agent_name`` alone (the trace-logging injection) is not a policy."""
    runtime = _runtime()
    _parent(runtime)
    child = _spawn_child(runtime, block={"agent_name": "documentalista"})

    assert not child._permission_scoped
    assert not runtime.permission_plugin._scoped_policies


def test_the_root_session_installs_no_scoped_policy():
    """The runtime policy IS the root's — scoping it would detach it from
    the operator's ``permissions allow|deny`` commands."""
    runtime = _runtime()
    parent = _parent(runtime)

    assert not parent._permission_scoped
    assert not runtime.permission_plugin._scoped_policies


def test_lifecycle_auto_approvals_reach_the_child():
    """``configure()`` whitelists the lifecycle tools on the runtime policy
    BEFORE the session becomes a subagent; the child's ``defaultPolicy:
    deny`` must not then deny its own ``signal_completion``."""
    runtime = _runtime()
    _parent(runtime)
    runtime.permission_plugin.add_whitelist_tools(["signal_completion"])
    child = _spawn_child(runtime)

    allowed, info = _verdict(child, "signal_completion")
    assert allowed is True and info["method"] == "whitelist"


def test_two_children_are_judged_separately():
    runtime = _runtime()
    _parent(runtime)
    writer = _spawn_child(runtime)
    fetcher = _spawn_child(runtime, block={
        "policy": {"defaultPolicy": "deny",
                   "whitelist": {"tools": ["web_fetch"]}}})

    assert _verdict(writer, "writeNewFile")[0] is True
    assert _verdict(writer, "web_fetch")[0] is False
    assert _verdict(fetcher, "web_fetch")[0] is True
    assert _verdict(fetcher, "writeNewFile")[0] is False


def test_closing_the_child_releases_its_policy():
    runtime = _runtime()
    _parent(runtime)
    child = _spawn_child(runtime)
    scope = child._permission_scope

    child.close_session()

    assert not runtime.permission_plugin.has_scoped_policy(scope)
    assert not child._permission_scoped


def test_a_revived_subagent_reinstalls_on_reconfigure():
    """A session that is ALREADY a subagent when configured — a revive —
    installs from ``configure`` itself, not only from ``set_agent_context``."""
    runtime = _runtime()
    _parent(runtime)
    child = JaatoSession(runtime, "test-model")
    child._agent_type = "subagent"
    child._agent_name = "documentalista"

    child.configure(skip_provider=True, plugins=["file_edit"],
                    plugin_configs={"permission": CHILD_BLOCK})

    assert child._permission_scoped
    assert _verdict(child, "writeNewFile")[0] is True


def test_ask_permission_agrees_with_the_gate():
    """The model's pre-check is dispatched ungated and fetched its context
    from nowhere — it answered from the runtime policy while the real call
    was judged by the child's."""
    from ..session_context import set_current_session, _current_session

    runtime = _runtime()
    _parent(runtime)
    child = _spawn_child(runtime)

    token = _current_session.set(child)
    try:
        result = runtime.permission_plugin._execute_ask_permission(
            {"tool_name": "writeNewFile", "intent": "write the report"})
    finally:
        _current_session.reset(token)

    assert result["allowed"] is True
    assert result["method"] == "whitelist"


def test_an_enforcer_without_the_seam_is_left_alone():
    """A stand-in permission plugin (a test double, an out-of-tree engine)
    that has no ``set_scoped_policy`` is neither re-initialized nor
    crashed into — the runtime policy judges, as before."""
    runtime = _runtime()

    class _Legacy:
        def check_permission(self, name, args, context=None, call_id=None):
            return True, {"method": "stub"}

        def add_whitelist_tools(self, tools):
            pass

        def add_framework_reserved_tools(self, tools):
            pass

    runtime.permission_plugin = _Legacy()
    child = _spawn_child(runtime)

    assert not child._permission_scoped
    assert "permission_scope" not in child._executor._permission_context
