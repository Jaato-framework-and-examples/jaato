"""One enforcer, one policy per session (#957).

The permission plugin is a registry-shared singleton: a parent and every
subagent it spawns are judged by the same object, and until #957 by the
same ``_policy`` — the one seeded at bootstrap from the ROOT profile.  A
subagent profile that declared its own ``plugin_configs.permission`` was
judged by the parent's policy regardless, so a profile that whitelisted
exactly the two tools it needed, and worked standalone, was denied
``method=default`` when spawned; the only edit that changed the verdict
was to the PARENT's whitelist.

These tests cover the plugin's half of the fix — ``set_scoped_policy``,
and ``_resolve_policy`` reading the caller's ``permission_scope`` out of
the per-session context the executor already passes:

- a context naming an installed scope is judged by that policy, and a
  context naming none (or an unknown scope) by the runtime policy;
- installing a scoped policy moves nothing on the runtime policy — the
  parent is not judged by the child's rules either (the in-process
  shape of the same defect);
- auto-approved tools (``add_whitelist_tools``) reach scoped policies
  installed before AND after the call — a ``defaultPolicy: deny`` child
  must not deny its own ``signal_completion``;
- an ``always`` answer from the channel lands on the policy that asked;
- the DECISION trace names which policy judged;
- evaluators declared in the scoped block are honoured;
- release and the slot-boundary reset fall back to the runtime policy.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pytest

from .. import plugin as permission_plugin_module
from ..channels import ChannelDecision, ChannelResponse
from ..plugin import PermissionPlugin


# ---------------------------------------------------------------- harness


@pytest.fixture
def traced(monkeypatch) -> List[Tuple[str, str]]:
    lines: List[Tuple[str, str]] = []

    def _capture(component: str, msg: str, **_kwargs: Any) -> None:
        lines.append((component, msg))

    monkeypatch.setattr(permission_plugin_module, "_trace_write", _capture)
    return lines


def _decisions(lines: List[Tuple[str, str]]) -> List[str]:
    return [msg for _c, msg in lines if "check_permission: DECISION" in msg]


PARENT_POLICY: Dict[str, Any] = {
    # the voice parent of #957: no file_edit at all
    "defaultPolicy": "deny",
    "whitelist": {"tools": ["store_memory", "spawn_subagent"]},
}
CHILD_BLOCK: Dict[str, Any] = {
    # ``_base_documentalista.yaml``, byte-identical to the issue
    "policy": {
        "defaultPolicy": "deny",
        "whitelist": {"tools": ["writeNewFile", "updateFile"]},
    },
}
PARENT_CTX = {"agent_type": "main", "agent_name": "escriba"}
CHILD_CTX = {"agent_type": "subagent", "agent_name": "documentalista",
             "permission_scope": "child-1"}


def _enforcer() -> PermissionPlugin:
    plugin = PermissionPlugin()
    plugin.initialize({"policy": PARENT_POLICY, "channel_type": "queue"})
    return plugin


# ------------------------------------------------------- the demonstration


def test_child_is_judged_by_its_own_policy():
    """#957's table, first row: ``writeNewFile`` from the child."""
    plugin = _enforcer()
    plugin.set_scoped_policy("child-1", CHILD_BLOCK)

    allowed, info = plugin.check_permission("writeNewFile", {}, CHILD_CTX, "c1")

    assert allowed is True
    assert info["method"] == "whitelist"


def test_without_a_scope_the_runtime_policy_still_judges():
    """The pre-#957 verdict, now reserved for a session that declared nothing."""
    plugin = _enforcer()
    plugin.set_scoped_policy("child-1", CHILD_BLOCK)

    allowed, info = plugin.check_permission("writeNewFile", {}, PARENT_CTX, "c2")

    assert allowed is False
    assert info["method"] == "default"


def test_unknown_scope_falls_back_to_the_runtime_policy():
    plugin = _enforcer()

    allowed, info = plugin.check_permission(
        "store_memory", {}, {**PARENT_CTX, "permission_scope": "nobody"}, "c3")

    assert allowed is True
    assert info["method"] == "whitelist"


def test_the_child_does_not_inherit_the_parents_whitelist():
    """A child's ``defaultPolicy: deny`` is not weakened by the parent's grants."""
    plugin = _enforcer()
    plugin.set_scoped_policy("child-1", CHILD_BLOCK)

    allowed, info = plugin.check_permission("store_memory", {}, CHILD_CTX, "c4")

    assert allowed is False
    assert info["method"] == "default"


def test_installing_a_scoped_policy_moves_nothing_on_the_runtime_one():
    """The in-process shape of #957: the child's block used to REPLACE
    the parent's policy (and its channel) through a re-``initialize``."""
    plugin = _enforcer()
    runtime_policy = plugin._policy
    channel = plugin._channel

    plugin.set_scoped_policy("child-1", CHILD_BLOCK)

    assert plugin._policy is runtime_policy
    assert plugin._channel is channel
    assert "writeNewFile" not in runtime_policy.whitelist_tools
    allowed, _ = plugin.check_permission("spawn_subagent", {}, PARENT_CTX, "c5")
    assert allowed is True


def test_two_subagents_get_two_policies():
    plugin = _enforcer()
    plugin.set_scoped_policy("child-1", CHILD_BLOCK)
    plugin.set_scoped_policy("child-2", {
        "policy": {"defaultPolicy": "deny",
                   "whitelist": {"tools": ["web_fetch"]}}})

    ctx2 = {**CHILD_CTX, "permission_scope": "child-2"}
    assert plugin.check_permission("web_fetch", {}, ctx2, "a")[0] is True
    assert plugin.check_permission("writeNewFile", {}, ctx2, "b")[0] is False
    assert plugin.check_permission("writeNewFile", {}, CHILD_CTX, "c")[0] is True
    assert plugin.check_permission("web_fetch", {}, CHILD_CTX, "d")[0] is False


# ------------------------------------------------ auto-approved tools reach it


def test_auto_approved_tools_reach_a_policy_installed_earlier():
    plugin = _enforcer()
    plugin.set_scoped_policy("child-1", CHILD_BLOCK)

    plugin.add_whitelist_tools(["signal_completion"])

    allowed, info = plugin.check_permission(
        "signal_completion", {}, CHILD_CTX, "c6")
    assert allowed is True
    assert info["method"] == "whitelist"


def test_auto_approved_tools_reach_a_policy_installed_later():
    """The lifecycle tools are whitelisted at ``configure()``, BEFORE the
    session becomes a subagent and installs its policy."""
    plugin = _enforcer()
    plugin.add_whitelist_tools(["signal_completion", "listReferences"])

    plugin.set_scoped_policy("child-1", CHILD_BLOCK)

    for tool in ("signal_completion", "listReferences"):
        allowed, info = plugin.check_permission(tool, {}, CHILD_CTX, "c7")
        assert allowed is True, tool
        assert info["method"] == "whitelist"


def test_auto_approved_tools_still_reach_the_runtime_policy():
    plugin = _enforcer()
    plugin.set_scoped_policy("child-1", CHILD_BLOCK)
    plugin.add_whitelist_tools(["listReferences"])

    assert "listReferences" in plugin._policy.whitelist_tools


# ------------------------------------------------------ session-level grants


def test_an_always_answer_lands_on_the_policy_that_asked():
    plugin = _enforcer()
    plugin.set_scoped_policy("child-1", CHILD_BLOCK)
    child_policy = plugin._scoped_policies["child-1"]

    plugin._handle_channel_response(
        "readFile", {}, ChannelResponse(request_id="r", decision=ChannelDecision.ALLOW_SESSION,
                                        reason="yes, always"),
        child_policy)

    assert "readFile" in child_policy.session_whitelist
    assert "readFile" not in plugin._policy.session_whitelist
    allowed, info = plugin.check_permission("readFile", {}, CHILD_CTX, "c8")
    assert allowed is True
    assert info["method"] == "session_whitelist"


def test_a_never_answer_lands_on_the_policy_that_asked():
    plugin = _enforcer()
    plugin.set_scoped_policy("child-1", CHILD_BLOCK)
    child_policy = plugin._scoped_policies["child-1"]

    plugin._handle_channel_response(
        "writeNewFile", {}, ChannelResponse(request_id="r", decision=ChannelDecision.DENY_SESSION,
                                            reason="never"),
        child_policy)

    assert plugin.check_permission("writeNewFile", {}, CHILD_CTX, "c9")[0] is False
    assert "writeNewFile" not in plugin._policy.session_blacklist


# ----------------------------------------------------------------- tracing


def test_decision_names_which_policy_judged(traced):
    plugin = _enforcer()
    plugin.set_scoped_policy("child-1", CHILD_BLOCK)

    plugin.check_permission("writeNewFile", {}, CHILD_CTX, "c10")
    plugin.check_permission("writeNewFile", {}, PARENT_CTX, "c11")

    child_line, parent_line = _decisions(traced)
    assert "policy=session" in child_line
    assert "agent=subagent:documentalista" in child_line
    assert "policy=runtime" in parent_line


def test_installing_is_traced(traced):
    plugin = _enforcer()
    plugin.set_scoped_policy("child-1", CHILD_BLOCK)

    line, = [m for _c, m in traced if m.startswith("set_scoped_policy:")]
    assert "scope=child-1" in line
    assert "default=deny" in line
    assert "writeNewFile" in line


# -------------------------------------------------------------- evaluators


def test_scoped_evaluators_are_honoured(tmp_path):
    evaluator = tmp_path / "gate.py"
    evaluator.write_text(
        "from shared.plugins.permission.evaluator import EvalResult, PolicyDecision\n"
        "def evaluate(tool_name, args, context):\n"
        "    return EvalResult(decision=PolicyDecision.DENY)\n"
    )
    plugin = _enforcer()
    plugin.set_scoped_policy("child-1", {
        **CHILD_BLOCK,
        "evaluators": {"writeNewFile": str(evaluator)},
    })

    allowed, info = plugin.check_permission("writeNewFile", {}, CHILD_CTX, "c12")
    assert allowed is False
    assert info["method"] == "evaluator"
    # the parent has no such evaluator
    assert not plugin._policy._evaluators


def test_a_block_with_only_evaluators_still_gets_a_policy():
    plugin = _enforcer()
    plugin.set_scoped_policy("child-1", {"evaluators": {}})

    assert plugin.has_scoped_policy("child-1")


# ---------------------------------------------------------------- lifetime


def test_release_falls_back_to_the_runtime_policy():
    plugin = _enforcer()
    plugin.set_scoped_policy("child-1", CHILD_BLOCK)
    assert plugin.check_permission("writeNewFile", {}, CHILD_CTX, "a")[0] is True

    plugin.release_scoped_policy("child-1")
    plugin.release_scoped_policy("child-1")  # idempotent

    assert not plugin.has_scoped_policy("child-1")
    assert plugin.check_permission("writeNewFile", {}, CHILD_CTX, "b")[0] is False


def test_reinstalling_under_the_same_scope_replaces():
    plugin = _enforcer()
    plugin.set_scoped_policy("child-1", CHILD_BLOCK)
    plugin.set_scoped_policy("child-1", {
        "policy": {"defaultPolicy": "deny", "whitelist": {"tools": ["readFile"]}}})

    assert plugin.check_permission("readFile", {}, CHILD_CTX, "a")[0] is True
    assert plugin.check_permission("writeNewFile", {}, CHILD_CTX, "b")[0] is False


def test_slot_boundary_reset_drops_scoped_policies():
    plugin = _enforcer()
    plugin.set_scoped_policy("child-1", CHILD_BLOCK)

    plugin.reset_for_next_session()

    assert not plugin.has_scoped_policy("child-1")
    assert plugin._policy is not None  # the runtime policy survives, as before


def test_shutdown_drops_scoped_policies():
    plugin = _enforcer()
    plugin.set_scoped_policy("child-1", CHILD_BLOCK)

    plugin.shutdown()

    assert not plugin._scoped_policies
    assert not plugin._auto_whitelisted


def test_show_names_the_scoped_policies():
    plugin = _enforcer()
    plugin.set_scoped_policy("child-1", CHILD_BLOCK)

    out = plugin._permissions_show()

    assert "Session-scoped policies: 1" in out
    assert "writeNewFile" not in out  # the child's rules are not the runtime's


# --------------------------------------------------------- askPermission


def test_ask_permission_is_judged_by_the_calling_sessions_policy(monkeypatch):
    """The pre-check the model runs must agree with the gate the real call
    meets — it is dispatched ungated, so it fetches the context itself."""
    from shared import session_context

    class _Session:
        def permission_context(self):
            return dict(CHILD_CTX)

        def get_session_state(self, _key):
            return None  # the reliability gate reads it

    plugin = _enforcer()
    plugin.set_scoped_policy("child-1", CHILD_BLOCK)
    token = session_context._current_session.set(_Session())
    try:
        result = plugin._execute_ask_permission(
            {"tool_name": "writeNewFile", "intent": "write the report"})
    finally:
        session_context._current_session.reset(token)

    assert result["allowed"] is True
    assert result["method"] == "whitelist"


def test_ask_permission_outside_a_session_uses_the_runtime_policy():
    plugin = _enforcer()
    plugin.set_scoped_policy("child-1", CHILD_BLOCK)

    result = plugin._execute_ask_permission(
        {"tool_name": "writeNewFile", "intent": "write the report"})

    assert result["allowed"] is False
    assert result["method"] == "default"
