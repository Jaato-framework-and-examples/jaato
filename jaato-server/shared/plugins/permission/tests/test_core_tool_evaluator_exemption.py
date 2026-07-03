"""Framework-reserved tools are EXEMPT from a business catch-all ``"default"``
evaluator.

A Daruma-style locked-down agent ships a ``default`` evaluator that DENIES any
tool not in its business whitelist.  Without this exemption that default-deny
also vetoes ``signal_completion`` (a framework lifecycle terminal the agent
needs to complete) and the framework's core infra tools, so the agent can do
its work but never finish.

The exemption keys on the permission plugin's ``_framework_reserved`` set
(populated at ``JaatoSession.configure`` from BOTH the registry's core tools
AND the session's lifecycle tool names) — NOT ``registry.is_core_tool``:
``signal_completion`` is registered SESSION-LEVEL (``executor.register``), so
``is_core_tool('signal_completion')`` is ``False`` (the #487 regression this
supersedes: its guard was a no-op for signal_completion, and its test hid that
by MOCKING is_core_tool).  Using a self-contained set with no registry lookup
also survives ``PermissionPlugin.shutdown()`` nulling ``_registry`` between
sessions on a reused daemon.

Contract pinned here:
- a framework-reserved tool is EXEMPT from the ``default`` evaluator → falls
  through to its normal whitelist allow;
- a business/plugin tool is still governed by the ``default`` evaluator;
- a tool-SPECIFIC evaluator keyed to a reserved tool STILL runs (explicit
  governance honored — only the catch-all collateral is prevented);
- the exemption needs NO registry (survives shutdown()/_registry=None).
"""
from shared.plugins.registry import PluginRegistry
from shared.plugins.permission.plugin import PermissionPlugin
from shared.plugins.permission.policy import PermissionPolicy
from shared.plugins.permission.evaluator import PolicyDecision


def _deny_all(tool_name, args, context):
    return PolicyDecision.DENY


def _fallback(tool_name, args, context):
    return PolicyDecision.FALLBACK


def _plugin(evaluators, reserved=("signal_completion",),
            whitelist=("signal_completion", "issue_refund")):
    p = PermissionPlugin()
    # default_policy="deny" (locked down) — only the whitelist grants access,
    # exactly the Daruma shape.
    p._policy = PermissionPolicy(
        default_policy="deny",
        whitelist_tools=set(whitelist),
    )
    p._policy.set_evaluators(evaluators)
    p.add_framework_reserved_tools(list(reserved))
    return p


def test_signal_completion_is_not_a_registry_core_tool():
    # Guards the whole premise: signal_completion is session-level, so
    # is_core_tool is False — which is exactly why the exemption can't key on
    # it and must use _framework_reserved instead.
    assert PluginRegistry().is_core_tool("signal_completion") is False


def test_reserved_tool_exempt_from_default_evaluator():
    p = _plugin({"default": _deny_all})

    # signal_completion (framework-reserved) → exempt from the deny-all default
    # → whitelist allow.
    allowed, meta = p.check_permission("signal_completion", {})
    assert allowed is True, meta
    assert meta["method"] != "evaluator"

    # issue_refund (business/plugin tool) → still governed → denied.
    allowed, meta = p.check_permission("issue_refund", {})
    assert allowed is False
    assert meta["method"] == "evaluator"


def test_tool_specific_evaluator_still_governs_reserved_tool():
    # Explicit per-tool governance of a reserved tool is honored; only the
    # catch-all "default" is bypassed.
    p = _plugin({"default": _fallback, "signal_completion": _deny_all})
    allowed, meta = p.check_permission("signal_completion", {})
    assert allowed is False
    assert meta["method"] == "evaluator"


def test_exemption_scoped_to_reserved_tools_only():
    # A tool NOT in the reserved set is governed by the default evaluator.
    p = _plugin({"default": _deny_all}, reserved=())
    allowed, meta = p.check_permission("signal_completion", {})
    assert allowed is False
    assert meta["method"] == "evaluator"


def test_exemption_needs_no_registry():
    # The reserved set is self-contained: even with _registry=None (the
    # shutdown() state on a reused daemon) the exemption still fires.
    p = _plugin({"default": _deny_all})
    p._registry = None
    allowed, meta = p.check_permission("signal_completion", {})
    assert allowed is True, meta
    assert meta["method"] != "evaluator"
