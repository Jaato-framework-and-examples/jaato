"""Framework core tools are EXEMPT from a business catch-all ``"default"``
evaluator.

A Daruma-style locked-down agent ships a ``default`` evaluator that DENIES any
tool not in its business whitelist.  Without this exemption that default-deny
also vetoes ``signal_completion`` (a framework lifecycle terminal registered via
``register_core_tool`` and auto-approved into the policy whitelist), so the
agent can do its work but never complete.

Contract pinned here:
- a core tool (``registry.is_core_tool`` True) is EXEMPT from the ``default``
  evaluator → falls through to its normal whitelist allow;
- a business/plugin tool is still governed by the ``default`` evaluator;
- a tool-SPECIFIC evaluator keyed to a core tool STILL runs (explicit
  governance is honored — only the catch-all collateral is prevented);
- without a registry we can't confirm core-ness, so the evaluator runs (safe
  default: govern, never silently bypass).
"""
from types import SimpleNamespace

from shared.plugins.permission.plugin import PermissionPlugin
from shared.plugins.permission.policy import PermissionPolicy
from shared.plugins.permission.evaluator import PolicyDecision


def _deny_all(tool_name, args, context):
    return PolicyDecision.DENY


def _fallback(tool_name, args, context):
    return PolicyDecision.FALLBACK


def _plugin(evaluators, core_tools=("signal_completion",),
            whitelist=("signal_completion", "issue_refund")):
    p = PermissionPlugin()
    # default_policy="deny" (locked down) — only the whitelist grants access,
    # exactly the Daruma shape.
    p._policy = PermissionPolicy(
        default_policy="deny",
        whitelist_tools=set(whitelist),
    )
    p._policy.set_evaluators(evaluators)
    p._registry = SimpleNamespace(is_core_tool=lambda n: n in core_tools)
    return p


def test_core_tool_exempt_from_default_evaluator():
    p = _plugin({"default": _deny_all})

    # signal_completion (core) → exempt from the deny-all default → whitelist allow
    allowed, meta = p.check_permission("signal_completion", {})
    assert allowed is True, meta
    assert meta["method"] != "evaluator"

    # issue_refund (business/plugin tool) → still governed → denied
    allowed, meta = p.check_permission("issue_refund", {})
    assert allowed is False
    assert meta["method"] == "evaluator"


def test_tool_specific_evaluator_still_governs_core_tool():
    # Explicit per-tool governance of a core tool is honored; only the catch-all
    # "default" is bypassed for core tools.
    p = _plugin({"default": _fallback, "signal_completion": _deny_all})
    allowed, meta = p.check_permission("signal_completion", {})
    assert allowed is False
    assert meta["method"] == "evaluator"


def test_exemption_scoped_to_core_tools_only():
    # If the same tool name is NOT core here, the default evaluator governs it —
    # the exemption keys strictly on registry.is_core_tool.
    p = _plugin({"default": _deny_all}, core_tools=())
    allowed, meta = p.check_permission("signal_completion", {})
    assert allowed is False
    assert meta["method"] == "evaluator"


def test_no_registry_runs_evaluator_safe_default():
    # Without a registry we cannot confirm core-ness → govern (never silently
    # bypass).  No crash.
    p = _plugin({"default": _deny_all})
    p._registry = None
    allowed, meta = p.check_permission("signal_completion", {})
    assert allowed is False
    assert meta["method"] == "evaluator"
