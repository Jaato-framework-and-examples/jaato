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
from jaato_sdk.plugins.model_provider.types import ToolSchema
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


# ---------------------------------------------------------------------------
# Regression (#487/#488 over-broadening): the reserved set must be sourced from
# ``register_core_tool`` machinery, NOT from every ``discoverability='core'``
# plugin tool.  The earlier tests above hardcode ``reserved=("signal_completion",)``
# and so never exercised the real ``JaatoSession.configure`` population path —
# which is exactly how the over-broad ``get_core_tool_schemas()`` source went
# unnoticed.  These tests drive the ACTUAL registry accessor.
# ---------------------------------------------------------------------------


class _FakePlugin:
    """Minimal plugin exposing its own tool schemas for registry introspection."""

    def __init__(self, schemas):
        self._schemas = schemas

    def get_tool_schemas(self):
        return self._schemas


def _registry_with_core_and_plugin_tools():
    """A registry holding one genuine framework core tool (registered via
    ``register_core_tool``) and one exposed *plugin* tool flagged
    ``discoverability='core'`` (the eager-load flag real business tools like
    ``readFile`` carry)."""
    reg = PluginRegistry()
    # Genuine framework machinery — the kind #487/#488 mean to exempt.
    # discoverability='core' so it also appears in the broad schema accessor,
    # isolating readFile as the ONE tool the two accessors disagree on.
    reg.register_core_tool(
        ToolSchema(name="stream_dismiss", description="", parameters={},
                   discoverability="core"),
        lambda args: None,
    )
    # A business/plugin tool that is merely eager-loaded (discoverability=core),
    # NOT framework machinery — must NOT be exempt.
    reg._plugins["file_edit"] = _FakePlugin(
        [ToolSchema(name="readFile", description="", parameters={},
                    discoverability="core")]
    )
    reg._exposed.add("file_edit")
    return reg


def test_registered_core_tool_names_excludes_discoverability_core_plugin_tools():
    reg = _registry_with_core_and_plugin_tools()

    # The broad schema accessor DOES surface the plugin's core-discoverability
    # tool (that is its job — eager context loading).
    broad_names = {s.name for s in reg.get_core_tool_schemas()}
    assert "readFile" in broad_names
    assert "stream_dismiss" in broad_names

    # The authoritative machinery accessor does NOT — only register_core_tool
    # names.  This is the crux of the fix.
    reserved_names = reg.get_registered_core_tool_names()
    assert reserved_names == {"stream_dismiss"}
    assert "readFile" not in reserved_names
    assert reg.is_core_tool("readFile") is False


def test_plugin_core_tool_still_governed_by_default_evaluator():
    # End-to-end: populate _framework_reserved exactly as JaatoSession.configure
    # does (lifecycle names + get_registered_core_tool_names()), then confirm a
    # whitelisted-but-conditionally-vetoed plugin tool is STILL denied by the
    # catch-all default evaluator, while genuine machinery is exempt.
    reg = _registry_with_core_and_plugin_tools()
    reserved = {"signal_completion"} | reg.get_registered_core_tool_names()

    p = PermissionPlugin()
    p._policy = PermissionPolicy(
        default_policy="deny",
        # Both tools are policy-whitelisted; the evaluator is the conditional
        # veto layer the operator relies on.
        whitelist_tools={"signal_completion", "stream_dismiss", "readFile"},
    )
    p._policy.set_evaluators({"default": _deny_all})
    p.add_framework_reserved_tools(list(reserved))

    # Genuine framework machinery → exempt from the default evaluator → allowed.
    allowed, meta = p.check_permission("stream_dismiss", {})
    assert allowed is True, meta
    assert meta["method"] != "evaluator"

    # A merely eager-loaded plugin tool → NOT exempt → the default evaluator
    # governs it → denied.  (Pre-fix this leaked through as exempt.)
    allowed, meta = p.check_permission("readFile", {})
    assert allowed is False, meta
    assert meta["method"] == "evaluator"
