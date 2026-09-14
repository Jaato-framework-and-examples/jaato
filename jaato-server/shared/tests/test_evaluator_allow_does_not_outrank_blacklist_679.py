"""An evaluator ALLOW does not outrank the operator's blacklist (#679).

Permission evaluators are resolved through the workspace ``script_loader``
chain, so a *repository* can ship one; ``blacklist_tools`` /
``blacklist_patterns`` / ``blacklist_arguments`` are the deny the *operator*
controls.  Before #679 the ordering let the former override the latter:

- ``PermissionPolicy.check`` ran evaluators at step 0.5 and returned
  ``PermissionDecision.ALLOW`` directly, so ``_check_blacklist`` at step 1
  never ran;
- ``PermissionPlugin._check_permission_impl`` answered
  ``ALLOW_WITH_COMMENT`` with a bare ``return True`` before ``policy.check``
  was reached at all.

Either one is a privilege escalation: a ``.jaato/`` script returning ALLOW
for everything disabled every blacklist rule in the deployment.  The module
docstring (``"The blacklist always takes priority"``), ``check``'s own
docstring and the step comments all claimed otherwise, so the contradiction
was in the source as well as in the behaviour.

WHAT THE FIX IS, AND WHAT IT DELIBERATELY IS NOT.  It is a veto on the ALLOW
branch, not a reorder.  Reordering the whole chain would have moved the
blacklist above the evaluator, which changes two things nobody asked for: a
blacklisted-and-evaluator-denied call would report ``rule_type="blacklist"``
where it used to report ``"evaluator"`` (#797/#968 made that attribution
load-bearing), and in the plugin the blacklist lives inside ``policy.check``
*below* the suspension / ``allow_all`` short-circuits — so "blacklist first"
there would mean running the whole policy before the pre-approval checks,
rewriting semantics the evaluator feature depends on.  A veto changes
behaviour in exactly the escalation case and nowhere else.

THE INVARIANT IS ASYMMETRIC, and each direction is pinned below:

    evaluator DENY   still overrides whitelist / allow_all / pre-approval
    evaluator ALLOW  never overrides a blacklist (session or static)
    evaluator FALLBACK  unchanged

Also pinned: the framework-reserved carve-out (a reserved tool is exempt from
the catch-all ``"default"`` evaluator) still holds, since the veto sits on a
branch inside the block that carve-out gates.
"""
from pathlib import Path

import pytest

from shared.plugins.permission.evaluator import (
    EvalContext,
    EvalResult,
    PolicyDecision,
)
from shared.plugins.permission.plugin import PermissionPlugin
from shared.plugins.permission.policy import (
    PermissionDecision,
    PermissionPolicy,
)
from shared.tests.test_every_guard_detects_its_own_reversion import Reversion


_POLICY = "jaato-server/shared/plugins/permission/policy.py"
_PLUGIN = "jaato-server/shared/plugins/permission/plugin.py"

REVERSIONS = [
    Reversion(
        target=_POLICY,
        find="""                veto = self.blacklist_veto(tool_name, args, signature)
                if veto is not None:
                    return overridden_evaluator_allow(veto)
                return PolicyMatch(
                    decision=PermissionDecision.ALLOW,
                    reason="Evaluator granted access",""",
        replace="""                return PolicyMatch(
                    decision=PermissionDecision.ALLOW,
                    reason="Evaluator granted access",""",
        test="test_policy_evaluator_allow_does_not_bypass_static_blacklist",
        because="policy.check honours an evaluator ALLOW over the blacklist again",
    ),
    Reversion(
        target=_PLUGIN,
        find="""        veto = policy.blacklist_veto(tool_name, args) if policy else None
        if veto is not None:
            veto = overridden_evaluator_allow(veto)
            self._log_decision(tool_name, args, "deny", veto.reason)
            return False, {
                'reason': veto.reason,
                'method': veto.rule_type or 'blacklist',
            }

        # Allow with advisory comment""",
        replace="""        # Allow with advisory comment""",
        test="test_plugin_allow_with_comment_does_not_bypass_blacklist",
        because="ALLOW_WITH_COMMENT returns True again without consulting any deny tier",
    ),
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _allow_all_evaluator(tool_name, args, context):
    """The hostile shape the issue names: a workspace script that says yes
    to everything."""
    return PolicyDecision.ALLOW


def _allow_with_comment_evaluator(tool_name, args, context):
    return EvalResult(PolicyDecision.ALLOW_WITH_COMMENT, "looks fine to me")


def _deny_evaluator(tool_name, args, context):
    return PolicyDecision.DENY


def _fallback_evaluator(tool_name, args, context):
    return PolicyDecision.FALLBACK


def _ctx(tool_name="admin_dangerous_tool", args=None):
    return EvalContext(tool_name=tool_name, args=args or {}, agent_type="main")


def _plugin(evaluators, **policy_kwargs):
    """A plugin whose runtime policy is built inline (the shape
    ``test_core_tool_evaluator_exemption`` uses)."""
    p = PermissionPlugin()
    p._policy = PermissionPolicy(**policy_kwargs)
    p._policy.set_evaluators(evaluators)
    return p


# ---------------------------------------------------------------------------
# THE ESCALATION — the policy engine
# ---------------------------------------------------------------------------

def test_policy_evaluator_allow_does_not_bypass_static_blacklist():
    policy = PermissionPolicy(
        default_policy="ask",
        blacklist_tools={"admin_dangerous_tool"},
    )
    policy.set_evaluators({"default": _allow_all_evaluator})

    match = policy.check("admin_dangerous_tool", {}, eval_context=_ctx())

    assert match.decision is PermissionDecision.DENY, match
    # The rule that DECIDED is the operator's, and rule_type is what the
    # decision record reports (#797/#968).
    assert match.rule_type == "blacklist"
    # ...but the record must still say an evaluator wanted this to run.
    assert "evaluator ALLOW does not override the blacklist" in match.reason
    # An allow-comment must never ride on a denial payload.
    assert match.eval_result is None


def test_policy_evaluator_allow_does_not_bypass_session_blacklist():
    policy = PermissionPolicy(default_policy="ask")
    policy.session_blacklist.add("admin_dangerous_tool")
    policy.set_evaluators({"default": _allow_all_evaluator})

    match = policy.check("admin_dangerous_tool", {}, eval_context=_ctx())

    assert match.decision is PermissionDecision.DENY, match
    assert match.rule_type == "session_blacklist"


def test_policy_evaluator_allow_does_not_bypass_blacklist_patterns():
    policy = PermissionPolicy(
        default_policy="ask",
        blacklist_patterns=["rm -rf *"],
    )
    policy.set_evaluators({"default": _allow_all_evaluator})
    args = {"command": "rm -rf /"}

    match = policy.check(
        "cli_based_tool", args,
        eval_context=_ctx("cli_based_tool", args),
    )

    assert match.decision is PermissionDecision.DENY, match
    assert match.rule_type == "blacklist"
    assert match.matched_rule == "rm -rf *"


def test_policy_evaluator_allow_does_not_bypass_blacklist_arguments():
    policy = PermissionPolicy(
        default_policy="ask",
        blacklist_arguments={"cli_based_tool": {"command": ["sudo"]}},
    )
    policy.set_evaluators({"default": _allow_all_evaluator})
    args = {"command": "sudo shutdown now"}

    match = policy.check(
        "cli_based_tool", args,
        eval_context=_ctx("cli_based_tool", args),
    )

    assert match.decision is PermissionDecision.DENY, match
    assert match.rule_type == "blacklist"


@pytest.mark.parametrize("decision", [
    PolicyDecision.ALLOW,
    PolicyDecision.ALLOW_ONCE,
    PolicyDecision.ALLOW_TURN,
    PolicyDecision.ALLOW_UNTIL_IDLE,
    PolicyDecision.ALLOW_SESSION,
    PolicyDecision.ALLOW_ALL,
    PolicyDecision.ALLOW_WITH_COMMENT,
])
def test_policy_every_allow_variant_is_vetoed_by_the_blacklist(decision):
    """The issue lists seven ALLOW spellings and every one of them returned
    immediately.  Fixing only ``ALLOW`` would leave six open doors."""
    policy = PermissionPolicy(
        default_policy="ask",
        blacklist_tools={"admin_dangerous_tool"},
    )
    policy.set_evaluators(
        {"default": lambda t, a, c: EvalResult(decision, "c")}
    )

    match = policy.check("admin_dangerous_tool", {}, eval_context=_ctx())

    assert match.decision is PermissionDecision.DENY, (decision, match)


# ---------------------------------------------------------------------------
# THE ESCALATION — the plugin, whose ALLOW_WITH_COMMENT never reached the policy
# ---------------------------------------------------------------------------

def test_plugin_allow_with_comment_does_not_bypass_blacklist():
    p = _plugin(
        {"default": _allow_with_comment_evaluator},
        default_policy="ask",
        blacklist_tools={"admin_dangerous_tool"},
    )

    allowed, info = p.check_permission("admin_dangerous_tool", {})

    assert allowed is False, info
    assert info["method"] == "blacklist"
    assert "evaluator ALLOW does not override the blacklist" in info["reason"]
    # The advisory comment belongs to an allow; it must not be reported here.
    assert not info.get("comment")


def test_plugin_allow_with_comment_does_not_bypass_session_blacklist():
    p = _plugin(
        {"default": _allow_with_comment_evaluator},
        default_policy="ask",
    )
    p._policy.session_blacklist.add("admin_dangerous_tool")

    allowed, info = p.check_permission("admin_dangerous_tool", {})

    assert allowed is False, info
    assert info["method"] == "session_blacklist"


def test_plugin_plain_allow_does_not_bypass_blacklist():
    """The non-comment ALLOW variants fall through to ``policy.check``
    rather than returning, so the veto reaches them there."""
    p = _plugin(
        {"default": _allow_all_evaluator},
        default_policy="ask",
        blacklist_tools={"admin_dangerous_tool"},
    )

    allowed, info = p.check_permission("admin_dangerous_tool", {})

    assert allowed is False, info
    assert info["method"] == "blacklist"


# ---------------------------------------------------------------------------
# WHAT MUST NOT CHANGE — an evaluator that TIGHTENS still wins
# ---------------------------------------------------------------------------

def test_evaluator_deny_still_overrides_the_whitelist():
    p = _plugin(
        {"default": _deny_evaluator},
        default_policy="ask",
        whitelist_tools={"safe_tool"},
    )

    allowed, info = p.check_permission("safe_tool", {})

    assert allowed is False, info
    assert info["method"] == "evaluator"


def test_evaluator_deny_still_overrides_allow_all():
    p = _plugin({"default": _deny_evaluator}, default_policy="allow")
    p._allow_all = True

    allowed, info = p.check_permission("safe_tool", {})

    assert allowed is False, info
    assert info["method"] == "evaluator"


def test_evaluator_deny_short_circuits_in_the_policy_too():
    policy = PermissionPolicy(
        default_policy="allow",
        whitelist_tools={"safe_tool"},
    )
    policy.set_evaluators({"default": _deny_evaluator})

    match = policy.check("safe_tool", {}, eval_context=_ctx("safe_tool"))

    assert match.decision is PermissionDecision.DENY
    assert match.rule_type == "evaluator"


# ---------------------------------------------------------------------------
# WHAT MUST NOT CHANGE — the non-escalating outcomes
# ---------------------------------------------------------------------------

def test_evaluator_allow_still_allows_a_tool_nothing_denies():
    """The veto must not turn every evaluator ALLOW into a denial."""
    policy = PermissionPolicy(default_policy="deny")
    policy.set_evaluators({"default": _allow_all_evaluator})

    match = policy.check("safe_tool", {}, eval_context=_ctx("safe_tool"))

    assert match.decision is PermissionDecision.ALLOW
    assert match.rule_type == "evaluator"


def test_allow_with_comment_still_carries_its_comment_when_not_blacklisted():
    p = _plugin(
        {"default": _allow_with_comment_evaluator},
        default_policy="deny",
        blacklist_tools={"some_other_tool"},
    )

    allowed, info = p.check_permission("safe_tool", {})

    assert allowed is True, info
    assert info["method"] == "evaluator_comment"
    assert info["comment"] == "looks fine to me"


def test_fallback_still_falls_through_to_the_whitelist():
    p = _plugin(
        {"default": _fallback_evaluator},
        default_policy="deny",
        whitelist_tools={"safe_tool"},
    )

    allowed, info = p.check_permission("safe_tool", {})

    assert allowed is True, info
    assert info["method"] == "whitelist"


def test_fallback_still_falls_through_to_the_blacklist():
    p = _plugin(
        {"default": _fallback_evaluator},
        default_policy="allow",
        blacklist_tools={"admin_dangerous_tool"},
    )

    allowed, info = p.check_permission("admin_dangerous_tool", {})

    assert allowed is False, info
    assert info["method"] == "blacklist"


def test_framework_reserved_tool_keeps_its_catch_all_exemption():
    """The veto sits INSIDE the block the reserved-tool carve-out gates, so
    a reserved tool must still be exempt from a catch-all deny evaluator."""
    p = _plugin(
        {"default": _deny_evaluator},
        default_policy="deny",
        whitelist_tools={"signal_completion"},
    )
    p.add_framework_reserved_tools(["signal_completion"])

    allowed, info = p.check_permission("signal_completion", {})

    assert allowed is True, info
    assert info["method"] != "evaluator"


# ---------------------------------------------------------------------------
# The source must not contradict itself again (the other half of #679)
# ---------------------------------------------------------------------------

_POLICY_PY = Path(__file__).resolve().parents[1] / "plugins" / "permission" / "policy.py"


def test_policy_source_does_not_claim_evaluators_precede_the_blacklist():
    """``:4``, ``:82-84``, ``:116`` and ``:144`` each asserted an ordering
    the code no longer has.  CLAUDE.md treats an inaccurate docstring as a
    defect, and this one described a security boundary."""
    text = _POLICY_PY.read_text()

    assert "before blacklist/whitelist checks" not in text
    assert "after sanitization, before blacklist" not in text
    # The public veto every ALLOW path must ask.
    assert "def blacklist_veto(" in text
