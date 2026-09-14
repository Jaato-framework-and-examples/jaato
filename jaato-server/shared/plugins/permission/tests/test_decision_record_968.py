"""A permission decision is a readable record, not a silence (issue #968).

#951 gave the plugin a DECISION trace line at a single exit, which is
what made an ALLOW and a DENY stop being byte-identical.  #968 is the
rest of that record, and it is three separate claims:

**1. The line is a contract, not prose.**  It is the ONE artefact every
deployment gets by default — the ledger's ``permission-check`` row needs
a ledger, ``_execution_log`` never leaves the process, and the event is
opt-in — so an operator must be able to read it back mechanically.
:func:`~..plugin.parse_decision_trace` is that reader, and these tests
use it rather than a regex of their own, which is the point.

**2. "Nobody was asked" is a fact, not an inference.**  #797 needed to
tell "policy said yes" from "the prompt was skipped", and ``method``
cannot answer: ``allow_all`` / ``turn_suspension`` / ``idle_suspension``
are each produced BOTH by a pre-approval short-circuit that asked nobody
and by a human answering ``a`` / ``t`` / ``i`` at the prompt.  The two
are now distinguished by ``asked=``, recorded where the call actually
reaches the channel.

**3. The event bus can carry every decision, at a price.**  Most
terminal decisions announce nothing on the hook — suspensions,
``allow_all``, the trusted bridge, an ``askPermission`` grant, an
evaluator's early exit, an uninitialized plugin — and in subagent mode
*nothing at all*, which is exactly the blind spot #951 reported.
``emit_decision_events`` closes that, and is OFF by default because the
hook fans out to every connected client and the daemon answers each one
with a second ``PermissionStatusEvent``.  Both halves are asserted: the
default emits nothing extra, the opt-in emits exactly one per decision.

#859's distinction is preserved throughout: ``user_id`` / ``approver``
are absent for a policy decision *by design*, so they are written to the
line only when present and passed as ``None`` on the event.

The three acceptance criteria of #968 have a test each, named for them:
:func:`test_acceptance_a_whitelist_approval_names_the_rule`,
:func:`test_acceptance_a_denial_names_the_rule`, and
:func:`test_acceptance_an_evaluator_decision_names_the_evaluator`; plus
:func:`test_acceptance_951_a_subagents_write_names_its_verdict`, the
case the issue says must become diagnosable.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
from unittest.mock import Mock

import pytest

from .. import plugin as permission_plugin_module
from ..channels import ChannelDecision, ChannelResponse
from ..evaluator import PolicyDecision as EvalDecision, EvalResult
from ..plugin import (
    PermissionPlugin,
    _describe_permission_decider,
    parse_decision_trace,
)


# ---------------------------------------------------------------- harness


@pytest.fixture
def traced(monkeypatch) -> List[Tuple[str, str]]:
    """Capture ``(component, message)`` pairs instead of writing a file."""
    lines: List[Tuple[str, str]] = []

    def _capture(component: str, msg: str, **_kwargs: Any) -> None:
        lines.append((component, msg))

    monkeypatch.setattr(permission_plugin_module, "_trace_write", _capture)
    return lines


def _records(lines: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
    """Every DECISION line, read back through the shipped parser."""
    parsed = [parse_decision_trace(msg) for _c, msg in lines]
    return [record for record in parsed if record is not None]


def _plugin(policy: Dict[str, Any], **config: Any) -> PermissionPlugin:
    plugin = PermissionPlugin()
    plugin.initialize({"policy": policy, "channel_type": "queue", **config})
    return plugin


def _asking_plugin(response: ChannelResponse, **config: Any) -> PermissionPlugin:
    """A plugin whose channel answers with *response* on the first prompt."""
    plugin = PermissionPlugin()
    plugin.initialize({"policy": {"defaultPolicy": "ask"}, **config})
    channel = Mock()
    channel.name = "webhook"
    channel.request_permission.return_value = response
    plugin._channel = channel
    return plugin


def _capture_hook(plugin: PermissionPlugin) -> List[Dict[str, Any]]:
    """Collect every resolved-hook invocation as a plain dict."""
    seen: List[Dict[str, Any]] = []

    def on_resolved(tool_name, request_id, granted, method, **kw):
        seen.append({
            "tool_name": tool_name, "request_id": request_id,
            "granted": granted, "method": method, **kw,
        })

    plugin.set_permission_hooks(on_resolved=on_resolved)
    return seen


SUBAGENT_CONTEXT = {
    "agent_type": "subagent",
    "agent_name": "documentalista",
    "session_id": "20260910_115239",
}


# ------------------------------------------------------------- acceptance


def test_acceptance_a_whitelist_approval_names_the_rule(traced) -> None:
    """#968 acceptance 1: approved, and by which rule, from the log."""
    plugin = _plugin({"defaultPolicy": "deny",
                      "whitelist": {"tools": ["writeNewFile"]}})

    allowed, _info = plugin.check_permission(
        "writeNewFile", {"path": "docs/x.md"}, SUBAGENT_CONTEXT, "call_1")

    assert allowed is True
    record, = _records(traced)
    assert record["tool"] == "writeNewFile"
    assert record["allowed"] is True
    assert record["method"] == "whitelist"
    assert record["asked"] is False
    assert record["call_id"] == "call_1"


def test_acceptance_a_denial_names_the_rule(traced) -> None:
    """#968 acceptance 2, denial half: a blacklist DENY says so."""
    plugin = _plugin({"defaultPolicy": "allow",
                      "blacklist": {"tools": ["removeFile"]}})

    allowed, _info = plugin.check_permission(
        "removeFile", {"path": "/etc/passwd"}, SUBAGENT_CONTEXT, "call_2")

    assert allowed is False
    record, = _records(traced)
    assert record["allowed"] is False
    assert record["method"] == "blacklist"
    assert record["asked"] is False
    assert record["reason"]


def test_acceptance_an_evaluator_decision_names_the_evaluator(traced) -> None:
    """#968 acceptance 2, evaluator half.

    An evaluator DENY exits before the policy match, which is one of the
    branches that announces nothing on the event bus — so the log is the
    only place it can be read, and it must name the evaluator rather
    than the generic default.
    """
    plugin = _plugin({"defaultPolicy": "allow"})
    plugin._policy.set_evaluators({
        "default": lambda tool_name, args, context: EvalResult(
            decision=EvalDecision.DENY),
    })

    allowed, info = plugin.check_permission(
        "writeNewFile", {"path": "docs/x.md"}, SUBAGENT_CONTEXT, "call_3")

    assert allowed is False
    assert info["method"] == "evaluator"
    record, = _records(traced)
    assert record["allowed"] is False
    assert record["method"] == "evaluator"
    assert record["asked"] is False


def test_acceptance_951_a_subagents_write_names_its_verdict(traced) -> None:
    """#968 acceptance 3: #951's author could now say which it was.

    #951 reported 38 ``check_permission`` calls on a subagent's
    ``writeNewFile``, zero prompts, no file on disk — and re-running
    with ``defaultPolicy: deny`` produced the same logs, so a genuine
    policy DENY and a call vanishing after the gate were one silence.
    The two profiles must now produce visibly different records, and
    each must name the subagent that asked.
    """
    denying = _plugin({"defaultPolicy": "deny", "whitelist": {"tools": []}})
    denying.check_permission("writeNewFile", {}, SUBAGENT_CONTEXT, "c")
    denied, = _records(traced)

    traced.clear()
    allowing = _plugin({"defaultPolicy": "deny",
                        "whitelist": {"tools": ["writeNewFile"]}})
    allowing.check_permission("writeNewFile", {}, SUBAGENT_CONTEXT, "c")
    approved, = _records(traced)

    assert (denied["allowed"], denied["method"]) == (False, "default")
    assert (approved["allowed"], approved["method"]) == (True, "whitelist")
    # And both are attributable to the subagent, not to the shared
    # singleton's last initializer.
    assert denied["agent"] == approved["agent"] == "subagent:documentalista"
    assert denied["session"] == "20260910_115239"


# ------------------------------------------- asked vs decided-for (#797)


def test_asked_is_false_for_a_policy_grant(traced) -> None:
    """``allow_all`` reached by pre-approval asked nobody."""
    plugin = _plugin({"defaultPolicy": "deny", "whitelist": {"tools": []}})
    plugin._allow_all = True

    plugin.check_permission("writeNewFile", {}, None, "c")

    record, = _records(traced)
    assert record["method"] == "allow_all"
    assert record["asked"] is False


def test_asked_is_true_when_a_human_answered_all(traced) -> None:
    """The same ``method``, reached by a prompt somebody answered.

    This pair is the #797 distinction in one file: ``method`` alone
    cannot separate them, ``asked`` can.
    """
    plugin = _asking_plugin(ChannelResponse(
        request_id="r", decision=ChannelDecision.ALLOW_ALL, reason="all"))

    plugin.check_permission("writeNewFile", {}, None, "c")

    record, = _records(traced)
    assert record["method"] == "allow_all"
    assert record["asked"] is True


def test_asked_resets_between_calls(traced) -> None:
    """A prompted call must not label the next, unprompted one."""
    plugin = _asking_plugin(ChannelResponse(
        request_id="r", decision=ChannelDecision.ALLOW_ALL, reason="all"))

    plugin.check_permission("first", {}, None, "c1")
    plugin.check_permission("second", {}, None, "c2")

    first, second = _records(traced)
    assert first["asked"] is True
    assert second["asked"] is False


# ----------------------------------------------------- who decided (#859)


def test_a_policy_decision_names_nobody_in_the_line(traced) -> None:
    """#859's distinction survives: absence, not ``user_id=None``."""
    plugin = _plugin({"defaultPolicy": "deny",
                      "whitelist": {"tools": ["readFile"]}})

    plugin.check_permission("readFile", {}, None, "c")

    line, = [msg for _c, msg in traced if "DECISION" in msg]
    assert "user_id=" not in line
    assert "approver=" not in line
    assert "user_id" not in _records(traced)[0]


def test_an_answered_prompt_names_who_answered(traced) -> None:
    plugin = _asking_plugin(ChannelResponse(
        request_id="r", decision=ChannelDecision.ALLOW, reason="ok",
        user_id="sso|alice", approver="Alice"))

    plugin.check_permission("deploy", {"env": "prod"}, None, "c")

    record, = _records(traced)
    assert record["user_id"] == "sso|alice"
    assert record["approver"] == "Alice"
    assert record["asked"] is True


def test_describe_decider_writes_only_what_is_there() -> None:
    assert _describe_permission_decider({}) == ""
    assert _describe_permission_decider({"user_id": None}) == ""
    assert _describe_permission_decider({"user_id": "u"}) == " user_id=u"
    assert _describe_permission_decider(
        {"user_id": "u", "approver": "a"}) == " user_id=u approver=a"


# ---------------------------------------------------- the event, opt-in


def test_by_default_an_unannounced_decision_stays_unannounced() -> None:
    """The cost guard.

    A suspension ALLOW fires no hook today, and must keep firing none
    unless the deployment asked for it: the hook reaches every connected
    client and the daemon answers it with a second event, per tool call.
    """
    plugin = _plugin({"defaultPolicy": "deny", "whitelist": {"tools": []}})
    plugin._idle_suspended = True
    seen = _capture_hook(plugin)

    allowed, _info = plugin.check_permission("writeNewFile", {}, None, "c")

    assert allowed is True
    assert seen == []


def test_opting_in_announces_the_decision_nobody_watched() -> None:
    plugin = _plugin({"defaultPolicy": "deny", "whitelist": {"tools": []}},
                     emit_decision_events=True)
    plugin._idle_suspended = True
    seen = _capture_hook(plugin)

    plugin.check_permission("writeNewFile", {}, None, "c")

    assert len(seen) == 1
    assert seen[0]["method"] == "idle_suspension"
    assert seen[0]["granted"] is True
    # #859 preserved on the opt-in path: nobody was asked.
    assert seen[0]["user_id"] is None
    assert seen[0]["approver"] is None


def test_opting_in_does_not_double_an_already_announced_decision() -> None:
    """A whitelist ALLOW already fires the hook; it must fire once."""
    plugin = _plugin({"defaultPolicy": "deny",
                      "whitelist": {"tools": ["readFile"]}},
                     emit_decision_events=True)
    seen = _capture_hook(plugin)

    plugin.check_permission("readFile", {}, None, "c")

    assert len(seen) == 1
    assert seen[0]["method"] == "whitelist"


def test_opting_in_announces_an_answered_prompt_once_with_identity() -> None:
    plugin = _asking_plugin(
        ChannelResponse(request_id="r", decision=ChannelDecision.ALLOW,
                        reason="ok", user_id="sso|alice", approver="Alice"),
        emit_decision_events=True)
    seen = _capture_hook(plugin)

    plugin.check_permission("deploy", {}, None, "c")

    assert len(seen) == 1
    assert seen[0]["user_id"] == "sso|alice"
    assert seen[0]["approver"] == "Alice"


def test_opting_in_announces_a_subagents_decision(traced) -> None:
    """#951's blind spot: in subagent mode the hook is suppressed.

    Suppression exists so a child's prompt does not drive the parent's
    UI into "waiting for input".  An automatic decision has no prompt to
    drive anything, and it is the decision an auditor most wants, so
    the opt-in surfaces it — with an empty ``request_id``, which is what
    keeps the daemon's pending-prompt guard from clobbering a parallel
    prompt.
    """
    from ..channels import ParentBridgedChannel

    plugin = _plugin({"defaultPolicy": "deny",
                      "whitelist": {"tools": ["writeNewFile"]}},
                     emit_decision_events=True)
    plugin._channel = ParentBridgedChannel()
    seen = _capture_hook(plugin)

    allowed, _info = plugin.check_permission(
        "writeNewFile", {}, SUBAGENT_CONTEXT, "c")

    assert allowed is True
    assert len(seen) == 1
    assert seen[0]["method"] == "whitelist"
    assert seen[0]["request_id"] == ""
    assert _records(traced)[0]["agent"] == "subagent:documentalista"


def test_an_uninitialized_plugin_never_announces() -> None:
    """No config was read, so the opt-in cannot have been given."""
    plugin = PermissionPlugin()
    seen = _capture_hook(plugin)

    allowed, info = plugin.check_permission("writeNewFile", {}, None, "c")

    assert allowed is True and info["method"] == "not_initialized"
    assert seen == []


# --------------------------------------------------------------- parser


def test_parser_ignores_a_non_decision_line() -> None:
    assert parse_decision_trace("check_permission: tool=x call_id=y") is None
    assert parse_decision_trace("") is None


def test_parser_round_trips_a_reason_containing_spaces(traced) -> None:
    """``reason`` is the only free-text field, so it is written last."""
    plugin = _plugin({"defaultPolicy": "deny", "whitelist": {"tools": []}})

    _allowed, info = plugin.check_permission("writeNewFile", {}, None, "c")

    record, = _records(traced)
    assert " " in record["reason"]
    assert record["reason"] == info["reason"]


def test_parser_reads_a_missing_call_id_as_none(traced) -> None:
    plugin = _plugin({"defaultPolicy": "allow"})

    plugin.check_permission("writeNewFile", {}, None)

    assert _records(traced)[0]["call_id"] is None
