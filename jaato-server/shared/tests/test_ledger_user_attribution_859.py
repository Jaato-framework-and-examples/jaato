"""The token ledger names the user a session runs as (issue #859).

``JaatoSession.set_client_user_id`` fed only the telemetry ``user.id``
attribute, so a deployment without an observability backend had no way
to attribute spend or approvals to a person after the fact.  The session
now stamps the same id on every ``response`` ledger record, and the tool
executor stamps the approver identity on ``permission-check`` records.
Both omit the key -- rather than write ``None`` -- when there is no user,
so keyless / IPC ledgers are unchanged.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from shared.ai_tool_runner import ToolExecutor
from shared.jaato_session import JaatoSession
from shared.token_accounting import TokenLedger


def _session(user_id: Any) -> JaatoSession:
    sess = JaatoSession.__new__(JaatoSession)
    sess._client_user_id = user_id
    sess._runtime = SimpleNamespace(ledger=TokenLedger())
    sess._budget_observe_response = lambda response: None  # type: ignore[method-assign]
    sess.get_session_env = lambda key, default=None: None  # type: ignore[method-assign]
    return sess


def _response() -> Any:
    return SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=10, output_tokens=5, total_tokens=15),
    )


def test_response_record_carries_the_session_user() -> None:
    sess = _session("sso|alice")
    sess._record_token_usage(_response())
    (record,) = sess._runtime.ledger.events()
    assert record["stage"] == "response"
    assert record["total_tokens"] == 15
    assert record["user_id"] == "sso|alice"


def test_response_record_without_user_has_no_key() -> None:
    sess = _session(None)
    sess._record_token_usage(_response())
    (record,) = sess._runtime.ledger.events()
    assert "user_id" not in record


def test_permission_check_record_names_the_approver() -> None:
    ledger = TokenLedger()
    plugin = SimpleNamespace(
        check_permission=lambda name, args, ctx, call_id: (
            True,
            {"reason": "ok", "method": "user_approved",
             "user_id": "sso|alice", "approver": "Alice"},
        ),
    )
    executor = ToolExecutor(ledger=ledger)
    executor.set_permission_plugin(plugin)
    executor.register("noop", lambda **kw: {"ok": True})

    executor.execute("noop", {})

    checks = [e for e in ledger.events() if e["stage"] == "permission-check"]
    assert len(checks) == 1
    assert checks[0]["method"] == "user_approved"
    assert checks[0]["user_id"] == "sso|alice"
    assert checks[0]["approver"] == "Alice"


def test_permission_check_record_for_a_policy_decision_names_nobody() -> None:
    ledger = TokenLedger()
    plugin = SimpleNamespace(
        check_permission=lambda name, args, ctx, call_id: (
            True, {"reason": "whitelisted", "method": "whitelist"},
        ),
    )
    executor = ToolExecutor(ledger=ledger)
    executor.set_permission_plugin(plugin)
    executor.register("noop", lambda **kw: {"ok": True})

    executor.execute("noop", {})

    (check,) = [e for e in ledger.events() if e["stage"] == "permission-check"]
    assert "user_id" not in check
    assert "approver" not in check
