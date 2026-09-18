"""A runner-served session must hold a ledger, or nothing is recorded.

``JaatoSession._record_token_usage`` returns early on a missing ledger and
``ToolExecutor`` is constructed with ``ledger=self._runtime.ledger``, so
the ``response`` and ``permission-check`` records the audit contract
promises (Arts. 12, 19) exist only if the runtime that owns the session
was given a :class:`TokenLedger`.  The runner's bootstrap passed ``None``
-- on the reading that token accounting is daemon-tier -- while the
daemon never received a runner session's records into its own ledger.
So on the default path ``trace.ledger`` named a file nobody wrote and
``explain audit <profile>`` reported it as written: a key parsed,
validated, rendered and enforced by nothing (the #735 shape), found by
driving a live daemon for the EU AI Act evidence manual.

The guard sits here rather than beside the Path D tests because the
reversion meta-guard walks ``shared/tests`` and ``server/tests`` only.
"""
from __future__ import annotations

from typing import Any, List, Optional

from shared.tests.test_every_guard_detects_its_own_reversion import Reversion
from shared.token_accounting import TokenLedger

_RUNNER_SESSION = "jaato-server/server/runner/session.py"

REVERSIONS = [
    Reversion(
        target=_RUNNER_SESSION,
        find="runtime.configure_plugins(registry, permission_plugin, TokenLedger())",
        replace="runtime.configure_plugins(registry, permission_plugin, None)",
        because=(
            "a runner runtime configured with no ledger records no response "
            "and no permission-check, whatever trace.ledger declares"
        ),
        test="test_the_runner_runtime_gets_a_ledger",
    ),
]


class _StubRuntime:
    """The four members ``_configure_runtime_plugins`` touches."""

    def __init__(self) -> None:
        self.is_connected = True
        self._registry: Any = None
        self.calls: List[dict] = []

    def configure_plugins(self, registry: Any, permission_plugin: Any = None,
                          ledger: Any = None, reliability_plugin: Any = None) -> None:
        self.calls.append({"registry": registry, "ledger": ledger})
        self._registry = registry


def _envelope(workspace: Optional[str]):
    from shared.session_envelope import SessionInitEnvelope
    return SessionInitEnvelope(
        session_id="sess-ledger", workspace_path=workspace, profile_name="p",
        provider_name="anthropic", model_name="claude-sonnet-4-6", plugins=[],
    )


def test_the_runner_runtime_gets_a_ledger(tmp_path):
    from server.runner.session import _configure_runtime_plugins
    stub = _StubRuntime()
    _configure_runtime_plugins(stub, _envelope(str(tmp_path)))
    assert len(stub.calls) == 1
    assert isinstance(stub.calls[0]["ledger"], TokenLedger)


def test_the_runner_ledger_writes_where_the_session_env_says(tmp_path, monkeypatch):
    """The ledger the runner holds resolves ``LEDGER_PATH`` per record, so
    the session env this bootstrap applies decides the file -- one per
    session for a relative ``trace.ledger``."""
    from server.runner.session import _configure_runtime_plugins
    stub = _StubRuntime()
    _configure_runtime_plugins(stub, _envelope(str(tmp_path)))
    ledger = stub.calls[0]["ledger"]
    monkeypatch.setenv("LEDGER_PATH", ".jaato/logs/ledger.jsonl")
    monkeypatch.setenv("JAATO_WORKSPACE_ROOT", str(tmp_path))
    ledger._record("response", {"prompt_tokens": 1, "output_tokens": 1, "total_tokens": 2})
    written = tmp_path / ".jaato" / "logs" / "ledger.jsonl"
    assert written.is_file()
    assert '"stage": "response"' in written.read_text(encoding="utf-8")
