"""An enrichment plugin's ``client_notice`` reaches the client as a typed event (1.31).

Before 1.31 what a tool-result enrichment plugin found reached the model and,
for a client, only as a formatted ``source="enrichment"`` line.  A client
that wanted to act on the finding had to detect it again.  A plugin now puts
``{"client_notice": {"kind", "data"}}`` in its enrichment metadata and the
session hands it to the UI hooks once the result is built; on the runner path
the shim forwards it as a ``tool_result_enriched`` notification and the
daemon turns that into ``ToolResultEnrichedEvent``.

Driven through a real ``PluginRegistry`` and the session's own
``_build_tool_result``, so the test asserts where the notice goes rather than
what a stub of the seam would do.
"""

from typing import Any, Dict, List

from jaato_sdk.events import ToolResultEnrichedEvent
from jaato_sdk.plugins.base import ToolResultEnrichmentResult
from jaato_sdk.plugins.model_provider.types import FunctionCall

from jaato_server.shared.enrichment_notice import MAX_NOTICE_BYTES, client_notices
from jaato_server.shared.plugins.registry import PluginRegistry
from jaato_server.shared.tests.reversion import Reversion


REVERSIONS = [
    Reversion(
        target="jaato-server/jaato_server/shared/jaato_session.py",
        find=(
            "                enrichment_metadata = None\n"
            "        self._emit_enrichment_client_notices(fc, enrichment_metadata)\n"
        ),
        replace="                enrichment_metadata = None\n",
        test="test_a_dict_result_notice_reaches_the_hooks",
        because="the notice staying in the enrichment metadata, where no client reads it",
    ),
    Reversion(
        target="jaato-server/jaato_server/shared/enrichment_notice.py",
        find="    if size > MAX_NOTICE_BYTES:\n",
        replace="    if False:\n",
        test="test_what_is_not_well_formed_is_dropped",
        because="an unbounded plugin payload travelling on every frame of the wire",
    ),
    Reversion(
        target="jaato-server/jaato_server/server/core.py",
        find='    "tool_result_enriched": _tool_result_enriched_from_payload,\n',
        replace="",
        test="test_the_daemon_turns_the_frame_into_the_event",
        because="the runner's frame reaching the daemon and being dropped, on the default path",
    ),
]


class _Offer:
    """A real enrichment plugin: appends a line and says so to the client."""

    name = "offer"

    def initialize(self, config=None):
        pass

    def shutdown(self):
        pass

    def reset_for_next_session(self):
        pass

    def subscribes_to_tool_result_enrichment(self) -> bool:
        return True

    def enrich_tool_result(self, tool_name, result, tool_args=None):
        if "command not found" not in result:
            return ToolResultEnrichmentResult(result=result)
        return ToolResultEnrichmentResult(
            result=result + "\n[hint] javac comes with Java",
            metadata={"client_notice": {"kind": "toolchain_offer", "data": {"command": "javac", "tool": "java"}}},
        )


class _Hooks:
    def __init__(self):
        self.calls: List[Dict[str, Any]] = []

    def on_tool_result_enriched(self, **kw):
        self.calls.append(kw)


def _session(hooks):
    """The session's own ``_build_tool_result`` over just the state it reads."""
    from jaato_server.shared.jaato_session import JaatoSession

    registry = PluginRegistry()
    registry.register_plugin(_Offer(), enrichment_only=True)

    class _Stub:
        _current_output_callback = None
        _terminal_width = 80
        _current_turn_span = None
        _agent_id = "main"

        def __init__(self):
            self._runtime = type("_RT", (), {"registry": registry})()
            self._ui_hooks = hooks

        def _trace(self, msg):
            pass

        def _check_and_pin_reference(self, metadata, text):
            pass

        _emit_enrichment_telemetry = JaatoSession._emit_enrichment_telemetry
        _enrich_tool_result_dict = JaatoSession._enrich_tool_result_dict
        _emit_enrichment_client_notices = JaatoSession._emit_enrichment_client_notices
        _build_tool_result = JaatoSession._build_tool_result

    return _Stub()


FAILED = {"stdout": "", "stderr": "bash: line 1: javac: command not found", "returncode": 127}


def test_a_dict_result_notice_reaches_the_hooks():
    hooks = _Hooks()
    result = _session(hooks)._build_tool_result(FunctionCall(id="c1", name="cli_based_tool", args={}), dict(FAILED))
    assert hooks.calls == [{
        "agent_id": "main", "call_id": "c1", "tool_name": "cli_based_tool",
        "plugin": "offer", "kind": "toolchain_offer", "data": {"command": "javac", "tool": "java"},
    }]
    # The model still gets the plugin's line; the notice is in addition to it.
    assert "[hint] javac comes with Java" in str(result.result)


def test_a_string_result_notice_reaches_the_hooks():
    hooks = _Hooks()
    _session(hooks)._build_tool_result(FunctionCall(id="c2", name="notebook_execute", args={}), "/bin/sh: 1: javac: command not found")
    assert [c["kind"] for c in hooks.calls] == ["toolchain_offer"]


def test_nothing_is_sent_when_no_plugin_has_a_notice():
    hooks = _Hooks()
    _session(hooks)._build_tool_result(FunctionCall(id="c3", name="cli_based_tool", args={}), {"stdout": "ok", "returncode": 0})
    assert hooks.calls == []


def test_hooks_without_the_method_receive_nothing():
    _session(object())._build_tool_result(FunctionCall(id="c4", name="cli_based_tool", args={}), dict(FAILED))


def test_what_is_not_well_formed_is_dropped():
    good = {"kind": "toolchain_offer", "data": {"command": "javac"}}
    meta = {
        "a": {"client_notice": [good, {"kind": "Bad Kind", "data": {}}]},
        "b": {"client_notice": {"kind": "x", "data": ["not", "a", "dict"]}},
        "c": {"client_notice": {"kind": "big", "data": {"blob": "x" * (MAX_NOTICE_BYTES + 1)}}},
        "d": {"client_notice": {"kind": "nan", "data": {"v": float("nan")}}},
        "e": {"other": 1},
    }
    assert client_notices(meta) == [("a", "toolchain_offer", {"command": "javac"})]


def test_the_runner_shim_forwards_it_as_a_notification():
    from jaato_server.server.runner.rpc import RunnerRPC, _AgentUIHooksNotificationShim

    sent = []

    class _RPC:
        _NOTIF_TOOL_RESULT_ENRICHED = RunnerRPC._NOTIF_TOOL_RESULT_ENRICHED

        def emit_notification(self, **kw):
            sent.append(kw)

    _AgentUIHooksNotificationShim(_RPC(), 7).on_tool_result_enriched(
        agent_id="main", call_id="c1", tool_name="cli_based_tool", plugin="offer",
        kind="toolchain_offer", data={"command": "javac"})
    assert sent == [{"request_id": 7, "event_type": "tool_result_enriched", "payload": {
        "agent_id": "main", "call_id": "c1", "tool_name": "cli_based_tool", "plugin": "offer",
        "kind": "toolchain_offer", "data": {"command": "javac"}}}]


def test_the_daemon_turns_the_frame_into_the_event():
    from jaato_server.server.core import _PURE_NOTIFICATION_EVENTS

    event = _PURE_NOTIFICATION_EVENTS["tool_result_enriched"](None, {
        "agent_id": "main", "call_id": "c1", "tool_name": "cli_based_tool", "plugin": "offer",
        "kind": "toolchain_offer", "data": {"command": "javac"}})
    assert isinstance(event, ToolResultEnrichedEvent)
    assert (event.kind, event.data, event.call_id) == ("toolchain_offer", {"command": "javac"}, "c1")
    # A payload from a runner of another vintage cannot take the demuxer down.
    assert _PURE_NOTIFICATION_EVENTS["tool_result_enriched"](None, {"data": "junk"}).data == {}
