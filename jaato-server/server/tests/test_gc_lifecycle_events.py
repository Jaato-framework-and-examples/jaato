"""GC lifecycle reaches clients as a TYPED event, not prose or silence.

Before this there was no GC lifecycle signal on the bus at all.  The framework
opened an OTel span carrying the trigger reason and strategy, and a client got
either a ``SystemMessageEvent`` reading "Context usage (84.2%) exceeds
threshold (80%). GC will run after this turn." -- or nothing.  Showing
"compacting..." meant substring-matching that sentence for the start and
guessing at the end, which is the parse-the-log shape typed events replace.

``GCConfigEvent`` did NOT cover this: its own docstring says it carries
configuration only.
"""
from types import SimpleNamespace

import pytest

from server.core import JaatoServer
from jaato_sdk.events import EventType
from shared.gc_support import run_gc
from shared.plugins.gc import GCResult
from shared.plugins.gc.base import GCTriggerReason


# ------------------------------------------------------- run_gc emits phases

class _Span:
    def set_attribute(self, k, v): pass


class _Telemetry:
    def gc_span(self, **kw):
        class _CM:
            def __enter__(_s): return _Span()
            def __exit__(_s, *e): return False
        return _CM()


class _Plugin:
    name = "gc_hybrid"

    def __init__(self, result): self.result = result

    def collect(self, history, usage, config, reason, budget=None):
        return (["after"], self.result)


def _result(**kw):
    base = dict(success=True, items_collected=4, tokens_before=1000,
                tokens_after=400, plugin_name="gc_hybrid",
                trigger_reason=GCTriggerReason.THRESHOLD)
    base.update(kw)
    return GCResult(**base)


def _phases(result=None, on_phase_raises=False):
    seen = []

    def _on_phase(phase, payload):
        if on_phase_raises:
            raise RuntimeError("observer exploded")
        seen.append((phase, payload))

    out = run_gc(
        gc_plugin=_Plugin(result or _result()),
        history=["before"],
        context_usage={"percent_used": 84.0, "context_limit": 120},
        gc_config=SimpleNamespace(),
        trigger_reason=GCTriggerReason.THRESHOLD,
        budget=None, cache_plugin=None,
        telemetry=_Telemetry(), on_trace=lambda m: None,
        on_phase=_on_phase,
    )
    return seen, out


def test_every_pass_emits_started_then_completed():
    seen, _ = _phases()
    assert [p for p, _ in seen] == ["started", "completed"]


def test_started_carries_the_before_framing():
    seen, _ = _phases()
    payload = seen[0][1]
    assert payload["trigger_reason"] == "threshold"
    assert payload["strategy"] == "gc_hybrid"
    assert payload["percent_used"] == 84.0


def test_completed_carries_the_outcome():
    seen, _ = _phases()
    payload = seen[1][1]
    assert payload["success"] is True
    assert payload["items_collected"] == 4
    assert payload["tokens_freed"] == 600
    assert payload["tokens_before"] == 1000 and payload["tokens_after"] == 400


def test_a_failed_pass_still_completes():
    seen, _ = _phases(_result(success=False, items_collected=0, error="boom"))
    assert [p for p, _ in seen] == ["started", "completed"]
    assert seen[1][1]["success"] is False
    assert seen[1][1]["error"] == "boom", (
        "a failed GC is the case an operator most needs to see")


def test_a_broken_observer_never_breaks_the_collection():
    # The GC must still return its result even if the listener raises.
    _, (history, result) = _phases(on_phase_raises=True)
    assert history == ["after"]
    assert result.items_collected == 4


# --------------------------------------------------- daemon emits a GCEvent

def _emit(payload):
    emitted = []
    srv = SimpleNamespace(emit=emitted.append, _main_agent_id="main")
    JaatoServer._emit_gc_phase_event(srv, payload)
    return emitted


def test_daemon_emits_a_typed_event():
    ev = _emit({"phase": "completed", "trigger_reason": "threshold",
                "strategy": "gc_hybrid", "success": True,
                "items_collected": 4, "tokens_freed": 600})
    assert len(ev) == 1
    assert ev[0].type == EventType.GC
    assert ev[0].phase == "completed"
    assert ev[0].tokens_freed == 600, (
        "a client must branch on values, not grep a sentence")


def test_about_to_run_carries_branchable_threshold_values():
    ev = _emit({"phase": "about_to_run", "percent_used": 84.2,
                "threshold": 80.0, "trigger_reason": "threshold"})
    assert ev[0].percent_used == 84.2 and ev[0].threshold == 80.0


@pytest.mark.parametrize("payload", [None, "text", {}, {"phase": ""}])
def test_malformed_payloads_emit_nothing(payload):
    assert _emit(payload) == []


def test_gc_event_is_distinct_from_gc_config():
    ev = _emit({"phase": "started"})
    assert ev[0].type != EventType.GC_CONFIG, (
        "GC_CONFIG carries configuration only, per its own docstring — it is "
        "not a lifecycle signal")


def test_the_notification_demuxer_actually_routes_gc_phase():
    """The DISPATCH, not just the handler.

    ``test_daemon_emits_a_typed_event`` calls ``_emit_gc_phase_event``
    directly, so deleting the ``event_type == "gc_phase"`` branch that routes
    to it leaves that test green — the handler still works, nothing calls it.
    Verified: removing the branch failed zero tests until this one existed.

    So build the real demuxer and feed it a real frame.
    """
    emitted = []
    srv = SimpleNamespace(
        emit=emitted.append,
        _main_agent_id="main",
        _emit_gc_phase_event=lambda payload: JaatoServer._emit_gc_phase_event(
            SimpleNamespace(emit=emitted.append, _main_agent_id="main"), payload),
    )
    handler = JaatoServer._build_send_message_notification_handler(srv)
    handler("gc_phase", {"phase": "started", "trigger_reason": "manual",
                         "strategy": "gc_truncate"})

    gc_events = [e for e in emitted if getattr(e, "type", None) == EventType.GC]
    assert len(gc_events) == 1, (
        "a gc_phase frame arrived from the runner and produced no event — the "
        "demuxer never routed it")
    assert gc_events[0].phase == "started"
    assert gc_events[0].strategy == "gc_truncate"
