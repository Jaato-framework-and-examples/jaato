"""Every GC pass goes through ``gc_support.run_gc``, so it is always observable.

The four collect paths on JaatoSession each wired their own telemetry by hand
and only half of them did: ``_maybe_collect_after_turn`` and
``_maybe_collect_before_send`` opened a span; ``_try_gc_for_context_recovery``
and ``manual_gc`` opened nothing.  So an operator watching spans saw a subset
of the GC that actually ran with nothing marking the difference -- and the two
invisible ones are exactly the passes someone debugging a context overflow goes
looking for.

A partially-firing observable is worse than none: it reads as complete.
"""
import ast
import pathlib
from types import SimpleNamespace

import pytest

from shared.gc_support import run_gc
from shared.plugins.gc import GCResult
from shared.plugins.gc.base import GCTriggerReason


class _Span:
    def __init__(self):
        self.attrs = {}

    def set_attribute(self, k, v):
        self.attrs[k] = v


class _Telemetry:
    """Records every gc_span opened, with the args it was opened with."""

    def __init__(self):
        self.spans = []

    def gc_span(self, *, trigger_reason, strategy, attributes):
        span = _Span()
        self.spans.append(
            {"trigger_reason": trigger_reason, "strategy": strategy,
             "attributes": attributes, "span": span})

        class _CM:
            def __enter__(_self):
                return span

            def __exit__(_self, *exc):
                return False

        return _CM()


def _result(**kw):
    base = dict(success=True, items_collected=3, tokens_before=100,
                tokens_after=40, plugin_name="fake",
                trigger_reason=GCTriggerReason.MANUAL)
    base.update(kw)
    return GCResult(**base)


class _Plugin:
    name = "fake"

    def __init__(self, result=None):
        self.result = result or _result()
        self.calls = []

    def collect(self, history, usage, config, reason, budget=None):
        self.calls.append(reason)
        return (["collected"], self.result)


def _run(plugin=None, telemetry=None, **kw):
    plugin = plugin or _Plugin()
    telemetry = telemetry or _Telemetry()
    out = run_gc(
        gc_plugin=plugin,
        history=["original"],
        context_usage={"percent_used": 84.0, "total_tokens": 100,
                       "context_limit": 120},
        gc_config=SimpleNamespace(),
        trigger_reason=GCTriggerReason.MANUAL,
        budget=None,
        cache_plugin=None,
        telemetry=telemetry,
        on_trace=lambda m: None,
        **kw,
    )
    return out, plugin, telemetry


def test_a_span_is_always_opened():
    _, _, tel = _run()
    assert len(tel.spans) == 1, (
        "a GC pass produced no span — the pass is invisible to any operator "
        "watching telemetry")


def test_span_carries_the_trigger_reason_and_strategy():
    _, _, tel = _run()
    assert tel.spans[0]["trigger_reason"] == GCTriggerReason.MANUAL.value
    assert tel.spans[0]["strategy"] == "fake"


def test_before_attributes_are_captured():
    _, _, tel = _run()
    attrs = tel.spans[0]["attributes"]
    assert attrs["gc.percent_used"] == 84.0
    assert attrs["gc.context_limit"] == 120


def test_result_is_populated_onto_the_span():
    _, _, tel = _run()
    span = tel.spans[0]["span"]
    assert span.attrs["gc.items_collected"] == 3
    assert span.attrs["gc.tokens_freed"] == 60, (
        "the span was opened but never populated — half an observation")


def test_on_collected_runs_inside_the_span_and_can_replace_history():
    seen = {}

    def _after(new_history, result, gc_span):
        seen["span"] = gc_span
        seen["history"] = new_history
        return ["replaced"]

    (history, result), _, tel = _run(on_collected=_after)
    assert seen["history"] == ["collected"]
    assert seen["span"] is tel.spans[0]["span"], "hook ran outside the span"
    assert history == ["replaced"]


def test_on_collected_returning_none_keeps_the_collected_history():
    (history, _), _, _ = _run(on_collected=lambda h, r, s: None)
    assert history == ["collected"]


def test_a_failed_gc_still_produces_a_span():
    plugin = _Plugin(_result(success=False, items_collected=0, error="boom"))
    _, _, tel = _run(plugin=plugin)
    assert len(tel.spans) == 1, (
        "a FAILED GC is the one an operator most needs to see")
    assert tel.spans[0]["span"].attrs["gc.success"] is False


# ---------------------------------------------------------------- drift guard

def test_no_gc_path_bypasses_the_shared_runner():
    """No direct ``_gc_plugin.collect(...)`` anywhere in jaato_session.

    The uniformity guard.  A NEW collect path added later would silently
    reintroduce exactly the state this module was written to end: a GC pass
    that runs without a span.  AST rather than grep so it matches real calls,
    not the word appearing in a comment or docstring.
    """
    src = pathlib.Path(__file__).resolve().parents[1] / "jaato_session.py"
    tree = ast.parse(src.read_text())
    offenders = [
        node.lineno for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "collect"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "_gc_plugin"
    ]
    assert not offenders, (
        f"jaato_session.py calls _gc_plugin.collect() directly at lines "
        f"{offenders}. Every GC pass must go through self._run_gc() or it "
        f"runs with no telemetry span, invisible to anyone watching."
    )
