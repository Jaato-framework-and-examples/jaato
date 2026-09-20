"""The call/result pairing invariant, and the repairs that restore it (#674).

Three layers, deliberately:

* **unit** — one test per defect kind, each named after the shape that
  produced it in the wild (cancellation mid-batch, resume from an orphaned
  leading result, a thought-only turn, an endpoint that streams no id);
* **property** — a seeded deterministic generator builds histories with
  multi-call turns, runs every GC strategy and every cancel-at-turn-N cut
  over them, and asserts the invariant survives.  ``hypothesis`` is not in
  this tree's dependency set, so the generator is a plain seeded
  :class:`random.Random`: reproducible from the seed printed in the
  failure, and no new dependency;
* **representation** — the same repaired histories are rendered through
  the *prose* tool-calling wire, because a repair that is pair-safe on the
  Anthropic wire is not automatically pair-safe on a wire that carries the
  pairing in text.

**Measured finding, recorded here because it is a negative result worth
keeping.**  On this corpus all four GC strategies are *already* pair-safe
— they cut on turn boundaries (``split_into_turns`` / ``flatten_turns``
keep an assistant turn and its results together), and across 40 seeds each
removed ~500 messages without once orphaning a call.  The corpus is not
vacuous (:func:`test_gc_collection_is_not_vacuous` fails if GC stops
collecting), so these tests are a **regression detector**: a future
strategy, or a change to turn splitting, that starts cutting mid-pair
fails here instead of 400-ing in production.

Where the invariant *is* broken today is cancellation mid-batch, resume
from a suffix, and the wire — see
:class:`TestComposesWithTheGcPathRepair`, which pins the one case the
pre-existing GC-path repair gets wrong.
"""

import contextlib
import logging
import random
from unittest.mock import MagicMock

import pytest


@contextlib.contextmanager
def caplog_at(level):
    """Capture records this module's logger emits at or above ``level``.

    A plain handler rather than pytest's ``caplog`` fixture because these
    assertions are about the LEVEL a repair is announced at, and the
    fixture's root-level capture makes an INFO record indistinguishable
    from a WARNING one unless the level is pinned on the logger itself.
    """
    records = []

    class _Collect(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _Collect(level=level)
    logger = logging.getLogger("shared.history_invariant")
    logger.addHandler(handler)
    previous, logger.level = logger.level, logging.DEBUG
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.level = previous

from shared.history_invariant import (
    CANCELLED_RESULT,
    SYNTHETIC_CALL_ID_PREFIX,
    new_tool_call_nonce,
    repair_history,
    synthetic_tool_call_id,
    validate_history,
)
from shared.plugins.gc.base import GCConfig, GCTriggerReason
from shared.plugins.gc.utils import ensure_tool_call_integrity
from shared.plugins.gc_budget.plugin import BudgetGCPlugin
from shared.plugins.gc_hybrid.plugin import HybridGCPlugin
from shared.plugins.gc_summarize.plugin import SummarizeGCPlugin
from shared.plugins.gc_truncate.plugin import TruncateGCPlugin
from shared.plugins.model_provider._prose_tools import messages_to_prose_chat

from shared.jaato_session import JaatoSession
from shared.plugins.model_provider.nim.provider import NIMProvider
from shared.tests.reversion import Reversion

from jaato import FunctionCall, Message, Part, Role, ToolResult


# --------------------------------------------------------------------------
# Meta-guard declarations (test_every_guard_detects_its_own_reversion)
# --------------------------------------------------------------------------
#
# A guard the meta-guard does not know about is the gap that list exists to
# close: these tests would keep passing if someone put the defect back, and
# nothing would say so.  Each entry puts ONE defect back and names the ONE
# test that must then fail.
#
# Two constraints that are easy to discover the hard way:
#   * ``find`` is the FIXED text and ``replace`` the BROKEN text;
#   * ``_run_guard`` invokes ``<the module declaring this>::<test>``, and
#     ``_PACKAGES`` covers only ``shared/tests`` and ``server/tests`` — so a
#     reversion cannot name a test living anywhere else.  That is why the
#     wire-seam case below is exercised by a test in THIS module rather than
#     from the provider suite under ``_openai_compat/tests/``.

REVERSIONS = [
    Reversion(
        target="jaato-server/shared/history_invariant.py",
        find="    out = _answer_unmatched_calls(out, synthesized)",
        replace="    out = out  # reverted: unmatched calls left unanswered",
        test="TestCancellationMidBatch::test_partial_batch_is_fully_answered",
        because="a cancelled mid-batch turn reaching the provider with a "
                "tool_use nothing answers, which is the 400 #674 is about",
    ),
    Reversion(
        target="jaato-server/shared/plugins/gc/utils.py",
        find="    return repair_history(history, trace_fn=trace_fn)",
        replace="    return history  # reverted: no producer-side repair",
        test="TestTheProducerRepairsStoredHistory::"
             "test_the_gc_path_answers_a_partially_answered_batch",
        because="the GC path failing to repair STORED history, whose output "
                "is persisted — so a session revived from an unrepaired "
                "record comes back orphaned and no later per-request repair "
                "changes what is on disk",
    ),
    Reversion(
        target="jaato-server/shared/plugins/model_provider/"
               "_openai_compat/base.py",
        find="                        tool_id = synthetic_tool_call_id("
             "idx, tool_id_nonce)",
        replace="                        tool_id = None  # reverted: no mint",
        test="TestTheWireSeamMintsAnId::"
             "test_a_streamed_call_without_an_id_still_gets_one",
        because="an OpenAI-compatible endpoint that streams no tool-call id "
                "putting an unmatchable call into history, which no result "
                "can ever be paired with",
    ),
]


# ==================== builders ====================


def user(text="go"):
    """A plain user turn."""
    return Message.from_text(Role.USER, text)


def calls(*ids, text=None, thought=None):
    """A MODEL turn emitting ``ids`` as a parallel tool-call batch.

    ``text`` and ``thought`` put a text block and a reasoning block on the
    same message, which is the shape ``replay_reasoning`` providers keep
    in history and require back verbatim.
    """
    parts = []
    if thought is not None:
        parts.append(Part.from_thought(thought))
    if text is not None:
        parts.append(Part.from_text(text))
    parts += [Part.from_function_call(FunctionCall(id=i, name=f"tool_{i}",
                                                   args={"n": i}))
              for i in ids]
    return Message(role=Role.MODEL, parts=parts)


def results(*ids):
    """A TOOL turn answering ``ids``."""
    return Message(role=Role.TOOL, parts=[
        Part.from_function_response(
            ToolResult(call_id=i, name=f"tool_{i}", result={"ok": i}))
        for i in ids])


def model_text(text="done"):
    """A MODEL turn that only speaks."""
    return Message(role=Role.MODEL, parts=[Part.from_text(text)])


def _defect_kinds(history):
    """The sorted set of defect codes ``history`` exhibits."""
    return sorted({d.kind for d in validate_history(history)})


def _call_ids(history):
    return [p.function_call.id for m in history for p in m.parts
            if p.function_call]


def _result_ids(history):
    return [p.function_response.call_id for m in history for p in m.parts
            if p.function_response]


# ==================== unit: each defect kind ====================


class TestCancellationMidBatch:
    """The shape parallel tool execution makes routine.

    A turn emits N calls, the cancel token fires, fewer than N ran.  With
    8-wide parallelism the partially-answered batch is the *normal* cancel
    outcome rather than an edge case.
    """

    def test_partial_batch_is_fully_answered(self):
        history = [user(), calls("a", "b", "c"), results("a")]
        assert _defect_kinds(history) == ["unmatched_call"]
        repaired = repair_history(history)
        assert _defect_kinds(repaired) == []
        assert sorted(_result_ids(repaired)) == ["a", "b", "c"]

    def test_the_executed_result_is_kept(self):
        """Repair must not throw away work that actually ran.

        The pre-existing GC repair removes the whole assistant message
        here, which orphans the one result that *did* arrive — turning a
        partially-answered batch into the leading-orphan defect.
        """
        repaired = repair_history([user(), calls("a", "b"), results("a")])
        real = [p.function_response for m in repaired for p in m.parts
                if p.function_response
                and p.function_response.call_id == "a"]
        assert real and real[0].result == {"ok": "a"}

    def test_synthetic_results_are_flagged_as_errors(self):
        repaired = repair_history([user(), calls("a", "b"), results("a")])
        synth = [p.function_response for m in repaired for p in m.parts
                 if p.function_response
                 and p.function_response.call_id == "b"][0]
        assert synth.is_error is True
        assert synth.result == CANCELLED_RESULT

    def test_synthetic_results_merge_into_the_batch_message(self):
        """One result block per call, in one place — the uninterrupted shape."""
        repaired = repair_history([user(), calls("a", "b"), results("a")])
        tool_msgs = [m for m in repaired if m.role == Role.TOOL]
        assert len(tool_msgs) == 1
        assert len(tool_msgs[0].parts) == 2

    def test_trailing_unanswered_batch(self):
        """Nothing followed the call batch at all."""
        repaired = repair_history([user(), calls("x", "y")])
        assert _defect_kinds(repaired) == []
        assert sorted(_result_ids(repaired)) == ["x", "y"]

    def test_repair_is_byte_stable_across_requests(self):
        """An unstable repair would invalidate the prompt cache every turn.

        ``repair_history`` runs on every request built from the same
        stored history, so a synthesised result carrying a timestamp or a
        random id would change the request prefix each time and cost a
        full cache re-read on Anthropic and Gemini upstreams.
        """
        history = [user(), calls("a", "b"), results("a")]
        first = _result_ids(repair_history(history))
        second = _result_ids(repair_history(history))
        assert first == second
        payloads = {str(p.function_response.result)
                    for m in repair_history(history) for p in m.parts
                    if p.function_response}
        assert str(CANCELLED_RESULT) in payloads


class TestLeadingOrphanResult:
    """Resume / rewind: history that *opens* with a tool result."""

    def test_leading_orphan_is_dropped(self):
        history = [results("gone"), user("continue")]
        assert _defect_kinds(history) == ["orphan_result"]
        repaired = repair_history(history)
        assert _defect_kinds(repaired) == []
        assert _result_ids(repaired) == []

    def test_mixed_batch_keeps_its_good_results(self):
        """A result message half of whose ids are valid loses only the half."""
        history = [user(), calls("a"),
                   Message(role=Role.TOOL, parts=[
                       Part.from_function_response(
                           ToolResult(call_id="a", name="t", result=1)),
                       Part.from_function_response(
                           ToolResult(call_id="stale", name="t", result=2)),
                   ])]
        repaired = repair_history(history)
        assert _defect_kinds(repaired) == []
        assert _result_ids(repaired) == ["a"]

    def test_gemini_shaped_results_on_a_user_message(self):
        """Results ride a USER message on the Gemini-shaped path."""
        history = [Message(role=Role.USER, parts=[
            Part.from_function_response(
                ToolResult(call_id="gone", name="t", result=1))])]
        assert _defect_kinds(history) == ["orphan_result"]
        assert _defect_kinds(repair_history(history)) == []


class TestEmptyContentBlocks:
    """*"text content blocks must be non-empty"*, and its whitespace variant."""

    def test_empty_text_beside_a_thinking_block_is_removed(self):
        history = [user(), Message(role=Role.MODEL, parts=[
            Part.from_thought("reasoning"), Part.from_text("")])]
        assert _defect_kinds(history) == ["empty_content"]
        repaired = repair_history(history)
        assert _defect_kinds(repaired) == []

    def test_whitespace_only_text_is_removed(self):
        """The abort-mid-stream variant: whitespace bypassed normalisation."""
        history = [user(), Message(role=Role.MODEL, parts=[
            Part.from_thought("r"), Part.from_text("   \n ")])]
        assert _defect_kinds(repair_history(history)) == []

    def test_a_thought_only_turn_survives(self):
        """Reasoning replay makes a thought-only turn a live shape.

        ``minimax`` / ``kimi`` / ``mimo`` set ``replay_reasoning = True``
        and require the assistant's ``reasoning_content`` back on the next
        request — MiMo answers 400 without it.  Removing the empty text
        block must not take the reasoning with it.
        """
        history = [user(), Message(role=Role.MODEL, parts=[
            Part.from_thought("only thinking"), Part.from_text("")])]
        repaired = repair_history(history)
        thoughts = [p.thought for m in repaired for p in m.parts if p.thought]
        assert thoughts == ["only thinking"]

    def test_a_message_with_nothing_left_is_dropped(self):
        history = [user(), Message(role=Role.MODEL,
                                   parts=[Part.from_text("  ")]), user("next")]
        repaired = repair_history(history)
        assert len(repaired) == 2
        assert all(m.role == Role.USER for m in repaired)


class TestReasoningReplayIsNotTradedAway:
    """A repair that fixes the pairing must not break reasoning replay.

    The obvious repair for an unanswered call — delete the assistant
    message — swaps a pairing 400 for a ``reasoning_content`` 400 on the
    three providers that replay thoughts.  So no MODEL message is ever
    removed: unanswered calls are *answered*.
    """

    def test_no_model_message_is_ever_removed(self):
        history = [user(), calls("a", "b", "c", thought="deep", text="hi"),
                   results("a")]
        repaired = repair_history(history)
        models = [m for m in repaired if m.role == Role.MODEL]
        assert len(models) == 1
        assert [p.thought for p in models[0].parts if p.thought] == ["deep"]
        assert [p.text for p in models[0].parts if p.text] == ["hi"]

    def test_call_ids_are_preserved_verbatim(self):
        """Kimi K3 wants the assistant message back as-is."""
        history = [user(), calls("a", "b", thought="t"), results("a")]
        assert _call_ids(repair_history(history)) == ["a", "b"]


class TestWireLevelMissingIds:
    """The fifth orphan source: an endpoint that streams no tool-call id."""

    def test_synthetic_ids_are_unique_within_a_response(self):
        nonce = new_tool_call_nonce()
        ids = {synthetic_tool_call_id(i, nonce) for i in range(8)}
        assert len(ids) == 8

    def test_synthetic_ids_do_not_collide_across_responses(self):
        """``index`` alone repeats every turn; the nonce is what separates them."""
        a = synthetic_tool_call_id(0, new_tool_call_nonce())
        b = synthetic_tool_call_id(0, new_tool_call_nonce())
        assert a != b

    def test_synthetic_ids_are_recognisable(self):
        assert synthetic_tool_call_id(3, "abcd1234").startswith(
            SYNTHETIC_CALL_ID_PREFIX)

    def test_boundary_backstop_mints_for_history_already_on_disk(self):
        """The seam fix cannot repair sessions written before it landed."""
        history = [user(),
                   Message(role=Role.MODEL, parts=[Part.from_function_call(
                       FunctionCall(id=None, name="t", args={}))]),
                   Message(role=Role.TOOL, parts=[Part.from_function_response(
                       ToolResult(call_id="", name="t", result=1))])]
        assert "missing_call_id" in _defect_kinds(history)
        repaired = repair_history(history)
        assert _defect_kinds(repaired) == []
        assert _call_ids(repaired) == _result_ids(repaired)
        assert _call_ids(repaired)[0].startswith(SYNTHETIC_CALL_ID_PREFIX)

    def test_minted_ids_are_stable_across_requests(self):
        """Minted from ``message_id``, which survives GC and the gate."""
        history = [user(), Message(role=Role.MODEL, parts=[
            Part.from_function_call(FunctionCall(id=None, name="t"))])]
        assert _call_ids(repair_history(history)) == _call_ids(
            repair_history(history))

    def test_minted_ids_are_namespaced_per_message(self):
        """Two id-less messages must not mint the same ids.

        The scope comes from ``message_id``; a ``Message`` carrying ``""``
        (hand-built, or an edited snapshot) falls back to a digest of its
        own shape rather than to a constant, which would give both
        messages ``..._0``.
        """
        history = [
            user(),
            Message(role=Role.MODEL, message_id="",
                    parts=[Part.from_function_call(
                        FunctionCall(id=None, name="alpha"))]),
            Message(role=Role.MODEL, message_id="",
                    parts=[Part.from_function_call(
                        FunctionCall(id=None, name="beta"))]),
        ]
        repaired = repair_history(history)
        assert len(set(_call_ids(repaired))) == 2
        assert validate_history(repaired) == []

    def test_two_id_less_calls_do_not_collapse_into_one(self):
        """``None == None`` made two distinct calls look like one pair."""
        history = [user(), Message(role=Role.MODEL, parts=[
            Part.from_function_call(FunctionCall(id=None, name="a")),
            Part.from_function_call(FunctionCall(id=None, name="b"))])]
        repaired = repair_history(history)
        assert len(set(_call_ids(repaired))) == 2
        assert _defect_kinds(repaired) == []


class TestNonDestructive:
    """Repair works on the per-request copy, following the #847 precedent."""

    def test_stored_history_is_not_mutated(self):
        stored = [user(), calls("a", "b"), results("a")]
        snapshot = [(m.role, len(m.parts)) for m in stored]
        repair_history(stored)
        assert [(m.role, len(m.parts)) for m in stored] == snapshot

    def test_stored_messages_are_not_rewritten_in_place(self):
        stored = [user(), calls("a", "b"), results("a")]
        model_msg = stored[1]
        repair_history(stored)
        assert model_msg.parts is stored[1].parts
        assert len(model_msg.parts) == 2

    def test_a_healthy_history_is_returned_unchanged(self):
        """The cheap path: the common case must allocate nothing."""
        healthy = [user(), calls("a"), results("a"), model_text()]
        assert repair_history(healthy) is healthy

    def test_unrepaired_messages_are_the_stored_objects(self):
        stored = [user(), calls("a", "b"), results("a")]
        repaired = repair_history(stored)
        assert repaired[0] is stored[0]
        assert repaired[1] is stored[1]

    def test_message_ids_survive_repair(self):
        """GC's history-budget sync keys on ``message_id``."""
        stored = [user(), calls("a", "b"), results("a")]
        ids = {m.message_id for m in stored}
        assert ids <= {m.message_id for m in repair_history(stored)}


class TestTracing:
    """A silent repair hides the subsystem that is producing bad histories."""

    def test_each_repair_kind_announces_itself(self):
        lines = []
        repair_history([results("gone"), user(), calls("a", "b"),
                        results("a")], trace_fn=lines.append)
        joined = "\n".join(lines)
        assert "HISTORY_INVARIANT" in joined
        assert "orphaned tool result" in joined
        assert "synthesised cancelled result" in joined

    def test_a_healthy_history_traces_nothing(self):
        lines = []
        repair_history([user(), calls("a"), results("a")],
                       trace_fn=lines.append)
        assert lines == []

    def test_every_repair_names_what_it_repaired_not_merely_that_it_ran(self):
        """A count is what a future producer defect moves; "it ran" is not."""
        lines = []
        repair_history([user(), calls("a", "b", "c"), results("a")],
                       trace_fn=lines.append)
        assert len(lines) == 1
        assert "2" in lines[0] and "'b'" in lines[0] and "'c'" in lines[0]

    def test_a_repair_is_logged_even_with_no_trace_sink(self, caplog):
        """The logger is the half a deployment aggregates.

        ``trace_fn`` is optional and a session that passes none must not
        repair in total silence — that is the silent-correction failure
        this reporting exists to prevent.
        """
        with caplog.at_level(logging.INFO, logger="shared.history_invariant"):
            repair_history([user(), calls("a", "b"), results("a")])
        assert any("HISTORY_INVARIANT" in r.message for r in caplog.records)

    def test_a_minted_id_warns_because_the_producer_should_have_minted(self):
        """WARNING is reserved for what the producer-side fix made a defect.

        Every streaming loop mints an id at the seam since #674, so a
        history still needing one here means either it predates that fix
        or a producer is not minting. A rising count is how that
        announces itself.
        """
        with caplog_at(logging.WARNING) as records:
            repair_history([user(), Message(role=Role.MODEL, parts=[
                Part.from_function_call(FunctionCall(id=None, name="t"))])])
        assert any("minted" in r.message for r in records)

    def test_a_cancelled_batch_does_not_warn(self):
        """Cancellation legitimately arrives here; it is not a defect.

        Warning on it would make the level meaningless — the backstop
        exists precisely because no producer sees this shape.
        """
        with caplog_at(logging.WARNING) as records:
            repair_history([user(), calls("a", "b"), results("a")])
        assert [r.message for r in records] == []


# ==================== property: generated histories ====================


def _random_history(rng, turns):
    """Build a well-formed history with multi-call turns and reasoning.

    Well-formed by construction, so any defect a later test finds was
    introduced by the transformation under test rather than by the
    generator.
    """
    history = [user("start")]
    counter = 0
    for _ in range(turns):
        kind = rng.random()
        if kind < 0.55:
            n = rng.choice([1, 1, 2, 3, 5, 8])
            ids = [f"c{counter + i}" for i in range(n)]
            counter += n
            history.append(calls(
                *ids,
                text=rng.choice([None, "working"]),
                thought=rng.choice([None, None, "reasoning"]),
            ))
            history.append(results(*ids))
        elif kind < 0.8:
            history.append(model_text(f"reply {counter}"))
            history.append(user(f"follow up {counter}"))
        else:
            history.append(Message(role=Role.MODEL,
                                   parts=[Part.from_thought("silent")]))
            history.append(user("go on"))
    return history


SEEDS = list(range(40))


@pytest.mark.parametrize("seed", SEEDS)
def test_generator_produces_valid_histories(seed):
    """The generator itself must not manufacture the defects under test."""
    rng = random.Random(seed)
    assert validate_history(_random_history(rng, 8)) == []


def _make_gc(name):
    """Instantiate one GC strategy with a summarizer that needs no model."""
    cls = {
        "truncate": TruncateGCPlugin,
        "summarize": SummarizeGCPlugin,
        "hybrid": HybridGCPlugin,
        "budget": BudgetGCPlugin,
    }[name]
    plugin = cls()
    plugin.initialize({
        "preserve_recent_turns": 2,
        # Deterministic stand-in for the model round-trip the summarizing
        # strategies would otherwise make.
        "summarizer": lambda text: "SUMMARY of prior conversation",
    })
    return plugin


GC_STRATEGIES = ["truncate", "summarize", "hybrid", "budget"]


def _usage(history):
    return {
        "model": "test-model",
        "context_limit": 1000,
        "total_tokens": 950,
        "prompt_tokens": 950,
        "output_tokens": 0,
        "turns": len(history),
        "percent_used": 95.0,
        "tokens_remaining": 50,
    }


@pytest.mark.parametrize("strategy", GC_STRATEGIES)
@pytest.mark.parametrize("seed", SEEDS)
def test_gc_output_satisfies_the_invariant_after_repair(strategy, seed):
    """Every GC strategy, over generated multi-call histories.

    This is the issue's point 4.  The assertion is on the **repaired**
    history, because repair-at-the-boundary is the contract being
    tested — a strategy is allowed to cut mid-pair as long as the
    boundary puts it right before the provider sees it.
    """
    rng = random.Random(seed)
    history = _random_history(rng, 10)
    plugin = _make_gc(strategy)
    collected, result = plugin.collect(
        history, _usage(history), GCConfig(preserve_recent_turns=2),
        GCTriggerReason.THRESHOLD)
    repaired = repair_history(collected)
    assert validate_history(repaired) == [], (
        f"strategy={strategy} seed={seed} result={result.plugin_name} "
        f"defects={validate_history(repaired)}")


@pytest.mark.parametrize("strategy", GC_STRATEGIES)
@pytest.mark.parametrize("seed", SEEDS)
def test_gc_never_invents_a_call_the_history_did_not_contain(strategy, seed):
    """Repair may add results; it may never add calls."""
    rng = random.Random(seed)
    history = _random_history(rng, 10)
    plugin = _make_gc(strategy)
    collected, _ = plugin.collect(
        history, _usage(history), GCConfig(preserve_recent_turns=2),
        GCTriggerReason.THRESHOLD)
    before = set(_call_ids(collected))
    assert set(_call_ids(repair_history(collected))) == before


@pytest.mark.parametrize("strategy", GC_STRATEGIES)
def test_gc_collection_is_not_vacuous(strategy):
    """A GC property test that collects nothing proves nothing.

    Guards the negative result in this module's docstring: the strategies
    pass the invariant because they cut on turn boundaries, not because
    ``collect`` is a no-op on these histories.
    """
    removed = 0
    shrank = 0
    for seed in SEEDS:
        rng = random.Random(seed)
        history = _random_history(rng, 10)
        collected, _ = _make_gc(strategy).collect(
            history, _usage(history), GCConfig(preserve_recent_turns=2),
            GCTriggerReason.THRESHOLD)
        removed += max(0, len(history) - len(collected))
        shrank += len(collected) != len(history)
    assert shrank >= len(SEEDS) - 2, f"{strategy} shrank only {shrank} histories"
    assert removed > 100, f"{strategy} removed only {removed} messages"


class TestComposesWithTheGcPathRepair:
    """Why a boundary backstop is needed even though a GC repair exists.

    ``shared.plugins.gc.utils.ensure_tool_call_integrity`` repairs stored
    history on the GC path, by **deletion**.  On a batch that was
    partially answered — the cancellation shape — deleting the assistant
    message leaves the result that *did* arrive with nothing to answer,
    converting an ``unmatched_call`` into an ``orphan_result``.  Measured
    across widths 2/3/5/8: every case with at least one real result, 14
    of 18.

    These tests assert the **composition** that production actually runs
    (GC repairs the store, the boundary repairs the per-request copy),
    not the intermediate state, so they stay green if the GC-path repair
    is improved later.
    """

    @pytest.mark.parametrize("n,k", [(2, 1), (3, 1), (3, 2), (5, 3), (8, 7)])
    def test_boundary_repair_fixes_what_the_gc_repair_leaves(self, n, k):
        ids = [f"c{i}" for i in range(n)]
        history = [user(), calls(*ids), results(*ids[:k])]
        after_gc = ensure_tool_call_integrity(history)
        assert validate_history(repair_history(after_gc)) == []

    def test_boundary_repair_alone_keeps_the_work_that_ran(self):
        """Composition is not the only route — the boundary alone is better.

        Reaching the boundary *without* the GC-path deletion keeps the
        assistant turn, its reasoning and the real result; the composed
        route is merely valid.
        """
        history = [user(), calls("a", "b", "c", thought="deep"), results("a")]
        repaired = repair_history(history)
        assert validate_history(repaired) == []
        assert [p.thought for m in repaired for p in m.parts if p.thought] \
            == ["deep"]
        real = [p.function_response.result for m in repaired for p in m.parts
                if p.function_response
                and p.function_response.call_id == "a"]
        assert real == [{"ok": "a"}]


@pytest.mark.parametrize("cut", list(range(1, 14)))
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5, 6, 7])
def test_cancel_at_turn_n_satisfies_the_invariant(seed, cut):
    """Cancel-at-turn-N: truncate the history at every possible point.

    The issue's second point.  Cutting at an arbitrary index reproduces
    every cancellation shape at once — mid-batch, between a call and its
    results, and immediately after an assistant turn — including cuts
    that leave the history *opening* on a tool result once combined with
    the suffix test below.
    """
    rng = random.Random(seed)
    history = _random_history(rng, 8)[:cut]
    assert validate_history(repair_history(history)) == []


@pytest.mark.parametrize("start", list(range(0, 12)))
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_resume_from_any_suffix_satisfies_the_invariant(seed, start):
    """Rewind / resume: keep only a suffix, which may open on a result."""
    rng = random.Random(seed)
    history = _random_history(rng, 8)[start:]
    assert validate_history(repair_history(history)) == []


@pytest.mark.parametrize("seed", SEEDS)
def test_partial_batch_execution_at_every_width(seed):
    """Answer k of n calls, for every k — the parallel-cancel window."""
    rng = random.Random(seed)
    n = rng.choice([2, 3, 5, 8])
    ids = [f"p{i}" for i in range(n)]
    for k in range(n + 1):
        history = [user(), calls(*ids, thought="mid-batch")]
        if k:
            history.append(results(*ids[:k]))
        repaired = repair_history(history)
        assert validate_history(repaired) == [], f"seed={seed} k={k}/{n}"
        assert sorted(_result_ids(repaired)) == sorted(ids)


# ==================== representation: the prose wire ====================


def _prose_pairs(history):
    """Count tool-call fences and rendered results in the prose rendering.

    The prose wire carries the pairing in *text*: a call is a
    ```` ```tool_call ```` fence and a result is a ``Tool result for ...
    (call <id>)`` line.  Nothing rejects a mismatch here the way a typed
    wire would, which is exactly why it needs its own assertion — a
    repair that is pair-safe on the Anthropic wire is not automatically
    pair-safe on this one.
    """
    rendered = messages_to_prose_chat(list(history))
    text = "\n".join(m["content"] for m in rendered)
    return text.count("```tool_call"), text.count("Tool result for ")


@pytest.mark.parametrize("seed", SEEDS)
def test_prose_wire_pairing_survives_repair(seed):
    """Every fence has a rendered result after repair, on the prose wire."""
    rng = random.Random(seed)
    history = _random_history(rng, 8)
    truncated = history[:rng.randint(1, max(1, len(history) - 1))]
    fences, rendered = _prose_pairs(repair_history(truncated))
    assert fences == rendered, f"seed={seed} fences={fences} results={rendered}"


def test_prose_wire_is_broken_without_repair():
    """The prose wire really can carry an orphan — the assertion has teeth."""
    fences, rendered = _prose_pairs([user(), calls("a", "b"), results("a")])
    assert fences == 2 and rendered == 1


def test_prose_wire_renders_a_synthetic_result_for_each_missing_call():
    repaired = repair_history([user(), calls("a", "b", "c"), results("a")])
    fences, rendered = _prose_pairs(repaired)
    assert fences == rendered == 3


def test_prose_wire_rendering_is_byte_stable_across_requests():
    """Byte-stability is claimed for the prose wire too, so it is tested.

    The typed-wire claim (a constant ``CANCELLED_RESULT``, so the request
    prefix does not move and the upstream prompt cache still hits) does
    **not** carry over for free: the prose wire re-serialises the whole
    result into text, including the ``(call <id>)`` label, so a minted id
    or a payload that varied would move the bytes here even if the typed
    blocks were stable.  Both halves are asserted: the repair renders
    identically on two separate calls, and the rendering contains the
    constant payload rather than anything per-request.
    """
    history = [user(), calls("a", "b", thought="deep"), results("a")]
    first = messages_to_prose_chat(list(repair_history(history)))
    second = messages_to_prose_chat(list(repair_history(history)))
    assert first == second
    text = "\n".join(m["content"] for m in first)
    assert CANCELLED_RESULT["error"] in text


def test_prose_wire_byte_stability_covers_a_minted_id():
    """A minted id must be stable too, or the prose bytes move each turn."""
    history = [user(), Message(role=Role.MODEL, parts=[
        Part.from_function_call(FunctionCall(id=None, name="t", args={}))])]
    first = messages_to_prose_chat(list(repair_history(history)))
    second = messages_to_prose_chat(list(repair_history(history)))
    assert first == second


def test_prose_wire_never_renders_an_unmatched_call_id():
    """Every ``(call <id>)`` the prose names is a call the history made."""
    repaired = repair_history([user(), calls("a", "b"), results("a")])
    text = "\n".join(m["content"]
                     for m in messages_to_prose_chat(list(repaired)))
    for cid in _call_ids(repaired):
        assert f"(call {cid})" in text


class TestTheProducerRepairsStoredHistory:
    """The GC path repairs the record, not just the per-request view.

    Distinct from :class:`TestComposesWithTheGcPathRepair`, which asserts
    the composition production runs.  This asserts the producer **alone**
    leaves a valid history, because its output is what gets *persisted*: a
    session revived from an unrepaired record comes back orphaned, and no
    later boundary repair changes what is on disk.
    """

    def test_the_gc_path_answers_a_partially_answered_batch(self):
        """The #674 shape, through the stored-history repair.

        ``{A,B}`` called, ``A`` answered, then a USER turn.  The old
        deletion policy removed the MODEL message here and left ``A``'s
        result answering a call present nowhere.
        """
        history = [user(), calls("a", "b"), results("a")]
        assert validate_history(ensure_tool_call_integrity(history)) == []

    def test_the_gc_path_keeps_the_assistant_turn(self):
        history = [user(), calls("a", "b", thought="deep"), results("a")]
        repaired = ensure_tool_call_integrity(history)
        assert [p.thought for m in repaired for p in m.parts if p.thought] \
            == ["deep"]


def _idless_tool_call_stream():
    """A streamed response whose tool call carries no id — the bad wire."""
    tc = MagicMock()
    tc.index = 0
    tc.id = None
    tc.type = "function"
    tc.function = MagicMock()
    tc.function.name = "tool_zero"
    tc.function.arguments = '{"x": 1}'

    def _chunk(tool_calls, finish_reason):
        chunk = MagicMock()
        chunk.usage = None
        choice = MagicMock()
        choice.finish_reason = finish_reason
        delta = MagicMock()
        delta.content = None
        delta.tool_calls = tool_calls
        delta.reasoning_content = None
        delta.audio = None
        choice.delta = delta
        chunk.choices = [choice]
        return chunk

    stream = MagicMock()
    chunks = [_chunk([tc], None), _chunk(None, "tool_calls")]
    stream.__iter__ = lambda self: iter(chunks)
    stream.close = MagicMock()
    return stream


class TestTheWireSeamMintsAnId:
    """The seam, asserted from THIS module so the meta-guard can reach it.

    The provider-level suite lives under
    ``shared/plugins/model_provider/_openai_compat/tests/``, which the
    meta-guard does not scan, so a reversion could not name a test there.
    That suite covers the seam far more thoroughly (three providers, five
    cases each); this is the one case CI can put the defect back into.
    """

    def test_a_streamed_call_without_an_id_still_gets_one(self):
        provider = NIMProvider()
        provider._client = MagicMock()
        provider._model_name = "meta/llama-3.3-70b-instruct"
        provider._enable_thinking = False
        provider._trace = lambda _m: None
        provider._client.chat.completions.create = (
            lambda **kw: _idless_tool_call_stream())
        result = provider.complete([], on_chunk=lambda _c: None)
        response = getattr(result, "response", result)
        ids = [p.function_call.id for p in response.parts if p.function_call]
        assert ids and all(ids)
        assert ids[0].startswith(SYNTHETIC_CALL_ID_PREFIX)


class TestManualGcIsTurnBoundaryOnly:
    """The second of the two guards that share one reason.

    ``manual_gc`` has one non-test caller — the public
    :meth:`JaatoClient.manual_gc` — and no daemon path reaches it, so the
    only way to run it mid-turn is an embedder calling it from a thread
    other than the one inside ``send_message``.  Narrow, but real, and the
    repair cannot tell a call pending a retry from one genuinely
    unanswered: collecting there would synthesise results for calls about
    to be answered for real.

    Constructed with ``__new__`` deliberately — the guard is reached before
    anything else in the method touches session state, and that ordering is
    itself asserted below.
    """

    def _session(self, *, running, plugin):
        session = JaatoSession.__new__(JaatoSession)
        session._gc_plugin = plugin
        session._is_running = running
        return session

    def test_a_running_session_refuses(self):
        session = self._session(running=True, plugin=object())
        with pytest.raises(RuntimeError, match="turn-boundary"):
            JaatoSession.manual_gc(session)

    def test_the_missing_plugin_error_still_comes_first(self):
        """Ordering is unchanged: a misconfiguration outranks a busy turn."""
        session = self._session(running=True, plugin=None)
        with pytest.raises(RuntimeError, match="No GC plugin"):
            JaatoSession.manual_gc(session)

    def test_an_idle_session_is_not_refused_by_this_guard(self):
        """Idle must get PAST the guard — answering in the store is the point."""
        session = self._session(running=False, plugin=object())
        with pytest.raises(Exception) as caught:
            JaatoSession.manual_gc(session)
        assert "turn-boundary" not in str(caught.value)
