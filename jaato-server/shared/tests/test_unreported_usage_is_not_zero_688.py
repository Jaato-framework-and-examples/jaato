"""A provider that reports no usage no longer reads as a free turn (#688).

``TokenUsage`` starts all-zero and is only overwritten when a usage block
arrives, so until this change **two completely different facts shared one
value**:

* the provider measured the call and it genuinely cost nothing;
* the provider — or a proxy in front of it — sent no ``usage`` at all, so
  nothing was ever measured.

``budget_control`` enforces its ``usd`` / ``tokens`` ceilings from exactly
that data.  Conflating the two therefore made an unmetered upstream
**silently disable spend enforcement**: the tracker was fed zero, no
dimension advanced, no rung fired, and a run that looked capped was
uncapped.  Failing open on a spend control is the wrong direction, and it
failed open quietly.

The exposure is wide by construction — ``nim``, ``nebius``, ``ovhcloud``,
``doubleword``, ``lmstudio``, ``tensorrt_llm``, ``triton``, ``vllm``,
``zhipuai_openai``, plus ``openrouter`` fronting 300+ upstreams and any
corporate gateway in front of those.  These are precisely the
"approximately OpenAI-compatible" endpoints where usage reporting is least
reliable, and the issue cites two separate upstream fixes for proxies that
drop ``usage`` from a delta event.

WHAT THESE TESTS DRIVE.  The wire-level cases go through the **real**
streaming loop of three providers via a mock SDK stream — the same
technique as ``test_streamed_tool_call_without_id_674.py`` — so they fail
if the flag stops being set at the seam.  ``openai`` was not installed in
the container when #688 was first sized, which is why the seam could not
be exercised then; it can now.

THE THREE-WAY DISTINCTION is what the first class asserts, and it is the
whole point: a stream carrying **no** usage, a stream carrying a
**genuine zero**, and a stream carrying real numbers must produce three
distinguishable results.  Before this change the first two were identical.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from shared.budget_control import (
    DEFAULT_UNMETERED_POLICY,
    UNMETERED_POLICIES,
    BudgetControlConfig,
    BudgetControlConfigError,
    BudgetTracker,
)
from shared.jaato_session import JaatoSession
from shared.plugins.model_provider.nim.provider import NIMProvider
from shared.plugins.model_provider.openrouter.provider import OpenRouterProvider
from jaato_sdk.plugins.model_provider.types import TokenUsage
from shared.plugins.model_provider.vllm.provider import VLLMProvider
from shared.tests.test_every_guard_detects_its_own_reversion import Reversion

_SESSION = "jaato-server/shared/jaato_session.py"
_BUDGET = "jaato-server/shared/budget_control.py"
_COMPAT = ("jaato-server/shared/plugins/model_provider/"
           "_openai_compat/base.py")
_TYPES = "jaato-sdk/jaato_sdk/plugins/model_provider/types.py"

REVERSIONS = [
    # The representation itself.  Without the field there is no distinction
    # to make, and every consumer reads an unmeasured turn as a measured zero.
    Reversion(
        target=_TYPES,
        find="    reported: bool = True",
        replace="    reported_REMOVED: bool = True",
        because="TokenUsage carries no reported flag, so 'measured zero' and "
                "'never measured' are the same value again",
        test="TestTheDistinctionExists::"
             "test_the_field_exists_and_defaults_true",
    ),
    # The seam.  An accumulator claiming to be a measurement IS the defect.
    Reversion(
        target=_COMPAT,
        find="        usage = TokenUsage(reported=False)\n"
             "        was_cancelled = False",
        replace="        usage = TokenUsage()\n"
                "        was_cancelled = False",
        because="the OpenAI-compatible streaming accumulator claims the turn "
                "was measured even when no chunk ever carried usage",
        test="TestTheWireSaysWhetherItMeasured::"
             "test_a_stream_that_never_sends_usage_is_unreported"
             "[openai_compat]",
    ),
    # The enforcement.  With the branch gone an unmetered turn is charged
    # zero again and the ceiling never fires -- the reported P1.
    Reversion(
        target=_SESSION,
        find='            if getattr(usage, "reported", True):',
        replace="            if True:",
        because="budget_control is fed an all-zero measurement for an "
                "unmetered provider, so no dimension advances and no rung "
                "fires",
        test="TestTheCeilingNoLongerFailsOpen::"
             "test_an_abort_rung_fires_on_an_unmetered_provider",
    ),
    # The asymmetry.  Charging usd from an estimate is precisely what the
    # house rule forbids, and nothing else in the suite would notice.
    Reversion(
        target=_SESSION,
        find="        return self._budget_tracker.observe(\n"
             "            tokens=self._estimate_response_tokens(response),\n"
             "        )",
        replace="        return self._budget_tracker.observe(\n"
                "            tokens=self._estimate_response_tokens(response),\n"
                "            usd=0.01,\n"
                "        )",
        because="a dollar figure invented from guessed tokens reaches the usd "
                "ceiling, which the house rule forbids",
        test="TestThePolicy::test_estimate_never_charges_usd",
    ),
    # The opt-in ceiling.  A 'halt' that does not halt is the fail-closed
    # posture silently not applying -- the #735 shape.
    Reversion(
        target=_SESSION,
        find='        if policy == "halt":',
        replace='        if policy == "halt_DISABLED":',
        because="on_unmetered: halt is configured, announced, and stops "
                "nothing",
        test="TestThePolicy::test_halt_stops_the_session",
    ),
    # The vocabulary guard.  ONE check, in ``__post_init__``, reached by
    # both doors -- ``from_dict`` constructs the dataclass, and direct
    # construction goes straight there.  An earlier draft also validated
    # inside the ``from_dict`` helper; the meta-guard then reported BOTH
    # reversions as decorative, correctly, because each copy caught what
    # the other would have let through.  Duplicated validation is not
    # defence in depth when it makes every copy individually unreachable.
    Reversion(
        target=_BUDGET,
        find="        if self.on_unmetered not in UNMETERED_POLICIES:",
        replace="        if False:",
        because="a misspelled on_unmetered policy is accepted and then "
                "silently means something else",
        test="TestTheKnob::test_an_unknown_policy_is_refused",
    ),
]


# --------------------------------------------------------------- wire

def _chunk(*, usage=None, content=None, finish_reason=None):
    """One streamed chunk.  ``usage=None`` is a chunk carrying no usage."""
    chunk = MagicMock()
    chunk.usage = usage
    choice = MagicMock()
    choice.finish_reason = finish_reason
    delta = MagicMock()
    delta.content = content
    delta.tool_calls = None
    delta.reasoning_content = None
    delta.audio = None
    choice.delta = delta
    chunk.choices = [choice]
    return chunk


def _usage_block(prompt, completion, total):
    """An SDK ``usage`` object, as the OpenAI client would present it."""
    u = MagicMock()
    u.prompt_tokens = prompt
    u.completion_tokens = completion
    u.total_tokens = total
    u.prompt_tokens_details = None
    u.completion_tokens_details = None
    return u


def _stream(chunks):
    stream = MagicMock()
    stream.__iter__ = lambda self: iter(chunks)
    stream.close = MagicMock()
    return stream


def _build(cls, model):
    provider = cls()
    provider._client = MagicMock()
    provider._model_name = model
    provider._enable_thinking = False
    provider._trace = lambda _msg: None
    return provider


def _complete(provider, chunks):
    """Drive one turn through the provider's **streaming** loop.

    ``on_chunk`` is what selects it — without a chunk callback ``complete``
    takes the batched path, which is a different accumulator.
    """
    provider._client.chat.completions.create = (
        lambda **kwargs: _stream(chunks))
    return provider.complete([], on_chunk=lambda _chunk: None)


def _usage_of(turn_result):
    response = getattr(turn_result, "response", turn_result)
    return response.usage


PROVIDERS = [
    pytest.param(NIMProvider, "meta/llama-3.3-70b-instruct", id="openai_compat"),
    pytest.param(OpenRouterProvider, "meta-llama/llama-3.3-70b-instruct",
                 id="openrouter"),
    pytest.param(VLLMProvider, "Qwen/Qwen2.5-7B-Instruct", id="vllm"),
]


class TestTheDistinctionExists:
    """The representation, before anything reads it."""

    def test_the_field_exists_and_defaults_true(self):
        """Default ``True`` is deliberate: an unmigrated or out-of-tree seam
        behaves exactly as it did before the field existed, rather than
        being marked unknown and having a policy applied its author never
        saw.  Every in-tree placeholder is migrated in the same change, so
        the default protects strangers, not this repository."""
        assert TokenUsage().reported is True
        assert TokenUsage(reported=False).reported is False

    def test_a_measured_zero_and_an_unmeasured_turn_differ(self):
        """The whole issue in one assertion: these were the same value."""
        measured_zero = TokenUsage(prompt_tokens=0, output_tokens=0,
                                   total_tokens=0)
        never_measured = TokenUsage(reported=False)
        assert measured_zero.total_tokens == never_measured.total_tokens == 0
        assert measured_zero != never_measured
        assert measured_zero.reported and not never_measured.reported


@pytest.mark.parametrize("cls,model", PROVIDERS)
class TestTheWireSaysWhetherItMeasured:
    """Driven through each provider's real streaming loop."""

    def test_a_stream_that_never_sends_usage_is_unreported(self, cls, model):
        """The #688 wire: chunks arrive, the turn completes, nothing is
        measured.  All-zero AND honest about it."""
        chunks = [_chunk(content="hi"), _chunk(finish_reason="stop")]
        usage = _usage_of(_complete(_build(cls, model), chunks))
        assert usage.reported is False
        assert usage.total_tokens == 0

    def test_a_stream_that_sends_usage_is_reported(self, cls, model):
        chunks = [
            _chunk(content="hi"),
            _chunk(finish_reason="stop"),
            _chunk(usage=_usage_block(10, 5, 15)),
        ]
        usage = _usage_of(_complete(_build(cls, model), chunks))
        assert usage.reported is True
        assert usage.total_tokens == 15

    def test_a_genuine_zero_is_reported(self, cls, model):
        """A provider CAN measure a turn at zero — a cached turn, a refusal
        short-circuited upstream.  That is a measurement and must not be
        mistaken for silence, which is the direction that would make this
        fix its own defect."""
        chunks = [
            _chunk(content=""),
            _chunk(finish_reason="stop"),
            _chunk(usage=_usage_block(0, 0, 0)),
        ]
        usage = _usage_of(_complete(_build(cls, model), chunks))
        assert usage.reported is True
        assert usage.total_tokens == 0


# ------------------------------------------------------------ policy

def _budget_session(policy=None, limits=None, degrade=None):
    """A session double carrying exactly what the budget path touches.

    Mirrors ``test_budget_runtime._session``'s approach — a SimpleNamespace
    with the real methods bound — rather than standing up a runtime.
    """
    cfg_data = {"limits": limits or {"tokens": 100}}
    if degrade:
        cfg_data["degrade"] = degrade
    if policy:
        cfg_data["on_unmetered"] = policy
    cfg = BudgetControlConfig.from_dict(cfg_data)
    s = SimpleNamespace(
        _budget_tracker=BudgetTracker(cfg),
        _budget_control=cfg,
        _budget_unmetered_warned=False,
        _budget_terminal_action=None,
        _budget_exhausted_reason=None,
        _budget_notice_sink=None,
        _budget_applied_rung_pct=0.0,
        _provider_name_override="nim",
        _model_name="some-model",
        _pricing_table=None,
        _history=SimpleNamespace(messages=[]),
        _tier_config=None,
        _active_tier=None,
        _trace=lambda *a, **k: None,
        _get_trace_prefix=lambda: "session:main",
        _current_output_callback=None,
        _ui_hooks=None,
        _stopped=None,
    )
    for name in ("_budget_observe_response", "_budget_observe_unmetered",
                 "_estimate_response_tokens", "_apply_budget_rungs",
                 "_surface_budget_event", "_budget_trace",
                 "_budget_trace_rung", "_budget_note_ceilings",
                 "_resolve_span_cost"):
        setattr(s, name, (lambda n: (lambda *a, **k:
                getattr(JaatoSession, n)(s, *a, **k)))(name))
    s.request_stop = lambda reason="": s.__setattr__("_stopped", reason) or True
    return s


def _response(*, reported, total_tokens=0, text="some answer text"):
    part = SimpleNamespace(text=text, function_call=None)
    return SimpleNamespace(
        parts=[part],
        text=text,
        usage=TokenUsage(total_tokens=total_tokens, reported=reported),
    )


class TestThePolicy:
    """``budget_control.on_unmetered`` — what an unmeasured turn costs."""

    def test_estimate_charges_tokens(self):
        s = _budget_session(policy="estimate")
        s._budget_observe_response(_response(reported=False))
        assert s._budget_tracker._usage.tokens > 0

    def test_estimate_never_charges_usd(self):
        """The house rule, stated in ``_budget_observe_response``'s own
        docstring: *a budget must never hard-stop on a number it invented*.
        A token estimate is a quantity GC already computes; a dollar figure
        derived from guessed tokens is a price nobody quoted."""
        s = _budget_session(policy="estimate")
        s._budget_observe_response(_response(reported=False))
        assert s._budget_tracker._usage.usd == 0

    def test_halt_stops_the_session(self):
        s = _budget_session(policy="halt")
        s._budget_observe_response(_response(reported=False))
        assert s._stopped is not None
        assert s._budget_terminal_action == "abort"

    def test_ignore_is_the_pre_688_behaviour(self):
        s = _budget_session(policy="ignore")
        s._budget_observe_response(_response(reported=False))
        assert s._budget_tracker._usage.tokens == 0
        assert s._stopped is None

    def test_estimate_is_the_default(self):
        """An existing profile gains enforcement without being edited, and
        without being stopped — which is why ``halt`` is not the default."""
        s = _budget_session()
        assert s._budget_control.on_unmetered == DEFAULT_UNMETERED_POLICY
        s._budget_observe_response(_response(reported=False))
        assert s._budget_tracker._usage.tokens > 0
        assert s._stopped is None

    def test_a_reported_turn_is_untouched_by_any_of_it(self):
        """The policy must not reach a provider that DOES report — including
        one reporting a real zero."""
        for policy in ("estimate", "halt", "ignore"):
            s = _budget_session(policy=policy)
            s._budget_observe_response(
                _response(reported=True, total_tokens=0))
            assert s._stopped is None, policy
            assert s._budget_tracker._usage.tokens == 0, policy
            assert s._budget_unmetered_warned is False, policy

    def test_the_warning_is_once_per_session(self, caplog):
        """An unmetered provider is unmetered on EVERY turn; a per-response
        line would bury the run's real output."""
        s = _budget_session(policy="estimate")
        for _ in range(5):
            s._budget_observe_response(_response(reported=False))
        lines = [r for r in caplog.records
                 if "reported no usage" in r.getMessage()]
        assert len(lines) == 1


class TestTheCeilingNoLongerFailsOpen:
    """The integrity claim the issue is actually about."""

    def test_an_abort_rung_fires_on_an_unmetered_provider(self):
        """Before #688 this ladder was fed zero on every turn, so it never
        fired however long the run went — the run looked capped and was
        uncapped."""
        s = _budget_session(
            policy="estimate",
            limits={"tokens": 50},
            degrade=[{"at": 100, "action": "abort"}],
        )
        for _ in range(20):
            s._budget_observe_response(
                _response(reported=False, text="x" * 400))
            if s._stopped is not None:
                break
        assert s._stopped is not None, (
            "an unmetered provider still sails past its ceiling")

    def test_and_a_ladder_on_usd_alone_still_does_not(self):
        """Stated rather than hidden: ``usd`` is never fed from an estimate,
        so a profile whose ONLY ceiling is a dollar figure is still
        unenforceable against an unmetered provider.  ``halt`` is the answer
        there, and that is why the knob exists."""
        s = _budget_session(
            policy="estimate",
            limits={"usd": 1.0},
            degrade=[{"at": 100, "action": "abort"}],
        )
        for _ in range(20):
            s._budget_observe_response(
                _response(reported=False, text="x" * 400))
        assert s._stopped is None
        assert s._budget_tracker._usage.usd == 0


class TestTheKnob:
    """Config surface for ``on_unmetered``."""

    def test_default_when_absent(self):
        cfg = BudgetControlConfig.from_dict({"limits": {"tokens": 10}})
        assert cfg.on_unmetered == DEFAULT_UNMETERED_POLICY

    def test_every_policy_parses(self):
        for policy in UNMETERED_POLICIES:
            cfg = BudgetControlConfig.from_dict(
                {"limits": {"tokens": 10}, "on_unmetered": policy})
            assert cfg.on_unmetered == policy

    def test_an_unknown_policy_is_refused(self):
        """Fail loud rather than degrade to a default nobody asked for — a
        silently-ignored policy is the family #910/#925/#947/#950 exists to
        catch."""
        with pytest.raises(BudgetControlConfigError) as exc:
            BudgetControlConfig.from_dict(
                {"limits": {"tokens": 10}, "on_unmetered": "halted"})
        assert "on_unmetered" in str(exc.value)

    def test_direct_construction_is_validated_too(self):
        """``from_dict`` is not the only door: jaato-premium and tests build
        the dataclass directly, so ``__post_init__`` carries its own check.
        Keeping both is why the two have separate reversions — an earlier
        draft guarded only one and was decorative for it."""
        with pytest.raises(BudgetControlConfigError):
            BudgetControlConfig(limits={"tokens": 10}, on_unmetered="halted")

    def test_a_non_string_policy_is_refused(self):
        with pytest.raises(BudgetControlConfigError):
            BudgetControlConfig.from_dict(
                {"limits": {"tokens": 10}, "on_unmetered": True})

    def test_it_round_trips_over_the_envelope(self):
        """The daemon parses eagerly and the runner re-parses ``to_dict``;
        a knob that did not survive that would apply on one side only."""
        cfg = BudgetControlConfig.from_dict(
            {"limits": {"tokens": 10}, "on_unmetered": "halt"})
        again = BudgetControlConfig.from_dict(cfg.to_dict())
        assert again.on_unmetered == "halt"

    def test_the_default_is_not_serialised(self):
        """Keeps the wire byte-identical for every profile that says
        nothing, so a runner predating the knob sees no new key."""
        cfg = BudgetControlConfig.from_dict({"limits": {"tokens": 10}})
        assert "on_unmetered" not in cfg.to_dict()

    def test_on_unmetered_alone_still_parses_to_none(self):
        """It enforces nothing by itself — it only says what to do about
        ceilings, and there are none.  "Absent" and "says nothing
        enforceable" stay identical, as the class docstring promises."""
        assert BudgetControlConfig.from_dict({"on_unmetered": "halt"}) is None
