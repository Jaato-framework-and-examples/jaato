"""A turn's cache traffic must be reported in full, and attributed to a tier.

THE GAP (closed here).  ``_accumulate_turn_tokens`` summed prompt/output
into ``spend_*`` but REPLACED ``cache_read`` / ``cache_creation``.  A turn
with a tool call has >= 2 billed responses, so the reported cache figures
were the last response's alone.

That is not a rounding error for a ``model_tiers`` session: ``enter_tier``
lands MID-TURN, and the switch re-reads the entire accumulated prefix cold
at the new model.  The expensive leg is the one before the last, so the
reported number hid exactly the miss the switch caused -- the measurement
you would reach for to decide whether tier mode pays for itself was
blind to the cost tier mode adds.

The level reading is kept as well as the spend reading, because they
answer different questions and the streaming usage-callback (which fires
per usage CHUNK, and so must not sum) writes the level one.

Also pins the span attribution: ``jaato.tier`` / ``jaato.tier.switches``,
without which a cache miss on a span cannot be told from an ordinary one.
Deriving the tier from ``llm.model_name`` does not work -- two tiers may
share a model, and a budget-control rung rebinds a tier's model underneath
it.

See ``docs/design/model-tier-prompt-cache.md`` §5.4.
"""

from __future__ import annotations

import ast
import pathlib

import pytest
from unittest.mock import MagicMock

from shared.jaato_session import JaatoSession
from shared.model_tiers import ModelTierConfig, TierEntry
from shared.tests.test_every_guard_detects_its_own_reversion import Reversion

#: The defect, put back: cache tokens replaced rather than accumulated.
REVERSIONS = [
    Reversion(
        target="jaato-server/shared/jaato_session.py",
        find="""            turn_tokens['spend_cache_read'] = (
                turn_tokens.get('spend_cache_read', 0)
                + response.usage.cache_read_tokens)""",
        replace="""            turn_tokens['spend_cache_read'] = (
                response.usage.cache_read_tokens)""",
        test="test_cache_reads_accumulate_across_a_turns_responses",
        because="a turn reporting only its final response's cache traffic",
    ),
    # The other half of the same measurement.  Accumulating a number the
    # provider never parses measures nothing, so the parse belongs under
    # the same guard as the accumulation.  Lives HERE rather than beside
    # the provider because #665's meta-guard scans `shared/tests` and
    # `server/tests` only -- a reversion declared in
    # `openrouter/tests/` would never be applied, and an unapplied
    # reversion reads as a passing one.  The detailed shape tests stay
    # with the provider; this pins the one fact cache accounting needs.
    Reversion(
        target=("jaato-server/shared/plugins/model_provider/openrouter"
                "/converters.py"),
        find='    creation = _read_details(details, "cache_write_tokens")',
        replace='    creation = None',
        test="test_the_provider_reports_the_write_count_at_all",
        because="cache writes billed but never reported (#699)",
    ),
]

#: Anchored to THIS FILE, not the process CWD.  Sibling guards use
#: repo-root-relative literals and therefore only pass when pytest is
#: invoked from the repo root; a guard that depends on how it was
#: launched reports its own environment, not the code.
ROOT = pathlib.Path(__file__).resolve().parents[3]
SESSION_SRC = (ROOT / "jaato-server" / "shared"
               / "jaato_session.py").read_text(encoding="utf-8")


def _usage(*, prompt=0, output=0, total=0, cache_read=None, cache_creation=None):
    u = MagicMock()
    u.prompt_tokens = prompt
    u.output_tokens = output
    u.total_tokens = total
    u.cache_read_tokens = cache_read
    u.cache_creation_tokens = cache_creation
    u.cost_usd = None
    u.thinking_tokens = 0
    return u


def _response(**kw):
    r = MagicMock()
    r.usage = _usage(**kw)
    return r


def _bare_session():
    s = JaatoSession.__new__(JaatoSession)
    s._update_thinking_budget = lambda *a, **k: None
    return s


class TestCacheSpendAccumulates:

    def test_cache_reads_accumulate_across_a_turns_responses(self):
        """The crux.  Two responses, and both are billed."""
        s = _bare_session()
        turn = {}

        # Leg 1: warm, on the tier the turn started in.
        s._accumulate_turn_tokens(
            _response(prompt=100, output=10, total=110, cache_read=58_000), turn)
        # Leg 2: after enter_tier -- the whole prefix re-read cold, so this
        # leg reads nothing from cache and WRITES the prefix at the new model.
        s._accumulate_turn_tokens(
            _response(prompt=60_000, output=20, total=60_020,
                      cache_read=0, cache_creation=58_000), turn)

        assert turn['spend_cache_read'] == 58_000, (
            'the first leg\'s cache read was dropped; a mid-turn tier switch '
            'is exactly when that leg is the expensive one'
        )
        assert turn['spend_cache_creation'] == 58_000

    def test_the_level_reading_is_still_the_last_leg(self):
        """Both shapes survive: the level one is what the streaming
        usage-callback writes, and replacing is correct for it."""
        s = _bare_session()
        turn = {}

        s._accumulate_turn_tokens(
            _response(prompt=1, output=1, total=2, cache_read=58_000), turn)
        s._accumulate_turn_tokens(
            _response(prompt=1, output=1, total=2, cache_read=0), turn)

        assert turn['cache_read'] == 0          # last leg
        assert turn['spend_cache_read'] == 58_000   # both legs

    def test_a_provider_reporting_nothing_stays_absent(self):
        """``None`` and ``0`` are different answers, as they are for cost:
        'this provider does not cache' is not 'it cached nothing'."""
        s = _bare_session()
        turn = {}

        s._accumulate_turn_tokens(_response(prompt=1, output=1, total=2), turn)

        assert 'spend_cache_read' not in turn
        assert 'spend_cache_creation' not in turn

    #: Both halves of the streaming path: the closure the provider is
    #: handed, and the method it delegates to.  A shape check that knew
    #: only about the closure went quiet the moment the body moved.
    STREAMING_FUNCS = (
        'usage_callback_with_turn_tracking', '_track_streaming_usage')

    @classmethod
    def _streaming_callbacks(cls):
        """The AST nodes of every function on the streaming path."""
        tree = ast.parse(SESSION_SRC)
        found = [
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name in cls.STREAMING_FUNCS
        ]
        names = {n.name for n in found}
        assert names == set(cls.STREAMING_FUNCS), (
            f'expected {sorted(cls.STREAMING_FUNCS)} on the streaming path, '
            f'found {sorted(names)}; something was renamed and this guard '
            f'no longer checks what it claims to'
        )
        return found

    def test_the_streaming_callback_never_mentions_a_spend_key(self):
        """It fires per usage CHUNK, so any write there double-counts.

        Deliberately BLUNT: no ``spend_``-prefixed string literal may
        appear anywhere inside the callback.  Every way of writing the key
        — ``=``, ``+=``, ``.update({...})``, ``.setdefault(...)`` — names
        it as a string, so one assertion covers the operation rather than
        one spelling of it.

        The narrow version of this check is what shipped first, and it did
        not discriminate: it collected ``ast.Assign`` targets only, so
        ``turn_data['spend_cache_creation'] += 1`` — an ``ast.AugAssign``,
        and the form a person would most naturally write a double-count in
        — passed it, along with the entire rest of the suite.  Found by a
        reviewer aiming a sabotage at it, not by reading it.
        """
        mentioned = {
            n.value
            for cb in self._streaming_callbacks()
            for n in ast.walk(cb)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and n.value.startswith('spend_')
        }
        assert not mentioned, (
            f'the streaming callback references spend keys {sorted(mentioned)}; '
            f'it fires per usage chunk, so anything it writes there is '
            f'counted once per chunk instead of once per response'
        )

    def test_driving_the_streaming_path_writes_no_spend_key(self):
        """The assertion no indirection can dodge.

        The shape check above reads function bodies, so moving the write
        into a helper the callback calls silences it — verified: an
        ordinary extract-a-helper refactor carrying a genuine per-chunk
        double-count passed the whole file.  This drives the real code
        with two chunks and looks at the result, so a write anywhere below
        it — helper, method, anything transitive — still lands in
        ``turn_data`` and is caught.
        """
        s = _bare_session()
        turn = {}

        s._track_streaming_usage(turn, _usage(
            prompt=100, output=10, total=110,
            cache_read=58_000, cache_creation=1_024))
        s._track_streaming_usage(turn, _usage(
            prompt=120, output=20, total=140,
            cache_read=58_000, cache_creation=1_024))

        spend = {k: v for k, v in turn.items() if k.startswith('spend_')}
        assert not spend, (
            f'the streaming path wrote {spend}; it fires once per usage '
            f'CHUNK and a provider may emit several per response, so spend '
            f'accumulated here counts one response many times'
        )

    def test_the_streaming_path_replaces_the_level_readings(self):
        """The other half of the same contract: it MUST replace, because
        each chunk's prompt_tokens already covers the whole context."""
        s = _bare_session()
        turn = {}

        s._track_streaming_usage(turn, _usage(
            prompt=100, output=10, total=110, cache_read=58_000))
        s._track_streaming_usage(turn, _usage(
            prompt=120, output=20, total=140, cache_read=0))

        assert turn['total'] == 140        # last chunk, not 250
        assert turn['cache_read'] == 0     # last chunk, not 58_000

    def test_the_closure_is_exactly_a_pass_through(self):
        """The closure is pinned by WHITELIST, not by forbidding writes.

        It is the one body neither other check reaches: the effect test
        drives ``_track_streaming_usage``, and a blacklist ("no assignment
        here") is dodged by a plain call —

            def _record_spend(tt, u): tt['spend_cache_read'] = ...
            def usage_callback_with_turn_tracking(usage):
                self._track_streaming_usage(turn_data, usage)
                _record_spend(turn_data, usage)          # <- a Call

        which is the reviewer's evasion applied one level up, and it
        passed a blacklist version of this test.  So the closure is
        allowed EXACTLY two statements and anything else fails, whatever
        its node type.  If the closure ever legitimately needs to grow,
        that should be a deliberate edit here, not a silent one there.
        """
        closure = next(
            n for n in self._streaming_callbacks()
            if n.name == 'usage_callback_with_turn_tracking')
        body = [n for n in closure.body
                if not (isinstance(n, ast.Expr)
                        and isinstance(n.value, ast.Constant))]  # docstring

        assert len(body) == 2, (
            f'the closure has {len(body)} statements; it must be exactly a '
            f'delegation to _track_streaming_usage plus the on_usage_update '
            f'pass-through, because anything else it does is invisible to '
            f'every other check in this file'
        )

        delegation, passthrough = body
        assert (isinstance(delegation, ast.Expr)
                and isinstance(delegation.value, ast.Call)
                and getattr(delegation.value.func, 'attr', None)
                == '_track_streaming_usage'), (
            'the closure no longer delegates to the method the effect test '
            'drives, so that test measures something the provider never calls'
        )
        assert isinstance(passthrough, ast.If), (
            'the second statement should be the guarded on_usage_update '
            'pass-through'
        )

    @pytest.mark.parametrize("form", ["=", "+=", ".update", ".setdefault"])
    def test_the_check_covers_every_write_form(self, form):
        """The guard above must reject all four, not just assignment.

        Parametrised so the guard's OWN coverage is asserted rather than
        assumed — each form is compiled into a stand-in callback and run
        through the same predicate the real check uses.
        """
        bodies = {
            "=": "turn_data['spend_cache_read'] = 1",
            "+=": "turn_data['spend_cache_read'] += 1",
            ".update": "turn_data.update({'spend_cache_read': 1})",
            ".setdefault": "turn_data.setdefault('spend_cache_read', 1)",
        }
        src = (
            "def usage_callback_with_turn_tracking(usage):\n"
            f"    {bodies[form]}\n"
        )
        cb = ast.parse(src).body[0]
        mentioned = {
            n.value for n in ast.walk(cb)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and n.value.startswith('spend_')
        }
        assert mentioned, f'the predicate does not detect the {form} form'


class TestTheParseFeedsTheChain:
    """Accumulating a figure nobody parses measures nothing.

    §5.4 sums the cache counts a provider reports.  If the provider does
    not report them, the sum is a well-tested zero -- which is exactly
    what OpenRouter did until #699: writes were billed at 1.25x and
    `cache_creation_tokens` was permanently `None`, invisible unless you
    cross-checked cost against published rates.

    One assertion, on the wire shape captured from a live call.  The
    provider's own tests cover dict/attr handling and the fallback; this
    pins the fact the accounting depends on.
    """

    def test_the_provider_reports_the_write_count_at_all(self):
        from shared.plugins.model_provider.openrouter.converters import (
            apply_cache_usage,
        )
        from jaato_sdk.plugins.model_provider.types import TokenUsage
        from types import SimpleNamespace

        usage = TokenUsage()
        # Captured 2026-08-29, anthropic/claude-sonnet-4.6, cold call.
        apply_cache_usage(
            SimpleNamespace(
                prompt_tokens=4412,
                prompt_tokens_details=SimpleNamespace(
                    cached_tokens=0, cache_write_tokens=4403),
            ),
            usage,
        )
        assert usage.cache_creation_tokens == 4403, (
            "the provider is not reporting cache writes, so every "
            "spend_cache_creation this module guards sums to nothing"
        )


class TestTheChainCarriesIt:
    """Every link, pinned per file AND per exit shape.

    Two vacuity traps here, both hit before this was correct.

    First: presence of the name is not enough.  In each of these files the
    name is also a PARAMETER of the function that receives it, so a link
    that accepts the value and then drops it reads identically to one that
    forwards it.  That is the exact state the cost chain was found in.

    Second, and the one that actually caught this suite out: asserting
    "the name appears as a keyword in SOME call" is still too loose when a
    file has more than one exit.  ``rpc.py`` both calls the hook AND
    builds the wire payload; deleting the payload keys left the hook call
    untouched, and the guard passed under a sabotage that would have
    dropped the field on the wire.  The anchor matched something, just not
    the thing under test.

    So each link is asserted at its OWN exit shape: a keyword in a call, a
    key in the wire dict literal, or a ``payload.get`` on the far side of
    the wire.  Verified by removing each in turn.
    """

    #: Every BILLED figure the chain must carry.  The prompt/output pair
    #: joined it in jaato #802: the session had accumulated
    #: ``spend_prompt`` / ``spend_output`` per response all along, but
    #: neither reached the wire, so a consumer had only the LAST
    #: response's ``prompt_tokens`` / ``output_tokens`` and summing those
    #: across turns undercounts for exactly the reason measured of
    #: ``total_tokens``.  Same chain, same links, same guard.
    FIELDS = ("spend_cache_read_tokens", "spend_cache_creation_tokens",
              "spend_prompt_tokens", "spend_output_tokens")

    @staticmethod
    def _tree(path):
        return ast.parse((ROOT / path).read_text(encoding="utf-8"))

    @classmethod
    def _keywords_of_calls_to(cls, path, func_name):
        """Keywords passed to calls of ONE named function.

        Not "keywords of any call in the file".  ``core.py`` both unpacks
        the payload into a hook call and hands the result to
        ``_build_usage``; a file-wide check is satisfied by either, so
        deleting the ``_build_usage`` arguments passed under it.  Verified:
        that sabotage was green until this became call-specific.
        """
        return {
            kw.arg
            for n in ast.walk(cls._tree(path)) if isinstance(n, ast.Call)
            if getattr(n.func, "attr", getattr(n.func, "id", None)) == func_name
            for kw in n.keywords if kw.arg
        }

    @classmethod
    def _dict_literal_keys(cls, path):
        return {
            k.value
            for n in ast.walk(cls._tree(path)) if isinstance(n, ast.Dict)
            for k in n.keys
            if isinstance(k, ast.Constant) and isinstance(k.value, str)
        }

    @classmethod
    def _get_string_args(cls, path):
        """String literals passed to a ``.get(...)`` — the unpack side."""
        return {
            n.args[0].value
            for n in ast.walk(cls._tree(path)) if isinstance(n, ast.Call)
            if isinstance(n.func, ast.Attribute) and n.func.attr == "get"
            and n.args and isinstance(n.args[0], ast.Constant)
            and isinstance(n.args[0].value, str)
        }

    @pytest.mark.parametrize("path,what", [
        ("jaato-server/shared/jaato_client.py", "the facade driver"),
        ("jaato-server/shared/plugins/subagent/plugin.py",
         "in-process subagents"),
    ])
    def test_the_hook_callers_forward_it(self, path, what):
        forwarded = self._keywords_of_calls_to(
            path, "on_agent_turn_completed")
        for field in self.FIELDS:
            assert field in forwarded, (
                f"{what} does not pass {field} to on_agent_turn_completed"
            )

    def test_the_runner_puts_it_on_the_wire(self):
        """``rpc.py``'s payload dict — its own exit, distinct from the
        hook call in the same file that an any-call check would match."""
        keys = self._dict_literal_keys("jaato-server/server/runner/rpc.py")
        for field in self.FIELDS:
            assert field in keys, (
                f"the runner's RPC payload does not carry {field}; it is "
                f"accepted by the hook and dropped before the wire"
            )

    def test_the_runner_reads_it_from_turn_accounting(self):
        """The other end of the same file: the session's turn dict."""
        reads = self._get_string_args("jaato-server/server/runner/rpc.py")
        # The accounting keys are NOT the wire names — the session's turn
        # dict predates them — so these are spelled out rather than derived.
        for key in ("spend_cache_read", "spend_cache_creation",
                    "spend_prompt", "spend_output"):
            assert key in reads, (
                f"the runner never reads {key!r} out of turn accounting, so "
                f"the wire field it feeds is always None"
            )

    def test_the_daemon_unpacks_it_from_the_wire(self):
        keys = self._get_string_args("jaato-server/server/core.py")
        for field in self.FIELDS:
            assert field in keys, (
                f"the daemon never reads {field} off the payload"
            )

    def test_the_daemon_passes_it_to_the_usage_builder(self):
        forwarded = self._keywords_of_calls_to(
            "jaato-server/server/core.py", "_build_usage")
        for field in self.FIELDS:
            assert field in forwarded, (
                f"the daemon unpacks {field} and never gives it to "
                f"_build_usage"
            )

    def test_the_event_declares_the_fields(self):
        """The terminal link: what a consumer actually reads."""
        from jaato_sdk.events import UsageBreakdown

        usage = UsageBreakdown(
            prompt_tokens=1, output_tokens=1, total_tokens=2,
            spend_cache_read_tokens=58_000,
            spend_cache_creation_tokens=1_024,
            spend_prompt_tokens=91_000,
            spend_output_tokens=2_048,
        )
        assert usage.spend_cache_read_tokens == 58_000
        assert usage.spend_cache_creation_tokens == 1_024
        # The billed split, distinct from prompt_tokens/output_tokens above,
        # which are this turn's LAST response only.
        assert usage.spend_prompt_tokens == 91_000
        assert usage.spend_output_tokens == 2_048
        assert usage.prompt_tokens == 1

    def test_the_fields_default_to_none_not_zero(self):
        from jaato_sdk.events import UsageBreakdown

        usage = UsageBreakdown(
            prompt_tokens=1, output_tokens=1, total_tokens=2)
        for field in self.FIELDS:
            assert getattr(usage, field) is None, (
                f"{field} defaults to 0, which claims the provider reported "
                f"none spent rather than reported nothing"
            )


class TestTierAttribution:

    @staticmethod
    def _tier_session(tiers, initial):
        s = JaatoSession.__new__(JaatoSession)
        s._turn_index = 3
        s._cache_plugin = None
        s._llm_span_attributes = {}
        s._tier_config = ModelTierConfig(
            tiers=tiers, initial_tier=initial, tier_fallback=initial)
        s._active_tier = initial
        s._tier_switch_count = 0
        s._tier_cache_rewire_failures = 0
        s._tier_reliability_retarget_failures = 0
        s._model_name = tiers[initial].model
        s._active_provider_name = 'anthropic'
        s._provider = MagicMock()
        s._provider.name = 'anthropic'
        s._provider.model_name = tiers[initial].model

        def _connect(m, **kw):
            s._provider.model_name = m

        s._provider.connect = _connect
        s._runtime = MagicMock()
        s._runtime.reliability_plugin = None
        s._wire_cache_plugin = lambda: None
        s._trace = lambda *a, **k: None
        return s

    def test_the_span_names_the_active_tier(self):
        s = self._tier_session({
            'dispatcher': TierEntry(model='claude-sonnet-4-5'),
            'executor': TierEntry(model='claude-haiku-4-5'),
        }, 'dispatcher')

        assert s._build_llm_span_attributes()['jaato.tier'] == 'dispatcher'

        s.switch_tier('executor')
        attrs = s._build_llm_span_attributes()
        assert attrs['jaato.tier'] == 'executor'
        assert attrs['jaato.tier.switches'] == 1

    def test_a_no_op_enter_tier_is_not_a_switch(self):
        """Re-entering the tier you are in costs nothing and must not
        inflate the count -- the count is the multiplier on tier mode's
        cost, so an inflated one argues against a feature that was free."""
        s = self._tier_session({
            'dispatcher': TierEntry(model='claude-sonnet-4-5'),
        }, 'dispatcher')

        result = s.switch_tier('dispatcher')

        assert result['status'] == 'already_at_tier'
        assert s._build_llm_span_attributes()['jaato.tier.switches'] == 0

    def test_a_budget_rebind_counts_as_a_switch(self):
        """It re-reads the prefix cold exactly as enter_tier does, and the
        tier NAME never changes -- so counting in ``switch_tier`` would
        have missed it."""
        s = self._tier_session({
            'planner': TierEntry(model='claude-opus-4-7'),
        }, 'planner')

        s._tier_config.tiers['planner'] = TierEntry(model='claude-haiku-4-5')
        s._reconnect_active_tier_if_rebound()

        assert s._build_llm_span_attributes()['jaato.tier.switches'] == 1

    def test_degraded_bookkeeping_is_visible_on_the_span(self):
        """The post-connect blocks cannot raise, so they must be counted.

        A cache plugin that fails to re-attach leaves the session running
        UNCACHED — a cost regression; a failed reliability retarget judges
        patterns against the wrong model — a correctness one.  Both are
        swallowed by design (the provider is already re-pointed; raising
        would leave the switch half-applied), which is exactly why they
        need a channel that is not a log line.
        """
        s = self._tier_session({
            'dispatcher': TierEntry(model='claude-sonnet-4-5'),
            'executor': TierEntry(model='claude-haiku-4-5'),
        }, 'dispatcher')

        def _boom():
            raise RuntimeError('entry points unreadable')

        s._wire_cache_plugin = _boom
        s._runtime.reliability_plugin = MagicMock()
        s._runtime.reliability_plugin.set_model_context.side_effect = (
            RuntimeError('detector gone'))

        result = s.switch_tier('executor')

        assert result['status'] == 'switched'      # still non-fatal
        attrs = s._build_llm_span_attributes()
        assert attrs['jaato.tier.cache_rewire_failures'] == 1
        assert attrs['jaato.tier.reliability_retarget_failures'] == 1

    def test_a_healthy_session_reports_zero_not_absence(self):
        """Present-and-zero, so ``> 0`` is a queryable condition and a
        consumer can tell a healthy span from an older build's."""
        s = self._tier_session({
            'dispatcher': TierEntry(model='claude-sonnet-4-5'),
            'executor': TierEntry(model='claude-haiku-4-5'),
        }, 'dispatcher')

        s.switch_tier('executor')
        attrs = s._build_llm_span_attributes()

        assert attrs['jaato.tier.cache_rewire_failures'] == 0
        assert attrs['jaato.tier.reliability_retarget_failures'] == 0

    def test_a_single_model_session_carries_no_tier_keys(self):
        """No dead attributes on the overwhelmingly common path."""
        s = JaatoSession.__new__(JaatoSession)
        s._turn_index = 0
        s._cache_plugin = None
        s._llm_span_attributes = {}
        s._active_tier = None

        attrs = s._build_llm_span_attributes()

        assert not [k for k in attrs if k.startswith('jaato.tier')]
